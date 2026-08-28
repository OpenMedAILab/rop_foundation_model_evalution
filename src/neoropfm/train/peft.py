"""LoRA PEFT 微调(E1.7,6 模型统一协议 × 4 折 LODO)。

协议(与 frozen probe 完全同口径评估,仅适配策略不同):
- LoRA 施加于**所有 nn.Linear 与 1×1 Conv2d**(即 ViT 的注意力/MLP 投影、CNN 的
  pointwise 层;head 除外):r=8, alpha=16, dropout=0.1。这是"全线性层 LoRA"口径,
  对 ViT 与 CNN 一视同仁,便于论文方法一节用一句话描述。
- 优化:AdamW(lr=3e-4, wd=0.01, 余弦退火);BCEWithLogits,pos_weight=训练折 neg/pos。
- 早停:训练折内部 patient-level 分层 90/10 留出验证集,val AUROC 连续 3 轮不提升
  (Δ>1e-4)即停;保存最优可训练参数到 fold_{heldout}_best.pt。
- 评估:同 probe.py —— 工作点阈值**锁定在训练折**预测上,另给 test-optimal 参考列;
  每折预测落盘 outputs/peft/{model}/fold_{heldout}_predictions.csv(列名与 probe 完全
  一致,aggregate.py 可直接消费)。

运行:
  python -m neoropfm.train.peft --config configs/peft.yaml --model retfound_green
  python -m neoropfm.train.peft --config configs/peft.yaml --model all
  python -m neoropfm.train.peft --config configs/peft.yaml --model dinov2_vitb14 --limit 64 --epochs 1
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import MANIFEST_V1, SPLITS_DIR, gpu_free_mb, load_yaml, seed_everything  # noqa: E402
from neoropfm.eval.metrics import compute_all_metrics, threshold_at_sensitivity  # noqa: E402
from neoropfm.models.backbones import get_backbone  # noqa: E402

from neoropfm.common import HELDOUTS  # noqa: E402


# ---- LoRA 层(模块级类,DataLoader 多进程不直接接触,但保持可 pickle 卫生)----

class LoRALinear(nn.Module):
    def __init__(self, lin: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        for p in lin.parameters():
            p.requires_grad_(False)
        self.lin = lin
        self.r = r
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(lin.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, lin.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lin(x) + self.scale * self.lora_B(self.lora_A(self.dropout(x)))


class LoRAConv1x1(nn.Module):
    def __init__(self, conv: nn.Conv2d, r: int, alpha: int):
        super().__init__()
        for p in conv.parameters():
            p.requires_grad_(False)
        self.conv = conv
        self.scale = alpha / r
        self.lora_A = nn.Conv2d(conv.in_channels, r, 1, bias=False)
        self.lora_B = nn.Conv2d(r, conv.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.conv(x) + self.scale * self.lora_B(self.lora_A(x))


def apply_lora(model: nn.Module, r: int, alpha: int, dropout: float, scope: str = "all") -> nn.Module:
    """就地用 LoRA 包装 Linear/1×1 Conv2d(可训练参数仅 LoRA 与后续 head)。

    先收集目标再统一包装:遍历中 setattr 会改变模块树,而新包装器内含名为
    lin/conv 的原模块子节点,再被匹配 → 无限递归。
    scope: "all" = 全部 Linear 与 1×1 Conv2d;"qv" = 仅注意力 qkv/proj
    (ViT 经典 LoRA 口径,扰动最小;全层 LoRA 在低数据跨数据集下严重损害泛化,
    见 LORA_V1_POSTMORTEM)。
    """
    targets: list[tuple[nn.Module, str, str]] = []
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                full = f"{name}.{child_name}" if name else child_name
                if scope == "qv" and ".attn.qkv" not in full and ".attn.proj" not in full:
                    continue
                targets.append((module, child_name, "lin"))
            elif isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1):
                if scope == "qv":
                    continue
                targets.append((module, child_name, "conv"))
    for module, child_name, kind in targets:
        child = getattr(module, child_name)
        if isinstance(child, (LoRALinear, LoRAConv1x1)):  # 防御:已包装
            continue
        if kind == "lin":
            setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
        else:
            setattr(module, child_name, LoRAConv1x1(child, r, alpha))
    return model


class PeftClassifier(nn.Module):
    """backbone(已加 LoRA) + 线性头。backbone 前向输出 (B, feature_dim)。"""

    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


# ---- 数据 ----

class LabeledImages(Dataset):
    """(path, label) 列表 + 裸 Compose(可 pickle,multiprocessing 安全)。"""

    def __init__(self, items, transform):
        self.items = items  # [(path, y), ...]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), float(y)


def _loader(items, transform, batch, workers, shuffle, seed, drop_last=False):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        LabeledImages(items, transform),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        generator=gen if shuffle else None,
        drop_last=drop_last,
        persistent_workers=workers > 0,
    )


def _make_items(split_df: pd.DataFrame, manifest: pd.DataFrame) -> list[tuple[str, int]]:
    """split_df 的 sample_id 顺序 → (image_path, strict_binary_label)。"""
    paths = manifest.set_index("sample_id")["image_path"]
    return [
        (paths[sid], int(y))
        for sid, y in zip(split_df["sample_id"], split_df["strict_binary_label"])
    ]


def _stratified_sample(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """按 strict_binary_label 分层抽 limit 行(debug 口径,固定种子可复现)。

    替代 head():head() 的截断与样本在 CSV 中的物理顺序耦合,导致各类别在
    患者/数据集维度上分布失衡,见 run_fold 中 --limit 分支的注释。
    """
    rng = np.random.RandomState(seed)
    picks = []
    for _, g in df.groupby("strict_binary_label", sort=False):
        n = max(1, int(round(limit * len(g) / len(df))))
        picks.append(g.sample(n=min(n, len(g)), random_state=rng))
    return pd.concat(picks).reset_index(drop=True)


def _val_split(train_df: pd.DataFrame, val_frac: float, seed: int):
    """训练折内部 patient-level 分层 90/10(复合患者键防跨数据集 id 碰撞)。"""
    train_df = train_df.copy()
    train_df["_unit"] = train_df["dataset"].astype(str) + "|" + train_df["patient_id"].astype(str)
    units = train_df.drop_duplicates("_unit").reset_index(drop=True)
    units["_y"] = train_df.groupby("_unit")["strict_binary_label"].max().loc[units["_unit"]].to_numpy()
    rng = np.random.RandomState(seed)
    val_units = set()
    for cls in [0, 1]:
        cls_units = units[units["_y"] == cls]["_unit"].tolist()
        if not cls_units:  # 极小切片/极端不平衡时某类无患者
            continue
        n_val = max(1, int(round(len(cls_units) * val_frac)))
        val_units.update(rng.choice(cls_units, size=n_val, replace=False))
    is_val = train_df["_unit"].isin(val_units)
    return train_df[~is_val], train_df[is_val]


# ---- 训练 ----

@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ps, ys = [], []
    for xb, yb in loader:
        ps.append(torch.sigmoid(model(xb.to(device))).float().cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def _train_fold(model: nn.Module, device, tr_loader, val_loader, cfg, seed, pos_weight):
    from sklearn.metrics import roc_auc_score

    epochs = cfg["epochs"]
    patience = cfg.get("patience", 3)
    # 参数分组:新随机初始化的线性头用更高的 lr(head_lr,默认 10×),LoRA 分支用 lr。
    # 第一版协议全体 lr=3e-4,头部 10 epoch 内远未收敛(训练子集 AUROC 仅 0.76 vs
    # probe 0.9+),是 LoRA 结果远差于冻结 probe 的主因(见 scripts/debug_lora_lr.py)。
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    # head-only 消融时 body 为空:AdamW 遇空参数组直接报错,需过滤
    groups = [{"params": head, "lr": cfg.get("head_lr", 10 * cfg["lr"])}]
    if body:
        groups.append({"params": body, "lr": cfg["lr"]})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.get("weight_decay", 0.01))
    total_steps = epochs * len(tr_loader)
    # 分档调度(替代共享 CosineAnnealingLR,v2.1 修复):
    # 调试定档(F:head_lr 1e-2×20ep=0.8704)时总步数仅 640(32 步/ep);真实折
    # 420 步/ep × 20ep = 8400 步,共享 cosine 让头部以 ≈1e-2 多走了 ~13 倍步数
    # → 头部发散(farfum 全折 test 0.5657 ≈ 调试 B 配置 0.567,训练集仅 0.814)。
    # 修复:head 的高 lr 预算按**绝对步数**给(head_steps,默认 640 = 调试口径),
    # 之后压到地板 1e-6;LoRA 分支照旧全程 cosine。这样头部轨迹与调试定档一致,
    # 与总步数/批大小无关。
    head_steps = cfg.get("head_steps", 640)
    head_lr = cfg.get("head_lr", 10 * cfg["lr"])
    body_lr = cfg["lr"]
    head_ids = {id(p) for p in head}
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    best_auc, best_state, bad = -1.0, None, 0
    step = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            body = body_lr * 0.5 * (1 + math.cos(math.pi * min(step, total_steps - 1) / total_steps)) + 1e-7
            head = head_lr * 0.5 * (1 + math.cos(math.pi * min(step, head_steps - 1) / head_steps)) + 1e-6
            for g in opt.param_groups:
                g["lr"] = head if any(id(p) in head_ids for p in g["params"]) else body
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            step += 1
        pv, yv = _predict(model, val_loader, device)
        # sklearn ≥1.9 对单类 y_true 返回 nan(仅告警不抛异常),NaN 会静默吞掉
        # best 分支;统一按 0.5 处理(正式运行分层后基本不会单类,防御为主)。
        auc = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else 0.5
        if not np.isfinite(auc):
            auc = 0.5
        print(f"  epoch {ep}: loss={loss.item():.4f} val_auc={auc:.4f} (best {best_auc:.4f})")
        if auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("训练未产生有效状态(val AUC 全程 NaN/无提升,检查数据与损失)")
    model.load_state_dict(best_state)
    return best_auc


def run_fold(model_key: str, heldout: str, cfg: dict, seed: int, limit: int | None):
    spec = get_backbone(model_key)
    device = cfg["device"]
    sp = pd.read_csv(Path(cfg["splits_dir"]) / f"lodo_test_{heldout}.csv")
    sp = sp.set_index("sample_id")
    manifest = pd.read_csv(MANIFEST_V1)
    manifest = manifest[manifest["include_strict_binary"] == 1]

    tr = sp[sp["split"] == "train"].reset_index()
    te = sp[sp["split"] == "test"].reset_index()
    if limit:
        # head() 按 sample_id 排序截断,会把阳性集中到个别患者(farfum 折前 2000 行
        # 全部 470 个阳性都属于患者 ridirp|043):患者级 90/10 验证切分必然把这唯一
        # 阳性单元整个划给验证集 → 训练集 0 阳性 → loss=0、预测全 0 的假性崩溃
        # (2026-08-20 v2.1 验证教训)。改为按标签分层抽样,阳性来自多位患者,
        # 验证切分后训练集仍含阳性。
        tr = _stratified_sample(tr, limit, seed)
        te = te.head(limit)
    tr_fit, tr_val = _val_split(tr, cfg.get("val_frac", 0.1), seed)

    tr_cfg = cfg["train"]
    batch = tr_cfg.get("batch_size_by_model", {}).get(model_key, tr_cfg["batch_size"])
    compose = spec.compose
    tr_loader = _loader(_make_items(tr_fit, manifest), compose, batch, cfg["num_workers"], True, seed)
    val_loader = _loader(_make_items(tr_val, manifest), compose, batch, cfg["num_workers"], False, seed)
    te_loader = _loader(_make_items(te, manifest), compose, batch, cfg["num_workers"], False, seed)

    backbone = spec.build(device)
    if cfg.get("full", False):
        # E1.8 全量微调:不做 LoRA,全部参数可训练(仅小模型使用)
        for p in backbone.parameters():
            p.requires_grad_(True)
    elif cfg.get("head_only", False):
        # 消融:骨干完全冻结,仅训练线性头——隔离"LoRA 分支"与"头部训练协议"
        # 对跨数据集泛化的各自贡献(LoRA 诊断用,非正式协议)。
        for p in backbone.parameters():
            p.requires_grad_(False)
    else:
        lora_cfg = dict(cfg["lora"])
        apply_lora(
            backbone,
            r=lora_cfg["r"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            scope=lora_cfg.get("scope", "all"),
        )
    model = PeftClassifier(backbone, spec.feature_dim).to(device)

    n_pos = int((tr_fit["strict_binary_label"] == 1).sum())
    n_neg = len(tr_fit) - n_pos
    pos_weight = max(n_neg, 1) / max(n_pos, 1)
    val_auc = _train_fold(model, device, tr_loader, val_loader, cfg["train"], seed, pos_weight)

    ptr, ytr = _predict(model, _loader(_make_items(tr, manifest), compose, batch, cfg["num_workers"], False, seed), device)
    pte, yte = _predict(model, te_loader, device)

    locked = {
        "sens95": threshold_at_sensitivity(ytr, ptr, 0.95),
        "sens98": threshold_at_sensitivity(ytr, ptr, 0.98),
    }
    test_opt = {
        "sens95": threshold_at_sensitivity(yte, pte, 0.95),
        "sens98": threshold_at_sensitivity(yte, pte, 0.98),
    }
    m = compute_all_metrics(yte, pte, thresholds=locked)
    m_opt = compute_all_metrics(yte, pte, thresholds=test_opt)

    row = {
        "heldout_dataset": heldout,
        "train_n": len(tr), "test_n": len(te),
        "train_negative": int((ytr == 0).sum()), "train_positive": int((ytr == 1).sum()),
        "test_negative": int((yte == 0).sum()), "test_positive": int((yte == 1).sum()),
        "val_auroc": val_auc,
        "auroc": m["auroc"], "auprc": m["auprc"],
        "brier": m["brier"], "ece": m["ece"],
        "cal_intercept": m["cal_intercept"], "cal_slope": m["cal_slope"],
        "spec@95sens_trainlocked": m["spec@sens95"],
        "sens@95sens_trainlocked": m["sens@sens95"],
        "spec@98sens_trainlocked": m["spec@sens98"],
        "sens@98sens_trainlocked": m["sens@sens98"],
        "spec@95sens_testopt_ref": m_opt["spec@sens95"],
        "spec@98sens_testopt_ref": m_opt["spec@sens98"],
        "threshold_sens95_locked": locked["sens95"],
        "threshold_sens98_locked": locked["sens98"],
    }
    fold_df = pd.DataFrame({
        "sample_id": tr["sample_id"].tolist() + te["sample_id"].tolist(),
        "patient_id": tr["patient_id"].tolist() + te["patient_id"].tolist(),
        "dataset": tr["dataset"].tolist() + te["dataset"].tolist(),
        "heldout_dataset": heldout,
        "subset": ["train"] * len(tr) + ["test"] * len(te),
        "y": np.concatenate([ytr, yte]),
        "p": np.concatenate([ptr, pte]),
    })
    return row, fold_df


def run_model(
    model_key: str, cfg: dict, limit: int | None = None, epochs: int | None = None,
    check_gpu_free: int | None = None,
) -> Path:
    out_dir = Path(cfg["output_dir"]) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 0)
    seed_everything(seed)
    if epochs is not None:
        cfg["train"] = {**cfg["train"], "epochs": epochs}

    device = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("config 要求 GPU 但当前不可用")
    min_free = cfg["check_gpu_free_mb"] if check_gpu_free is None else check_gpu_free
    if device.startswith("cuda") and min_free:
        ok, free = gpu_free_mb(min_free)
        if not ok:
            raise RuntimeError(f"GPU 空闲显存 {free}MB < 要求 {min_free}MB;由 gpu_queue.sh 排队重试")

    mode = "full-FT" if cfg.get("full", False) else f"LoRA r={cfg['lora']['r']}"
    print(f"[{model_key}] {mode} epochs={cfg['train']['epochs']} device={device}")
    rows = []
    for held in HELDOUTS:
        row, fold_df = run_fold(model_key, held, cfg, seed, limit)
        rows.append(row)
        fold_df.to_csv(out_dir / f"fold_{held}_predictions.csv", index=False)
        print(
            f"  [{held}] val_auc={row['val_auroc']:.4f} auroc={row['auroc']:.4f} "
            f"auprc={row['auprc']:.4f} spec@95(锁)={row['spec@95sens_trainlocked']:.4f} "
            f"spec@95(opt)={row['spec@95sens_testopt_ref']:.4f}"
        )
    stem = "fullft" if cfg.get("full", False) else "peft"
    out = pd.DataFrame(rows)
    out["model"] = model_key
    out.to_csv(out_dir / f"{model_key}_{stem}_lodo_metrics.csv", index=False)
    print(f"saved: {out_dir / f'{model_key}_{stem}_lodo_metrics.csv'}")
    return out_dir / f"{model_key}_{stem}_lodo_metrics.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="all")
    ap.add_argument("--limit", type=int, default=None, help="每折每子集仅用前 N 条(冒烟测试)")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epochs(冒烟测试)")
    ap.add_argument("--check-gpu-free", type=int, default=None,
                    help="覆盖配置 check_gpu_free_mb(调试小模型时放宽门禁;0=跳过检查)")
    ap.add_argument("--head-only", action="store_true",
                    help="消融:骨干完全冻结仅训练线性头(隔离 LoRA 分支影响)")
    ap.add_argument("--output-dir", default=None,
                    help="覆盖配置 output_dir(消融/调试输出隔离)")
    ap.add_argument("--lora-scope", default=None, choices=["all", "qv"],
                    help="覆盖 lora.scope(qv = 仅注意力 qkv/proj,扰动最小)")
    ap.add_argument("--lora-alpha", type=int, default=None,
                    help="覆盖 lora.alpha(scale=alpha/r,减半即扰动减半)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    cfg.setdefault("splits_dir", str(SPLITS_DIR))
    if args.head_only:
        cfg["head_only"] = True
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.lora_scope:
        cfg["lora"]["scope"] = args.lora_scope
    if args.lora_alpha:
        cfg["lora"]["alpha"] = args.lora_alpha
    models = list(cfg["models"]) if args.model == "all" else [args.model]
    for m in models:
        run_model(m, cfg, limit=args.limit, epochs=args.epochs,
                  check_gpu_free=args.check_gpu_free)
    print("done.")


if __name__ == "__main__":
    main()
