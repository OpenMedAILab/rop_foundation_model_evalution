"""E2 新生儿继续自监督预训练主脚本(iBOT / MAE 双路线,方案 §8.2)。

iBOT 路线(DINOv2-S/14 起点,主):动量教师 + 多裁剪(2×224 全局 + 2×96 局部),
DINO 全局 CLS 蒸馏(排除同视图配对,官方口径)+ iBOT 块状掩码 patch 自蒸馏;
可选 PMA 回归 / visit-consistency 辅助头(§8.3/§8.4,E2.4 消融开关)。

MAE 路线(RETFound-Green 起点,备选):75% 随机掩码,编码器只吃可见 patch +
cls/reg,轻量解码器重构被掩 patch 的 per-patch 归一化像素(MAE 原文)。

两条路线都只保存 **裸 backbone** 权重到 ckpt_ep{N:03d}.pth(不含 SSL 头/解码器),
之后用 backbones.get_ssl_backbone() 挂回 E1 的 frozen probe 管线评估(E2.5)。

运行:
  python -m neoropfm.train.ssl_pretrain --config configs/ssl_ibot_dinov2.yaml
  # 冒烟测试(CPU/小占用):
  python -m neoropfm.train.ssl_pretrain --config configs/ssl_ibot_dinov2.yaml \
      --device cpu --limit 16 --max-steps 3 --batch-size 4 --accum 1 --no-amp
"""
from __future__ import annotations

import argparse
import contextlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import REPO_DIR, gpu_free_mb, load_yaml, seed_everything  # noqa: E402
from neoropfm.ssl.augment import MAETransform, MultiCropSSL  # noqa: E402
from neoropfm.ssl.dino import (  # noqa: E402
    BlockwiseMask, Center, DINOHead, dino_total_loss, iBOTHead, ibot_loss,
)
from neoropfm.ssl.heads import PMAHead, pma_loss, visit_consistency_loss  # noqa: E402
from neoropfm.ssl.mae import MAEModel  # noqa: E402

# iBOT 起点:timm DINOv2-S/14 lvd142m(与 E1 dinov2_vits14 同一权重)
IBOT_BASE_TIMID = "vit_small_patch14_dinov2.lvd142m"


# ---- 数据 ----

class SSLImages(Dataset):
    """无标签语料:(image_path, pma_z, visit_idx)。

    pma_z 为发育年龄 z-score(NaN = 缺失,PMA 头按掩码处理);visit_idx 为
    (dataset, patient_id) 复合键的整数编码(visit-consistency 头用)。
    """

    def __init__(self, items, transform, multi_crop: bool):
        # items: [(path, pma_z, visit_idx), ...]
        self.items = items
        self.transform = transform
        self.multi_crop = multi_crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, pma_z, visit = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), float(pma_z), int(visit)


def _collate_multi(batch):
    """多裁剪批次:按裁剪序号组 batch → ([g1, g2, l1, l2], pma_z, visit)。"""
    n_crops = len(batch[0][0])
    tensors = [torch.stack([b[0][i] for b in batch]) for i in range(n_crops)]
    pma = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    visit = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return tensors, pma, visit


# ---- iBOT 学生/教师 ----

class IBOTStudent(nn.Module):
    """backbone + 掩码 token + 投影头(可 EMA 至 TeacherBranch 的部分按同名对齐)。"""

    def __init__(self, backbone: nn.Module, mask_token: nn.Parameter,
                 dino_head: DINOHead, ibot_head: iBOTHead, pma_head: PMAHead | None):
        super().__init__()
        self.backbone = backbone
        self.mask_token = mask_token
        self.dino_head = dino_head
        self.ibot_head = ibot_head
        self.pma_head = pma_head  # None = 不挂(E2.4 消融开关)

    @property
    def prefix(self) -> int:
        return self.backbone.num_prefix_tokens

    def forward_masked(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """timm forward_features 等价流程,在 pos-embed 后把被掩 patch 换成 mask_token。

        mask: (B, N) bool,True = 掩码。返回 (B, prefix+N, D)。
        """
        b = self.backbone
        emb = b.patch_embed(x)
        emb = b._pos_embed(emb)
        emb = b.patch_drop(emb)
        emb = b.norm_pre(emb)
        m = mask.unsqueeze(-1)  # (B, N, 1)
        emb[:, self.prefix:] = torch.where(m, self.mask_token, emb[:, self.prefix:])
        for blk in b.blocks:
            emb = blk(emb)
        return b.norm(emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)


class TeacherBranch(nn.Module):
    """教师:backbone + 两个投影头(与学生的对应部分同名,供 EMA)。"""

    def __init__(self, backbone: nn.Module, dino_head: DINOHead, ibot_head: iBOTHead):
        super().__init__()
        self.backbone = backbone
        self.dino_head = dino_head
        self.ibot_head = ibot_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)


# ---- 调度器 ----

def _lr_at(step: int, warmup_steps: int, total_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(warmup_steps, 1)
    t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * t))


def _wd_at(step: int, total_steps: int, wd: float, wd_end: float) -> float:
    t = min(step / max(total_steps - 1, 1), 1.0)
    return wd_end + 0.5 * (wd - wd_end) * (1 + math.cos(math.pi * t))


def _make_optimizer(model: nn.Module, lr: float, wd: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "mask_token" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def _ema_by_name(student: nn.Module, teacher: nn.Module, m: float, init: bool = False) -> None:
    """按参数名对齐做 EMA(学生含教师没有的 mask_token/pma_head,只动同名部分)。"""
    t_named = dict(teacher.named_parameters())
    with torch.no_grad():
        for name, ps in student.named_parameters():
            if name not in t_named:
                continue
            if init:
                t_named[name].data.copy_(ps.data)
            else:
                t_named[name].data.mul_(m).add_(ps.data, alpha=1.0 - m)


# ---- 训练循环 ----

def _load_items(cfg: dict, limit: int | None) -> tuple[list, float, float]:
    """读 SSL manifest → [(path, pma_z, visit_idx)];返回 PMA 统计(全局 z-score)。"""
    df = pd.read_csv(REPO_DIR / cfg["manifest"])
    if limit:
        df = df.head(limit)
    pma = pd.to_numeric(df["pma"], errors="coerce")
    mean, std = float(pma.mean()), float(pma.std())
    visit_ids = (df["dataset"].astype(str) + "|" + df["patient_id"].astype(str))
    visit_map = {v: i for i, v in enumerate(visit_ids.unique())}
    items = [
        (p, (pma.iloc[i] - mean) / std if np.isfinite(pma.iloc[i]) else float("nan"),
         visit_map[v])
        for i, (p, v) in enumerate(zip(df["image_path"], visit_ids))
    ]
    return items, mean, std


def _run_ibot(cfg: dict, items: list, out_dir: Path, device: str, limit, steps_override, batch_override, accum_override, amp: bool, log_every: int):
    import timm

    t_cfg = cfg["train"]
    d_cfg = cfg["data"]
    h_cfg = cfg["heads"]
    ib = cfg["ibot"]
    B = batch_override if batch_override is not None else t_cfg["batch_size"]
    accum = accum_override if accum_override is not None else t_cfg["accum_steps"]
    n_local = d_cfg["n_local"]

    ds = SSLImages(items, MultiCropSSL(
        global_size=d_cfg["global_size"], local_size=d_cfg["local_size"],
        n_global=d_cfg["n_global"], n_local=n_local,
    ), multi_crop=True)
    loader = DataLoader(ds, batch_size=B, shuffle=True, num_workers=d_cfg["num_workers"],
                        collate_fn=_collate_multi, drop_last=True,
                        persistent_workers=d_cfg["num_workers"] > 0)

    # 学生/教师 backbone(同一 pretrained 起点;教师随后 EMA 初始化)。
    # dynamic_img_size:多裁剪的 96² 局部视图 grid=96//14=6 ≠ 224 的 16,
    # timm 按输入尺寸动态插值 pos_embed(官方 DINO 多裁剪同款做法)。
    stu_bb = timm.create_model(IBOT_BASE_TIMID, pretrained=True, img_size=d_cfg["global_size"],
                               num_classes=0, dynamic_img_size=True)
    tch_bb = timm.create_model(IBOT_BASE_TIMID, pretrained=True, img_size=d_cfg["global_size"],
                               num_classes=0, dynamic_img_size=True)
    emb_dim = stu_bb.embed_dim
    mask_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
    nn.init.normal_(mask_token, std=0.02)
    pma_head = PMAHead(emb_dim) if h_cfg["pma_w"] > 0 else None
    student = IBOTStudent(stu_bb, mask_token, DINOHead(emb_dim, h_cfg["out_dim"], h_cfg["hidden"], h_cfg["bottleneck"]),
                          iBOTHead(emb_dim, h_cfg["out_dim"], h_cfg["patch_hidden"]), pma_head).to(device)
    teacher = TeacherBranch(tch_bb, DINOHead(emb_dim, h_cfg["out_dim"], h_cfg["hidden"], h_cfg["bottleneck"]),
                            iBOTHead(emb_dim, h_cfg["out_dim"], h_cfg["patch_hidden"])).to(device)
    _ema_by_name(student, teacher, t_cfg["momentum_teacher"], init=True)

    center_cls = Center(h_cfg["out_dim"], t_cfg["center_momentum"], device)
    center_patch = Center(h_cfg["out_dim"], t_cfg["center_momentum"], device)
    masker = BlockwiseMask(ib["mask_ratio"], tuple(ib["mask_aspect"]), cfg.get("seed", 0))

    opt = _make_optimizer(student, t_cfg["lr"], t_cfg["weight_decay"])
    steps_per_epoch = len(loader)
    epochs = t_cfg["max_epochs"]
    total_steps = min(epochs * steps_per_epoch, steps_override) if steps_override else epochs * steps_per_epoch
    warmup_steps = t_cfg["warmup_epochs"] * steps_per_epoch
    amp_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
               if amp and device.startswith("cuda") else contextlib.nullcontext())
    grad_enabled = torch.enable_grad()

    print(f"[ibot] {len(ds)} 张 × {epochs}ep,bs={B}×accum={accum}(有效 {B * accum}),"
          f"steps={total_steps},lr={t_cfg['lr']},out_dim={h_cfg['out_dim']}")
    log_rows = []
    step = 0
    for ep in range(epochs):
        if step >= total_steps:
            break
        if ep >= t_cfg["freeze_last_layer_epochs"]:
            student.dino_head.unfreeze_last_layer()
        for crops, pma_z, visit in loader:
            if step >= total_steps:
                break
            batch_g = torch.cat(crops[:d_cfg["n_global"]], dim=0).to(device)  # (2B,3,224,224)
            batch_l = torch.cat(crops[d_cfg["n_global"]:], dim=0).to(device)  # (2B,3,96,96)
            pma_z, visit = pma_z.to(device), visit.to(device)
            # 负对照:batch 内打乱,破坏图像↔PMA / 图像↔身份配对(保留分布)
            if h_cfg.get("pma_shuffle"):
                pma_z = pma_z[torch.randperm(pma_z.size(0), device=pma_z.device)]
            if h_cfg.get("visit_shuffle"):
                visit = visit[torch.randperm(visit.size(0), device=visit.device)]
            # visit id 扩展到全局视图(每图 2 个全局视图)
            visit_g = visit.repeat_interleave(d_cfg["n_global"])
            pma_g = pma_z.repeat_interleave(d_cfg["n_global"])

            mask_g = masker(batch_g.shape[0], batch_g.shape[-1] // stu_bb.patch_embed.patch_size[0]).to(device)

            lr = _lr_at(step, warmup_steps, total_steps, t_cfg["lr"], t_cfg["min_lr"])
            for g in opt.param_groups:
                g["lr"] = lr
                if g["weight_decay"]:
                    g["weight_decay"] = _wd_at(step, total_steps, t_cfg["weight_decay"], t_cfg["weight_decay_end"])

            with amp_ctx, grad_enabled:
                # 学生:全局视图掩码前向 + 局部视图
                toks_g = student.forward_masked(batch_g, mask_g)
                z_cls_g = student.dino_head(toks_g[:, 0])
                patch_masked = toks_g[:, student.prefix:][mask_g]
                z_patch_s = student.ibot_head(patch_masked)
                toks_l = student(batch_l)
                z_cls_l = student.dino_head(toks_l[:, 0])

                # 教师:全局视图(不掩码)
                with torch.no_grad():
                    toks_t = teacher(batch_g)
                    z_cls_t = teacher.dino_head(toks_t[:, 0])
                    z_patch_t_all = teacher.ibot_head(toks_t[:, teacher.backbone.num_prefix_tokens:])
                    z_patch_t = z_patch_t_all[mask_g]

                loss_cls, _ = dino_total_loss(
                    torch.cat([z_cls_g, z_cls_l], dim=0), z_cls_t,
                    t_cfg["teacher_temp"], t_cfg["student_temp"], center_cls.center, n_local,
                )
                loss_ibot, _ = ibot_loss(z_patch_s, z_patch_t, t_cfg["teacher_temp"],
                                         t_cfg["student_temp"], center_patch.center)
                total = loss_cls + loss_ibot
                loss_pma = loss_visit = torch.tensor(0.0, device=device)
                if pma_head is not None:
                    loss_pma = h_cfg["pma_w"] * pma_loss(pma_head(toks_g[:, 0]), pma_g, ~torch.isnan(pma_g))
                    total = total + loss_pma
                if h_cfg["visit_w"] > 0:
                    loss_visit = h_cfg["visit_w"] * visit_consistency_loss(z_cls_g, visit_g, h_cfg["visit_tau"])
                    total = total + loss_visit
                total = total / accum

            total.backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), t_cfg["clip_grad"])
                opt.step()
                opt.zero_grad(set_to_none=True)
                _ema_by_name(student, teacher, t_cfg["momentum_teacher"])
            with torch.no_grad():
                center_cls.update(z_cls_t)
                center_patch.update(z_patch_t)

            step += 1
            if step % log_every == 0 or step == total_steps:
                msg = (f"  ep{ep} step{step}/{total_steps} lr={lr:.2e} "
                       f"cls={loss_cls.item():.4f} ibot={loss_ibot.item():.4f} "
                       f"pma={loss_pma.item():.4f} visit={loss_visit.item():.4f}")
                print(msg)
                log_rows.append({"epoch": ep, "step": step, "lr": lr,
                                 "loss_cls": loss_cls.item(), "loss_ibot": loss_ibot.item(),
                                 "loss_pma": loss_pma.item(), "loss_visit": loss_visit.item()})

        if (ep + 1) % cfg["ckpt_every"] == 0 or step >= total_steps:
            ckpt = out_dir / f"ckpt_ep{ep + 1:03d}.pth"
            torch.save({"backbone": stu_bb.state_dict(), "epoch": ep + 1, "step": step,
                        "route": "ibot", "base_model": IBOT_BASE_TIMID}, ckpt)
            print(f"saved: {ckpt}")

    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)
    torch.save({"backbone": stu_bb.state_dict(), "epoch": ep + 1, "step": step,
                "route": "ibot", "base_model": IBOT_BASE_TIMID}, out_dir / "ckpt_final.pth")
    print(f"saved: {out_dir / 'ckpt_final.pth'} + train_log.csv")


def _run_mae(cfg: dict, items: list, out_dir: Path, device: str, steps_override, batch_override, accum_override, amp: bool, log_every: int):
    from neoropfm.models.backbones import _build_retfound_green

    t_cfg = cfg["train"]
    d_cfg = cfg["data"]
    m_cfg = cfg["mae"]
    B = batch_override if batch_override is not None else t_cfg["batch_size"]
    accum = accum_override if accum_override is not None else t_cfg["accum_steps"]

    ds = SSLImages(items, MAETransform(cfg["img_size"]), multi_crop=False)
    loader = DataLoader(ds, batch_size=B, shuffle=True, num_workers=d_cfg["num_workers"],
                        drop_last=True, persistent_workers=d_cfg["num_workers"] > 0)

    encoder = _build_retfound_green(cfg["img_size"])
    encoder.global_pool = ""  # MAE 需要全部 token(特征提取口径在 get_ssl_backbone 恢复)
    encoder.train()
    mae = MAEModel(encoder, prefix_tokens=encoder.num_prefix_tokens,
                   decoder_dim=m_cfg["decoder_dim"], decoder_depth=m_cfg["decoder_depth"],
                   decoder_heads=m_cfg["decoder_heads"]).to(device)

    opt = _make_optimizer(mae, t_cfg["lr"], t_cfg["weight_decay"])
    steps_per_epoch = len(loader)
    epochs = t_cfg["max_epochs"]
    total_steps = min(epochs * steps_per_epoch, steps_override) if steps_override else epochs * steps_per_epoch
    warmup_steps = t_cfg["warmup_epochs"] * steps_per_epoch
    amp_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
               if amp and device.startswith("cuda") else contextlib.nullcontext())

    print(f"[mae] {len(ds)} 张 × {epochs}ep,bs={B}×accum={accum},mask={m_cfg['mask_ratio']},"
          f"steps={total_steps},lr={t_cfg['lr']}")
    log_rows = []
    step = 0
    for ep in range(epochs):
        if step >= total_steps:
            break
        for x, _pma, _visit in loader:
            if step >= total_steps:
                break
            lr = _lr_at(step, warmup_steps, total_steps, t_cfg["lr"], t_cfg["min_lr"])
            for g in opt.param_groups:
                g["lr"] = lr
            with amp_ctx:
                loss, _mask = mae(x.to(device), m_cfg["mask_ratio"])
                loss = loss / accum
            loss.backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(mae.parameters(), t_cfg["clip_grad"])
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0 or step == total_steps:
                print(f"  ep{ep} step{step}/{total_steps} lr={lr:.2e} loss={loss.item():.4f}")
                log_rows.append({"epoch": ep, "step": step, "lr": lr, "loss": loss.item()})

        if (ep + 1) % cfg["ckpt_every"] == 0 or step >= total_steps:
            ckpt = out_dir / f"ckpt_ep{ep + 1:03d}.pth"
            torch.save({"backbone": encoder.state_dict(), "epoch": ep + 1, "step": step,
                        "route": "mae", "base_model": "retfound_green"}, ckpt)
            print(f"saved: {ckpt}")

    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)
    torch.save({"backbone": encoder.state_dict(), "epoch": ep + 1, "step": step,
                "route": "mae", "base_model": "retfound_green"}, out_dir / "ckpt_final.pth")
    print(f"saved: {out_dir / 'ckpt_final.pth'} + train_log.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="覆盖 config(冒烟测试用 cpu)")
    ap.add_argument("--limit", type=int, default=None, help="仅用前 N 张(冒烟测试)")
    ap.add_argument("--max-steps", type=int, default=None, help="覆盖总步数(冒烟测试)")
    ap.add_argument("--manifest", default=None, help="覆盖语料 manifest(E2.5 leave-test-out 用)")
    ap.add_argument("--output-dir", default=None, help="覆盖输出目录(E2.5 各折独立 run)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖 config 的训练种子(训练随机性报告用)")
    ap.add_argument("--pma-shuffle", action="store_true",
                    help="负对照:batch 内打乱 pma_z,破坏图像↔发育年龄配对")
    ap.add_argument("--visit-shuffle", action="store_true",
                    help="负对照:batch 内打乱 visit id,破坏图像↔身份配对")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.manifest:
        cfg["manifest"] = args.manifest
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.pma_shuffle or args.visit_shuffle:
        cfg.setdefault("heads", {})
        cfg["heads"]["pma_shuffle"] = bool(args.pma_shuffle)
        cfg["heads"]["visit_shuffle"] = bool(args.visit_shuffle)
    device = args.device or cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("config 要求 GPU 但当前不可用")
    min_free = cfg.get("check_gpu_free_mb", 0)
    if device.startswith("cuda") and min_free:
        ok, free = gpu_free_mb(min_free)
        if not ok:
            raise RuntimeError(f"GPU 空闲显存 {free}MB < 要求 {min_free}MB;由 gpu_queue.sh 排队重试")

    seed_everything(cfg.get("seed", 0))
    out_dir = REPO_DIR / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    items, pma_mean, pma_std = _load_items(cfg, args.limit)
    with open(out_dir / "run_info.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"config": str(args.config), "n_images": len(items),
                        "pma_mean_weeks": pma_mean, "pma_std_weeks": pma_std,
                        "seed": cfg.get("seed", 0),
                        "pma_shuffle": cfg.get("heads", {}).get("pma_shuffle", False),
                        "visit_shuffle": cfg.get("heads", {}).get("visit_shuffle", False)}, f)

    log_every = args.log_every or cfg.get("log_every", 50)
    amp = (not args.no_amp) and cfg["train"].get("amp", True)
    if cfg["route"] == "ibot":
        _run_ibot(cfg, items, out_dir, device, args.limit, args.max_steps,
                  args.batch_size, args.accum, amp, log_every)
    elif cfg["route"] == "mae":
        _run_mae(cfg, items, out_dir, device, args.max_steps,
                 args.batch_size, args.accum, amp, log_every)
    else:
        raise ValueError(f"未知 route {cfg['route']!r}(ibot|mae)")
    print("done.")


if __name__ == "__main__":
    main()
