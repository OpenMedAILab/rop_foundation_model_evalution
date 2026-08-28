"""E4.2 标签效率(方案 §20):冻结特征 + 患者级分层子采样训练标签的线性探针。

口径与 E1 主基准一致(LODO 4 折、StandardScaler(fit subsampled train)+ LR balanced、
同超参),唯一变量 = 训练标签比例 f ∈ {1%, 5%, 10%, 25%, 50%, 100%}。
抽样单位 = 患者(数据集, patient_id):按患者阳性/阴性分层保留 f×n_patients 个患者,
保留其全部图像(标签效率在临床上以"多少患者需要标注"计,故用患者级口径);
每 (fold, f) 组合 5 个随机种子(seed 0–4),评估在完整测试折上进行。

输出:
  outputs/label_efficiency/{model}_label_efficiency.csv   逐 (fold, f, seed) 明细
  outputs/label_efficiency/all_models_label_efficiency.csv  聚合(mean ± std across 20 组)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import load_yaml, seed_everything  # noqa: E402
from neoropfm.eval.metrics import auroc, auprc  # noqa: E402
from neoropfm.train.probe import load_features  # noqa: E402

from neoropfm.common import parse_heldouts  # noqa: E402
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
N_SEEDS = 5


def patient_subsample(tr: pd.DataFrame, f: float, seed: int) -> pd.DataFrame:
    """训练折患者级分层子采样:保留 f×n_patients 个患者及其全部图像。

    患者标签 = 该患者任一阳性图像 → 阳性(保守口径);分层保持阳性/阴性患者比。
    f=1.0 时返回原表(无随机性)。
    """
    if f >= 1.0:
        return tr
    rng = np.random.default_rng(seed)
    pids = (tr["dataset"].astype(str) + "|" + tr["patient_id"].astype(str)).to_numpy()
    tr = tr.assign(_pid=pids)
    y_patient = tr.groupby("_pid")["strict_binary_label"].max()
    pos_pids = y_patient[y_patient == 1].index.to_numpy()
    neg_pids = y_patient[y_patient == 0].index.to_numpy()
    n_pos = max(1, int(round(f * len(pos_pids))))
    n_neg = max(1, int(round(f * len(neg_pids))))
    keep_pos = rng.choice(pos_pids, size=n_pos, replace=False)
    keep_neg = rng.choice(neg_pids, size=n_neg, replace=False)
    return tr[tr["_pid"].isin(np.concatenate([keep_pos, keep_neg]))].drop(columns="_pid")


def run_one(feat: np.ndarray, ids: list[str], heldout: str, f: float, seed: int,
            splits_dir: Path, lr_cfg: dict) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sp = pd.read_csv(splits_dir / f"lodo_test_{heldout}.csv").set_index("sample_id").loc[ids].reset_index()
    tr = patient_subsample(sp[sp["split"] == "train"], f, seed)
    te = sp[sp["split"] == "test"]
    itr = np.where(np.isin(np.array(ids), tr["sample_id"]))[0]
    ite = np.where(np.isin(np.array(ids), te["sample_id"]))[0]
    Xtr, ytr = feat[itr], tr["strict_binary_label"].to_numpy()
    Xte, yte = feat[ite], te["strict_binary_label"].to_numpy()

    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(
        class_weight=lr_cfg.get("class_weight", "balanced"),
        C=lr_cfg.get("C", 1.0),
        max_iter=lr_cfg.get("max_iter", 5000),
        random_state=seed,
    )
    lr.fit(scaler.transform(Xtr), ytr)
    pte = lr.predict_proba(scaler.transform(Xte))[:, 1]
    return {
        "heldout_dataset": heldout, "fraction": f, "seed": seed,
        "train_n": len(tr), "train_positive": int(ytr.sum()),
        "n_patients_kept": tr.assign(_p=(tr["dataset"] + "|" + tr["patient_id"].astype(str)))["_p"].nunique(),
        "test_n": len(yte), "test_positive": int(yte.sum()),
        "auroc": auroc(yte, pte), "auprc": auprc(yte, pte),
    }


def resolve_ckpt(model: str, held: str) -> str:
    """R3 修订:model 名带 _iso 后缀 → 按折读取隔离选中的 c*;否则原样返回。"""
    if model.endswith("_iso"):
        base = model[: -len("_iso")]
        sel = pd.read_csv(
            Path(__file__).resolve().parents[3] / "outputs/checkpoint_iso"
            / f"{base}_iso" / f"selection_{held}.csv")
        return str(sel.loc[sel["selected"], "ckpt"].iloc[0])
    return model


def load_one(ckpt: str, cfg: dict) -> tuple[np.ndarray, list[str]]:
    # E1 六模型特征位置不一:本仓库 outputs/features 与 yesterday 缓存各存一部分;
    # 先按配置源读取,缺失则回退另一源
    try:
        return load_features(cfg["feature_source"], ckpt,
                             Path(cfg["cache_dir"]), Path(cfg["extract_dir"]))
    except FileNotFoundError:
        other = "cache" if cfg["feature_source"] == "extract" else "extract"
        print(f"[{ckpt}] 特征取自 {other} 源", flush=True)
        return load_features(other, ckpt,
                             Path(cfg["cache_dir"]), Path(cfg["extract_dir"]))


def run_model(model: str, cfg: dict, heldouts: list[str],
              fractions: list[float]) -> Path:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    feat_cache: dict[str, tuple[np.ndarray, list[str]]] = {}
    for held in heldouts:
        ckpt = resolve_ckpt(model, held)
        if ckpt not in feat_cache:
            feat_cache[ckpt] = load_one(ckpt, cfg)
        feat, ids = feat_cache[ckpt]
        for f in fractions:
            for seed in range(N_SEEDS):
                rows.append(run_one(feat, ids, held, f, seed,
                                    Path(cfg["splits_dir"]), cfg.get("lr", {})))
                print(f"[{model}] {held} f={f:.2f} seed={seed} "
                      f"auroc={rows[-1]['auroc']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    path = out_dir / f"{model}_label_efficiency.csv"
    df.to_csv(path, index=False)
    print(f"[{model}] saved → {path}", flush=True)
    return path


def aggregate(models: list[str], out_dir: Path) -> None:
    rows = []
    for model in models:
        df = pd.read_csv(out_dir / f"{model}_label_efficiency.csv")
        g = df.groupby("fraction")["auroc"].agg(["mean", "std"]).reset_index()
        g["model"] = model
        g["n_runs"] = df.groupby("fraction")["auroc"].count().values
        rows.append(g)
    all_df = pd.concat(rows)[["model", "fraction", "mean", "std", "n_runs"]]
    all_df = all_df.sort_values(["model", "fraction"])
    all_df.to_csv(out_dir / "all_models_label_efficiency.csv", index=False)
    print("聚合 →", out_dir / "all_models_label_efficiency.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None, help="单模型;缺省 = 全部 models")
    ap.add_argument("--heldouts", default=None,
                    help="逗号分隔折(默认 4 折;LOTO 敏感性只跑对角线折)")
    ap.add_argument("--fractions", default=None,
                    help="逗号分隔比例(默认 1/5/10/25/50/100%%);如 0.10,0.25,1.00")
    ap.add_argument("--no-aggregate", action="store_true")
    args = ap.parse_args()

    heldouts = parse_heldouts(args.heldouts)
    fractions = (FRACTIONS if args.fractions is None
                 else [float(x) for x in args.fractions.split(",")])
    cfg = load_yaml(args.config)
    models = [args.model] if args.model else cfg["models"]
    for model in models:
        run_model(model, cfg, heldouts, fractions)
    if not args.no_aggregate and not args.model:
        aggregate(models, Path(cfg["output_dir"]))


if __name__ == "__main__":
    main()
