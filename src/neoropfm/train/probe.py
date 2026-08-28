"""LODO 冻结特征线性探针(E1 主评估管线)。

协议(已用 yesterday 缓存特征逐格验证,14/16 格 AUROC 精确复现 ≤1e-4,其余 ≤1e-3):
- StandardScaler(fit on train) + LogisticRegression(class_weight="balanced", C=1.0)
- 筛查工作点(spec@sens)阈值**锁定在训练折**(方案 §14/§24 要求,严格 > 语义);
  另输出 test-optimal 参考列与 yesterday 报告口径保持一致(该口径乐观,仅作对照)。
- 每折预测落盘 outputs/probes/{model}/fold_{heldout}_predictions.csv,
  统计阶段(aggregate.py: bootstrap/DeLong/校准)零重算。

运行:
  python -m neoropfm.train.probe --config configs/probe.yaml            # 全部模型
  python -m neoropfm.train.probe --config configs/probe.yaml --model retfound_mae_cfp
  python -m neoropfm.train.probe --config configs/probe.yaml --model all --compare <ref.csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import SPLITS_DIR, load_yaml, parse_heldouts, seed_everything  # noqa: E402
from neoropfm.eval.metrics import compute_all_metrics, threshold_at_sensitivity  # noqa: E402


def load_features(source: str, model: str, cache_dir: Path, extract_dir: Path):
    """按配置加载特征与 sample_ids。source=cache 读 yesterday 缓存,extract 读本仓库提取缓存。"""
    if source == "cache":
        d = Path(cache_dir) / model
        feat = np.load(d / f"{model}_features.npy")
        ids = pd.read_csv(d / f"{model}_sample_ids.csv")["sample_id"].tolist()
    elif source == "extract":
        d = Path(extract_dir) / model
        feat = np.load(d / f"{model}_features.npy")
        ids = pd.read_csv(d / f"{model}_sample_ids.csv")["sample_id"].tolist()
    else:
        raise ValueError(f"unknown feature_source: {source}")
    return feat, ids


def run_fold(feat, ids, heldout: str, splits_dir: Path, lr_cfg: dict, seed: int):
    """单折:训练探针、锁定阈值、测试评估。返回 (metrics_row, fold_predictions_df)。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sp = pd.read_csv(Path(splits_dir) / f"lodo_test_{heldout}.csv")
    sp = sp.set_index("sample_id").loc[ids].reset_index()
    tr = sp[sp["split"] == "train"]
    te = sp[sp["split"] == "test"]
    if len(tr) == 0 or len(te) == 0:
        raise RuntimeError(f"empty split for {heldout}")

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
    ptr = lr.predict_proba(scaler.transform(Xtr))[:, 1]
    pte = lr.predict_proba(scaler.transform(Xte))[:, 1]

    # 筛查工作点:训练折锁定(主口径)+ test-optimal(与 yesterday 报告对照)
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


def compare_with_reference(out: pd.DataFrame, ref_path: Path) -> None:
    """与参考 CSV(如 yesterday 的 {model}_lodo_metrics.csv)逐格对比。"""
    ref = pd.read_csv(ref_path)
    print(f"\n--- 对比 {ref_path} ---")
    worst = 0.0
    for _, r in ref.iterrows():
        held = r["heldout_dataset"]
        mine = out.loc[out["heldout_dataset"] == held].iloc[0]
        d_auc = abs(mine["auroc"] - r["auroc"])
        worst = max(worst, d_auc)
        print(f"  [{held}] auroc mine={mine['auroc']:.6f} ref={r['auroc']:.6f} Δ={d_auc:.2e}")
    print(f"  max |ΔAUROC| = {worst:.2e}(判定:≤2e-3 视为协议一致)")


def run_model(model: str, cfg: dict, heldouts: list[str]) -> Path:
    out_dir = Path(cfg["output_dir"]) / model
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 0)
    seed_everything(seed)

    feat, ids = load_features(
        cfg["feature_source"], model,
        Path(cfg["cache_dir"]), Path(cfg["extract_dir"]),
    )
    print(f"[{model}] features {feat.shape} from {cfg['feature_source']}")

    rows = []
    for held in heldouts:
        row, fold_df = run_fold(feat, ids, held, cfg["splits_dir"], cfg.get("lr", {}), seed)
        rows.append(row)
        fold_df.to_csv(out_dir / f"fold_{held}_predictions.csv", index=False)
        print(
            f"  [{held}] auroc={row['auroc']:.4f} auprc={row['auprc']:.4f} "
            f"spec@95(锁)={row['spec@95sens_trainlocked']:.4f} "
            f"spec@95(opt)={row['spec@95sens_testopt_ref']:.4f}"
        )

    out = pd.DataFrame(rows)
    out["model"] = model
    out.to_csv(out_dir / f"{model}_probe_lodo_metrics.csv", index=False)
    print(f"saved: {out_dir / f'{model}_probe_lodo_metrics.csv'}")
    return out_dir / f"{model}_probe_lodo_metrics.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="all")
    ap.add_argument("--compare", default=None, help="参考 CSV 路径(逐格对比 AUROC)")
    ap.add_argument("--heldouts", default=None,
                    help="逗号分隔的 heldout 折(默认 4 折;外部折如 hvdropdb 用此追加)")
    args = ap.parse_args()

    heldouts = parse_heldouts(args.heldouts)
    cfg = load_yaml(args.config)
    models = list(cfg["models"]) if args.model == "all" else [args.model]
    for m in models:
        out_path = run_model(m, cfg, heldouts)
        if args.compare:
            compare_with_reference(pd.read_csv(out_path), Path(args.compare))
    print("done.")


if __name__ == "__main__":
    main()
