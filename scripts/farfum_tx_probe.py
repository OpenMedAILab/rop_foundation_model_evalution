"""P3b FARFUM 治疗决策敏感性终点探针:患者分层 5 折 CV(冻结特征,主探针协议)。

背景:audit_farfum_labels.py 从 Dataset_Labels.xlsx 5 位专家标注构造治疗共识终点:
  y_tx_d1 = 显式诊断标注多数票(treatment vs 非 treatment;平局→0);
  y_tx_d2 = 协议口径(grade==0 且 diag 空 → 隐式 no_treatment;grade>0 且 diag 空 → 缺失)。
本脚本在**同一患者分层 5 折分割**下评估 y_tx_d1 / y_tx_d2 / strict_binary_label
三终点,输出 AUROC/AUPRC + patient-cluster bootstrap CI(2,000 次,seed=42)。
与 strict 主探针的区别:这里是 FARFUM 内部 5 折 CV(而非 3 数据集训练),
回答"同一模型在两个终点定义下表现如何"这一敏感性问题。

前提:outputs/features_tx/{model}/{model}_features.npy 已生成
(extract_features.py --manifest outputs/audit/farfum_grade_audit.csv --no-strict-filter,
见 experiments/e3_farfum_tx_extract.sh)。

运行:
  python3 scripts/farfum_tx_probe.py --model all
  python3 scripts/farfum_tx_probe.py --model retfound_green
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoropfm.common import OUTPUTS_DIR, seed_everything  # noqa: E402
from neoropfm.stats.bootstrap import patient_level_bootstrap  # noqa: E402

AUDIT_CSV = OUTPUTS_DIR / "audit" / "farfum_grade_audit.csv"
FEATURES_TX_DIR = OUTPUTS_DIR / "features_tx"
OUT_DIR = OUTPUTS_DIR / "probe_tx"

MODELS = [
    "retfound_green", "retfound_mae_cfp", "dinov2_vits14", "dinov2_vitb14",
    "convnext_tiny", "efficientnet_b0",
    "ibot_dinov2s_v1_ckpt_ep040", "ibot_dinov2s_heads_ckpt_ep040",
    "mae_retfound_green_v1_ckpt_ep090",
]
ENDPOINTS = ["y_tx_d1", "y_tx_d2", "strict_binary_label"]
N_SPLITS = 5
SEED = 0
BOOT_SEED = 42
N_BOOT = 2000


def run_model(model: str) -> pd.DataFrame:
    feat = np.load(FEATURES_TX_DIR / model / f"{model}_features.npy")
    ids = pd.read_csv(FEATURES_TX_DIR / model / f"{model}_sample_ids.csv")["sample_id"].tolist()
    audit = pd.read_csv(AUDIT_CSV)
    audit = audit.dropna(subset=["sample_id"]).reset_index(drop=True)
    audit = audit.set_index("sample_id").loc[ids].reset_index()
    assert len(audit) == len(ids), f"feature/sample 对齐失败: {len(audit)} vs {len(ids)}"
    patients = audit["patient"].astype(str).to_numpy()
    print(f"[{model}] {len(audit)} images, {len(np.unique(patients))} patients")

    rows = []
    oof = {"sample_id": ids, "patient": patients}
    gkf = GroupKFold(n_splits=N_SPLITS)
    for ep in ENDPOINTS:
        y_all = audit[ep].to_numpy(float)
        valid = ~np.isnan(y_all)
        valid_idx = np.where(valid)[0]
        y_v = y_all[valid]
        p_oof = np.full(len(audit), np.nan)
        for tr_idx, te_idx in gkf.split(np.zeros(valid.sum()), y_v, groups=patients[valid]):
            tr_v, te_v = tr_idx, te_idx  # 索引相对 valid 子集
            Xtr, ytr = feat[valid][tr_v], y_v[tr_v]
            Xte = feat[valid][te_v]
            scaler = StandardScaler().fit(Xtr)
            lr = LogisticRegression(class_weight="balanced", C=1.0,
                                    max_iter=5000, random_state=SEED)
            lr.fit(scaler.transform(Xtr), ytr)
            p_oof[valid_idx[te_v]] = lr.predict_proba(scaler.transform(Xte))[:, 1]
        mask = ~np.isnan(p_oof)
        y_eval, p_eval, pat_eval = y_all[mask], p_oof[mask], patients[mask]
        auc = patient_level_bootstrap(y_eval, p_eval, pat_eval, roc_auc_score,
                                      n_boot=N_BOOT, seed=BOOT_SEED)
        ap = patient_level_bootstrap(y_eval, p_eval, pat_eval, average_precision_score,
                                     n_boot=N_BOOT, seed=BOOT_SEED)
        rows.append({
            "model": model, "endpoint": ep,
            "n": int(mask.sum()), "n_pos": int(y_eval.sum()),
            "auroc": round(auc["point"], 4), "auroc_lo": round(auc["lower"], 4),
            "auroc_hi": round(auc["upper"], 4),
            "auprc": round(ap["point"], 4), "auprc_lo": round(ap["lower"], 4),
            "auprc_hi": round(ap["upper"], 4), "n_boot_valid": auc["n_valid"],
        })
        oof[f"p_{ep}"] = p_oof
        print(f"  {ep}: auroc={auc['point']:.4f} [{auc['lower']:.4f},{auc['upper']:.4f}] "
              f"auprc={ap['point']:.4f} [{ap['lower']:.4f},{ap['upper']:.4f}] "
              f"(n={mask.sum()}, pos={int(y_eval.sum())})")
    oof.update({ep: audit[ep].to_numpy() for ep in ENDPOINTS})
    pd.DataFrame(oof).to_csv(OUT_DIR / f"{model}_oof_predictions.csv", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    global N_SPLITS, N_BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    N_SPLITS, N_BOOT = args.n_splits, args.n_boot
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = MODELS if args.model == "all" else [args.model]
    all_rows = pd.concat([run_model(m) for m in models], ignore_index=True)
    all_rows.to_csv(OUT_DIR / "farfum_tx_probe_metrics.csv", index=False)
    print(f"saved: {OUT_DIR / 'farfum_tx_probe_metrics.csv'}")


if __name__ == "__main__":
    main()
