"""P4 决策单位(decision-unit)重分析:图像级 → 患者/访视/眼级重聚合 + 层级计数 + 统一预处理表。

背景:主表为图像级 AUROC;临床决策发生在
患者-访视(或眼)级。本脚本:
  A. hierarchy_counts.csv —— 每数据集 患者/眼/访视/图像 层级计数 + 阳性数(决策单位透明化);
  B. decision_unit_metrics.csv —— 读 outputs/checkpoint_iso/*_iso/fold_{held}_predictions.csv
     (test 行),按数据集规则聚合决策单位,单位标签=max(图像 y),单位分=mean/max(图像 p),
     输出单位级 AUROC/AUPRC + patient-cluster bootstrap CI(2000 次,seed=42),
     图像级 AUROC 作敏感性对照列;
  C. unified_pipeline_table.csv —— green@392(官方管线)vs green@224 vs mae_cfp@224(官方)
     同分辨率对照,消除"基线预处理不公平"。

决策单位规则:ridirp → (patient,visit);rop_vl → (patient,eye);farfum/szeh → patient。

运行:python3 scripts/decision_unit_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoropfm.common import MANIFEST_V2, OUTPUTS_DIR  # noqa: E402
from neoropfm.stats.bootstrap import patient_level_bootstrap  # noqa: E402

ISO_DIR = OUTPUTS_DIR / "checkpoint_iso"
OUT_DIR = OUTPUTS_DIR / "aggregate_e2"
N_BOOT = 2000
BOOT_SEED = 42

UNIT_RULES = {
    "ridirp": "patient|visit",
    "rop_vl": "patient|eye",
    "farfum_rop": "patient",
    "szeh_irops": "patient",
}
NOTES = {
    "farfum_rop": "无 eye/visit 列;决策单位=患者(单次检查集)",
    "ridirp": "无 eye;决策单位=(患者,访视)对;仅 5 阳性患者 → 患者池化将退化(见警示)",
    "rop_vl": "eye 全覆盖且 1 图/眼 → eye 级≈图像级;决策单位=eye",
    "szeh_irops": "1 图/患者;决策单位=患者",
}


def part_a() -> None:
    mf = pd.read_csv(MANIFEST_V2)
    m = mf[mf["include_strict_binary"] == 1].copy()
    rows = []
    for d in UNIT_RULES:
        s = m[m["dataset"] == d]
        n_visits = s.groupby(["patient_id", "visit_id"]).ngroups if s["visit_id"].notna().any() else 0
        pos_visits = (s[s["strict_binary_label"] == 1]
                      .groupby(["patient_id", "visit_id"]).ngroups
                      if s["visit_id"].notna().any() else 0)
        rows.append({
            "dataset": d,
            "n_images": len(s),
            "n_patients": s["patient_id"].nunique(),
            "n_eyes": int(s["eye"].notna().sum()),
            "n_visits": int(n_visits),
            "n_pos_images": int((s["strict_binary_label"] == 1).sum()),
            "n_pos_patients": int(s.loc[s["strict_binary_label"] == 1, "patient_id"].nunique()),
            "n_pos_visits": int(pos_visits),
            "decision_unit": UNIT_RULES[d],
            "notes": NOTES[d],
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "hierarchy_counts.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"saved: {OUT_DIR / 'hierarchy_counts.csv'}")


def unit_metrics(y_img, p_img, unit_ids, patient_ids, n_boot=N_BOOT):
    """图像级 → 单位级聚合后 AUROC/AUPRC + CI。"""
    df = pd.DataFrame({"u": unit_ids, "pat": patient_ids, "y": y_img, "p": p_img})
    g = df.groupby("u")
    y_u = g["y"].max().to_numpy()
    p_mean = g["p"].mean().to_numpy()
    p_max = g["p"].max().to_numpy()
    pat_u = g["pat"].first().to_numpy()
    if len(np.unique(y_u)) < 2:
        return None
    auc_mean = patient_level_bootstrap(y_u, p_mean, pat_u, roc_auc_score,
                                       n_boot=n_boot, seed=BOOT_SEED)
    auc_max = patient_level_bootstrap(y_u, p_max, pat_u, roc_auc_score,
                                      n_boot=n_boot, seed=BOOT_SEED)
    ap_mean = patient_level_bootstrap(y_u, p_mean, pat_u, average_precision_score,
                                      n_boot=n_boot, seed=BOOT_SEED)
    return {
        "n_units": len(y_u), "n_pos_units": int(y_u.sum()),
        "auroc_meanp": round(auc_mean["point"], 4),
        "auroc_meanp_lo": round(auc_mean["lower"], 4),
        "auroc_meanp_hi": round(auc_mean["upper"], 4),
        "auroc_maxp": round(auc_max["point"], 4),
        "auprc_meanp": round(ap_mean["point"], 4),
    }


def part_b() -> None:
    mf = pd.read_csv(MANIFEST_V2)[["sample_id", "eye", "visit_id"]]
    pools = sorted(p.name[:-4] for p in ISO_DIR.glob("*_iso") if p.is_dir())
    rows = []
    for pool in pools:
        pool_dir = ISO_DIR / f"{pool}_iso"
        for pred_csv in sorted(pool_dir.glob("fold_*_predictions.csv")):
            held = pred_csv.stem.replace("fold_", "").replace("_predictions", "")
            pred = pd.read_csv(pred_csv)
            test = pred[pred["subset"] == "test"].copy()
            if len(test) == 0:
                continue
            test = test.merge(mf, on="sample_id", how="left")
            if held == "ridirp":
                unit = (test["patient_id"].astype(str) + "|" + test["visit_id"].astype(str)).to_numpy()
            elif held == "rop_vl":
                unit = (test["patient_id"].astype(str) + "|" + test["eye"].astype(str)).to_numpy()
            else:
                unit = test["patient_id"].astype(str).to_numpy()
            m = unit_metrics(test["y"].to_numpy(), test["p"].to_numpy(), unit,
                             test["patient_id"].astype(str).to_numpy())
            if m is None:
                continue
            # 图像级参照(同一折,同模型,隔离选点表)
            lock = pd.read_csv(pool_dir / "locked_metrics_all.csv")
            img_ref = lock.loc[lock["heldout_dataset"] == held, "auroc"].iloc[0]
            # 严格隔离口径:LOTO 池只有对角线折(池名 minus_{held})的预测是
            # "训练语料不含该数据集"的;非对角线折训练集含被排除数据集,论文仅用 diagonal=True。
            is_loto = pool.startswith("loto_")
            diagonal = (not is_loto) or f"minus_{held}" in pool
            m.update({"model": pool, "heldout_dataset": held, "unit_rule": UNIT_RULES[held],
                      "auroc_image_ref": round(img_ref, 4), "diagonal": diagonal})
            rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "decision_unit_metrics.csv", index=False)
    print(df[["model", "heldout_dataset", "unit_rule", "n_units", "n_pos_units",
              "auroc_meanp", "auroc_meanp_lo", "auroc_meanp_hi", "auroc_image_ref"]].to_string(index=False))
    print(f"saved: {OUT_DIR / 'decision_unit_metrics.csv'}")


def part_c() -> None:
    """统一预处理受控表:green@392 vs green@224 vs mae@224(同分辨率对照)。"""
    pools = {"retfound_green": "392(官方管线)", "retfound_green_224": "224",
             "retfound_mae_cfp": "224(官方管线)"}
    rows = []
    for pool, res in pools.items():
        lock = pd.read_csv(ISO_DIR / f"{pool}_iso" / "locked_metrics_all.csv")
        for _, r in lock.iterrows():
            rows.append({"model": pool, "input_resolution": res,
                         "heldout_dataset": r["heldout_dataset"],
                         "auroc": round(r["auroc"], 4),
                         "auroc_lo": round(r["auroc_lo"], 4),
                         "auroc_hi": round(r["auroc_hi"], 4)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "unified_pipeline_table.csv", index=False)
    print(df.to_string(index=False))
    print(f"saved: {OUT_DIR / 'unified_pipeline_table.csv'}")


if __name__ == "__main__":
    part_a()
    print()
    part_b()
    print()
    part_c()
