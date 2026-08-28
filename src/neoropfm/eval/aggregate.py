"""统计聚合阶段(probe 输出 → 带 CI 的结果表,零重算)。

输入:outputs/probes/{model}/fold_{heldout}_predictions.csv + {model}_probe_lodo_metrics.csv
输出:outputs/aggregate/
  - {model}_fold_metrics_with_ci.csv      每折点估计 + patient-level bootstrap CI(≥2000, seed 42)
  - all_models_lodo_metrics.csv           合并(与 yesterday 口径兼容的列)
  - all_models_mean_metrics.csv           4 折均值(auroc/auprc/spec@sens)
  - pairwise_delong.csv                   同折同测试集两两模型配对 DeLong(报告阶段做 Holm)

Bootstrap 单位:patient_id(方案 §18);SZEH 无患者分组(1 图 1 patient_id)→ 自动降级为
image-level,结果 CSV 中 bootstrap_unit 列注明。
筛查工作点:阈值来自训练折锁定(probe_lodo_metrics.csv 的 threshold_sens95_locked),
bootstrap 时阈值固定不重估。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import load_yaml  # noqa: E402
from neoropfm.eval.metrics import auroc, auprc, brier, ece  # noqa: E402
from neoropfm.eval.metrics import sensitivity_specificity_at_threshold  # noqa: E402
from neoropfm.stats.bootstrap import patient_level_bootstrap  # noqa: E402
from neoropfm.stats.delong import delong_test  # noqa: E402

from neoropfm.common import HELDOUTS  # noqa: E402


def _spec_at_th(metric_fn_y_p, th):
    """固定阈值下的特异性(供 bootstrap 使用)。"""

    def f(y, p):
        sens, spec = sensitivity_specificity_at_threshold(y, p, th)
        return spec

    return f


def fold_metrics_with_ci(model: str, held: str, cfg: dict) -> dict:
    probe_dir = Path(cfg["probe_dir"]) / model
    pred = pd.read_csv(probe_dir / f"fold_{held}_predictions.csv")
    te = pred[pred["subset"] == "test"]
    y = te["y"].to_numpy()
    p = te["p"].to_numpy()
    # 患者单位 = (dataset, patient_id),防跨数据集 id 命名空间碰撞
    pids = (te["dataset"].astype(str) + "|" + te["patient_id"].astype(str)).to_numpy()

    n_boot = cfg.get("n_boot", 2000)
    seed = cfg.get("seed", 42)

    # SZEH 无患者分组 → image-level bootstrap
    unit = "patient"
    if held == "szeh_irops" and len(np.unique(pids)) == len(y):
        unit = "image"

    row = {
        "model": model, "heldout_dataset": held, "bootstrap_unit": unit,
        "test_n": len(y), "test_positive": int(y.sum()),
    }
    for name, fn in [("auroc", auroc), ("auprc", auprc), ("brier", brier), ("ece", ece)]:
        ci = patient_level_bootstrap(y, p, pids, fn, n_boot=n_boot, seed=seed)
        row[name] = ci["point"]
        row[f"{name}_lo"] = ci["lower"]
        row[f"{name}_hi"] = ci["upper"]
        row[f"{name}_n_valid_boot"] = ci["n_valid"]

    # 训练折锁定的筛查工作点(spec, sens 一起报;bootstrap 固定阈值)
    m = pd.read_csv(probe_dir / f"{model}_{cfg.get('metrics_stem', 'probe')}_lodo_metrics.csv")
    m = m[m["heldout_dataset"] == held].iloc[0]
    for tgt in ["sens95", "sens98"]:
        th = m[f"threshold_{tgt}_locked"]
        if th is None or (isinstance(th, float) and np.isinf(th)):
            row[f"spec@{tgt}_locked"], row[f"sens@{tgt}_locked"] = 0.0, 0.0
            continue
        sens, spec = sensitivity_specificity_at_threshold(y, p, th)
        row[f"spec@{tgt}_locked"], row[f"sens@{tgt}_locked"] = spec, sens
        ci = patient_level_bootstrap(
            y, p, pids, _spec_at_th(None, th), n_boot=n_boot, seed=seed
        )
        row[f"spec@{tgt}_locked_lo"] = ci["lower"]
        row[f"spec@{tgt}_locked_hi"] = ci["upper"]
    return row


def pairwise_delong(models: list[str], cfg: dict) -> pd.DataFrame:
    """同折、同测试集(按 sample_id 对齐)两两配对 DeLong。"""
    probe_dir = Path(cfg["probe_dir"])
    rows = []
    for held in HELDOUTS:
        preds = {}
        for m in models:
            preds[m] = pd.read_csv(probe_dir / m / f"fold_{held}_predictions.csv")
        for i, ma in enumerate(models):
            for mb in models[i + 1:]:
                a = preds[ma][preds[ma]["subset"] == "test"].set_index("sample_id")
                b = preds[mb][preds[mb]["subset"] == "test"].set_index("sample_id")
                common = a.index.intersection(b.index)
                a, b = a.loc[common], b.loc[common]
                y = a["y"].to_numpy()
                if len(np.unique(y)) < 2:
                    continue
                r = delong_test(y, a["p"].to_numpy(), b["p"].to_numpy())
                rows.append({
                    "heldout_dataset": held, "model_a": ma, "model_b": mb,
                    "auc_a": r["auc1"], "auc_b": r["auc2"],
                    "diff": r["diff"], "z": r["z"], "p": r["p"],
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(cfg["models"])
    print(f"models: {models}")

    # 1) 每模型每折 bootstrap CI
    ci_rows = []
    for m in models:
        for held in HELDOUTS:
            row = fold_metrics_with_ci(m, held, cfg)
            ci_rows.append(row)
            print(
                f"  [{m}/{held}] auroc={row['auroc']:.4f} "
                f"[{row['auroc_lo']:.4f},{row['auroc_hi']:.4f}] (unit={row['bootstrap_unit']})"
            )
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(out_dir / "all_models_fold_metrics_with_ci.csv", index=False)

    # 2) 合并 lodo 表(与 yesterday 口径兼容)
    lodo = []
    for m in models:
        lodo.append(pd.read_csv(
            Path(cfg["probe_dir"]) / m / f"{m}_{cfg.get('metrics_stem', 'probe')}_lodo_metrics.csv"
        ))
    all_lodo = pd.concat(lodo, ignore_index=True)
    all_lodo.to_csv(out_dir / "all_models_lodo_metrics.csv", index=False)

    # 3) 4 折均值
    mean_rows = []
    for m in models:
        s = all_lodo[all_lodo["model"] == m]
        mean_rows.append({
            "model": m,
            "mean_auroc": s["auroc"].mean(),
            "mean_auprc": s["auprc"].mean(),
            "mean_spec@95sens_trainlocked": s["spec@95sens_trainlocked"].mean(),
            "mean_spec@98sens_trainlocked": s["spec@98sens_trainlocked"].mean(),
            "mean_sens@95sens_trainlocked": s["sens@95sens_trainlocked"].mean(),
            "mean_spec@95sens_testopt_ref": s["spec@95sens_testopt_ref"].mean(),
            "n_splits": len(s),
        })
    mean_df = pd.DataFrame(mean_rows)
    mean_df.to_csv(out_dir / "all_models_mean_metrics.csv", index=False)
    print("\n均值表:")
    print(mean_df.to_string(index=False))

    # 4) 配对 DeLong
    if len(models) >= 2:
        delong_df = pairwise_delong(models, cfg)
        delong_df.to_csv(out_dir / "pairwise_delong.csv", index=False)
        print(f"\npairwise DeLong: {len(delong_df)} 组(报告阶段做 Holm 校正)")

    print(f"\nsaved → {out_dir}")


if __name__ == "__main__":
    main()
