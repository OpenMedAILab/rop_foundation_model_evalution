"""E5 不确定性分流(方案 §17 / §3.5):selective prediction + DCA + 工作量模拟。

输入:E1/E2 冻结探针的 LODO 逐折预测(outputs/probes/{model}/fold_*_predictions.csv,
subset=test)+ 各折训练锁定阈值({model}_probe_lodo_metrics.csv)。
患者级:同患者多图取均值 p、标签取 max。

三块输出(outputs/uncertainty/):
1. **risk–coverage**:不确定度 u = 1 − max(p, 1−p)(margin)升序保留患者,
   记录各 coverage 下保留子集的 AUROC/AUPRC(selective_{model}.csv);
2. **DCA**:阈值概率 0.02–0.40 的净获益——treat-none / treat-all / 模型
   按 sens95 锁定阈值 / 模型+不确定性分流(dca_{model}.csv);
3. **工作量模拟**:margin 阈值 τ ∈ {0, .05, .1, .15, .2, .25, .3} 下,不确定
   患者交人工;其余按 sens95 锁定阈值自动决策——报告人工比例、管线灵敏度/
   特异度/阳性预测值(workload_{model}.csv)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import load_yaml  # noqa: E402
from neoropfm.eval.metrics import auroc, auprc  # noqa: E402

from neoropfm.common import HELDOUTS  # noqa: E402


def load_patients(model: str, probe_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """患者级 (y, p, u, th95):u = margin 不确定度,th95 = 该折 sens95 锁定阈值。"""
    ys, ps, ths, pids = [], [], [], []
    m = pd.read_csv(probe_dir / model / f"{model}_probe_lodo_metrics.csv").set_index("heldout_dataset")
    for held in HELDOUTS:
        pred = pd.read_csv(probe_dir / model / f"fold_{held}_predictions.csv")
        te = pred[pred["subset"] == "test"]
        g = te.groupby(["dataset", "patient_id"]).agg(p=("p", "mean"), y=("y", "max"))
        ys.append(g["y"].to_numpy()); ps.append(g["p"].to_numpy())
        ths.append(np.repeat(float(m.loc[held, "threshold_sens95_locked"]), len(g)))
        pids.append(g.index.to_numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    th = np.concatenate(ths); u = 1 - np.maximum(p, 1 - p)
    return y, p, u, th


def net_benefit(y: np.ndarray, decide: np.ndarray, pt: float) -> float:
    """净获益 = TP/n − FP/n × pt/(1−pt)。decide ∈ {0,1}。"""
    tp = float(((decide == 1) & (y == 1)).sum())
    fp = float(((decide == 1) & (y == 0)).sum())
    n = len(y)
    return tp / n - fp / n * pt / (1 - pt)


def run_model(model: str, cfg: dict) -> None:
    probe_dir = Path(cfg["probe_dir"])
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    y, p, u, th = load_patients(model, probe_dir)

    # ---- 1. risk–coverage ----
    order = np.argsort(u)
    rows = []
    for cov in np.arange(1.0, 0.44, -0.05):
        k = int(round(cov * len(y)))
        keep = order[:k]
        rows.append({"coverage": cov, "n_patients": k,
                     "auroc": auroc(y[keep], p[keep]),
                     "auprc": auprc(y[keep], p[keep]),
                     "n_pos": int(y[keep].sum())})
    pd.DataFrame(rows).to_csv(out_dir / f"selective_{model}.csv", index=False)

    # ---- 2. DCA ----
    dca_rows = []
    treat_none = np.zeros(len(y))
    treat_all = np.ones(len(y))
    for pt in np.arange(0.02, 0.41, 0.02):
        model_dec = (p >= th).astype(int)  # sens95 锁定阈值决策
        dca_rows.append({
            "threshold_prob": pt,
            "nb_treat_none": net_benefit(y, treat_none, pt),
            "nb_treat_all": net_benefit(y, treat_all, pt),
            "nb_model": net_benefit(y, model_dec, pt),
        })
    pd.DataFrame(dca_rows).to_csv(out_dir / f"dca_{model}.csv", index=False)

    # ---- 3. 工作量模拟(不确定患者交人工,假定人工正确处理)----
    wl_rows = []
    for tau in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        certain = u <= tau
        auto_dec = (p >= th).astype(int)
        auto_keep = certain  # 自动决策的患者(确定且不交人工)
        auto_tp = ((auto_dec == 1) & (y == 1) & auto_keep).sum()
        auto_tn = ((auto_dec == 0) & (y == 0) & auto_keep).sum()
        auto_pos = ((y == 1) & auto_keep).sum()   # 自动决策的阳性数
        auto_neg = ((y == 0) & auto_keep).sum()   # 自动决策的阴性数
        missed = ((auto_dec == 0) & (y == 1) & auto_keep).sum()  # 自动放走的阳性
        wl_rows.append({
            "margin_tau": tau, "n_total": len(y),
            "n_to_human": int((~certain).sum()), "human_share": float((~certain).mean()),
            "auto_share": float(certain.mean()),
            "auto_sens": auto_tp / max(1, auto_pos),
            "auto_spec": auto_tn / max(1, auto_neg),
            "auto_dismissed_positives": int(missed),
            "auto_dismissed_positive_rate": missed / max(1, (y == 1).sum()),
        })
    pd.DataFrame(wl_rows).to_csv(out_dir / f"workload_{model}.csv", index=False)
    print(f"[{model}] selective/dca/workload → {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    models = [args.model] if args.model else cfg["models"]
    for model in models:
        run_model(model, cfg)


if __name__ == "__main__":
    main()
