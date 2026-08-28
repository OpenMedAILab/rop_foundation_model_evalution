"""E5 安全自动化 max-τ 敏感性分析(部署模拟口径附录)。

背景:锁定版 safe-auto 用"最小 τ 使管线灵敏度 ≥99%"规则——τ=0 时被自动化集合≈空、
管线灵敏度恒为 1.0,故恒选 τ=0("安全自动化≈0"是该规则的构造性结果)。本脚本
计算**最大自动化比例**规则:在 {0, 0.05, …, 0.5} 网格上取**最大的** τ 使管线
灵敏度 ≥99%,报告该 τ 下的自动化比例(与锁定版最小-τ 口径并存,作为敏感性)。

口径:只读 outputs/checkpoint_iso/{model}_iso/fold_{held}_predictions.csv 的
train/test 子集,访视级(与锁定 E5 主口径一致);θ99 阈值与锁定版相同
(pos[n_miss], n_miss = floor(0.01·n_pos));患者簇不重采样(确定性,seed 无关)。

输出(新文件,不覆盖锁定产物):outputs/e5_locked/all_models_e5_max_tau.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoropfm.common import HELDOUTS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ISO = REPO / "outputs/checkpoint_iso"
OUT = REPO / "outputs/e5_locked"


def main() -> None:
    rows = []
    for iso_dir in sorted(d for d in ISO.iterdir() if d.is_dir() and d.name.endswith("_iso")):
        model = iso_dir.name[:-4]
        for held in HELDOUTS:
            f = iso_dir / f"fold_{held}_predictions.csv"
            if not f.exists():
                continue
            pred = pd.read_csv(f)
            tr = pred[pred["subset"] == "train"]
            te = pred[pred["subset"] == "test"]
            pos = np.sort(tr.loc[tr["y"] == 1, "p"].to_numpy())
            if len(pos) == 0 or len(te) < 5:
                continue
            th99 = pos[int(np.floor(0.01 * len(pos)))]
            y, p = te["y"].to_numpy(), te["p"].to_numpy()
            n_pos = int(y.sum())
            u = 1 - np.maximum(p, 1 - p)
            best = None
            for tau in np.arange(0.0, 0.51, 0.05):
                certain = u <= tau
                missed = ((p < th99) & (y == 1) & certain).sum()
                pipe_sens = 1 - missed / n_pos if n_pos else float("nan")
                if pipe_sens >= 0.99:
                    best = (tau, certain.mean(), pipe_sens)
                else:
                    break  # τ 单调递增,certain 集合单调增大 → 再大的 τ 不可能恢复
            if best is None:
                rows.append({"model": model, "heldout_dataset": held, "margin_tau": float("nan"),
                             "auto_share": float("nan"), "pipeline_sens": float("nan"),
                             "n_test_units": len(te), "n_test_pos": n_pos})
            else:
                rows.append({"model": model, "heldout_dataset": held, "margin_tau": best[0],
                             "auto_share": round(best[1], 4), "pipeline_sens": round(best[2], 4),
                             "n_test_units": len(te), "n_test_pos": n_pos})
    df = pd.DataFrame(rows)
    agg = (df.groupby("model")
             .agg(n_folds=("margin_tau", "size"),
                  mean_margin_tau=("margin_tau", "mean"),
                  mean_auto_share=("auto_share", "mean"),
                  n_folds_no_feasible=("auto_share", lambda s: int(s.isna().sum())))
             .reset_index())
    agg["mean_auto_share"] = agg["mean_auto_share"].round(4)
    agg["mean_margin_tau"] = agg["mean_margin_tau"].round(3)
    df.to_csv(OUT / "e5_max_tau_per_fold.csv", index=False)
    agg.to_csv(OUT / "all_models_e5_max_tau.csv", index=False)
    print("per-fold →", OUT / "e5_max_tau_per_fold.csv")
    print("aggregate →", OUT / "all_models_e5_max_tau.csv")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
