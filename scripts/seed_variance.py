"""P1b 配套:heads 路线 3 种子方差(单种子 → 种子敏感性)。

口径:每种子各自完成隔离选点(seed 0 = outputs/checkpoint_iso/ibot_dinov2s_heads_iso,
seed 1/2 = ibot_dinov2s_heads_seed{1,2}_iso,由 e2c 训练 + checkpoint_selection_isolated 产出),
读取各折 locked_metrics_{held}.csv 的访视级 AUROC。
输出:outputs/aggregate_e2/seed_variance.csv(长格式:model, heldout_dataset, seed, auroc)
- 正文 Table 10:按 heldout_dataset 聚合 → 均值±SD + min–max(3 种子)
- S17:按 model 聚合 → 每种子 4 折均值,再跨种子 均值±SD
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoropfm.common import HELDOUTS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ISO = REPO / "outputs/checkpoint_iso"
OUT = REPO / "outputs/aggregate_e2"
OUT.mkdir(parents=True, exist_ok=True)

# (模型池名, 种子号);seed 0 = 既有全语料 heads 训练(无 _seed0 后缀)
SEEDS = [
    ("ibot_dinov2s_heads", 0),
    ("ibot_dinov2s_heads_seed1", 1),
    ("ibot_dinov2s_heads_seed2", 2),
]


def main() -> None:
    rows = []
    for model, seed in SEEDS:
        iso_dir = ISO / f"{model}_iso"
        for held in HELDOUTS:
            f = iso_dir / f"locked_metrics_{held}.csv"
            if not f.exists():
                print(f"[skip] {model} @ {held}(隔离选点缺失)")
                continue
            auroc = float(pd.read_csv(f).iloc[0]["auroc"])
            rows.append({"model": "ibot_dinov2s_heads", "heldout_dataset": held,
                         "seed": seed, "auroc": auroc})
            print(f"{model} seed{seed} @ {held}: {auroc:.4f}")
    df = pd.DataFrame(rows)
    if len(df) < 3 * len(HELDOUTS):
        print(f"[warn] 仅 {len(df)}/{3 * len(HELDOUTS)} 行——种子/折缺失,不写输出")
        return
    df = df.sort_values(["heldout_dataset", "seed"])
    df.to_csv(OUT / "seed_variance.csv", index=False)
    print("saved →", OUT / "seed_variance.csv")
    per_fold = df.groupby("heldout_dataset")["auroc"].agg(["mean", "std", "min", "max"])
    print(per_fold.round(4).to_string())


if __name__ == "__main__":
    main()
