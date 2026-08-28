"""E2.5 leave-test-out 适应敏感度:为每个 LODO 折生成剔除测试集数据集的 SSL 语料。

方案 §2.4 泄漏控制:主分析 SSL 语料含全部 4 数据集;leave-test-out 对照 = 每个
LODO 折只用其余 3 个数据集做继续预训练(排除"同数据集无标签预训练增益")。
输出 data/manifests/ssl_corpus_loto_{heldout}.csv(heldout ∈ farfum_rop/ridirp/
rop_vl/szeh_irops),对应 SSL config 把 manifest 字段指过来即可复跑 4 次。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import MANIFEST_DIR  # noqa: E402

from neoropfm.common import HELDOUTS  # noqa: E402


def main() -> None:
    src = MANIFEST_DIR / "ssl_corpus_manifest.csv"
    df = pd.read_csv(src)
    for held in HELDOUTS:
        sub = df[df["dataset"] != held].reset_index(drop=True)
        out = MANIFEST_DIR / f"ssl_corpus_loto_{held}.csv"
        sub.to_csv(out, index=False)
        print(f"{out.name}: {len(sub)} 张(全量 {len(df)} − {held} {(df['dataset'] == held).sum()})")


if __name__ == "__main__":
    main()
