"""E0 第 3 项:标签规则近似敏感性——ROP-VL 按方案 §2.1 记载的稿件规则重建标签。

背景:附件稿件原文不可获取(2026-08-20 用户确认)。方案 §2.1 记载稿件 D4 采用
"A-ROP 或 stage 4/5"。本脚本按此*记载*(非原文)重建 ROP-VL 标签做近似敏感性:
  - 阳性:A-ROP 或 Stage 4/5(v1 中已如此)
  - 阴性:Stage 3(无 A-ROP)从阳性翻为阴性 —— 241 张
其余数据集规则不变。仅重写 split CSV 的 strict_binary_label 列(rop_vl 行),
复用于所有折(rop_vl 在 3 折中为训练、1 折中为测试)。

产物:data/manifests/splits_stage45_sens/lodo_test_*.csv(供 probe 配置 splits_dir 指向)
运行:python -m neoropfm.data.label_rule_sensitivity
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import MANIFEST_V1, REPO_DIR  # noqa: E402

SPLITS = REPO_DIR / "data" / "manifests" / "splits"
OUT_DIR = REPO_DIR / "data" / "manifests" / "splits_stage45_sens"
STAGE45 = {"Stage 4 ROP", "Stage 5 ROP"}


def main() -> None:
    m = pd.read_csv(MANIFEST_V1)
    rv = m[(m["dataset"] == "rop_vl") & (m["include_strict_binary"] == 1)]
    variant = (
        (rv["original_a_rop"] == 1) | (rv["original_stage"].isin(STAGE45))
    ).astype(int)
    flipped = (variant != rv["strict_binary_label"])
    print(f"rop_vl strict {len(rv)} 行:稿件近似规则阳性 {variant.sum()}"
          f"(v1 {rv['strict_binary_label'].sum()});翻转 {flipped.sum()} 行")
    print(f"翻转明细(Stage3→阴性):{flipped.sum()} 行")
    assert (rv.loc[flipped, "original_stage"] == "Stage 3 ROP").all(), \
        "预期只有 Stage 3 无 A-ROP 的行被翻转"
    label_map = dict(zip(rv["sample_id"], variant))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for held in ["farfum_rop", "ridirp", "rop_vl", "szeh_irops"]:
        sp = pd.read_csv(SPLITS / f"lodo_test_{held}.csv")
        mask = (sp["dataset"] == "rop_vl") & (sp["sample_id"].isin(label_map))
        sp.loc[mask, "strict_binary_label"] = sp.loc[mask, "sample_id"].map(label_map)
        sp.to_csv(OUT_DIR / f"lodo_test_{held}.csv", index=False)
        print(f"  wrote splits_stage45_sens/lodo_test_{held}.csv")
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
