"""LODO 划分质检(E0.2):患者跨折重叠、类别平衡、规模核对。

检查项:
1. 每个 heldout 折内 train 与 test 无患者重叠(patient leakage 守卫);
2. test 全部来自 heldout 数据集;
3. 各折 train/test 数量与 yesterday 记录一致;
4. 类别比例(严格二分类阳性率)。
输出:outputs/qc/splits_qc.csv + stdout 报告。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import OUTPUTS_DIR, SPLITS_DIR  # noqa: E402

from neoropfm.common import HELDOUTS  # noqa: E402


def main() -> None:
    rows = []
    all_ok = True
    for held in HELDOUTS:
        sp = pd.read_csv(SPLITS_DIR / f"lodo_test_{held}.csv")
        # 患者单位 = (dataset, patient_id):不同数据集 id 命名空间独立
        # (实测 RIDIRP/ROP-VL 数字 id 相互碰撞,非真实泄漏)
        sp["patient_unit"] = sp["dataset"] + "|" + sp["patient_id"].astype(str)
        tr = sp[sp["split"] == "train"]
        te = sp[sp["split"] == "test"]

        # 1) train/test 复合患者键重叠(真泄漏守卫)
        overlap = set(tr["patient_unit"]) & set(te["patient_unit"])
        ok_no_overlap = len(overlap) == 0
        # 1b) 同数据集内裸 id 重叠(排除跨数据集命名空间碰撞后的真正危险形态)
        same_ds_overlap = set(
            tr[tr["dataset"] == held]["patient_id"]
        ) & set(te["patient_id"])
        # 2) test 仅来自 heldout
        ok_test_source = (te["dataset"] == held).all()
        # 3) 所有行都属于严格二分类样本(v1 manifest include 列在此文件未带,仅核对规模)
        n_total = len(sp)
        ok_scale = n_total == 7815

        pos_rate_tr = tr["strict_binary_label"].mean()
        pos_rate_te = te["strict_binary_label"].mean()
        ok = ok_no_overlap and len(same_ds_overlap) == 0 and bool(ok_test_source) and ok_scale
        all_ok &= ok
        rows.append({
            "heldout_dataset": held,
            "train_n": len(tr), "test_n": len(te), "total": n_total,
            "train_pos_rate": round(float(pos_rate_tr), 4),
            "test_pos_rate": round(float(pos_rate_te), 4),
            "train_test_patient_unit_overlap": len(overlap),
            "same_dataset_bare_id_overlap": len(same_ds_overlap),
            "test_all_from_heldout": bool(ok_test_source),
            "total_equals_7815": ok_scale,
            "ok": ok,
        })

    qc = pd.DataFrame(rows)
    out = OUTPUTS_DIR / "qc"
    out.mkdir(parents=True, exist_ok=True)
    qc.to_csv(out / "splits_qc.csv", index=False)
    print(qc.to_string(index=False))
    print(f"\n质检结论: {'全部通过' if all_ok else '存在异常,见上表'};saved → {out / 'splits_qc.csv'}")


if __name__ == "__main__":
    main()
