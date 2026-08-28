"""P6:HVDROPDB 外部折清单构建(untouched 外部队列,Mendeley v3,CC BY 4.0)。

落地验证结论(2026-08-23):
- 公开发布版分类子集 = **185 张 png**(RetCam_Normal 50 / RetCam_ROP 50 /
  Neo_Normal 50 / Neo_ROP 35),文件名仅为序号(1.png…),**无患者标识**、
  无 eye/visit/stage 字段 → 外部折降级为**图像级**基准(如实说明;
  每个图像单独成簇,bootstrap 即图像级)。
- Segmentation 子集(600 文件=300 图+mask,BV/OD/RIDGE)全为 ROP 病灶图,
  无 Normal 对照 → 不用于分类外部折。
- 许可 CC BY 4.0;DOI 10.17632/xw5xc7xrmp.3 (v3, 2024-06-17)。
- strict 映射 = 文件夹标签本身(ROP→1, Normal→0)——数据原生二值,
  比 4 个内部数据集的近似映射更干净,无需专家再标。

输出:
  data/manifests/ext_hvdropdb_manifest_v2.csv   (30 列 v2 schema,含 device)
  data/manifests/splits/lodo_test_hvdropdb.csv  (train=4 内部数据集 strict 全量
                                                 7,815 行;test=hvdropdb 185 行)
运行:python3 scripts/build_hvdropdb_manifest.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoropfm.common import HELDOUTS, PUBLIC_DATA_ROOT, SPLITS_DIR  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "data/manifests"
CLS_DIR = PUBLIC_DATA_ROOT / "hvdropdb/raw/hvdropdb_classification_v3"
V2_COLS = [
    "sample_id", "dataset", "source_name", "image_path", "patient_id", "eye",
    "visit_id", "image_id", "original_label", "original_stage", "original_plus",
    "original_preplus", "original_a_rop", "original_laser", "strict_group",
    "strict_binary_label", "severity_group", "include_strict_binary",
    "exclusion_reason", "split_unit", "license", "source_doi", "notes",
    "pma", "ga", "bw", "sex", "exam_date", "device", "series",
]
STRICT_COLS = V2_COLS[:23]  # 与 public_rop_strict_manifest.csv 列一致
NOTE = ("公开发布版仅 185 图(RetCam 100 + Neo 85),无患者标识/eye/visit/stage → "
        "仅图像级外部基准;label=文件夹标签(数据原生二值);v3 (2024-06-17)")


def build_ext() -> pd.DataFrame:
    rows = []
    for folder, label, dev in [
        ("RetCam_Normal", 0, "RetCam"), ("RetCam_ROP", 1, "RetCam"),
        ("Neo_Normal", 0, "Neo"), ("Neo_ROP", 1, "Neo"),
    ]:
        for f in sorted((CLS_DIR / folder).glob("*.png")):
            path = str(f.resolve())
            sid = f"hvdropdb_{hashlib.md5(path.encode()).hexdigest()[:12]}"
            rows.append({
                "sample_id": sid, "dataset": "hvdropdb", "source_name": "HVDROPDB",
                "image_path": path, "patient_id": sid,  # 无患者标识 → 每图独立成簇
                "eye": "", "visit_id": "", "image_id": f.name,
                "original_label": "Normal" if label == 0 else "ROP",
                "original_stage": "", "original_plus": "", "original_preplus": "",
                "original_a_rop": "", "original_laser": "",
                "strict_group": "negative" if label == 0 else "positive",
                "strict_binary_label": label,
                "severity_group": "Normal" if label == 0 else "ROP",
                "include_strict_binary": 1, "exclusion_reason": "",
                "split_unit": "image", "license": "CC BY 4.0",
                "source_doi": "10.17632/xw5xc7xrmp.3", "notes": NOTE,
                "pma": "", "ga": "", "bw": "", "sex": "", "exam_date": "",
                "device": dev, "series": "",
            })
    df = pd.DataFrame(rows, columns=V2_COLS)
    assert len(df) == 185 and df["sample_id"].is_unique, "185 图且 sample_id 唯一"
    return df


def build_split(ext: pd.DataFrame) -> None:
    # train = 4 个内部数据集 strict 全量(与内部 LODO split 的 train 行组成一致)
    mf = pd.read_csv(MANIFESTS / "public_rop_strict_manifest.csv")
    mf = mf[mf["include_strict_binary"] == 1]
    tr = mf[mf["dataset"].isin(HELDOUTS)][
        ["sample_id", "dataset", "patient_id", "strict_binary_label", "image_path"]].copy()
    tr["split"] = "train"
    te = ext[["sample_id", "dataset", "patient_id", "strict_binary_label",
              "image_path"]].copy()
    te["split"] = "test"
    sp = pd.concat([tr, te], ignore_index=True)
    assert len(tr) == 7815 and len(te) == 185, (len(tr), len(te))
    sp.to_csv(SPLITS_DIR / "lodo_test_hvdropdb.csv", index=False)
    print(f"split saved: {SPLITS_DIR / 'lodo_test_hvdropdb.csv'} "
          f"(train {len(tr)} / test {len(te)})")


def main() -> None:
    ext = build_ext()
    ext.to_csv(MANIFESTS / "ext_hvdropdb_manifest_v2.csv", index=False)
    ext[STRICT_COLS].to_csv(MANIFESTS / "ext_hvdropdb_strict_manifest.csv", index=False)
    print(ext.groupby(["device", "strict_binary_label"]).size())
    print(f"saved: ext_hvdropdb_manifest_v2.csv ({len(ext)} rows)")
    build_split(ext)


if __name__ == "__main__":
    main()
