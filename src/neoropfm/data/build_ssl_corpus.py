"""E0.3 / E2.1:无标签新生儿眼底 SSL 语料 manifest(计划 §2.4)。

构成:
- 4 个主数据集全部图像(含 strict-binary 排除类,共 10,656 张);
- 辅助:hvdropdb、macretina、PLOS 连续严重度子集、血管/视盘分割原始图像;
- **排除 coph100**(RIDIRP 衍生,防重复);排除掩码/标注文件(*_mask、Segmentation、
  OD Detection 裁剪片、非图像文件)。
- PMA:RIDIRP 有逐访次 PMA(来自 E0 审计元数据),其余数据集缺失 → NaN,E4 掩码处理。
- 许可证:主 4 数据集沿用 strict manifest 的 license 字段;辅助数据集标 'verify',
  E2 正式开跑前人工核对(仅用于本地无标签预训练,不重分发)。

运行:
  python -m neoropfm.data.build_ssl_corpus
产物:data/manifests/ssl_corpus_manifest.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import MANIFEST_V1, PUBLIC_DATA_ROOT, REPO_DIR  # noqa: E402

PUBLIC = PUBLIC_DATA_ROOT
OUT = REPO_DIR / "data" / "manifests" / "ssl_corpus_manifest.csv"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

AUX_DIRS = {
    "hvdropdb": PUBLIC / "hvdropdb" / "raw",
    "macretina": PUBLIC / "macretina" / "raw",
    "plos_severity": PUBLIC / "plos_digital_health_rop_continuous_severity_subset" / "raw",
    "rop_vessel_seg": PUBLIC / "rop_blood_vessel_segmentation" / "raw",
    "rop_disc_seg": PUBLIC / "rop_optic_disc_segmentation" / "raw",
}
EXCLUDED_DIRS = {"coph100": PUBLIC / "coph100" / "raw"}


def is_usable_image(path: Path) -> bool:
    """原始眼底图判定:图像扩展名 + 排除掩码/裁剪/标注。"""
    if path.suffix.lower() not in IMG_EXTS:
        return False
    if "mask" in path.name.lower():
        return False
    p = str(path).lower()
    # macretina:排除 OD Detection(YOLO 裁剪片)与 Bv Segmentation/Masks
    if "macretina" in p and ("od detection" in p or "masks" in p):
        return False
    # hvdropdb:只收分类子集 + 分割子集的原始图像(*_images 目录),排除 *_masks
    # (父目录名含 "Segmentation",不能按子串排除)
    if "hvdropdb" in p:
        parts = [part.lower() for part in path.parts]
        if not any("classification_extracted" in part for part in parts) and \
           not any(part.endswith("_images") for part in parts):
            return False
    return True


def main() -> None:
    rows = []

    # 1) 主 4 数据集:strict manifest 全部行(含被 strict 排除的 2,841 张)
    m = pd.read_csv(MANIFEST_V1)
    pma = pd.read_csv(REPO_DIR / "outputs" / "audit" / "ridirp_image_metadata.csv",
                      dtype={"sample_id": str})
    pma_map = pma.set_index("sample_id")["pa"].to_dict()
    for _, r in m.iterrows():
        rows.append({
            "sample_id": r["sample_id"],
            "dataset": r["dataset"],
            "image_path": r["image_path"],
            "patient_id": r["patient_id"],
            "pma": pma_map.get(r["sample_id"], float("nan")),
            "source_doi": r.get("source_doi", ""),
            "license": r.get("license", ""),
            "notes": "",
        })

    # 2) 辅助数据集(忽略标签;coph100 排除)
    for ds, root in AUX_DIRS.items():
        n = 0
        for f in sorted(root.rglob("*")):
            if not f.is_file() or not is_usable_image(f):
                continue
            rows.append({
                "sample_id": f"{ds}|{n}",
                "dataset": ds,
                "image_path": str(f),
                "patient_id": float("nan"),
                "pma": float("nan"),
                "source_doi": "",
                "license": "verify",
                "notes": "aux SSL corpus; license to be verified before E2 run",
            })
            n += 1
        print(f"  {ds}: {n} 张")

    n_excl = sum(1 for f in EXCLUDED_DIRS["coph100"].rglob("*")
                 if f.is_file() and is_usable_image(f))
    print(f"  coph100(排除,RIDIRP 衍生):{n_excl} 张")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\nsaved: {OUT}")
    print(out.groupby("dataset").agg(n=("sample_id", "size"),
                                     pma_known=("pma", lambda s: s.notna().sum())))
    print(f"总图像数: {len(out)}")


if __name__ == "__main__":
    main()
