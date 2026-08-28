"""公共工具:路径、配置加载、随机种子。

路径约定见 README.md「数据路径约定」。所有数据目录只读,中间产物写 REPO_DIR/outputs。
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

# ---- 仓库根目录 ----
REPO_DIR = Path(__file__).resolve().parents[2]

# ---- 只读数据路径 ----
# 公开数据集根目录:环境变量 NEOROPFM_PUBLIC_DATA_ROOT 优先,
# 默认仓库内 data/public_data(见 README「Data preparation」)。
PUBLIC_DATA_ROOT = Path(
    os.environ.get("NEOROPFM_PUBLIC_DATA_ROOT", str(REPO_DIR / "data" / "public_data"))
)
RIDIRP_INFO_XLSX = PUBLIC_DATA_ROOT / "ridirp/raw/infant_retinal_database_info.xlsx"
ROP_VL_IMG_INFO = PUBLIC_DATA_ROOT / "rop_vl/raw/rop_vl_extracted/img_info.xlsx"
FARFUM_LABELS = (
    PUBLIC_DATA_ROOT
    / "farfum_rop/raw/figshare_article_23609643_dataset_labels.xlsx/Dataset_Labels.xlsx"
)
FARFUM_DETAILS = (
    PUBLIC_DATA_ROOT
    / "farfum_rop/raw/figshare_article_23609646_dataset_details.xlsx/Dataset_Details.xlsx"
)

# ---- 仓库内路径 ----
DATA_DIR = REPO_DIR / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
SPLITS_DIR = MANIFEST_DIR / "splits"
OUTPUTS_DIR = REPO_DIR / "outputs"
FEATURES_DIR = OUTPUTS_DIR / "features"
CONFIGS_DIR = REPO_DIR / "configs"

MANIFEST_V1 = MANIFEST_DIR / "public_rop_strict_manifest.csv"
MANIFEST_V2 = MANIFEST_DIR / "public_rop_manifest_v2.csv"

# ---- 评估折(单一来源;外部折通过 --heldouts 追加,不写入此处)----
HELDOUTS = ["farfum_rop", "ridirp", "rop_vl", "szeh_irops"]


def parse_heldouts(heldouts: str | None) -> list[str]:
    """逗号分隔的 heldout 列表;None → 默认 HELDOUTS。

    外部折(如 hvdropdb)允许超出 HELDOUTS,由对应 lodo_test_*.csv 存在性兜底。
    """
    if not heldouts:
        return list(HELDOUTS)
    hs = [h.strip() for h in heldouts.split(",") if h.strip()]
    if not hs:
        return list(HELDOUTS)
    return hs


def load_yaml(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 0) -> None:
    """统一随机种子(numpy/torch/python)。bootstrap 等统计过程独立固定种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_free_mb(min_free_mb: int = 30_000) -> tuple[bool, int]:
    """检查 GPU 显存是否足够空闲。CPU-only 环境返回 (True, -1)。"""
    if not torch.cuda.is_available():
        return True, -1
    free = torch.cuda.mem_get_info()[0] // (1024 * 1024)  # free MiB
    return free >= min_free_mb, free
