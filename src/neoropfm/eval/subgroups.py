"""E1.6 亚组分析:RIDIRP 测试折内的 sex/GA/BW/PA/device/图像分辨率分箱 AUROC。

输入:outputs/audit/ridirp_image_metadata.csv(E0 审计产物)+
     outputs/probes/{model}/fold_ridirp_predictions.csv(测试折预测)。
注意:亚组只在 ridirp 测试折内评估(4330 图、754 阳),不涉及其他数据集。
每个亚组:样本数、阳性数、AUROC/AUPRC + patient-level bootstrap CI(1000 次,
亚组分析属探索性,主结论以主表 2000 次为准;少样本亚组 CI 宽是预期)。

运行:
  python -m neoropfm.eval.subgroups --config configs/subgroups.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import load_yaml  # noqa: E402
from neoropfm.eval.metrics import auroc, auprc  # noqa: E402
from neoropfm.stats.bootstrap import patient_level_bootstrap  # noqa: E402

META = Path("outputs/audit/ridirp_image_metadata.csv")
SIZE_CACHE = Path("outputs/audit/ridirp_image_sizes.csv")

BINS = {
    "sex": lambda m: m["sex"],
    "GA(周)": lambda m: pd.cut(m["ga"], [0, 28, 32, 100], labels=["<28", "28-31", ">=32"]),
    "BW(g)": lambda m: pd.cut(m["bw"], [0, 1000, 1500, 10000], labels=["<1000", "1000-1499", ">=1500"]),
    "PA(周)": lambda m: pd.cut(m["pa"], [0, 33, 37, 200], labels=["<=33", "34-36", ">=37"]),
    "device": lambda m: m["device"].astype(int).astype(str).map(lambda s: f"D{s}"),
    "img_width": lambda m: pd.cut(
        m["width"], [0, 800, 1200, 10000], labels=["<800", "800-1199", ">=1200"]),
}


def load_sizes(sample_ids: pd.Series, paths: pd.Series) -> pd.DataFrame:
    """图像原始尺寸缓存(plan §E1 第 6 项:image quality 分箱)。"""
    if SIZE_CACHE.exists():
        cached = pd.read_csv(SIZE_CACHE, dtype={"sample_id": str})
        if set(sample_ids).issubset(set(cached["sample_id"])):
            return cached.set_index("sample_id")
    rows = []
    for sid, p in zip(sample_ids, paths):
        with Image.open(p) as im:
            rows.append((sid, im.size[0], im.size[1]))
    df = pd.DataFrame(rows, columns=["sample_id", "width", "height"])
    df.to_csv(SIZE_CACHE, index=False)
    return df.set_index("sample_id")


def subgroup_row(y, p, pids, name, value, n_boot=1000, seed=42):
    row = {"subgroup": name, "value": str(value), "n": len(y), "n_pos": int(y.sum())}
    if len(y) >= 2 and y.sum() >= 1 and (1 - y).sum() >= 1:
        for metric, fn in [("auroc", auroc), ("auprc", auprc)]:
            ci = patient_level_bootstrap(y, p, pids, fn, n_boot=n_boot, seed=seed)
            row[metric] = ci["point"]
            row[f"{metric}_lo"] = ci["lower"]
            row[f"{metric}_hi"] = ci["upper"]
    else:
        for m in ["auroc", "auprc", "auroc_lo", "auroc_hi", "auprc_lo", "auprc_hi"]:
            row[m] = np.nan
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    meta = pd.read_csv(META, dtype={"sample_id": str})
    meta = meta[meta["include_strict_binary"] == 1]
    sizes = load_sizes(meta["sample_id"], meta["image_path"])
    meta["width"] = meta["sample_id"].map(sizes["width"])

    all_rows = []
    for model in cfg["models"]:
        pred = pd.read_csv(Path(cfg["probe_dir"]) / model / "fold_ridirp_predictions.csv")
        te = pred[pred["subset"] == "test"].merge(
            meta, on="sample_id", how="left", validate="one_to_one")
        if te["ga"].isna().any():
            raise RuntimeError(f"{model}: 测试折样本未匹配到元数据")
        y = te["y"].to_numpy()
        p = te["p"].to_numpy()
        pids = (te["dataset"].astype(str) + "|" + te["patient_id"].astype(str)).to_numpy()

        row = subgroup_row(y, p, pids, "overall", "ridirp_test")
        row["model"] = model
        all_rows.append(row)

        for name, fn in BINS.items():
            groups = fn(te)
            for value in groups.dropna().unique():
                mask = (groups == value).to_numpy()
                row = subgroup_row(y[mask], p[mask], pids[mask], name, value)
                row["model"] = model
                all_rows.append(row)
        print(f"[{model}] ridirp 亚组完成")

    out = pd.DataFrame(all_rows)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "ridirp_subgroup_metrics.csv", index=False)
    print(f"saved: {out_dir / 'ridirp_subgroup_metrics.csv'}")


if __name__ == "__main__":
    main()
