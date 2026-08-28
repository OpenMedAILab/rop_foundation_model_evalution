"""RIDIRP 审计(E0 第 2 项):DG/PF 代码核验 + 临床元数据提取。

RIDIRP 图像文件名自带临床信息: {ID}_{SEX}_GA{ga}_BW{bw}_PA{pa}_DG{dg}_PF{pf}_D{device}_S{serie}_{n}.jpg
官方临床表 infant_retinal_database_info.xlsx 每行一次访视 (ID, SEX, GA, BW, PA, DG, PF, D, S)。

核验三件事:
1. 文件名元数据 vs 官方表(按 ID+PA 连接)是否一致(GA/BW/SEX/DG/PF/DEVICE);
2. strict_binary_label 编码与 (DG, PF) 的映射是否自洽(任何 (DG,PF) 组合必须
   要么全阳性、要么全阴性,否则编码规则有异常);
3. 输出 per-image 临床元数据 CSV → E1.6 亚组分析(sex/GA/BW/PA/device 分箱)直接消费。

运行:
  python -m neoropfm.data.audit_ridirp
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import MANIFEST_V1, PUBLIC_DATA_ROOT, REPO_DIR  # noqa: E402

XLSX = PUBLIC_DATA_ROOT / "ridirp/raw/infant_retinal_database_info.xlsx"
OUT_DIR = REPO_DIR / "outputs" / "audit"

PATH_RE = re.compile(
    r"/(?P<id>\d+)_(?P<sex>[MF])_GA(?P<ga>\d+)_BW(?P<bw>\d+)_PA(?P<pa>\d+(?:\.\d+)?)"
    r"_DG(?P<dg>\d+)_PF(?P<pf>\d+)_D(?P<device>\d+)_S(?P<serie>\d+)_\d+\.jpg$"
)


def parse_paths(paths: pd.Series) -> pd.DataFrame:
    """从 image_path 解析临床字段;无法解析的行保留为 NaN 并报告。"""
    parsed = paths.str.extract(PATH_RE)
    for col in ["id", "ga", "bw", "dg", "pf", "device", "serie"]:
        parsed[col] = pd.to_numeric(parsed[col], errors="coerce")
    parsed["pa"] = pd.to_numeric(parsed["pa"], errors="coerce")
    return parsed


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MANIFEST_V1)
    r = m[m["dataset"] == "ridirp"].copy()
    print(f"ridirp 总行数: {len(r)}")

    parsed = parse_paths(r["image_path"])
    n_bad = parsed["id"].isna().sum()
    print(f"路径解析失败: {n_bad} / {len(r)}")
    if n_bad:
        print(r.loc[parsed["id"].isna(), "image_path"].head(5).to_list())

    # 1) 与官方表连接(ID 零填充 3 位 + PA 取整)
    xlsx = pd.read_excel(XLSX)
    xlsx = xlsx.rename(columns=str.strip)
    xlsx["id_str"] = xlsx["ID"].astype(int).astype(str).str.zfill(3)
    xlsx["pa_int"] = xlsx["POSTCONCEPTUAL AGE (PA)"].astype(float).round().astype(int)
    parsed["id_str"] = parsed["id"].astype("Int64").astype(str).str.zfill(3)
    parsed["pa_int"] = parsed["pa"].astype(float).round().astype(int)

    dup = xlsx[xlsx.duplicated(["id_str", "pa_int"], keep=False)]
    print(f"xlsx 重复 (ID, PA) 行数: {len(dup)}(同一访视多条记录,连接前取首条)")
    if len(dup):
        print(dup[["ID", "pa_int", "DIAGNOSIS CODE (DG)", "PLUS FORM (PF)", "DEVICE (D)"]].to_string(index=False))
    xlsx_u = xlsx.drop_duplicates(["id_str", "pa_int"], keep="first")

    joined = parsed.merge(
        xlsx_u, on=["id_str", "pa_int"], how="left",
        suffixes=("_file", "_xlsx"),
    )
    joined.index = r.index
    n_unmatched = joined["ID"].isna().sum()
    print(f"官方表连接失败(按 ID+PA): {n_unmatched} / {len(r)}")
    if n_unmatched:
        print(joined.loc[joined["ID"].isna(), ["image_path", "id_str", "pa_int"]].head(5).to_string())

    # 字段一致性对比(文件名解析值 vs 官方表;列名无碰撞故无 _file 后缀)
    for file_col, xlsx_col, label in [
        ("sex", "SEX", "sex"), ("ga", "GESTATIONAL AGE (GA)", "GA"),
        ("bw", "BIRTH WEIGHT (BW)", "BW"), ("dg", "DIAGNOSIS CODE (DG)", "DG"),
        ("pf", "PLUS FORM (PF)", "PF"), ("device", "DEVICE (D)", "DEVICE"),
    ]:
        a = joined[file_col].astype(str)
        b = joined[xlsx_col].astype(str).str.replace(r"\.0$", "", regex=True)
        mismatch = (a != b).sum()
        print(f"  字段不一致 {label}: {mismatch} / {len(r)}")

    # 2) strict_binary_label 与 (DG, PF) 映射自洽性(仅 include_strict_binary==1 行)
    s = r[r["include_strict_binary"] == 1].copy()
    s["_dg"] = parsed.loc[s.index, "dg"].astype(int)
    s["_pf"] = parsed.loc[s.index, "pf"].astype(int)
    cross = s.groupby(["_dg", "_pf"])["strict_binary_label"].agg(["count", "min", "max"])
    inconsistent = cross[cross["min"] != cross["max"]]
    print(f"\nstrict 行 (DG,PF) 组合数: {len(cross)},编码不自洽组合数: {len(inconsistent)}")
    if len(inconsistent):
        print(inconsistent.to_string())
    print("\n(DG, PF) -> 标签映射(来自 manifest 实际编码):")
    mapping = cross.reset_index()
    mapping["label"] = mapping["min"].astype(int)
    print(mapping[["_dg", "_pf", "count", "label"]].to_string(index=False))

    # 3) 输出 per-image 元数据(全部 ridirp 行,供 E1.6 亚组分析)
    out = r[["sample_id", "image_path", "strict_binary_label", "include_strict_binary"]].copy()
    out["sex"] = parsed["sex"].values
    for col in ["ga", "bw", "pa", "dg", "pf", "device", "serie"]:
        out[col] = parsed[col].values
    out_path = OUT_DIR / "ridirp_image_metadata.csv"
    out.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")

    cross_path = OUT_DIR / "ridirp_dg_pf_label_mapping.csv"
    mapping.to_csv(cross_path, index=False)
    print(f"saved: {cross_path}")


if __name__ == "__main__":
    main()
