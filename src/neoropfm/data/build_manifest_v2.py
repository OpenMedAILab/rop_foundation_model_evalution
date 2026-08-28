"""在 v1 manifest 基础上构建 v2:补充纵向/临床字段。

新增字段(缺失为 NaN,原 v1 全部列保留):
- pma          :检查时胎龄后年龄(周)。RIDIRP info 表提供;其余数据集不可得。
- ga / bw / sex:胎龄(周)/ 出生体重(g)/ 性别。RIDIRP、ROP-VL、FARFUM(患者级)。
- exam_date    :检查日期(字符串)。ROP-VL 文件名解析;RIDIRP 无日期、以 info 表 PA 替代。
- eye          :眼别(L/R/unknown)。ROP-VL 文件名解析;RIDIRP 从文件夹名解析(如存在)。
- device       :采集设备(RIDIRP info 表 DEVICE 编码)。
- series       :RIDIRP 序列号(访次)。visit_id 已有,此处保留原始 serie 号。

匹配规则(已实测):
- RIDIRP 文件夹 images/{ID:03d}/{S:02d}/ 与 info 表按 (ID, S) 零填充后 join;
  info 表含 17 个重复 (ID,S) 键 → drop_duplicates(keep='first')。
- ROP-VL 文件名 {pid}_{YYYY-MM-DD}_{L|R}_{n}.jpg。
- FARFUM 按 patient_id(patientNN)join Dataset_Details(患者级 GA/BW/gender)。

运行:python -m neoropfm.data.build_manifest_v2
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import (  # noqa: E402
    FARFUM_DETAILS,
    MANIFEST_DIR,
    MANIFEST_V1,
    MANIFEST_V2,
    RIDIRP_INFO_XLSX,
    ROP_VL_IMG_INFO,
)


def _join_ridirp(m: pd.DataFrame) -> pd.DataFrame:
    """RIDIRP:按 (ID, S) 零填充 join info 表,取 PMA/GA/BW/SEX/DEVICE/DG/PF。"""
    r = m["dataset"] == "ridirp"
    ids = m.loc[r, "image_path"].str.extract(r"/images/(\d+)/", expand=False)
    sers = m.loc[r, "image_path"].str.extract(r"/images/\d+/(\d+)/", expand=False)
    # 注意:文件夹 serie 号零填充("01"),info 表为整数;统一转 int→str 后再拼 key
    m.loc[r, "series"] = sers.astype(int).astype(str)

    x = pd.read_excel(RIDIRP_INFO_XLSX)
    rename = {
        "POSTCONCEPTUAL AGE (PA)": "pma",
        "DIAGNOSIS CODE (DG)": "dg",
        "PLUS FORM (PF)": "pf",
        "SERIE NUMBER (S)": "series",
        "DEVICE (D)": "device",
        "SEX": "sex",
    }
    for c in x.columns:  # GA/BW 列名带括号
        if "GEST" in c.upper():
            rename[c] = "ga"
        if "BIRTH" in c.upper():
            rename[c] = "bw"
    x = x.rename(columns=rename)
    x["ID"] = x["ID"].astype(int).astype(str)
    x["series"] = x["series"].astype(int).astype(str)
    x = x.drop_duplicates(["ID", "series"], keep="first")  # 实测 17 个重复键

    m.loc[r, "ridirp_id"] = ids.astype(int).astype(str)
    m.loc[r, "_key"] = m.loc[r, "ridirp_id"] + "_" + m.loc[r, "series"]
    x["_key"] = x["ID"] + "_" + x["series"]
    for col in ["pma", "ga", "bw", "sex", "device", "dg", "pf"]:
        m.loc[r, col] = m.loc[r, "_key"].map(
            x.set_index("_key")[col].to_dict()
        )
    # 文件夹名中的 eye 标记(如 20240109_L 形式在内部数据;公开 RIDIRP 无 → 留空)
    return m


def _join_rop_vl(m: pd.DataFrame) -> pd.DataFrame:
    """ROP-VL:文件名解析 exam_date/eye,img_info 提供 GA/BW/gender。"""
    r = m["dataset"] == "rop_vl"
    m.loc[r, "exam_date"] = m.loc[r, "image_path"].str.extract(
        r"_(\d{4}-\d{2}-\d{2})_", expand=False
    )
    eye = m.loc[r, "image_path"].str.extract(r"_([LR])_\d+\.jpg$", expand=False)
    m.loc[r, "eye"] = eye.fillna("unknown")

    info = pd.read_excel(ROP_VL_IMG_INFO)
    info = info.rename(
        columns={
            "patient_id": "pid",
            "gestational_age": "ga",
            "birth_weight": "bw",
            "gender": "sex",
        }
    )
    info = info.drop_duplicates("pid", keep="first")  # 每图一行 → 每患者一行
    m.loc[r, "pid"] = m.loc[r, "patient_id"].astype(str)
    for col in ["ga", "bw", "sex"]:
        m.loc[r, col] = m.loc[r, "pid"].map(info.set_index(info["pid"].astype(str))[col])
    return m


def _join_farfum(m: pd.DataFrame) -> pd.DataFrame:
    """FARFUM:患者级 GA/BW/gender(出生体重 g/胎龄周)。"""
    r = m["dataset"] == "farfum_rop"
    det = pd.read_excel(FARFUM_DETAILS)
    det = det.rename(
        columns={
            "Patient.id": "pid",
            "Patient.BirthWeight": "bw",
            "Patient.Gestation Age": "ga",
            "Patient.Gender": "sex",
        }
    )
    det["pid"] = det["pid"].astype(str)
    m.loc[r, "pid"] = m.loc[r, "patient_id"].astype(str)
    for col in ["ga", "bw", "sex"]:
        m.loc[r, col] = m.loc[r, "pid"].map(det.set_index("pid")[col])
    return m


def main() -> None:
    v1 = pd.read_csv(MANIFEST_V1)
    n0 = len(v1)
    for col in ["pma", "ga", "bw", "sex", "exam_date", "eye", "device", "series", "ridirp_id", "pid", "_key"]:
        v1[col] = pd.NA
    v1 = _join_ridirp(v1)
    v1 = _join_rop_vl(v1)
    v1 = _join_farfum(v1)

    # 清理临时列
    v1 = v1.drop(columns=["_key", "ridirp_id", "pid"])

    v1.to_csv(MANIFEST_V2, index=False)

    # ---- 覆盖度报告 ----
    rep = {}
    for ds in ["ridirp", "rop_vl", "farfum_rop", "szeh_irops"]:
        s = v1[v1["dataset"] == ds]
        rep[ds] = {
            "images": len(s),
            "pma": int(s["pma"].notna().sum()),
            "ga": int(s["ga"].notna().sum()),
            "bw": int(s["bw"].notna().sum()),
            "sex": int(s["sex"].notna().sum()),
            "exam_date": int(s["exam_date"].notna().sum()),
            "eye_LR": int(s["eye"].isin(["L", "R"]).sum()),
            "device": int(s["device"].notna().sum()),
        }
    rep_df = pd.DataFrame(rep).T
    print(f"v1 rows: {n0} → v2 rows: {len(v1)}")
    print(rep_df.to_string())
    rep_df.to_csv(MANIFEST_DIR / "manifest_v2_coverage.csv")
    print(f"\nsaved: {MANIFEST_V2}")


if __name__ == "__main__":
    main()
