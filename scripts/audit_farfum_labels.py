"""P3 FARFUM 标签审计:5 位专家逐图标注 → 治疗决策敏感性终点 + 专家间一致性。

背景:现有 strict 主终点对 FARFUM 用最终 Label(1/2/3=Normal/Pre-plus/Plus,
已对照 Sci Data 2024 论文核实)近似"治疗指征",但此前未使用原始表中 5 位专家的
grade/stage/diagnostic 列。本脚本:
1. 解析 Dataset_Labels.xlsx(1,533 行 × 5 raters A–E × grade/stage/diagnostic);
2. 与 public_rop_manifest_v2.csv 的 farfum_rop 行 join(覆盖率验收 ≥99%);
3. 归一化诊断值(含原始拼写错误 'teatment'→'treatment'),构建多数票:
   治疗决策终点 y_tx = majority diagnostic == treatment(平局→0);
4. 专家间一致性:Fleiss κ(diagnostic,覆盖度最高的 rater 三元组)+ 两两 Cohen κ;
5. 核对最终 Label 与专家多数 grade 的映射一致性(交叉表 + 偏差计数);
6. 输出 outputs/audit/farfum_grade_audit.csv(逐图)与 farfum_label_audit_summary.csv(汇总)。

运行:python3 scripts/audit_farfum_labels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoropfm.common import FARFUM_LABELS, MANIFEST_V2, OUTPUTS_DIR  # noqa: E402

RATERS = ["A", "B", "C", "D", "E"]
DIAG_CATS = ["no_treatment", "revisit", "treatment"]


def normalize_diag(v: object) -> str | None:
    """归一化诊断值:小写去空格,'teatment'→'treatment'(原始表拼写错误)。"""
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in ("teatment", "treatmnt"):
        s = "treatment"
    if s not in DIAG_CATS:
        return None
    return s


def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss κ。table: (n_items, n_raters) 的类别标签矩阵(0..k-1),NaN 剔除对应格。"""
    n, m = table.shape
    # 对每个 item,只用非缺失 rater
    nij = np.zeros((n, len(DIAG_CATS)))
    row_valid = np.zeros(n, dtype=int)
    for i in range(n):
        vals = table[i][~np.isnan(table[i])]
        row_valid[i] = len(vals)
        for v in vals:
            nij[i, int(v)] += 1
    if (row_valid == 0).any():
        raise RuntimeError("存在整行缺失的 item")
    pj = nij.sum(axis=0) / row_valid.sum()
    # P_i: item 内配对一致率
    Pi = np.zeros(n)
    for i in range(n):
        m_i = row_valid[i]
        if m_i < 2:
            Pi[i] = np.nan
            continue
        Pi[i] = (np.sum(nij[i] ** 2) - m_i) / (m_i * (m_i - 1))
    mask = ~np.isnan(Pi)
    Pbar = Pi[mask].mean()
    Pe = float(np.sum(pj**2))
    if abs(1.0 - Pe) < 1e-12:
        return 1.0 if Pbar > 0.999 else 0.0
    return (Pbar - Pe) / (1.0 - Pe)


def majority(values: pd.Series) -> object:
    """多数票;平局返回 None。"""
    vc = values.dropna().value_counts()
    if len(vc) == 0:
        return None
    top = vc.iloc[0]
    if (vc == top).sum() > 1:
        return None
    return vc.index[0]


def main() -> None:
    out_dir = OUTPUTS_DIR / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 解析 xlsx(两行表头)
    raw = pd.read_excel(FARFUM_LABELS, sheet_name="Labels", header=None)
    raw = raw.iloc[2:].reset_index(drop=True)  # 去掉两行表头
    cols = ["patient", "image"]
    for r in RATERS:
        cols += [f"grade_{r}", f"stage_{r}", f"diag_{r}"]
    cols += ["Label"]
    raw.columns = cols
    assert len(raw) == 1533, f"期望 1,533 行,实际 {len(raw)}"
    for r in RATERS:
        raw[f"grade_{r}"] = pd.to_numeric(raw[f"grade_{r}"], errors="coerce")
        raw[f"stage_{r}"] = pd.to_numeric(raw[f"stage_{r}"], errors="coerce")
        raw[f"diag_{r}"] = raw[f"diag_{r}"].map(normalize_diag)
    raw["Label"] = pd.to_numeric(raw["Label"], errors="coerce")

    # 2. join manifest v2(farfum 行)
    mf = pd.read_csv(MANIFEST_V2)
    mf = mf[mf["dataset"] == "farfum_rop"].copy()
    mf["_img_stem"] = mf["image_id"].str.rsplit(".", n=1).str[0]
    mf["_join_key"] = mf["patient_id"].astype(str) + "|" + mf["_img_stem"]
    raw["_join_key"] = raw["patient"].astype(str) + "|" + raw["image"].astype(str)
    n_dup = raw["_join_key"].duplicated().sum()
    if n_dup:
        print(f"警告: xlsx 侧重复 join key {n_dup} 行")
    joined = raw.merge(
        mf[["sample_id", "_join_key", "image_path", "original_label", "strict_binary_label",
            "strict_group", "include_strict_binary", "exclusion_reason"]],
        on="_join_key", how="left",
    )
    coverage = joined["sample_id"].notna().mean()
    print(f"join 覆盖率: {joined['sample_id'].notna().sum()}/{len(joined)} = {coverage:.4%}")
    unjoined = joined.loc[joined["sample_id"].isna(), ["patient", "image"]]
    if len(unjoined):
        print("未 join 行(前 10):")
        print(unjoined.head(10).to_string())

    # 3. 多数票 + 治疗终点(双口径)
    # 观测:每位 rater 的 diag/stage 非空行 ≈ 其 grade>0 的行(grade==0 → 不记录任何行动),
    # 即 blank diag 在协议上等价于"无 ROP、无需处理"。
    #   D1 = 仅显式标注的多数票;D2 = 协议口径(grade==0 且 diag 空 → 隐式 no_treatment;
    #   grade>0 且 diag 空 → 缺失)。
    for r in RATERS:
        joined[f"diag_{r}"] = joined[f"diag_{r}"].map(normalize_diag)
        joined[f"vote_tx_{r}"] = joined[f"diag_{r}"].map(
            lambda v: np.nan if v is None else int(v == "treatment"))
        blank_g0 = joined[f"diag_{r}"].isna() & (joined[f"grade_{r}"] == 0)
        joined.loc[blank_g0, f"vote_tx_{r}"] = 0
    joined["maj_grade"] = joined[[f"grade_{r}" for r in RATERS]].apply(majority, axis=1)
    joined["maj_diag"] = joined[[f"diag_{r}" for r in RATERS]].apply(majority, axis=1)
    joined["n_diag_nonnull"] = joined[[f"diag_{r}" for r in RATERS]].notna().sum(axis=1)
    joined["n_diag_treatment"] = joined[[f"diag_{r}" for r in RATERS]].apply(
        lambda row: (row == "treatment").sum(), axis=1)
    joined["y_tx_d1"] = (joined["maj_diag"] == "treatment").astype(int)
    joined["y_tx_d1"] = joined["y_tx_d1"].where(joined["n_diag_nonnull"] >= 1)
    joined["y_tx_d2"] = joined[[f"vote_tx_{r}" for r in RATERS]].apply(majority, axis=1)
    # strict 近似:Label 3 = plus = 治疗指征
    joined["strict_label3"] = (joined["Label"] == 3).astype(int)

    joined.to_csv(out_dir / "farfum_grade_audit.csv", index=False)

    # 4. 一致性
    summ = []
    # 4a. Fleiss κ:覆盖度最高的 rater 三元组/四元组(完整格)
    cat_idx = {c: i for i, c in enumerate(DIAG_CATS)}
    from itertools import combinations
    for k in (5, 4, 3, 2):
        best = None
        for combo in combinations(RATERS, k):
            sub = joined[[f"diag_{c}" for c in combo]]
            n_complete = sub.notna().all(axis=1).sum()
            if best is None or n_complete > best[1]:
                best = (combo, n_complete)
        combo, n_complete = best
        if n_complete < 10:
            print(f"Fleiss κ (diagnostic, raters {''.join(combo)}): 完整标注行不足,跳过")
            continue
        sub_num = joined[[f"diag_{c}" for c in combo]].replace(cat_idx).to_numpy(float)
        complete_rows = ~np.isnan(sub_num).any(axis=1)
        # 只对 complete rows 做 Fleiss(kappa 要求每 item 的 rater 数固定)
        kappa = fleiss_kappa(sub_num[complete_rows])
        summ.append({"metric": f"fleiss_kappa_diag_{''.join(combo).lower()}",
                     "value": round(kappa, 4), "n_images_complete": int(n_complete)})
        print(f"Fleiss κ (diagnostic, raters {''.join(combo)}): {kappa:.4f} "
              f"(n={n_complete} 完整标注)")
        if k == 2:  # 两两全矩阵
            from sklearn.metrics import cohen_kappa_score
            pairs = []
            for a, b in combinations(RATERS, 2):
                sub = joined[[f"diag_{a}", f"diag_{b}"]].dropna()
                if len(sub) < 10:
                    pairs.append((a + b, np.nan, len(sub)))
                    continue
                pairs.append((a + b, cohen_kappa_score(sub[f"diag_{a}"], sub[f"diag_{b}"]),
                              len(sub)))
            for name, kap, n in pairs:
                summ.append({"metric": f"cohen_kappa_diag_{name.lower()}",
                             "value": None if pd.isna(kap) else round(kap, 4),
                             "n_images_complete": n})
            break

    # 4b. grade 一致性(两两 Cohen κ,grade 0–3)
    from sklearn.metrics import cohen_kappa_score
    for a, b in [("A", "B"), ("A", "D"), ("B", "D")]:
        sub = joined[[f"grade_{a}", f"grade_{b}"]].dropna()
        kap = cohen_kappa_score(sub[f"grade_{a}"], sub[f"grade_{b}"])
        summ.append({"metric": f"cohen_kappa_grade_{a.lower()}{b.lower()}",
                     "value": round(kap, 4), "n_images_complete": len(sub)})
        print(f"Cohen κ (grade, raters {a}{b}): {kap:.4f} (n={len(sub)})")

    # 5. 最终 Label 与多数 grade 的映射核对
    print("\n== 最终 Label × 多数 grade 交叉表 ==")
    xt = pd.crosstab(joined["Label"], joined["maj_grade"], dropna=False)
    print(xt.to_string())
    # Label 3 (Plus) 中多数 grade <2 的占比;Label 1 中多数 grade>=2 的占比
    mism = {
        "n_label3": int((joined["Label"] == 3).sum()),
        "n_label3_majgrade_lt2": int(((joined["Label"] == 3) & (joined["maj_grade"] < 2)).sum()),
        "n_label1": int((joined["Label"] == 1).sum()),
        "n_label1_majgrade_ge2": int(((joined["Label"] == 1) & (joined["maj_grade"] >= 2)).sum()),
    }
    summ.append({"metric": "mismatch_label3_majgrade_lt2", "value": mism["n_label3_majgrade_lt2"],
                 "n_images_complete": mism["n_label3"]})
    summ.append({"metric": "mismatch_label1_majgrade_ge2", "value": mism["n_label1_majgrade_ge2"],
                 "n_images_complete": mism["n_label1"]})
    print(f"Label3(Plus)中多数grade<2: {mism['n_label3_majgrade_lt2']}/{mism['n_label3']}")
    print(f"Label1(Normal)中多数grade>=2: {mism['n_label1_majgrade_ge2']}/{mism['n_label1']}")

    # 6. 治疗终点 × strict Label3 交叉(双口径)
    for d in ("d1", "d2"):
        y = joined[f"y_tx_{d}"]
        print(f"\n== y_tx_{d}(多数治疗票)× Label3 ==")
        print(pd.crosstab(y, joined["strict_label3"], dropna=False).to_string())
        n_tx = int((y == 1).sum())
        n_def = int(y.notna().sum())
        # 方向一致性:Plus(Label3)图中多数票"不治疗"的数量
        n_plus_notreat = int(((joined["strict_label3"] == 1) & (y == 0)).sum())
        n_plus_treat = int(((joined["strict_label3"] == 1) & (y == 1)).sum())
        summ.append({"metric": f"y_tx_{d}_n_treatment", "value": n_tx,
                     "n_images_complete": n_def})
        summ.append({"metric": f"y_tx_{d}_n_defined", "value": n_def,
                     "n_images_complete": len(joined)})
        summ.append({"metric": f"y_tx_{d}_plus_notreat", "value": n_plus_notreat,
                     "n_images_complete": int(joined["strict_label3"].sum())})
        summ.append({"metric": f"y_tx_{d}_plus_treat", "value": n_plus_treat,
                     "n_images_complete": int(joined["strict_label3"].sum())})
        # 非 Plus 图中治疗票的 Label 分布
        nonplus_tx = joined[(y == 1) & (joined["strict_label3"] == 0)]
        print(f"  非 Plus 图中多数治疗票 {len(nonplus_tx)} 张,Label 分布:")
        print(nonplus_tx["Label"].value_counts(dropna=False).to_string())
        summ.append({"metric": f"y_tx_{d}_nonplus_tx_label1",
                     "value": int((nonplus_tx["Label"] == 1).sum()),
                     "n_images_complete": len(nonplus_tx)})
        summ.append({"metric": f"y_tx_{d}_nonplus_tx_label2",
                     "value": int((nonplus_tx["Label"] == 2).sum()),
                     "n_images_complete": len(nonplus_tx)})

    # 7. 每位 rater 治疗票(D2)vs strict Label3 的一致性
    print("\n== rater 治疗票(D2)vs strict Label3 ==")
    from sklearn.metrics import cohen_kappa_score
    for r in RATERS:
        sub = joined[[f"vote_tx_{r}", "strict_label3"]].dropna()
        if len(sub) < 10:
            continue
        kap = cohen_kappa_score(sub[f"vote_tx_{r}"], sub["strict_label3"])
        agree = (sub[f"vote_tx_{r}"] == sub["strict_label3"]).mean()
        summ.append({"metric": f"rater_{r}_vote_tx_vs_label3_kappa",
                     "value": round(kap, 4), "n_images_complete": len(sub)})
        print(f"  rater {r}: κ={kap:.4f} 一致率={agree:.3f} (n={len(sub)})")

    # 8. stage 量表异质性:各 rater stage 值域(说明为何不用 stage 做终点)
    print("\n== stage 值域(各 rater)== ")
    for r in RATERS:
        vals = joined[f"stage_{r}"].dropna().unique()
        print(f"  rater {r}: {sorted(vals, key=str)}")

    pd.DataFrame(summ).to_csv(out_dir / "farfum_label_audit_summary.csv", index=False)
    print(f"\nsaved: {out_dir/'farfum_grade_audit.csv'} + {out_dir/'farfum_label_audit_summary.csv'}")


if __name__ == "__main__":
    main()
