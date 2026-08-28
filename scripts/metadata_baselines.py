"""P2(CPU)元数据基线:量化捷径学习天花板 + 严重度-年龄混杂检查。

辅助头增益可能来自捷径(pma/center/device/visit 序数),而非视网膜表征。
本脚本按主探针协议(LODO 4 折、StandardScaler + balanced LR、train=其余 3 数据集)
用纯元数据特征拟合探针,给出"不看图"的上限:
  pma        : 单标量(出生后月龄)
  center     : source_name one-hot(数据集/机构)
  device     : device one-hot(RetCam/Neo/...)
  visit      : visit_id 序数(仅 RIDIRP 有;其余折该列为 NaN → 该折跳过)
  pma+center, pma+center+device
另输出 strict 标签 × PMA 的分布摘要(严重度-年龄混杂检查)。

输出:
  outputs/aggregate_e2/metadata_baselines.csv
  outputs/audit/pma_label_correlation.csv
运行:python3 scripts/metadata_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoropfm.common import HELDOUTS, MANIFEST_V2, OUTPUTS_DIR, SPLITS_DIR, seed_everything  # noqa: E402

SEED = 0
FEATURES = ["pma", "center", "device", "visit", "pma+center", "pma+center+device"]


def run_fold(mf, held: str, feat_names: list[str]) -> list[dict]:
    sp = pd.read_csv(SPLITS_DIR / f"lodo_test_{held}.csv")
    # 只取元数据列,避免与 split 自带 strict_binary_label 列名冲突
    meta = mf[["sample_id", "source_name", "device", "pma", "visit_id"]]
    sp = sp.merge(meta, on="sample_id", how="left")
    tr, te = sp[sp["split"] == "train"], sp[sp["split"] == "test"]

    # OneHotEncoder 只 fit train(handle_unknown=ignore),两侧 transform 保证列数一致
    enc_c = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(tr[["source_name"]])
    enc_d = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        tr[["device"]].astype(str))

    def feats(df: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {"pma": df[["pma"]].to_numpy(float)}
        out["center"] = enc_c.transform(df[["source_name"]])
        out["device"] = enc_d.transform(df[["device"]].astype(str))
        if df["visit_id"].notna().any():
            out["visit"] = df[["visit_id"]].to_numpy(float)
        out["pma+center"] = np.hstack([out["pma"], out["center"]])
        out["pma+center+device"] = np.hstack([out["pma"], out["center"], out["device"]])
        return out

    feats_tr, feats_te = feats(tr), feats(te)
    rows = []
    for name in feat_names:
        if name not in feats_tr or name not in feats_te:
            continue
        Xtr, ytr = feats_tr[name], tr["strict_binary_label"].to_numpy()
        Xte, yte = feats_te[name], te["strict_binary_label"].to_numpy()
        # pma 等元数据有缺失:逐侧剔除含 NaN 的行
        vtr = ~np.isnan(Xtr).any(axis=1)
        vte = ~np.isnan(Xte).any(axis=1)
        Xtr, ytr = Xtr[vtr], ytr[vtr]
        Xte, yte = Xte[vte], yte[vte]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        scaler = StandardScaler().fit(Xtr)
        lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=5000,
                                random_state=SEED)
        lr.fit(scaler.transform(Xtr), ytr)
        pte = lr.predict_proba(scaler.transform(Xte))[:, 1]
        rows.append({"heldout_dataset": held, "feature": name, "dim": Xtr.shape[1],
                     "train_n": len(ytr), "test_n": len(yte),
                     "test_positive": int(yte.sum()), "auroc": roc_auc_score(yte, pte)})
    # 折内元数据 rank 上限(不经训练,仅在 test 侧有该元数据时成立):
    # 回答"该折 test 集上,纯 pma/visit 排序能拿多少 AUROC"(LOTO 下 center/device
    # 在 test 侧为常数 → 恒 0.5,已由上面 LR 行体现)
    for name in ("pma", "visit"):
        Xte = feats_te.get(name)
        if Xte is None:
            continue
        yte = te["strict_binary_label"].to_numpy()
        vte = ~np.isnan(Xte).any(axis=1)
        if vte.sum() < 50 or len(np.unique(yte[vte])) < 2:
            continue
        rows.append({"heldout_dataset": held, "feature": f"{name}_rank_testonly",
                     "dim": -1, "train_n": 0, "test_n": int(vte.sum()),
                     "test_positive": int(yte[vte].sum()),
                     "auroc": roc_auc_score(yte[vte], Xte[vte])})
    return rows


def pma_correlation(mf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in HELDOUTS:
        s = mf[(mf["dataset"] == d) & (mf["include_strict_binary"] == 1)].dropna(subset=["pma"])
        if len(s) < 2 or s["strict_binary_label"].nunique() < 2:
            rows.append({"dataset": d, "n": len(s), "pma_mean_neg": None, "pma_mean_pos": None,
                         "point_biserial_r": None, "p": None,
                         "note": "pma 无值或单类(manifest v2 仅 RIDIRP 有 pma)"})
            continue
        r, p = pointbiserialr(s["strict_binary_label"], s["pma"])
        rows.append({
            "dataset": d, "n": len(s),
            "pma_mean_neg": s.loc[s["strict_binary_label"] == 0, "pma"].mean(),
            "pma_mean_pos": s.loc[s["strict_binary_label"] == 1, "pma"].mean(),
            "point_biserial_r": round(r, 4), "p": round(p, 4), "note": "",
        })
    return pd.DataFrame(rows)


def main() -> None:
    seed_everything(SEED)
    mf = pd.read_csv(MANIFEST_V2)
    rows = []
    for held in HELDOUTS:
        rows += run_fold(mf, held, FEATURES)
        print(f"[{held}] done", flush=True)
    df = pd.DataFrame(rows)
    (OUTPUTS_DIR / "aggregate_e2").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUTS_DIR / "aggregate_e2" / "metadata_baselines.csv", index=False)
    print(df.to_string(index=False))
    corr = pma_correlation(mf)
    (OUTPUTS_DIR / "audit").mkdir(parents=True, exist_ok=True)
    corr.to_csv(OUTPUTS_DIR / "audit" / "pma_label_correlation.csv", index=False)
    print(corr.to_string(index=False))
    print("saved → aggregate_e2/metadata_baselines.csv + audit/pma_label_correlation.csv")


if __name__ == "__main__":
    main()
