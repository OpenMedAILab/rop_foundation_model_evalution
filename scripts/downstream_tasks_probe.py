"""多下游任务线性探针:表征广度补充(any_rop / plus / stage_multi)。

任务(全部在 7,815 张 strict-binary 基准子集上,特征已缓存):
  any_rop     stage>=1 vs stage 0(ridirp / rop_vl / szeh 有 stage 标签)
  plus        plus vs normal(ridirp PF2 vs PF0;farfum Label 3 vs Label 1)
  stage_multi ridirp 5 类 OvR(0/1/2/3/AP-ROP),macro-AUROC
  (pre-plus 在 strict-binary 协议中已被排除,无基准子集标签 → 归 Version B;zone/quality 无标签同)

协议:与 E1 相同的 LODO 冻结探针(StandardScaler + LogReg balanced C=1);
  某数据集无该任务标签时从 train 折剔除(如实记录 train_n);heldout 无标签则跳过该折。
  评估:患者级 bootstrap CI(2,000, seed 42)。

模型:retfound_green(_224)、dinov2_vits14、dinov2_vitb14、retfound_mae_cfp、
     ibot_dinov2s_v1_iso / ibot_dinov2s_heads_iso(R3 每折隔离选点,读 outputs/checkpoint_iso)。

输出:outputs/downstream_tasks/{task}/{model}_{heldout}_metrics.csv + all_{task}.csv
"""
from __future__ import annotations

import re
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoropfm.stats.bootstrap import patient_level_bootstrap  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXTRACT = REPO / "outputs/features"
CACHE = Path(os.environ.get("NEOROPFM_BENCHMARK_CACHE", str(REPO / "outputs" / "benchmark_cache")))  # reference baseline cache (see README)
ISO = REPO / "outputs/checkpoint_iso"
SPLITS = REPO / "data/manifests/splits"
OUT = REPO / "outputs/downstream_tasks"
from neoropfm.common import HELDOUTS  # noqa: E402

MODELS = [
    ("retfound_green", "extract", None),
    ("retfound_green_224", "extract", None),
    ("dinov2_vits14", "cache", None),
    ("dinov2_vitb14", "extract", None),
    ("retfound_mae_cfp", "cache", None),
    ("ibot_dinov2s_v1", "extract", "iso"),
    ("ibot_dinov2s_heads", "extract", "iso"),
]


def build_labels() -> pd.DataFrame:
    """从 v2 manifest 清洗 stage/plus 标签(仅 strict-binary 包含行)。"""
    m = pd.read_csv(REPO / "data/manifests/public_rop_manifest_v2.csv", dtype=str)
    m = m[m["include_strict_binary"] == "1"].copy()

    def ridirp_stage(s: str) -> str:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return ""
        s = str(s)
        if "AP-ROP" in s:
            return "AP"
        mt = re.search(r"ROP (\d)", s)
        if mt:
            return mt.group(1)
        if "physiological" in s or "ROP 0" in s:
            return "0"
        return ""

    def vl_stage(s: str) -> str:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return ""
        s = str(s)
        if "A-ROP" in s:
            return "AP"
        mt = re.search(r"Stage (\d)", s)
        return mt.group(1) if mt else ("0" if "Normal" in s else "")

    def szeh_stage(s: str) -> str:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return ""
        s = str(s)
        if "Normal" in s:
            return "0"
        mt = re.search(r"Stage\s*(\d)", s)
        return mt.group(1) if mt else ""

    def plus(v: str) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        v = str(v).lower()
        return "1" if "plus" in v and "pre" not in v else ("0" if "normal" in v else "")

    def farfum_label(v: str) -> str:
        # Label 1=Normal, Label 3=Plus(strict 子集内无 Label 2)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        v = str(v)
        return {"Label 3": "1", "Label 1": "0"}.get(v, "")

    m["stage"] = np.where(m["dataset"] == "ridirp", m["original_stage"].map(ridirp_stage),
                 np.where(m["dataset"] == "rop_vl", m["original_stage"].map(vl_stage),
                 np.where(m["dataset"] == "szeh_irops", m["original_stage"].map(szeh_stage), "")))
    m["plus"] = np.where(m["dataset"] == "farfum_rop", m["original_label"].map(farfum_label),
                np.where(m["dataset"] == "ridirp", m["original_plus"].map(plus), ""))
    return m[["sample_id", "dataset", "patient_id", "stage", "plus"]]


def task_labels(lab: pd.DataFrame, task: str) -> pd.DataFrame:
    """按任务生成样本级标签表(y 缺失 = 无标签)。"""
    out = lab.copy()
    if task == "any_rop":
        out["y"] = np.where(out["stage"] == "", np.nan,
                            np.where(out["stage"] == "0", 0, 1).astype(float))
    elif task == "plus":
        out["y"] = np.where(out["plus"] == "", np.nan,
                            np.where(out["plus"] == "0", 0, 1).astype(float))
    elif task == "stage_multi":
        out["cls"] = out["stage"]
        out["cls"] = np.where(out["cls"].isin(["0", "1", "2", "3", "AP"]), out["cls"], np.nan)
    else:
        raise ValueError(task)
    return out


def load_feat(key: str, source: str):
    d = (EXTRACT if source == "extract" else CACHE) / key
    feat = np.load(d / f"{key}_features.npy")
    ids = pd.read_csv(d / f"{key}_sample_ids.csv")["sample_id"].tolist()
    return feat, ids


def fit_probe(Xtr, ytr, Xte):
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=5000, random_state=0)
    lr.fit(scaler.transform(Xtr), ytr)
    return lr.predict_proba(scaler.transform(Xte))[:, 1]


def row_idx(ids: list[str], wanted) -> np.ndarray:
    """逐样本特征行定位(见 checkpoint_selection_isolated.py 同名前注)。

    不能用 np.where(np.isin(ids, wanted))——返回 ids 顺序位置,与 wanted
    顺序(merge 后标签行序)错位,缓存特征 ids 字典序时 AUROC ≈0.5。
    """
    pos = pd.Index(ids).get_indexer(np.asarray(wanted))
    assert (pos >= 0).all(), "存在 sample_id 不在特征 ids 中"
    return pos


def macro_auc(y: np.ndarray, p_all: np.ndarray) -> float:
    """多分类 OvR macro-AUROC;p_all 形状 (n, n_classes)。"""
    aucs = []
    for c in range(p_all.shape[1]):
        if len(np.unique(y)) < 2:
            continue
        yc = (y == c).astype(int)
        if yc.sum() == 0:
            continue
        aucs.append(roc_auc_score(yc, p_all[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def run_task(model, source, iso, task: str, lab: pd.DataFrame) -> None:
    tlab = task_labels(lab, task)
    out_dir = OUT / task
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    key = model if not iso else f"{model}_iso"
    for held in HELDOUTS:
        sp = pd.read_csv(SPLITS / f"lodo_test_{held}.csv")
        if iso:
            sel_file = ISO / f"{model}_iso" / f"selection_{held}.csv"
            if not sel_file.exists():
                print(f"[{task}] {key} @ {held}: 跳过(R3 selection 尚未产出)", flush=True)
                continue
            sel = pd.read_csv(sel_file)
            ck = sel.loc[sel["selected"], "ckpt"].iloc[0]
        else:
            ck = model
        feat, ids = load_feat(ck, source)
        is_multi = task == "stage_multi"
        lab_col = "cls" if is_multi else "y"
        lab_held = tlab[tlab["sample_id"].isin(sp.loc[sp["split"] == "test", "sample_id"])]
        if lab_held[lab_col].notna().sum() == 0:
            continue  # heldout 折无该任务标签
        tr = sp[sp["split"] == "train"].merge(tlab[["sample_id", lab_col]],
                                              on="sample_id", how="inner")
        te = sp[sp["split"] == "test"].merge(tlab[["sample_id", lab_col]],
                                             on="sample_id", how="inner")
        tr = tr[tr[lab_col].notna()]
        te = te[te[lab_col].notna()]
        if len(tr) < 10 or len(te) < 5:
            continue
        itr = row_idx(ids, tr["sample_id"])
        ite = row_idx(ids, te["sample_id"])
        if is_multi:
            classes = ["0", "1", "2", "3", "AP"]
            ytr_c = np.array([classes.index(c) for c in tr["cls"]])
            scaler = StandardScaler().fit(feat[itr])
            lr = OneVsRestClassifier(
                LogisticRegression(class_weight="balanced", C=1.0, max_iter=5000, random_state=0))
            lr.fit(scaler.transform(feat[itr]), ytr_c)
            pte_all = lr.predict_proba(scaler.transform(feat[ite]))
            yte_c = np.array([classes.index(c) for c in te["cls"]])
            unit = (te["dataset"] + "|" + te["patient_id"]).to_numpy()
            ci = patient_level_bootstrap(yte_c, pte_all, unit, macro_auc, 2000, 42)
            rows.append({"model": key, "heldout_dataset": held, "selected_ckpt": ck,
                         "task": task, "train_n": len(tr), "test_n": len(te),
                         "auroc": ci["point"], "auroc_lo": ci["lower"], "auroc_hi": ci["upper"]})
        else:
            ytr, yte = tr["y"].to_numpy(float), te["y"].to_numpy(float)
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            pte = fit_probe(feat[itr], ytr, feat[ite])
            unit = (te["dataset"] + "|" + te["patient_id"]).to_numpy()
            ci = patient_level_bootstrap(yte, pte, unit, roc_auc_score, 2000, 42)
            rows.append({"model": key, "heldout_dataset": held, "selected_ckpt": ck,
                         "task": task, "train_n": len(tr), "test_n": len(te),
                         "auroc": ci["point"], "auroc_lo": ci["lower"], "auroc_hi": ci["upper"]})
        print(f"[{task}] {key} @ {held}: {rows[-1]['auroc']:.4f} "
              f"(train_n={len(tr)}, test_n={len(te)}, ck={ck})", flush=True)
    return rows


def main() -> None:
    lab = build_labels()
    for task in ["any_rop", "plus", "stage_multi"]:
        all_rows = []
        for model, source, iso in MODELS:
            all_rows.extend(run_task(model, source, iso, task, lab))
        if not all_rows:
            continue
        out_dir = OUT / task
        all_df = pd.DataFrame(all_rows)
        all_df.to_csv(out_dir / f"all_{task}.csv", index=False)
        # 每模型均值(全部模型行一次性聚合,避免逐模型覆盖写)
        mean_df = all_df.groupby("model").agg(
            mean_auroc=("auroc", "mean"), n_folds=("auroc", "count"),
            mean_train_n=("train_n", "mean")).reset_index()
        mean_df.to_csv(out_dir / f"mean_{task}.csv", index=False)
        print(f"[{task}] saved → {out_dir / f'all_{task}.csv'} "
              f"({len(all_df)} 折行, {len(mean_df)} 模型)", flush=True)


if __name__ == "__main__":
    main()
