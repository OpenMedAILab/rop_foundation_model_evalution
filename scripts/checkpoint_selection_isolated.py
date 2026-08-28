"""checkpoint 选择完全隔离最终测试折(隔离协议)。

协议(与 probe.py 保持同一探针配置,仅选择环节隔离):
- 每个模型 × 每个 heldout 折:
  1. inner 选择:train 折(其余 3 数据集)患者级 inner LODO CV,
     每 checkpoint 得 3 个数据集 AUROC 的均值,argmax 选 c*;
     单权重基线(无 ckpt 池)跳过选择,仅产 train 侧 OOF 预测(供阈值锁定)。
  2. 锁模:以 c* 在全部 train 折上拟合 probe
     (StandardScaler + LogisticRegression(class_weight=balanced, C=1.0, max_iter=5000),
     与 src/neoropfm/train/probe.py 完全一致,random_state=0)。
  3. 一次性评估 heldout 折:AUROC/AUPRC + 患者级 bootstrap CI(2,000, seed 42)。
- 基线模型 locked AUROC 应与原 E1 数字一致(协议一致性自检,≤2e-3)。

输出 outputs/checkpoint_iso/{model}/:
  selection_{heldout}.csv          # 各 ckpt 的 inner AUROC 明细与选中 c*
  fold_{heldout}_predictions.csv   # train(OOF)+ test 预测(sample_id/patient_id/dataset/subset/y/p)
  locked_metrics_{heldout}.csv     # 锁定评估指标 + CI
  locked_metrics_all.csv           # 4 折汇总
"""
from __future__ import annotations

import json
import re
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoropfm.stats.bootstrap import patient_level_bootstrap  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXTRACT = REPO / "outputs/features"
CACHE = Path(os.environ.get("NEOROPFM_BENCHMARK_CACHE", str(REPO / "outputs" / "benchmark_cache")))  # reference baseline cache (see README)
SPLITS = REPO / "data/manifests/splits"
OUT = REPO / "outputs/checkpoint_iso"
from neoropfm.common import HELDOUTS  # noqa: E402

CACHE_BASELINES = ["efficientnet_b0", "convnext_tiny", "dinov2_vits14", "retfound_mae_cfp"]
CKPT_RE = re.compile(r"^(.*)_ckpt_(ep\d+|final)$")


def discover_models() -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """pools: {prefix: [ckpt_keys 排序]}; singles: [(key, source)]。"""
    pools: dict[str, list[str]] = {}
    singles: list[tuple[str, str]] = []
    for d in sorted(p for p in EXTRACT.iterdir() if p.is_dir()):
        if not (d / f"{d.name}_features.npy").exists():
            continue
        m = CKPT_RE.match(d.name)
        if m:
            pools.setdefault(m.group(1), []).append(d.name)
        else:
            singles.append((d.name, "extract"))
    for name in CACHE_BASELINES:
        if (CACHE / name).exists():
            singles.append((name, "cache"))

    def ckpt_sort(key: str) -> int:
        ep = CKPT_RE.match(key).group(2)
        return 10**9 if ep == "final" else int(ep[2:])

    for p in pools:
        pools[p].sort(key=ckpt_sort)
    return pools, singles


def load_feat(key: str, source: str) -> tuple[np.ndarray, list[str]]:
    d = (EXTRACT if source == "extract" else CACHE) / key
    feat = np.load(d / f"{key}_features.npy")
    ids = pd.read_csv(d / f"{key}_sample_ids.csv")["sample_id"].tolist()
    return feat, ids


def row_idx(ids: list[str], wanted: np.ndarray) -> np.ndarray:
    """逐样本特征行定位:返回 wanted 中每个 sample_id 在 ids(即特征行)中的位置。

    注意:不能用 np.where(np.isin(ids, wanted))——它返回的是 ids 顺序的位置,
    与 wanted(按 split 表顺序排列的标签)错位。缓存特征的 ids 常按字典序
    排列而 split 表按 manifest 顺序,二者不一致时该 bug 使 AUROC ≈0.5。
    """
    pos = pd.Index(ids).get_indexer(wanted)
    assert (pos >= 0).all(), "存在 sample_id 不在特征 ids 中"
    return pos


def patient_auc(y: np.ndarray, p: np.ndarray, unit_ids: np.ndarray) -> float:
    """患者级(均值池化 probs,取 max 标签)AUROC。"""
    df = pd.DataFrame({"u": unit_ids, "y": y, "p": p})
    g = df.groupby("u").agg(y=("y", "max"), p=("p", "mean"))
    if len(np.unique(g["y"])) < 2:
        return float("nan")
    return float(roc_auc_score(g["y"], g["p"]))


def fit_probe(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=5000, random_state=0)
    lr.fit(scaler.transform(Xtr), ytr)
    return lr.predict_proba(scaler.transform(Xte))[:, 1]


def run_inner_cv(feat, ids, sp_tr, ckpt_key, source) -> tuple[float, dict[str, float], pd.DataFrame]:
    """train 折上 3 数据集 inner LODO CV。返回 (mean_auc, {dataset: auc}, oof_df)。"""
    per_ds: dict[str, float] = {}
    oof_rows = []
    inner_dss = sorted(sp_tr["dataset"].unique())
    for ds in inner_dss:
        tr_mask = (sp_tr["dataset"] != ds).to_numpy()
        te_mask = (sp_tr["dataset"] == ds).to_numpy()
        itr = row_idx(ids, sp_tr.loc[tr_mask, "sample_id"].to_numpy())
        ite = row_idx(ids, sp_tr.loc[te_mask, "sample_id"].to_numpy())
        p = fit_probe(feat[itr], sp_tr.loc[tr_mask, "strict_binary_label"].to_numpy(), feat[ite])
        unit = (sp_tr.loc[te_mask, "dataset"] + "|" + sp_tr.loc[te_mask, "patient_id"]).to_numpy()
        y = sp_tr.loc[te_mask, "strict_binary_label"].to_numpy()
        per_ds[ds] = patient_auc(y, p, unit)
        oof_rows.append(pd.DataFrame({
            "sample_id": sp_tr.loc[te_mask, "sample_id"].to_numpy(),
            "patient_id": sp_tr.loc[te_mask, "patient_id"].to_numpy(),
            "dataset": sp_tr.loc[te_mask, "dataset"].to_numpy(),
            "y": y, "p": p,
        }))
    oof = pd.concat(oof_rows, ignore_index=True)
    return float(np.nanmean(list(per_ds.values()))), per_ds, oof


def run_model(pool_name: str, ckpt_keys: list[str] | None, source: str) -> None:
    out_dir = OUT / f"{pool_name}_iso"
    out_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_keys:
        (out_dir / "pool.json").write_text(
            json.dumps(ckpt_keys, ensure_ascii=False, indent=2))
    feat_cache: dict[str, tuple[np.ndarray, list[str]]] = {}

    def get_feat(key: str):
        if key not in feat_cache:
            feat_cache[key] = load_feat(key, source)
        return feat_cache[key]

    all_metrics = []
    for held in HELDOUTS:
        sp = pd.read_csv(SPLITS / f"lodo_test_{held}.csv")
        sp_tr = sp[sp["split"] == "train"].reset_index(drop=True)
        sp_te = sp[sp["split"] == "test"].reset_index(drop=True)

        sel_rows = []
        candidates = ckpt_keys if ckpt_keys else [pool_name]
        for ck in candidates:
            feat, ids = get_feat(ck)
            mean_auc, per_ds, _ = run_inner_cv(feat, ids, sp_tr, ck, source)
            row = {"ckpt": ck, "mean_inner_auc": mean_auc}
            row.update({f"inner_auc_{ds}": auc for ds, auc in per_ds.items()})
            sel_rows.append(row)
        sel = pd.DataFrame(sel_rows)
        if ckpt_keys:
            c_star = sel.loc[sel["mean_inner_auc"].idxmax(), "ckpt"]
        else:
            c_star = pool_name
        sel["selected"] = sel["ckpt"] == c_star
        sel.to_csv(out_dir / f"selection_{held}.csv", index=False)

        # 选中 c* 的 train OOF(锁定拟合前的 inner 预测)+ 锁定评估
        feat, ids = get_feat(c_star)
        _, _, oof = run_inner_cv(feat, ids, sp_tr, c_star, source)
        itr = row_idx(ids, sp_tr["sample_id"].to_numpy())
        ite = row_idx(ids, sp_te["sample_id"].to_numpy())
        ytr = sp_tr["strict_binary_label"].to_numpy()
        yte = sp_te["strict_binary_label"].to_numpy()
        pte = fit_probe(feat[itr], ytr, feat[ite])

        pred = pd.concat([
            pd.DataFrame({
                "sample_id": sp_tr["sample_id"].to_numpy(),
                "patient_id": sp_tr["patient_id"].to_numpy(),
                "dataset": sp_tr["dataset"].to_numpy(),
                "subset": "train",
                "y": ytr,
                "p": oof["p"].to_numpy(),
            }),
            pd.DataFrame({
                "sample_id": sp_te["sample_id"].to_numpy(),
                "patient_id": sp_te["patient_id"].to_numpy(),
                "dataset": sp_te["dataset"].to_numpy(),
                "subset": "test",
                "y": yte, "p": pte,
            }),
        ], ignore_index=True)
        pred.to_csv(out_dir / f"fold_{held}_predictions.csv", index=False)

        unit_ids = (sp_te["dataset"] + "|" + sp_te["patient_id"]).to_numpy()
        auroc = patient_level_bootstrap(yte, pte, unit_ids, roc_auc_score, 2000, 42)
        auprc = patient_level_bootstrap(yte, pte, unit_ids, average_precision_score, 2000, 42)
        row = {
            "heldout_dataset": held, "selected_ckpt": c_star,
            "test_n": len(sp_te), "test_positive": int(yte.sum()),
            "auroc": auroc["point"], "auroc_lo": auroc["lower"], "auroc_hi": auroc["upper"],
            "auprc": auprc["point"], "auprc_lo": auprc["lower"], "auprc_hi": auprc["upper"],
        }
        all_metrics.append(row)
        pd.DataFrame([row]).to_csv(out_dir / f"locked_metrics_{held}.csv", index=False)
        print(f"[{pool_name}_iso] {held}: c*={c_star} auroc={auroc['point']:.4f} "
              f"[{auroc['lower']:.3f},{auroc['upper']:.3f}]")
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(out_dir / "locked_metrics_all.csv", index=False)
    print(f"[{pool_name}_iso] 4 折均值 AUROC = {metrics['auroc'].mean():.4f}")


def sanity_check() -> None:
    """基线模型锁定数字应与原 E1 一致(协议一致性自检)。"""
    ref = pd.read_csv(REPO / "outputs/aggregate_e2/all_models_fold_metrics_with_ci.csv")
    worst = 0.0
    for _, r in ref.iterrows():
        f = OUT / f"{r['model']}_iso" / f"locked_metrics_{r['heldout_dataset']}.csv"
        if not f.exists():
            continue
        mine = pd.read_csv(f).iloc[0]
        worst = max(worst, abs(mine["auroc"] - r["auroc"]))
    print(f"协议一致性自检:基线 locked AUROC 与原 E1 最大 |Δ| = {worst:.2e}(判定 ≤2e-3)")


def main() -> None:
    only = sys.argv[1:]  # 可选:只跑指定模型名(修复后重跑用)
    pools, singles = discover_models()
    if only:
        pools = {k: v for k, v in pools.items() if k in only}
        singles = [s for s in singles if s[0] in only]
    print(f"pools: {list(pools)}")
    print(f"singles: {[s[0] for s in singles]}")
    for pool, keys in pools.items():
        run_model(pool, keys, "extract")
    for name, source in singles:
        run_model(name, None, source)
    if not only:
        sanity_check()


if __name__ == "__main__":
    main()
