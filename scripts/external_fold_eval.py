"""P6b: HVDROPDB 外部折评估(untouched external cohort,图像级)。

协议(镜像 probe.py 的探针协议,训练集 = 4 个内部数据集 strict 全量 7,815 行):
- 每个模型:StandardScaler(fit train)+ LogisticRegression(balanced, C=1.0,
  max_iter=5000, random_state=0),7,815 内部图训练 → 185 张 hvdropdb 图一次性评估。
- 特征:主源(cache/extract)+ outputs/features_hvdropdb/{model}/ 追加
  (训练侧特征零触碰,内部 4 折锁定数字不受影响)。
- ckpt 口径:各 SSL run 按 outputs/ssl/{run}/checkpoint_selection.csv 的**内部 4 折
  LODO 均值最优**(与 loto_analyze 同口径),无外部测试集选点泄漏;基线单权重模型无选择。
- CI:HVDROPDB 无患者标识 → 每图独立成簇,patient_level_bootstrap 退化为
  图像级 bootstrap(2,000 次,seed 42),如实标注。
- 工作点:spec@95/98 阈值锁定在训练折(与主探针一致),输出外部折 spec/PPV/NPV。

输出:
  outputs/aggregate_e2/external_hvdropdb_metrics.csv
  outputs/external_fold/{model}_hvdropdb_predictions.csv

运行:python3 scripts/external_fold_eval.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neoropfm.common import OUTPUTS_DIR, SPLITS_DIR, seed_everything  # noqa: E402
from neoropfm.eval.metrics import threshold_at_sensitivity  # noqa: E402
from neoropfm.stats.bootstrap import patient_level_bootstrap  # noqa: E402

EXTRACT = Path("outputs/features")
CACHE = Path(os.environ.get("NEOROPFM_BENCHMARK_CACHE", str(REPO / "outputs" / "benchmark_cache")))  # reference baseline cache (see README)
EXTRA = Path("outputs/features_hvdropdb")
OUT_DIR = OUTPUTS_DIR / "external_fold"
AGG = OUTPUTS_DIR / "aggregate_e2" / "external_hvdropdb_metrics.csv"

# (model_key, feature_source);source=cache → yesterday 基线缓存(CPU 提取)
MODELS: list[tuple[str, str]] = [
    ("efficientnet_b0", "cache"),
    ("convnext_tiny", "cache"),
    ("dinov2_vits14", "cache"),
    ("retfound_mae_cfp", "cache"),
    ("retfound_green", "extract"),
    ("retfound_green_224", "extract"),
    ("dinov2_vitb14", "extract"),
    ("dinov2_vits14_392", "extract"),
    ("ibot_dinov2s_v1_ckpt_ep040", "extract"),
    ("ibot_dinov2s_heads_ckpt_ep040", "extract"),
    ("mae_retfound_green_v1_ckpt_ep090", "extract"),
    ("loto_ibot_dinov2s_heads_minus_farfum_rop_ckpt_final", "extract"),
    ("loto_ibot_dinov2s_heads_minus_ridirp_ckpt_ep050", "extract"),
    ("loto_ibot_dinov2s_heads_minus_rop_vl_ckpt_ep050", "extract"),
    # P1a 重训池(checkpoint_selection.csv 最优 = ckpt_final):
    ("loto_ibot_dinov2s_heads_minus_szeh_irops_rerun_20260823_ckpt_final", "extract"),
]
N_BOOT, BOOT_SEED = 2000, 42


def load_feat(key: str, source: str) -> tuple[np.ndarray, list[str]]:
    d = (EXTRACT if source == "extract" else CACHE) / key
    feat = np.load(d / f"{key}_features.npy")
    ids = pd.read_csv(d / f"{key}_sample_ids.csv")["sample_id"].tolist()
    x = EXTRA / key
    if (x / f"{key}_features.npy").exists():
        fx = np.load(x / f"{key}_features.npy")
        ix = pd.read_csv(x / f"{key}_sample_ids.csv")["sample_id"].tolist()
        assert not (set(ids) & set(ix)), f"{key}: extra 与主特征重叠"
        feat = np.concatenate([feat, fx], axis=0)
        ids = ids + ix
    return feat, ids


def main() -> None:
    seed_everything(0)
    sp = pd.read_csv(SPLITS_DIR / "lodo_test_hvdropdb.csv")
    tr = sp[sp["split"] == "train"]
    te = sp[sp["split"] == "test"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, source in MODELS:
        feat, ids = load_feat(key, source)
        have_tr = np.isin(np.array(ids), tr["sample_id"])
        have_te = np.isin(np.array(ids), te["sample_id"])
        if have_te.sum() != len(te):
            print(f"[skip] {key}: 外部特征缺失({have_te.sum()}/{len(te)})")
            continue
        ids_arr = np.array(ids)
        Xtr = feat[have_tr]
        ytr = tr.set_index("sample_id").loc[ids_arr[have_tr]]["strict_binary_label"].to_numpy()
        Xte = feat[have_te]
        yte = te.set_index("sample_id").loc[ids_arr[have_te]]["strict_binary_label"].to_numpy()

        scaler = StandardScaler().fit(Xtr)
        lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=5000,
                                random_state=0)
        lr.fit(scaler.transform(Xtr), ytr)
        ptr = lr.predict_proba(scaler.transform(Xtr))[:, 1]
        pte = lr.predict_proba(scaler.transform(Xte))[:, 1]

        auc = patient_level_bootstrap(yte, pte, ids_arr[have_te], roc_auc_score,
                                      n_boot=N_BOOT, seed=BOOT_SEED)
        ap = patient_level_bootstrap(yte, pte, ids_arr[have_te], average_precision_score,
                                     n_boot=N_BOOT, seed=BOOT_SEED)
        t95 = threshold_at_sensitivity(ytr, ptr, 0.95)
        t98 = threshold_at_sensitivity(ytr, ptr, 0.98)
        pred95 = (pte >= t95).astype(int)
        pred98 = (pte >= t98).astype(int)
        pos = yte.astype(bool)

        def op_at(pred: np.ndarray) -> tuple[float, float, float]:
            tp = int(((pred == 1) & pos).sum())
            tn = int(((pred == 0) & ~pos).sum())
            fp = int(((pred == 1) & ~pos).sum())
            fn = int(((pred == 0) & pos).sum())
            sens = tp / max(1, tp + fn)
            spec = tn / max(1, tn + fp)
            ppv = tp / max(1, tp + fp)
            npv = tn / max(1, tn + fn)
            return sens, spec, ppv, npv

        s95, sp95, ppv95, npv95 = op_at(pred95)
        s98, sp98, ppv98, npv98 = op_at(pred98)
        rows.append({
            "model": key, "train_n": len(ytr), "test_n": len(yte),
            "test_positive": int(yte.sum()),
            "auroc": round(auc["point"], 4), "auroc_lo": round(auc["lower"], 4),
            "auroc_hi": round(auc["upper"], 4),
            "auprc": round(ap["point"], 4), "auprc_lo": round(ap["lower"], 4),
            "auprc_hi": round(ap["upper"], 4),
            "spec@95sens": round(sp95, 4), "sens@95sens": round(s95, 4),
            "ppv@95sens": round(ppv95, 4), "npv@95sens": round(npv95, 4),
            "spec@98sens": round(sp98, 4), "sens@98sens": round(s98, 4),
            "ppv@98sens": round(ppv98, 4), "npv@98sens": round(npv98, 4),
            "note": "图像级外部折;无患者标识 → 图像级 bootstrap CI",
        })
        pd.DataFrame({
            "sample_id": ids_arr[have_te], "dataset": "hvdropdb",
            "patient_id": ids_arr[have_te], "subset": "test",
            "y": yte, "p": pte,
        }).to_csv(OUT_DIR / f"{key}_hvdropdb_predictions.csv", index=False)
        print(f"{key}: AUROC={auc['point']:.4f} [{auc['lower']:.4f},{auc['upper']:.4f}] "
              f"AUPRC={ap['point']:.4f} pos={int(yte.sum())}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(AGG, index=False)
    print(df.to_string(index=False))
    print(f"saved → {AGG}")


if __name__ == "__main__":
    main()
