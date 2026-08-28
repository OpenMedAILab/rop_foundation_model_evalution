"""纵向预测基线模型(E3.2)。

5 个基线(方案 §15 / 实验计划 E3.2):
  1. clinical_only         : GA/BW/sex/PMA/访视次数(无图像)
  2. current_image_only    : index 访视冻结特征(横断面)
  3. static_multimodal     : clinical + current image
  4. longitudinal_image    : 历史访视特征聚合(mean/max/last/trend/delta)
  5. longitudinal_multimodal: clinical + longitudinal image

评估协议:
  - 患者级 5-fold 交叉验证(StratifiedGroupKFold,按 patient_id 分组,
    按 y_next 分层),禁止患者泄漏
  - StandardScaler(fit on train) + LogisticRegression(class_weight="balanced", C=1.0)
  - 筛查工作点阈值锁定在训练折
  - 输出 AUROC/AUPRC/Brier/ECE/spec@sens + bootstrap 95% CI

运行:
  python -m neoropfm.train.longitudinal_baselines
  python -m neoropfm.train.longitudinal_baselines --feature-model retfound_green
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.common import OUTPUTS_DIR, seed_everything  # noqa: E402
from neoropfm.eval.metrics import compute_all_metrics, threshold_at_sensitivity  # noqa: E402

LONG_DIR = OUTPUTS_DIR / "longitudinal"

# 临床变量列
CLINICAL_COLS = ["ga", "bw", "sex_code", "pma_or_time", "n_history"]


def encode_sex(sex_val) -> float:
    """sex → 0/1/0.5(boy=1, girl=0, unknown=0.5)。"""
    if pd.isna(sex_val):
        return 0.5
    s = str(sex_val).lower().strip()
    if s in ("boy", "male", "m", "1"):
        return 1.0
    if s in ("girl", "female", "f", "0"):
        return 0.0
    return 0.5


def build_clinical_features(manifest: pd.DataFrame) -> np.ndarray:
    """构建临床变量矩阵 [N, 5]。"""
    X = np.zeros((len(manifest), len(CLINICAL_COLS)), dtype=np.float32)
    for i, (_, row) in enumerate(manifest.iterrows()):
        X[i, 0] = row.ga if pd.notna(row.ga) else np.nan
        X[i, 1] = row.bw if pd.notna(row.bw) else np.nan
        X[i, 2] = encode_sex(row.sex)
        # pma_or_time: RIDIRP 用 PMA, ROP-VL 用距首次访视天数(在 seq times 中)
        X[i, 3] = np.nan  # 稍后从 seq 填充
        X[i, 4] = row.n_history
    return X


def build_longitudinal_image_features(
    seq_feat: np.ndarray, seq_mask: np.ndarray
) -> np.ndarray:
    """从访次特征序列构建聚合特征 [N, D*5]。

    聚合方式:
    - last:  最近一次(index)访视特征
    - mean:  所有历史访视均值
    - max:   所有历史访视逐维最大值
    - trend: last - first(变化趋势)
    - delta: last - previous(最近变化)
    """
    N, T, D = seq_feat.shape
    feats = np.zeros((N, D * 5), dtype=np.float32)

    for i in range(N):
        m = seq_mask[i].astype(bool)
        f = seq_feat[i, m]  # [t, D]
        t = len(f)

        last = f[-1]
        mean = f.mean(axis=0)
        mx = f.max(axis=0)
        first = f[0]
        prev = f[-2] if t >= 2 else f[-1]

        feats[i, :D] = last
        feats[i, D:2*D] = mean
        feats[i, 2*D:3*D] = mx
        feats[i, 3*D:4*D] = last - first
        feats[i, 4*D:5*D] = last - prev

    return feats


def bootstrap_ci(y_true, y_prob, n_boot=1000, seed=0, alpha=0.05):
    """患者级 bootstrap 95% CI(AUROC/AUPRC)。"""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    aps = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
            aps.append(average_precision_score(y_true[idx], y_prob[idx]))
        except Exception:
            pass
    if not aucs:
        return (np.nan, np.nan), (np.nan, np.nan)
    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100
    return (
        (np.percentile(aucs, lo), np.percentile(aucs, hi)),
        (np.percentile(aps, lo), np.percentile(aps, hi)),
    )


def run_baseline(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """运行单个基线配置,返回 (metrics_df, predictions_df)。"""
    # 用 -999 填充缺失值(StandardScaler 会处理)
    X = np.nan_to_num(X, nan=-999.0)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_preds = []
    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, y, groups)):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]

        scaler = StandardScaler().fit(Xtr)
        lr = LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=5000, random_state=seed
        )
        lr.fit(scaler.transform(Xtr), ytr)

        ptr = lr.predict_proba(scaler.transform(Xtr))[:, 1]
        pte = lr.predict_proba(scaler.transform(Xte))[:, 1]

        # 训练折锁定阈值
        thresholds = {
            "sens95": threshold_at_sensitivity(ytr, ptr, 0.95),
            "sens98": threshold_at_sensitivity(ytr, ptr, 0.98),
        }
        m = compute_all_metrics(yte, pte, thresholds=thresholds)
        m["fold"] = fold
        m["train_n"] = len(tr_idx)
        m["test_n"] = len(te_idx)
        m["train_pos"] = int(ytr.sum())
        m["test_pos"] = int(yte.sum())
        m["threshold_sens95_locked"] = thresholds["sens95"]
        m["threshold_sens98_locked"] = thresholds["sens98"]
        fold_metrics.append(m)

        for j, idx in enumerate(te_idx):
            all_preds.append({
                "model": name,
                "fold": fold,
                "patient_id": groups[idx],
                "y": yte[j],
                "p": pte[j],
            })

    fold_df = pd.DataFrame(fold_metrics)
    preds_df = pd.DataFrame(all_preds)

    # 汇总: 所有测试折预测拼接后计算总体指标
    y_all = preds_df.y.values
    p_all = preds_df.p.values
    auc_ci, ap_ci = bootstrap_ci(y_all, p_all, seed=seed)

    # 阈值: 用训练折锁定阈值的中位数
    th95 = fold_df["threshold_sens95_locked"].median()
    th98 = fold_df["threshold_sens98_locked"].median()
    overall = compute_all_metrics(y_all, p_all, thresholds={
        "sens95": th95, "sens98": th98
    })
    overall["model"] = name
    overall["n_samples"] = len(y_all)
    overall["n_events"] = int(y_all.sum())
    overall["n_patients"] = preds_df.patient_id.nunique()
    overall["auroc_ci_lo"] = auc_ci[0]
    overall["auroc_ci_hi"] = auc_ci[1]
    overall["auprc_ci_lo"] = ap_ci[0]
    overall["auprc_ci_hi"] = ap_ci[1]

    return pd.DataFrame([overall]), preds_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-model", default="retfound_green")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    LONG_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    manifest = pd.read_csv(LONG_DIR / f"longitudinal_manifest_{args.feature_model}.csv")
    seq = np.load(LONG_DIR / f"visit_sequences_{args.feature_model}.npz")
    seq_feat = seq["features"]
    seq_mask = seq["masks"]
    seq_time = seq["times"]

    print(f"Loaded {len(manifest)} samples, features {seq_feat.shape}")
    print(f"Events: {manifest.y_next.sum()} / {len(manifest)}")

    y = manifest.y_next.values.astype(int)
    groups = manifest.patient_id.values

    # 临床特征
    X_clin = build_clinical_features(manifest)
    # 填充 PMA/time: RIDIRP 用 index_time(PMA), ROP-VL 用距首次访视天数
    for i, row in manifest.iterrows():
        if row.dataset == "ridirp":
            X_clin[i, 3] = row.index_time  # PMA 周
        else:
            # ROP-VL: index_time 是日期字符串,用 seq_time 的最后一个有效值(距首次天数)
            m = seq_mask[i].astype(bool)
            X_clin[i, 3] = seq_time[i, m][-1] / 7.0  # 天→周

    # 当前图像特征(index 访视 = 序列最后一个有效位置)
    D = seq_feat.shape[2]
    X_current = np.zeros((len(manifest), D), dtype=np.float32)
    for i in range(len(manifest)):
        m = seq_mask[i].astype(bool)
        X_current[i] = seq_feat[i, m][-1]

    # 纵向图像特征
    X_long_img = build_longitudinal_image_features(seq_feat, seq_mask)

    # 5 个基线配置
    configs = [
        ("clinical_only", X_clin),
        ("current_image_only", X_current),
        ("static_multimodal", np.concatenate([X_clin, X_current], axis=1)),
        ("longitudinal_image", X_long_img),
        ("longitudinal_multimodal", np.concatenate([X_clin, X_long_img], axis=1)),
    ]

    all_metrics = []
    all_preds = []

    for name, X in configs:
        print(f"\n--- {name} (X shape: {X.shape}) ---")
        mdf, pdf = run_baseline(name, X, y, groups, args.n_splits, args.seed)
        all_metrics.append(mdf)
        all_preds.append(pdf)
        print(f"  AUROC={mdf.auroc.iloc[0]:.4f} [{mdf.auroc_ci_lo.iloc[0]:.4f}-{mdf.auroc_ci_hi.iloc[0]:.4f}]  "
              f"AUPRC={mdf.auprc.iloc[0]:.4f}  "
              f"spec@95sens={mdf['spec@sens95'].iloc[0]:.4f}  "
              f"Brier={mdf.brier.iloc[0]:.4f}")

    # 保存
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    preds_df = pd.concat(all_preds, ignore_index=True)

    metrics_path = LONG_DIR / f"baseline_metrics_{args.feature_model}.csv"
    preds_path = LONG_DIR / f"baseline_predictions_{args.feature_model}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)

    print(f"\n=== Summary ===")
    print(metrics_df[["model", "n_samples", "n_events", "auroc", "auroc_ci_lo", "auroc_ci_hi",
                       "auprc", "spec@sens95", "brier", "ece"]].to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {preds_path}")


if __name__ == "__main__":
    main()
