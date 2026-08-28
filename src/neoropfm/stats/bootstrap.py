"""Patient-level bootstrap 置信区间(方案 §18:≥2,000 次,固定种子)。

以患者为单位重采样(同患者全部记录同进同出),避免 patient leakage。
无 patient-ID 的数据(如 SZEH)按 image-level bootstrap,调用方显式说明。
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def patient_level_bootstrap(
    y: np.ndarray,
    p: np.ndarray,
    patient_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """对 metric_fn 做 patient-level bootstrap,返回 2.5/97.5 百分位 CI。

    Returns: {"point": ..., "lower": ..., "upper": ...}
    """
    rng = np.random.default_rng(seed)
    uniq_patients = np.unique(patient_ids)
    n_pat = len(uniq_patients)
    point = metric_fn(y, p)
    stats = []
    # 预分组:每患者 → 行索引
    groups = {pid: np.where(patient_ids == pid)[0] for pid in uniq_patients}
    for _ in range(n_boot):
        sample_pids = rng.choice(uniq_patients, size=n_pat, replace=True)
        idx = np.concatenate([groups[pid] for pid in sample_pids])
        ys, ps = y[idx], p[idx]
        if len(np.unique(ys)) < 2:
            continue  # 该次重采样全为单类,指标不可定义
        stats.append(metric_fn(ys, ps))
    stats = np.array(stats)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "lower": float(lo), "upper": float(hi), "n_valid": int(len(stats))}


def paired_bootstrap_diff(
    y: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    patient_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """两模型配对比较的 bootstrap(用于 AUPRC、工作点指标等无 DeLong 的指标)。

    Returns: {"diff": metric1-metric2, "lower": ..., "upper": ..., "p": 双侧经验 p 值}
    CI 为对称 bootstrap 区间(θ̂ ± Q_{1-α}(|d*−θ̂|)),与中心化 p 检验严格一致;
    逐折表仅作描述。
    """
    rng = np.random.default_rng(seed)
    uniq_patients = np.unique(patient_ids)
    n_pat = len(uniq_patients)
    groups = {pid: np.where(patient_ids == pid)[0] for pid in uniq_patients}
    diffs = []
    for _ in range(n_boot):
        sample_pids = rng.choice(uniq_patients, size=n_pat, replace=True)
        idx = np.concatenate([groups[pid] for pid in sample_pids])
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        diffs.append(metric_fn(ys, p1[idx]) - metric_fn(ys, p2[idx]))
    diffs = np.array(diffs)
    point = metric_fn(y, p1) - metric_fn(y, p2)
    # 对称 bootstrap CI:θ̂ ± Q_{1-α}(|d* − θ̂|)。与中心化 p 检验逐样本一致
    # (CI 排除 0 ⟺ |θ̂| > Q_{1-α} ⟺ p < α,至 +1 下限内)。
    # 百分位/basic CI 在偏态分布下都会与中心化检验矛盾(CI 排除 0 但 p>α)。
    q = np.quantile(np.abs(diffs - point), 1.0 - alpha)
    lo, hi = point - q, point + q
    # 双侧经验 p:以观测差为中心计数(|d* − d_obs| ≥ |d_obs|),下限 (k+1)/(n_boot+1)
    # —— k=0 时 p = 1/(n_boot+1),论文写法 "p < 1/2001"。
    # 中心化计数对 tie(两模型预测恒等)稳健——旧式 2·min(share≥0, share≤0) 在 tie 下
    # 会给出 p>1。
    k = int((np.abs(diffs - point) >= abs(point)).sum())
    p_val = (k + 1.0) / (len(diffs) + 1.0)
    return {"diff": float(point), "lower": float(lo), "upper": float(hi), "p": float(p_val)}


def fold_mean_delta_bootstrap(
    folds: list[dict],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """跨折均值差值 CI(摘要级"总体差值"的 CI)。

    输入 folds:每折一个 dict:
      {"y": np.ndarray, "p1": np.ndarray, "p2": np.ndarray,
       "units": np.ndarray(患者簇 id,同簇同进同出)}
    每折独立做 patient-level bootstrap(同一迭代内 4 折用同一重采样种子流),
    取各折差值的均值 → 均值差值分布。点估 = 各折点差的等权均值。
    p = (k+1)/(n_boot+1),k = 中心化计数(|d* − d_obs| ≥ |d_obs|)。

    Returns: {"diff", "lower", "upper", "p", "n_folds"}
    """
    n_folds = len(folds)
    assert n_folds > 0, "folds 为空"
    points = []
    rngs = [np.random.default_rng(seed + i * 10_000) for i in range(n_folds)]
    groups = []
    for fold in folds:
        uniq = np.unique(fold["units"])
        g = {pid: np.where(fold["units"] == pid)[0] for pid in uniq}
        groups.append((uniq, g))
        points.append(roc_auc_guard(fold["y"], fold["p1"]) - roc_auc_guard(fold["y"], fold["p2"]))

    def _fold_diff(fi: int, rng) -> float:
        fold = folds[fi]
        uniq, g = groups[fi]
        idx = np.concatenate([g[pid] for pid in rng.choice(uniq, size=len(uniq), replace=True)])
        ys = fold["y"][idx]
        if len(np.unique(ys)) < 2:
            return np.nan
        return roc_auc_guard(ys, fold["p1"][idx]) - roc_auc_guard(ys, fold["p2"][idx])

    means = []
    for _ in range(n_boot):
        ds = [_fold_diff(fi, rngs[fi]) for fi in range(n_folds)]
        ds = [d for d in ds if not np.isnan(d)]
        if not ds:
            continue
        means.append(float(np.mean(ds)))
    means = np.array(means)
    point = float(np.mean(points))
    # 对称 bootstrap CI:与中心化 p 一致(见 paired_bootstrap_diff 注释)
    q = np.quantile(np.abs(means - point), 1.0 - alpha)
    lo, hi = point - q, point + q
    k = int((np.abs(means - point) >= abs(point)).sum())
    p_val = (k + 1.0) / (len(means) + 1.0)
    return {"diff": point, "lower": float(lo), "upper": float(hi),
            "p": float(p_val), "n_folds": n_folds}


def roc_auc_guard(y: np.ndarray, p: np.ndarray) -> float:
    """AUROC 的轻量本地实现(避免 stats 模块对 sklearn 的耦合)。"""
    from sklearn.metrics import roc_auc_score  # 延迟导入,仅此处使用

    return float(roc_auc_score(y, p))
