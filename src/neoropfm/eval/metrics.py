"""评估指标(方案 §16 落实):判别 + 筛查工作点 + 校准。

所有函数输入为真实标签 y(0/1)与预测概率 p。
筛查工作点(spec@sens)的阈值由调用方在训练数据上确定后传入。
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def threshold_at_sensitivity(y: np.ndarray, p: np.ndarray, target_sens: float) -> float:
    """返回满足 灵敏度≥target_sens 的**最大**阈值(等价于该灵敏度约束下特异性最大)。

    筛查工作点:阈值只能在训练折确定并锁定,测试折不得重新选择(方案 §14/§24)。
    """
    pos = y == 1
    n_pos = pos.sum()
    if n_pos == 0:
        return float("nan")
    candidates = np.sort(np.unique(p))  # 升序
    for th in candidates[::-1]:  # 从高到低:第一个满足条件的就是最大阈值
        sens = float((p[pos] > th).mean())
        if sens >= target_sens:
            return float(th)
    return float(-np.inf)  # 全部判阳才能达标(此时特异度=0)


def sensitivity_specificity_at_threshold(
    y: np.ndarray, p: np.ndarray, th: float
) -> tuple[float, float]:
    pred = p > th
    tp = int((pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    sens = tp / pos if pos else float("nan")
    spec = tn / neg if neg else float("nan")
    return sens, spec


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error(等宽分箱)。"""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p > bins[i]) & (p <= bins[i + 1])
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += mask.sum() / len(y) * abs(acc - conf)
    return float(ece)


def calibration_summary(
    y: np.ndarray, p: np.ndarray
) -> dict[str, float]:
    """calibration-in-the-large(intercept)与 calibration slope。

    以 logit(p) 对 y 做单变量 logistic 回归:intercept 应≈0、slope 应≈1。
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    from sklearn.linear_model import LogisticRegression

    logit = np.log(p / (1 - p))
    if len(np.unique(y)) < 2:
        return {"cal_intercept": float("nan"), "cal_slope": float("nan")}
    lr = LogisticRegression(C=np.inf, max_iter=2000)  # 无正则(sklearn≥1.8 弃用 penalty=None)
    lr.fit(logit.reshape(-1, 1), y)
    return {
        "cal_intercept": float(lr.intercept_[0]),
        "cal_slope": float(lr.coef_[0][0]),
    }


def compute_all_metrics(
    y: np.ndarray, p: np.ndarray, thresholds: dict[str, float] | None = None
) -> dict[str, float]:
    """一键计算全部指标。thresholds = {'sens95': th, 'sens98': th}(训练折锁定)。"""
    out = {
        "auroc": auroc(y, p),
        "auprc": auprc(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p),
        **calibration_summary(y, p),
    }
    if thresholds:
        for name, th in thresholds.items():
            sens, spec = sensitivity_specificity_at_threshold(y, p, th)
            out[f"spec@{name}"] = spec
            out[f"sens@{name}"] = sens
    return out
