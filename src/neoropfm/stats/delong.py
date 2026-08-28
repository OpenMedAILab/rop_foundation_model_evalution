"""配对 DeLong 检验(方案 §18:不同模型 AUROC 比较)。"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _auc_components(y: np.ndarray, p: np.ndarray):
    """DeLong 结构分量:V10/V01 矩阵的按样本聚合。"""
    y = np.asarray(y)
    p = np.asarray(p)
    pos = y == 1
    neg = y == 0
    p_pos = p[pos]
    p_neg = p[neg]
    n_pos, n_neg = len(p_pos), len(p_neg)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("DeLong 需要两类样本均存在")
    # 对每个正样本:x_i = (#负样本 低于 p_i) / n_neg 的贡献
    # V10_i = (1/n_neg) * sum_j I(p_neg_j < p_pos_i) ; V01_j = (1/n_pos) * sum_i I(p_pos_i > p_neg_j)
    v10 = np.array([np.mean(p_neg < pp) for pp in p_pos])  # 每个正样本的列分量
    v01 = np.array([np.mean(p_pos > pn) for pn in p_neg])  # 每个负样本的行分量
    return v10, v01, n_pos, n_neg


def delong_test(
    y: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> dict[str, float]:
    """配对 DeLong 检验。返回 {"auc1", "auc2", "diff", "z", "p"}。"""
    v10_1, v01_1, n_pos, n_neg = _auc_components(y, p1)
    v10_2, v01_2, _n_pos2, _n_neg2 = _auc_components(y, p2)
    auc1 = float(np.mean(v10_1))
    auc2 = float(np.mean(v10_2))
    # S10 与 S01 的合并协方差(DeLong et al. 1988)
    s10 = np.column_stack([v10_1, v10_2])  # (n_pos, 2)
    s01 = np.column_stack([v01_1, v01_2])  # (n_neg, 2)
    cov10 = np.cov(s10, rowvar=False) * (n_pos - 1) / n_pos if n_pos > 1 else np.zeros((2, 2))
    cov01 = np.cov(s01, rowvar=False) * (n_neg - 1) / n_neg if n_neg > 1 else np.zeros((2, 2))
    S = cov10 / n_pos + cov01 / n_neg
    diff = auc1 - auc2
    L = np.array([1.0, -1.0])
    var = float(L @ S @ L)
    z = diff / np.sqrt(var) if var > 0 else 0.0
    p_val = 2 * (1 - norm.cdf(abs(z)))
    return {"auc1": auc1, "auc2": auc2, "diff": diff, "z": float(z), "p": float(p_val)}
