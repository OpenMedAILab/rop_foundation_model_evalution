"""Holm 校正(方案 §18:多重 primary model comparisons)。"""
from __future__ import annotations

import numpy as np


def holm_correct(p_values: list[float]) -> list[float]:
    """Holm–Bonferroni 校正。返回与输入同序的校正后 p 值。"""
    n = len(p_values)
    order = np.argsort(p_values)
    corrected = np.zeros(n)
    for rank, idx in enumerate(order):
        corrected[idx] = min(1.0, p_values[idx] * (n - rank))
    # 单调性修正(按排序后的次序累计取 max)
    for rank in range(1, n):
        corrected[order[rank]] = max(corrected[order[rank]], corrected[order[rank - 1]])
    return corrected.tolist()
