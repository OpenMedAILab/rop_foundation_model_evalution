"""E2.4 消融辅助头(PMA 回归 + visit-consistency InfoNCE;方案 §8.3/§8.4)。

主路线(iBOT)默认不挂(w=0),E2.4 消融开启后对比:
- PMAHead:[CLS] → MLP → 发育年龄回归(MSE;PMA 缺失行掩码)。RIDIRP 约 6,004 张
  带逐访次 PMA,其余数据集 NaN——正是 masked-regression 的目标场景。
- visit_consistency_loss:同 (dataset, patient) 全局视图互为正对的 InfoNCE;
  multi-crop 保证每张图 2 个全局视图 → 每个 visit 至少 2 视图,正对恒存在。
  多图 visit 时天然构成 set-level 对比(方案 §8.4 轻量实现)。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PMAHead(nn.Module):
    """[CLS] → MLP → PMA 标量(z-score 目标,由脚本从 manifest 统计传入)。"""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.mlp(cls).squeeze(-1)


def pma_loss(pred: torch.Tensor, pma_z: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """masked MSE:PMA 缺失的行不参与(valid 为该行有 PMA)。"""
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.mse_loss(pred[valid], pma_z[valid])


def visit_consistency_loss(z: torch.Tensor, visit_ids: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """visit-level InfoNCE(SupCon 式:同 visit 的视图互为正对,其余为负)。

    z: (M, D) 学生全局视图 CLS 投影;visit_ids: (M,) 长整型 visit 编号。
    每图 2 个全局视图 → 单图 visit 也有正对;多图 visit 构成 set-level 对比。
    """
    z = F.normalize(z, dim=-1)
    M = z.shape[0]
    sim = z @ z.T / tau  # (M, M)
    same_visit = visit_ids[:, None] == visit_ids[None, :]
    self_mask = torch.eye(M, dtype=torch.bool, device=z.device)
    pos = same_visit & ~self_mask
    if not pos.any():
        return torch.tensor(0.0, device=z.device)
    log_num = torch.logsumexp(sim.masked_fill(~pos, float("-inf")), dim=1)
    log_den = torch.logsumexp(sim.masked_fill(self_mask, float("-inf")), dim=1)
    return (log_den - log_num).mean()
