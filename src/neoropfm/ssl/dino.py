"""DINO/iBOT 组件(E2.2 continued pretraining,iBOT 路线)。

对应方案 §8.2「teacher–student self-distillation + masked-token representation
learning」:DINO 全局视图 CLS 蒸馏 + iBOT 掩码 patch token 自蒸馏,动量教师 + 居中
(centering)。实现按 facebookresearch/dinov2 训练口径简化:

- DINOHead:backbone 输出 → 2×hidden MLP → L2norm → wn(bottleneck) → wn(out_dim)
- iBOTHead:patch token → 2 层 MLP → out_dim
- BlockwiseMask:iBOT 官方块状掩码(BEiT 生成器同款:随机长宽比块覆盖 mask_ratio)
- dino_loss / ibot_loss:交叉熵自蒸馏(教师软目标 detach + 居中 + 低温)
- update_teacher:EMA 动量更新(教师含头,与学生同构)
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def _trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.02) -> None:
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-2 * std, b=2 * std)


class DINOHead(nn.Module):
    """DINO 投影头;weight_g 置 0 梯度,配合 freeze_last_layer_epochs 先冻结。

    weight_norm 的重参数 (weight_g, weight_v):推理时由两者合成权重,训练时
    冻结 weight_g(scale)只学方向,是 DINO 的稳定性技巧。
    """

    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int = 1024, bottleneck: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(hidden, bottleneck, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False
        self.head = nn.utils.weight_norm(nn.Linear(bottleneck, out_dim, bias=False))
        self.head.weight_g.data.fill_(1)
        self.head.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return self.head(x)

    def unfreeze_last_layer(self) -> None:
        """冻结期(默认第 1 个 epoch)结束后放开投影层。"""
        self.last_layer.weight_g.requires_grad = True


class iBOTHead(nn.Module):
    """patch token 投影头(iBOT 口径:2 层 MLP,无 L2norm/wn 链)。"""

    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int = 384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class BlockwiseMask:
    """iBOT 块状掩码(BEiT masking generator 同款随机化)。

    在 grid×grid 的 patch 网格上反复采样随机长宽比块,直至覆盖 ≥ mask_ratio。
    返回形状 (N,) 的 bool 向量,True = 被掩码。
    """

    def __init__(self, mask_ratio: float = 0.35, aspect: tuple = (0.75, 1.5), seed: int = 0):
        self.mask_ratio = mask_ratio
        self.aspect = aspect
        self.rng = np.random.default_rng(seed)

    def _mask_one(self, grid: int) -> torch.Tensor:
        mask = torch.zeros(grid * grid, dtype=torch.bool)
        n_masked = int(grid * grid * self.mask_ratio)
        log_aspect = math.log(self.aspect[1] / self.aspect[0])
        for _ in range(100):  # 接受-拒绝采样上限,防死循环
            if int(mask.sum()) >= n_masked:
                break
            aspect = math.exp(self.rng.uniform(-log_aspect, log_aspect))
            target_area = grid * grid * self.mask_ratio
            bh = int(round(math.sqrt(target_area * aspect)))
            bw = int(round(math.sqrt(target_area / aspect)))
            if bw >= grid or bh >= grid:
                continue
            top = int(self.rng.integers(0, grid - bh + 1))
            left = int(self.rng.integers(0, grid - bw + 1))
            mask_2d = mask.view(grid, grid)
            mask_2d[top:top + bh, left:left + bw] = True
        return mask

    def __call__(self, batch_size: int, grid: int) -> torch.Tensor:
        return torch.stack([self._mask_one(grid) for _ in range(batch_size)])


def dino_total_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_temp: float,
    student_temp: float,
    center: torch.Tensor | None = None,
    n_local_crops: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DINO CLS 自蒸馏(facebookresearch/dinov2 官方口径)。

    学生 (B×(2+n_local), K):[g1, g2, l1, l2] 按视图顺序堆叠;
    教师 (2B, K):[g1, g2]。每个学生视图与每个教师视图配对求 CE,**跳过同一裁剪的
    配对**(v == iq)——避免"学生 g1 抄教师 g1 自身"的平凡解。

    返回 (loss, teacher_prob):后者为居中+低温后的教师软目标(2B, K),
    detach,供 centering EMA 更新(与官方一致:对全部教师视图取 batch 均值滑入)。
    """
    teacher_out = teacher_logits.detach()
    if center is not None:
        teacher_out = teacher_out - center
    teacher_prob = torch.softmax(teacher_out / teacher_temp, dim=-1).detach()
    t_q = teacher_prob.chunk(2)  # [g1, g2],各 (B, K)
    student_out = student_logits / student_temp
    s_v = student_out.chunk(2 + n_local_crops)  # [g1, g2, l1, ...],各 (B, K)
    total = torch.zeros((), device=student_logits.device)
    n_terms = 0
    for iq, q in enumerate(t_q):
        for v, sv in enumerate(s_v):
            if v == iq:  # 学生与教师作用于同一裁剪 → 跳过
                continue
            total = total + torch.sum(-q * torch.log_softmax(sv, dim=-1), dim=-1).mean()
            n_terms += 1
    return total / n_terms, teacher_prob


def ibot_loss(
    student_patch_logits: torch.Tensor,
    teacher_patch_logits: torch.Tensor,
    teacher_temp: float,
    student_temp: float,
    center: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """iBOT patch 自蒸馏(仅被掩码位置;输入已按位置对齐)。

    返回 (loss, teacher_prob),同 dino_loss。
    """
    teacher_out = teacher_patch_logits.detach()
    if center is not None:
        teacher_out = teacher_out - center
    teacher_prob = torch.softmax(teacher_out / teacher_temp, dim=-1).detach()
    student_prob = torch.log_softmax(student_patch_logits / student_temp, dim=-1)
    return -(teacher_prob * student_prob).sum(dim=-1).mean(), teacher_prob


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    """教师 = EMA(学生)(含投影头;DINO 口径教师参数动量更新)。"""
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


@torch.no_grad()
def init_teacher(student: nn.Module, teacher: nn.Module) -> None:
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.copy_(ps.data)


class Center:
    """居中缓冲:教师输出的指数滑动均值(CLS 与 patch 各一份)。"""

    def __init__(self, dim: int, momentum: float = 0.9, device: str = "cpu"):
        self.center = torch.zeros(1, dim, device=device)
        self.momentum = momentum

    @torch.no_grad()
    def update(self, teacher_logits: torch.Tensor) -> None:
        # 平均到 batch 维后按动量滑入;教师输出取每个样本均值(含多视图)
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center = self.center * self.momentum + batch_center * (1 - self.momentum)

    def to(self, device) -> "Center":
        self.center = self.center.to(device)
        return self
