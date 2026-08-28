"""MAE 掩码建模(RETFound-Green 起点,E2.3 备选路线;方案 §8.2 MAE 分支)。

green 架构:timm vit_small_patch14_reg4_dinov2 @392²,14×14 patch → 28² = 784 个
patch token;token 布局 [cls(1), reg(4), patches(784)](实测:timm reg4-dinov2 的
pos_embed 只覆盖 patch,cls/reg 无位置编码、前置拼接)。

MAE 标准做法:随机掩码 75% patch(不掩 cls/reg),编码器只吃 [cls, reg, 可见 patch]
(注意力成本降到 1/4);小型解码器以可学习 mask token 补齐全部 784 位置 + 解码器
位置编码,重构被掩 patch 的 per-patch 归一化像素,MSE 只算被掩位置。

绿色路线的 PMA 辅助头不在本模块(方案 §8.3 的头挂在主路线 iBOT 上,见 heads.py)。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from torch import nn


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, 3, H, W) → (B, N, patch²·3) 展平像素(unfold 语义,无投影)。"""
    b, c, h, w = x.shape
    p = patch_size
    assert h % p == 0 and w % p == 0
    x = x.reshape(b, c, h // p, p, w // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // p) * (w // p), c * p * p)
    return x


class MAEModel(nn.Module):
    """green 编码器 + 轻量解码器的掩码自编码器。

    encoder: timm vit_small_patch14_reg4_dinov2(global_pool='',需手动拿 token)。
    forward 返回 (loss, mask):mask (B, N) bool,True = 被掩码(供日志/可视化)。
    """

    def __init__(
        self,
        encoder: nn.Module,
        prefix_tokens: int,  # 5 = cls + 4 reg(green);1 = cls(无 reg 架构)
        patch_size: int = 14,
        in_chans: int = 3,
        decoder_dim: int = 192,
        decoder_depth: int = 4,
        decoder_heads: int = 6,
    ):
        super().__init__()
        self.encoder = encoder
        self.prefix_tokens = prefix_tokens
        self.patch_size = patch_size
        emb_dim = encoder.embed_dim
        grid = encoder.patch_embed.grid_size[0]
        self.n_patches = grid * grid
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.dec_embed = nn.Linear(emb_dim, decoder_dim)
        self.dec_pos = nn.Parameter(torch.zeros(1, self.n_patches, decoder_dim))
        self.dec_blocks = nn.ModuleList([
            Block(dim=decoder_dim, num_heads=decoder_heads, mlp_ratio=4.0,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(decoder_depth)
        ])
        self.dec_norm = nn.LayerNorm(decoder_dim)
        self.dec_pred = nn.Linear(decoder_dim, patch_size * patch_size * in_chans)

        import math as _m
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.dec_pos, std=0.02)
        nn.init.xavier_uniform_(self.dec_pred.weight)
        nn.init.zeros_(self.dec_pred.bias)

    def _encode_visible(self, x: torch.Tensor, ids_keep: torch.Tensor):
        """patch_embed + patch pos_embed → [cls, reg, 可见 patch] → blocks → norm。

        返回 (visible_encoded (B, K, D), ids_keep)。
        """
        enc = self.encoder
        emb = enc.patch_embed(x)  # (B, N, D)
        emb = emb + enc.pos_embed  # reg4-dinov2:pos_embed 仅覆盖 patch
        visible = torch.gather(
            emb, 1, ids_keep[:, :, None].expand(-1, -1, emb.shape[-1])
        )
        to_cat = []
        if enc.cls_token is not None:
            to_cat.append(enc.cls_token.expand(x.shape[0], -1, -1))
        if enc.reg_token is not None:
            to_cat.append(enc.reg_token.expand(x.shape[0], -1, -1))
        to_cat.append(visible)
        tokens = torch.cat(to_cat, dim=1)
        tokens = enc.pos_drop(tokens)
        if enc.norm_pre is not None:
            tokens = enc.norm_pre(tokens)
        for blk in enc.blocks:
            tokens = blk(tokens)
        tokens = enc.norm(tokens)
        return tokens[:, self.prefix_tokens:], ids_keep  # 去掉 cls/reg 前缀

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.75):
        B, _, H, W = x.shape
        n = self.n_patches
        # 随机掩码(patch 均匀随机,MAE 原文;不加块结构)
        noise = torch.rand(B, n, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        n_masked = int(n * mask_ratio)
        ids_keep = ids_shuffle[:, n_masked:].sort(dim=1).values  # 可见 patch 索引(升序)
        mask = torch.ones(B, n, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_keep, False)

        visible_enc, _ = self._encode_visible(x, ids_keep)  # (B, K, D)
        target = patchify(x, self.patch_size)  # (B, N, 588) 原始(归一化后)像素

        # ---- 解码器:全部 784 位置,可见位置嵌入编码器输出 ----
        # amp 下 dtype 对齐要点(2026-08-21 实测修复):green 编码器末层 LayerNorm
        # 在 autocast 白名单内输出 fp32,而 dec_embed(Linear)输出 bf16;scatter_
        # 是普通原地算子不受 autocast 管辖,必须以 Linear 输出 emb 的 dtype 为
        # 准初始化 mask 网格(以 visible_enc.dtype 对齐是错的——它会因 LayerNorm
        # 落到 fp32,首修仍崩,冒烟测试须在 amp 上下文内做)。
        emb = self.dec_embed(visible_enc)
        dec = self.mask_token.expand(B, n, -1).clone().to(emb.dtype)
        dec.scatter_(1, ids_keep[:, :, None].expand(-1, -1, dec.shape[-1]), emb)
        dec = dec + self.dec_pos.to(dec.dtype)
        for blk in self.dec_blocks:
            dec = blk(dec)
        dec = self.dec_norm(dec)
        pred = self.dec_pred(dec)  # (B, N, 588)

        loss = self._loss(pred, target, mask)
        return loss, mask

    @staticmethod
    def _loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """被掩位置 per-patch 归一化 MSE(MAE 原文口径)。"""
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5
        se = ((pred - target) ** 2).mean(dim=-1)  # (B, N)
        return (se * mask).sum() / (mask.sum() + 1e-8)
