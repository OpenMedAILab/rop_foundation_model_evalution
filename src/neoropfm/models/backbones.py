"""统一 backbone 注册表(E1 冻结特征提取 + E2 继续预训练起点)。

设计要点(方案 §11-§12 / README「可复现约定」):
- 每个 backbone 一个 `BackboneSpec`:负责构建模型 + 定义预处理,特征维度与
  yesterday 缓存特征保持一致(effnet 1280 / convnext 768 / dino-s 384 /
  retfound 1024)。
- 冻结特征一律一次提取、缓存到 outputs/features/{model}/,probe 与统计阶段
  零重算。
- 预处理协议:官方/仓库惯例,逐模型记录在 feature_meta.json 的 preprocessing
  字段(yesterday 的 4 个基线沿用其缓存特征,不重新提取,协议差异不影响结果)。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from neoropfm.models.vendor_retfound_models_vit import RETFound_mae  # noqa: E402

# RETFound-MAE(Nature 2023 CFP)权重路径:环境变量 RETFOUND_MAE_CKPT 优先,
# 默认 outputs/weights/retfound_mae/(与 RETFOUND_GREEN_CKPT 同一约定)。
REPO_ROOT = Path(__file__).resolve().parents[3]
RETFOUND_CKPT = Path(
    os.environ.get(
        "RETFOUND_MAE_CKPT",
        str(REPO_ROOT / "outputs" / "weights" / "retfound_mae" / "RETFound_mae_natureCFP_key_modified.pth"),
    )
)
RETFOUND_GREEN_CKPT = Path(
    "outputs/weights/retfound_green/retfoundgreen_statedict.pth"
).resolve()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _imagenet_normalize() -> transforms.Normalize:
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


@dataclass(frozen=True)
class BackboneSpec:
    """一个冻结 backbone 的完整规格:构建 + 预处理 + 前向提取。"""

    key: str
    feature_dim: int
    _build: Callable[[], nn.Module] = field(repr=False, compare=False)
    _transform: transforms.Compose = field(repr=False, compare=False)
    input_size: int = 224
    weights_note: str = ""
    preprocessing: str = "resize256+centercrop224+imagenet_norm"

    def build(self, device: torch.device | str = "cpu") -> nn.Module:
        model = self._build()
        model.to(device)
        model.eval()
        return model

    def transform(self, image: Image.Image) -> torch.Tensor:
        return self._transform(image)

    @property
    def compose(self) -> transforms.Compose:
        """返回裸 Compose(可 pickle,供 DataLoader 多进程使用)。

        绑定的 spec.transform 方法会把整个 BackboneSpec(含 _build)带入
        multiprocessing pickle,spawn/forkserver 下失败;Dataset 应持有 Compose 本体。
        """
        return self._transform

    @torch.no_grad()
    def extract(self, model: nn.Module, batch: torch.Tensor) -> np.ndarray:
        """batch: (B,3,H,W) 已归一化 tensor → (B, feature_dim) float32 numpy。"""
        out = model(batch)
        return out.float().cpu().numpy()


# ---- 构建函数 ----

def _build_timm_encoder(name: str) -> nn.Module:
    import timm

    return timm.create_model(name, pretrained=True, num_classes=0)


def _build_timm_dinov2(name: str, dim: int, img_size: int = 224) -> nn.Module:
    """timm dinov2:无分类头,forward_features 返回 CLS token。

    lvd142m 权重原生 518×518,timm 自动插值 pos_embed 到任意 img_size
    (392 为分辨率敏感度实验变量,与 yesterday DINOv2-S/14 协议一致的 224 为默认)。
    """
    import timm

    base = timm.create_model(name, pretrained=True, img_size=img_size)

    class _CLSWrapper(nn.Module):
        """必须把 base 注册为子模块:闭包捕获的模型不会被 .to(device) 移动,
        GPU 提取时会报 Input type (cuda) vs weight type (cpu) 不匹配。"""

        def __init__(self, base, dim):
            super().__init__()
            self.base = base
            self.feature_dim = dim

        def forward(self, x):
            return self.base.forward_features(x)[:, 0]

    return _CLSWrapper(base, dim)


class _FwdFeatWrapper(nn.Module):
    """绕过 timm 新版 forward,直接调用 vendor 类的 forward_features。

    RETFound_mae(vendor,旧版 timm 签名)的 forward_features 只接受 x,而继承自
    timm 1.x 的 forward 会以 attn_mask= 关键字调用它 → TypeError;且其自带
    分类头(head)权重不在 checkpoint 中(随机初始化),绝不能走 model(x)
    完整前向。冻结特征与 head-only 训练均只用 CLS 表征(1024 维)。
    """

    def __init__(self, base: nn.Module, dim: int):
        super().__init__()
        self.base = base
        self.feature_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.forward_features(x)


def _build_retfound_mae_cfp() -> nn.Module:
    """RETFound MAE CFP(Nature 2023 权重,global_pool=False → CLS token)。

    checkpoint 实际结构为 {'state_dict': {'backbone.<vit键>': ...}}(2026-08-21
    实测),需先解包 state_dict/model 包装层,再剥离 "backbone." 前缀;前向
    经 _FwdFeatWrapper 走 forward_features(见类注释)。
    """
    model = RETFound_mae()
    state = torch.load(RETFOUND_CKPT, map_location="cpu", weights_only=False)
    for wrapper in ("state_dict", "model"):
        if isinstance(state, dict) and isinstance(state.get(wrapper), dict):
            state = state[wrapper]
    state = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    # head.weight/bias 属分类头,checkpoint 只有骨干——预期缺失,其余键必须齐全
    missing = [k for k in msg.missing_keys if not k.startswith("head.")]
    if missing:
        raise RuntimeError(
            f"RETFound checkpoint missing keys: {missing[:5]} "
            f"(unexpected: {msg.unexpected_keys[:5]})")
    return _FwdFeatWrapper(model, 1024)


# ---- 命名的构建函数(lambda 不可 pickle,DataLoader 多进程需用模块级函数)----

def _build_efficientnet_b0() -> nn.Module:
    return _build_timm_encoder("efficientnet_b0")


def _build_convnext_tiny() -> nn.Module:
    return _build_timm_encoder("convnext_tiny")


def _build_dinov2_vits14() -> nn.Module:
    return _build_timm_dinov2("vit_small_patch14_dinov2.lvd142m", 384)


def _build_dinov2_vitb14() -> nn.Module:
    return _build_timm_dinov2("vit_base_patch14_dinov2.lvd142m", 768)


def _build_dinov2_vits14_392() -> nn.Module:
    """DINOv2-S/14 @392² — 分辨率敏感度实验对照(green 同分辨率、不同域)。"""
    return _build_timm_dinov2("vit_small_patch14_dinov2.lvd142m", 384, img_size=392)


def _build_retfound_green_224() -> nn.Module:
    """RETFound-Green @224² — 分辨率敏感度实验对照(同域、同模型、低分辨率)。"""
    return _build_retfound_green(img_size=224)


# ---- 预处理 ----

def _t_cnn() -> transforms.Compose:
    """timm CNN 惯例:resize 256 → centercrop 224 → ImageNet 归一化。"""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        _imagenet_normalize(),
    ])


def _t_dinov2() -> transforms.Compose:
    """DINOv2 224:方形 resize(保留整个视野,轻微纵横比压缩)。

    transforms.Resize(224) 只缩短短边,非方形输入会触发 timm 尺寸断言;
    故用 Resize((224,224)) 强制方形。+ ImageNet 归一化。
    """
    return _t_dinov2_sq(224)


def _interp_pos_embed(model: nn.Module, state: dict) -> None:
    """把 green 权重的 pos_embed 插值到模型尺寸(392→224 敏感度实验用)。

    该 checkpoint 的 pos_embed 仅含 patch 网格(784 = 28²,无 cls/reg 前缀行)。
    按 timm 惯例:网格双线性插值到模型 grid_size,前缀行补零。
    """
    if "pos_embed" not in state or state["pos_embed"].shape == model.pos_embed.shape:
        return
    pe = state["pos_embed"]  # (1, N_grid, D)
    grid = int(round(pe.shape[1] ** 0.5))
    assert grid * grid == pe.shape[1], f"unexpected pos_embed tokens {pe.shape[1]}"
    d = pe.shape[-1]
    n_total = model.pos_embed.shape[1]
    side = model.patch_embed.grid_size[0]
    prefix = n_total - side * side
    if side * side != pe.shape[1]:
        pe = pe.reshape(1, grid, grid, d).permute(0, 3, 1, 2)
        pe = torch.nn.functional.interpolate(
            pe, size=(side, side), mode="bicubic", align_corners=False
        )
        pe = pe.permute(0, 2, 3, 1).reshape(1, side * side, d)
    if prefix:
        pe = torch.cat([torch.zeros(1, prefix, d, dtype=pe.dtype), pe], dim=1)
    state["pos_embed"] = pe


def _build_retfound_green(img_size: int = 392) -> nn.Module:
    """RETFound-Green(JAMA Ophthalmol 2024,CNCRL 非商用许可)。

    官方用法:timm vit_small_patch14_reg4_dinov2,输入 392×392,num_classes=0,
    global_pool='avg'(所有 token 均值池化,含 reg/cls)。输出 384 维。
    注意:归一化为 0.5/0.5(非 ImageNet 统计),预处理见 _t_green()。
    权重:GitHub release v0.1(justinengelmann/RETFound_Green)。
    img_size=224 仅用于分辨率敏感度实验(隔离"分辨率"与"域"的贡献,
    pos_embed 由 _interp_pos_embed 插值,与 timm 前缀补零惯例一致)。
    """
    import timm

    model = timm.create_model(
        "vit_small_patch14_reg4_dinov2",
        img_size=img_size,
        num_classes=0,
    )
    state = torch.load(RETFOUND_GREEN_CKPT, map_location="cpu", weights_only=False)
    state = state["state_dict"] if "state_dict" in state else state.get("model", state)
    _interp_pos_embed(model, state)
    msg = model.load_state_dict(state, strict=True)
    if msg.missing_keys:
        raise RuntimeError(f"RETFound-Green checkpoint missing keys: {msg.missing_keys[:5]}")
    model.global_pool = "avg"
    model.eval()
    return model


def _t_green(size: int = 392) -> transforms.Compose:
    """RETFound-Green 官方预处理:resize 392×392 + 0.5 均值/标准差归一化。"""
    return transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def _t_dinov2_sq(size: int = 224) -> transforms.Compose:
    """DINOv2 方形 resize(392 为分辨率敏感度实验变量)。"""
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        _imagenet_normalize(),
    ])


# ---- 注册表 ----

BACKBONES: dict[str, BackboneSpec] = {
    "efficientnet_b0": BackboneSpec(
        key="efficientnet_b0",
        feature_dim=1280,
        weights_note="timm efficientnet_b0 (ImageNet-1k, pretrained=True)",
        preprocessing="resize256+centercrop224+imagenet_norm",
        _build=_build_efficientnet_b0,
        _transform=_t_cnn(),
    ),
    "convnext_tiny": BackboneSpec(
        key="convnext_tiny",
        feature_dim=768,
        weights_note="timm convnext_tiny (ImageNet-1k, pretrained=True)",
        preprocessing="resize256+centercrop224+imagenet_norm",
        _build=_build_convnext_tiny,
        _transform=_t_cnn(),
    ),
    "dinov2_vits14": BackboneSpec(
        key="dinov2_vits14",
        feature_dim=384,
        weights_note="timm vit_small_patch14_dinov2.lvd142m (LVD-142M self-supervised)",
        preprocessing="resize224sq+imagenet_norm",
        _build=_build_dinov2_vits14,
        _transform=_t_dinov2(),
    ),
    "dinov2_vitb14": BackboneSpec(
        key="dinov2_vitb14",
        feature_dim=768,
        weights_note="timm vit_base_patch14_dinov2.lvd142m (LVD-142M self-supervised)",
        preprocessing="resize224sq+imagenet_norm",
        _build=_build_dinov2_vitb14,
        _transform=_t_dinov2(),
    ),
    "retfound_mae_cfp": BackboneSpec(
        key="retfound_mae_cfp",
        feature_dim=1024,
        weights_note="RETFound MAE CFP (Nature 2023, natureCFP_key_modified.pth, CLS token)",
        preprocessing="resize256+centercrop224+imagenet_norm",
        _build=_build_retfound_mae_cfp,
        _transform=_t_cnn(),
    ),
    "retfound_green": BackboneSpec(
        key="retfound_green",
        feature_dim=384,
        input_size=392,
        weights_note=(
            "RETFound-Green (JAMA Ophthalmol 2024, GitHub release v0.1, "
            "vit_small_patch14_reg4_dinov2, avg-pool 全 token;CNCRL 非商用许可)"
        ),
        preprocessing="resize392+norm0.5",
        _build=_build_retfound_green,
        _transform=_t_green(),
    ),
    # ---- 分辨率敏感度实验对照(E1.6 遗留项)----
    "retfound_green_224": BackboneSpec(
        key="retfound_green_224",
        feature_dim=384,
        weights_note=(
            "RETFound-Green @224²(同权重,pos_embed 插值)——分辨率敏感度对照:"
            "分离 green 的 392² 输入优势与婴儿域优势"
        ),
        preprocessing="resize224+norm0.5",
        _build=_build_retfound_green_224,
        _transform=_t_green(224),
    ),
    "dinov2_vits14_392": BackboneSpec(
        key="dinov2_vits14_392",
        feature_dim=384,
        input_size=392,
        weights_note=(
            "DINOv2-S/14 @392²(同权重,pos_embed 插值)——分辨率敏感度对照:"
            "通用域模型提到 green 同分辨率后还剩多少差距"
        ),
        preprocessing="resize392sq+imagenet_norm",
        _build=_build_dinov2_vits14_392,
        _transform=_t_dinov2_sq(392),
    ),
}


def get_backbone(key: str) -> BackboneSpec:
    if key not in BACKBONES:
        raise KeyError(f"unknown backbone {key!r}; available: {sorted(BACKBONES)}")
    return BACKBONES[key]


def get_ssl_backbone(key: str, ckpt: Path, base_key: str) -> BackboneSpec:
    """E2 继续预训练产物 → BackboneSpec(与基座同预处理、同特征提取口径)。

    ckpt 由 ssl_pretrain.py 保存,含 {"backbone": 裸 timm 模型 state_dict};
    构建时复用基座的提取包装(CLS 包装 / avg-pool),只换权重,保证 E2.5 的
    probe 评估与 E1 完全同口径(仅编码器权重不同)。
    """
    ckpt = Path(ckpt)
    base = BACKBONES[base_key]

    def _build():
        model = base._build()  # 同基座构建(含特征提取包装)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state["backbone"] if "backbone" in state else state
        raw = model.base if hasattr(model, "base") else model  # 剥 CLS 包装
        msg = raw.load_state_dict(state, strict=True)
        if msg.missing_keys:
            raise RuntimeError(f"{key} SSL checkpoint missing keys: {msg.missing_keys[:5]}")
        return model

    return BackboneSpec(
        key=key,
        feature_dim=base.feature_dim,
        input_size=base.input_size,
        weights_note=(
            f"E2 SSL continued pretraining({ckpt.name};起点 {base_key} "
            f"{base.weights_note})"
        ),
        preprocessing=base.preprocessing,
        _build=_build,
        _transform=base.compose,
    )
