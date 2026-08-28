"""E2 SSL 数据增强(模块级类;属性均为可 pickle 对象,DataLoader 多进程安全)。

DINO/iBOT 多裁剪协议(facebookresearch/dinov2 官方口径):
- 全局视图 ×2:RandomResizedCrop(scale 0.4–1.0)+ 翻转 + 颜色抖动 + 灰度 + 模糊(p=0.1)
- 局部视图 ×2:RandomResizedCrop(scale 0.05–0.4)+ 同色彩扰动 + 模糊(p=0.5)+ solarize(p=0.2)
归一化沿用基座口径:dinov2 路线 = ImageNet 统计;green 路线 = 0.5/0.5。

MAE 协议:单视图 RandomResizedCrop(scale 0.2–1.0,长宽比 3/4–4/3)+ 翻转 + 0.5 归一化。
"""
from __future__ import annotations

from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


class MultiCropSSL:
    """DINO/iBOT 多裁剪:返回 [g1, g2, l1, l2] 归一化 tensor 列表。"""

    def __init__(
        self,
        global_size: int = 224,
        local_size: int = 96,
        n_global: int = 2,
        n_local: int = 2,
        norm: tuple = (IMAGENET_MEAN, IMAGENET_STD),
    ):
        mean, std = norm
        flip = transforms.RandomHorizontalFlip(p=0.5)
        color = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        normalize = transforms.Normalize(mean=mean, std=std)
        bicubic = transforms.InterpolationMode.BICUBIC
        self.global_tf = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0), interpolation=bicubic),
            flip,
            transforms.Compose([
                color,
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.1),
                transforms.ToTensor(),
                normalize,
            ]),
        ])
        self.local_tf = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.05, 0.4), interpolation=bicubic),
            flip,
            transforms.Compose([
                color,
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
                transforms.RandomSolarize(threshold=128, p=0.2),
                transforms.ToTensor(),
                normalize,
            ]),
        ])
        self.n_global = n_global
        self.n_local = n_local

    def __call__(self, image: Image.Image) -> list:
        crops = [self.global_tf(image) for _ in range(self.n_global)]
        crops += [self.local_tf(image) for _ in range(self.n_local)]
        return crops


class MAETransform:
    """MAE 单视图增强(green 392 口径;多裁剪会改变 token 数,不适用)。"""

    def __init__(self, size: int = 392, norm: tuple = HALF):
        mean, std = norm
        self.tf = transforms.Compose([
            transforms.RandomResizedCrop(
                size, scale=(0.2, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image: Image.Image):
        return self.tf(image)
