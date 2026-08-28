"""E2 新生儿继续自监督预训练组件(方案 §8.2–§8.4)。

- augment: DINO/iBOT 多裁剪与 MAE 单视图增强(模块级类,可 pickle)
- dino: DINO/iBOT 头、块状掩码、自蒸馏损失、EMA 教师(iBOT 路线)
- mae: MAE 掩码建模(RETFound-Green 起点,备选路线)
- heads: PMA 回归头 + visit-consistency InfoNCE(E2.4 消融辅助头)
"""
