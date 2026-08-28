#!/usr/bin/env bash
# E4 消融与表征分析
# 对应计划 §E4:消融 A-F;标签效率 1/5/10/25/50/100%;表征分析(UMAP/DDS/轨迹)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== E4.1 消融 A-F =="
echo "  A 继续预训练有无 | B PMA 头 | C infant-consistency | D 时间模型部件 | E 临床变量 | F 集成/选择性预测"
# [待实现] python3 -m neoropfm.train.ablation_matrix --config configs/ablations.yaml
echo "  [待实现] ablation_matrix(与 E1-E3 结果对照表)"

echo "== E4.2 标签效率(1/5/10/25/50/100% 训练标签下采样)=="
# [待实现] python3 -m neoropfm.train.label_efficiency --config configs/label_eff.yaml
echo "  [待实现] label_efficiency(冻结特征 probe,仅重训探针)"

echo "== E4.3 表征分析(UMAP / DDS / 发育轨迹与恢复模式)=="
# [待实现] python3 -m neoropfm.eval.representation_analysis --config configs/repr.yaml
echo "  [待实现] representation_analysis(UMAP、DDS 随 PMA 轨迹、跨数据集域差)"
