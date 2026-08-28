#!/bin/bash
# E3 纵向预测 POC 复现脚本
# 数据: RIDIRP (PMA 纵向) + ROP-VL (检查日期纵向)
# 标签: index 访视阴性 → 下次访视阳性 (y_next)
# 评估: 患者级 5-fold CV, 禁止患者泄漏
#
# 前置: E0 manifest v2 + E1 冻结特征缓存已生成
#
# 用法: bash experiments/e3_longitudinal.sh [feature_model]
#   feature_model: retfound_green (默认) / dinov2_vitb14 / ibot_dinov2s_v1_ckpt_ep025 / ...

set -e
cd "$(dirname "$0")/.."
PY=${PY:-python3}
FEAT_MODEL=${1:-retfound_green}

echo "=== E3.1 构建纵向样本 ==="
$PY -m neoropfm.data.build_longitudinal_samples --feature-model "$FEAT_MODEL"

echo "=== E3.2 五个基线 (LR) ==="
$PY -m neoropfm.train.longitudinal_baselines --feature-model "$FEAT_MODEL"

echo "=== E3.3+E3.4 Temporal Transformer + 深度集成 ==="
$PY -m neoropfm.train.temporal_transformer --feature-model "$FEAT_MODEL" --device cpu --n-ensemble 5

echo "=== E3.5 Lead-time 分析 ==="
$PY -m neoropfm.train.lead_time_analysis --feature-model "$FEAT_MODEL"

echo "=== E3 完成 ==="
echo "结果: outputs/longitudinal/"
echo "  longitudinal_manifest_${FEAT_MODEL}.csv     # 纵向样本清单"
echo "  visit_sequences_${FEAT_MODEL}.npz           # 访次特征序列"
echo "  baseline_metrics_${FEAT_MODEL}.csv          # 5 基线指标"
echo "  baseline_predictions_${FEAT_MODEL}.csv      # 基线预测"
echo "  temporal_transformer_metrics_${FEAT_MODEL}.csv  # TT 指标"
echo "  temporal_transformer_predictions_${FEAT_MODEL}.csv # TT 预测"
echo "  lead_time_analysis_${FEAT_MODEL}.csv        # 提前量分析"
