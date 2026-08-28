#!/usr/bin/env bash
# E1 新模型全流程:CPU 特征提取 → 线性探针 → 6 模型统计聚合
# 用途:GPU 被 vLLM 占用时,新模型特征用 CPU 提取(与 yesterday 4 基线口径一致,
# feature_meta.device=cpu),约 30–45 分钟,不影响共享 GPU。
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/3 特征提取(retfound_green 392×392 / dinov2_vitb14 224)=="
for m in retfound_green dinov2_vitb14; do
  python3 -m neoropfm.train.extract_features --config configs/extract_cpu.yaml --model "$m"
done

echo "== 2/3 线性探针(新模型,零重算)=="
python3 -m neoropfm.train.probe --config configs/probe_extract.yaml

echo "== 3/3 统计聚合(6 模型:4 基线 + 2 新模型)=="
python3 -m neoropfm.eval.aggregate --config configs/aggregate_all6.yaml

echo "done → outputs/probes / outputs/aggregate"
