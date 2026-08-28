#!/usr/bin/env bash
# E1.8 全量微调:2 个小模型 × 4 折 LODO(与 e1_peft.sh 同门控/重试策略)。
set -u
cd "$(dirname "$0")/.."

MODELS=(efficientnet_b0 convnext_tiny)
mkdir -p outputs/logs

for m in "${MODELS[@]}"; do
  while true; do
    out=$(python -m neoropfm.train.peft --config configs/fullft.yaml --model "$m" 2>&1)
    echo "$out" | grep -v "Warning\|warn" >> "outputs/logs/fullft_${m}.txt"
    if echo "$out" | grep -q "GPU 空闲显存"; then
      echo "[$m] $(date +%H:%M:%S) GPU busy -> sleep 300s" >> "outputs/logs/fullft_${m}.txt"
      sleep 300
      continue
    fi
    break
  done
done
echo "E1.8 all done"
