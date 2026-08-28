#!/usr/bin/env bash
# E1.7 适配策略:6 模型 × 4 折 LODO。
# 协议(2026-08-21 定档):head-only——骨干完全冻结,仅训练线性头
# (head_lr 1e-2、head_steps 640 绝对步数预算,其余同 peft.yaml)。
# 背景:LoRA 五配置(全层 α16/α8、qv-only、v1/v2 调度)在本小样本跨数据集
# LODO 场景全部崩盘(test 0.55–0.64 vs 冻结 probe 0.88),仅 head-only 胜出
# (4 折平均 0.888 ≥ 冻结 probe 0.884),详见 outputs/logs/LORA_V1_POSTMORTEM.md。
# 每模型开跑前校验 GPU 空闲 ≥30GB(peft.yaml 门控);仅"门控拒绝"才等待重试,
# 其余错误直接跳过该模型(日志可查,便于人工恢复)。
set -u
cd "$(dirname "$0")/.."

MODELS=(retfound_green retfound_mae_cfp dinov2_vits14 dinov2_vitb14 convnext_tiny efficientnet_b0)
mkdir -p outputs/logs

for m in "${MODELS[@]}"; do
  while true; do
    out=$(python -m neoropfm.train.peft --config configs/peft.yaml --model "$m" --head-only 2>&1)
    echo "$out" | grep -v "Warning\|warn" >> "outputs/logs/peft_${m}.txt"
    if echo "$out" | grep -q "GPU 空闲显存"; then
      echo "[$m] $(date +%H:%M:%S) GPU busy -> sleep 300s" >> "outputs/logs/peft_${m}.txt"
      sleep 300
      continue
    fi
    break
  done
done
echo "E1.7 all done"
