#!/usr/bin/env bash
# E1 基线基准测试(frozen+linear probe 主评估;LoRA / full-FT 次级评估)
# 对应计划 §E1。probe 为 CPU;特征提取需 GPU,自动排队。
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== E1.1 权重准备 =="
echo "  DINOv2-B/14: timm 已预下载 ✅(2026-08-20)"
echo "  RETFound-Green: GitHub release v0.1 已下载 ✅ → outputs/weights/retfound_green/(CNCRL 非商用)"

echo "== E1.2 冻结特征提取(新模型;GPU 排队,空闲 ≥30GB 时执行)=="
for m in retfound_green dinov2_vitb14; do
  MIN_FREE_MB=30000 experiments/gpu_queue.sh \
    python3 -m neoropfm.train.extract_features --config configs/extract.yaml --model "$m"
done

echo "== E1.3 线性探针(全部模型,零重算)=="
python3 -m neoropfm.train.probe --config configs/probe_cache.yaml
# 新模型特征就绪后(特征提取完成时):
python3 -m neoropfm.train.probe --config configs/probe_extract.yaml

echo "== E1.4 统计聚合(bootstrap CI / DeLong / 均值表)=="
python3 -m neoropfm.eval.aggregate --config configs/aggregate.yaml

echo "== E1.5 LoRA PEFT + 全量微调(次级评估)=="
# [待实现] python3 -m neoropfm.train.lora_probe --config configs/lora.yaml
# [待实现] python3 -m neoropfm.train.full_ft --config configs/ft.yaml
echo "  [待实现] lora_probe / full_ft(小模型先做全量微调)"

echo "== E1.6 标签规则敏感性分析 =="
# [待实现] python3 -m neoropfm.data.label_rule_sensitivity --probe
echo "  [待实现] label_rule_sensitivity(等待附件稿件)"
