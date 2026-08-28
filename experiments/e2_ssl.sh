#!/usr/bin/env bash
# E2 新生儿继续自监督预训练(NeoROP-FM 初始化编码器)
# 对应计划 §E2:iBOT(DINOv2 resume,主)或 MAE(RETFound-Green resume,备)
# + PMA 辅助头 + infant-consistency 头 → 产出 outputs/ssl/{run}/ckpt_*.pth
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs

echo "== E2.1 SSL 语料 manifest(4 主数据集 + 辅助,排除 coph100)=="
python3 -m neoropfm.data.build_ssl_corpus
echo "  → data/manifests/ssl_corpus_manifest.csv(已完成,12,074 张)"

echo "== E2.2 iBOT 继续预训练(DINOv2-S/14 起点,teacher–student self-distillation)=="
MIN_FREE_MB=40000 experiments/gpu_queue.sh \
  python3 -m neoropfm.train.ssl_pretrain --config configs/ssl_ibot_dinov2.yaml \
  >> outputs/logs/ssl_ibot_dinov2s_v1.txt 2>&1

echo "== E2.3 MAE 继续预训练(RETFound-Green 起点,备选路线)=="
MIN_FREE_MB=40000 experiments/gpu_queue.sh \
  python3 -m neoropfm.train.ssl_pretrain --config configs/ssl_mae_retfound_green.yaml \
  >> outputs/logs/ssl_mae_retfound_green_v1.txt 2>&1

echo "== E2.4 PMA 辅助头 / infant-consistency 头消融 =="
MIN_FREE_MB=40000 experiments/gpu_queue.sh \
  python3 -m neoropfm.train.ssl_pretrain --config configs/ssl_ibot_abl_heads.yaml \
  >> outputs/logs/ssl_ibot_abl_heads.txt 2>&1

echo "== E2.5 leave-test-out 适应敏感度 + checkpoint 选择(probe)=="
echo "  [待实现] 用 E1 probe 管线评估 outputs/ssl/*/ckpt_*.pth,"
echo "           按 mean AUROC 选最优 epoch;并做 leave-test-out 适应(每折只用其余 3 数据集)"
