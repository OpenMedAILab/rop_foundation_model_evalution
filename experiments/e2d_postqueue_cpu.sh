#!/usr/bin/env bash
# E2d: e2c 训练队列完成后的 CPU 收尾管线(纯 CPU,不占 GPU)。
#
# 1. 隔离选点(自动发现池):pma_only(可能已单独跑过,幂等重跑无妨)+
#    cons_only / shuf_pma / shuf_visit / seed1 / seed2
# 2. 消融族配对检验 → outputs/checkpoint_iso/ablation_paired_tests.csv(逐折,S16)
#                      + ablation_overall.csv(总体 + 独立 Holm 族,正文消融句)
# 3. 种子方差 → outputs/aggregate_e2/seed_variance.csv(长格式;正文 Table 10 + S17)
#
# 注意:绝不触碰锁定输出目录/outputs/ssl/loto_*、基线 iso 目录。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs

# ---- 1. 隔离选点(5 个新池;pma_only 已单独启动,重跑幂等)----
for pool in ibot_dinov2s_heads_pma_only ibot_dinov2s_heads_cons_only \
            ibot_dinov2s_heads_shuf_pma ibot_dinov2s_heads_shuf_visit \
            ibot_dinov2s_heads_seed1 ibot_dinov2s_heads_seed2; do
  if [ ! -d "outputs/checkpoint_iso/${pool}_iso" ]; then
    echo "===== E2d iso selection: ${pool} ====="
    python3 scripts/checkpoint_selection_isolated.py "$pool" \
      >> "outputs/logs/e2d_iso_${pool}.txt" 2>&1
  else
    echo "===== E2d skip(已存在):${pool} ====="
  fi
done

# ---- 2. 消融族配对检验 ----
echo "===== E2d ablation paired tests ====="
python3 scripts/iso_paired_tests.py --ablation >> outputs/logs/e2d_ablation.txt 2>&1

# ---- 3. 种子方差 ----
echo "===== E2d seed variance ====="
python3 scripts/seed_variance.py >> outputs/logs/e2d_seed_variance.txt 2>&1

echo "===== E2d 全部完成 ====="
