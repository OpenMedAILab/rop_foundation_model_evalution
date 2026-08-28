#!/usr/bin/env bash
# E2c: heads 路线消融 ×2 + 打乱负对照 ×2(P2)+ 种子 ×2(P1b)
# 全部在 szeh 重训完成后启动(内层队列 45GB 门控:szeh 训练持有 ~3GB,
# 空闲 43.9GB < 45GB → 等它退出后才放行,避免与另一并行会话的 LOTO 重训抢卡)。
#
# 顺序(与计划 GPU 队列一致):
#   0. FARFUM 全量 1,533 图特征提取(9 模型 → outputs/features_tx,~10min,P3b 前提)
#   1. pma_only(visit_w=0)
#   2. cons_only(pma_w=0)
#   3. shuf_pma(heads + batch 内打乱 pma_z 配对)
#   4. shuf_visit(heads + batch 内打乱 visit 配对)
#   5. seed 1 / seed 2(heads 同超参,种子方差)
# 每个训练:50 ep ≈ 2.5h;评估只保留 ep040/ep050/final(与 heads 全语料/LOTO 同口径)。
# 若打乱对照后增益仍在 → 辅助头未学习被打乱的信号(捷径学习排除)。
#
# 输出:
#   outputs/ssl/ibot_dinov2s_heads_{pma_only,cons_only,shuf_pma,shuf_visit,seed1,seed2}/
#   → 各自 checkpoint_selection.csv + outputs/features/<run>_ckpt_* (隔离选点自动发现)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs
GATE=45000
TIMEOUT_S=43200

eval_ckpts() {
  local run_dir="$1"
  mkdir -p "$run_dir/all_ckpts"
  mv "$run_dir"/ckpt_ep*.pth "$run_dir/all_ckpts/" 2>/dev/null || true
  mv "$run_dir"/ckpt_final.pth "$run_dir/all_ckpts/" 2>/dev/null || true
  for ck in ckpt_ep040.pth ckpt_ep050.pth ckpt_final.pth; do
    cp "$run_dir/all_ckpts/$ck" "$run_dir/"
  done
  python3 scripts/ssl_eval_checkpoints.py \
    --run-dir "$run_dir" \
    --base-key dinov2_vits14 \
    --extract-config configs/extract_ssl_eval.yaml \
    --probe-config configs/probe_extract.yaml
}

# ---- 0. FARFUM 全量特征提取(9 模型;外部 45GB 门控已由内层队列承担)----
echo "===== P3b FARFUM 全量特征提取(9 模型)====="
MIN_FREE_MB=$GATE TIMEOUT_S=$TIMEOUT_S experiments/gpu_queue.sh bash -c '
set -euo pipefail
# cwd 已由外层脚本置为仓库根
MANIFEST=outputs/audit/farfum_grade_audit.csv
for m in retfound_green retfound_mae_cfp dinov2_vits14 dinov2_vitb14 convnext_tiny efficientnet_b0; do
  python3 -m neoropfm.train.extract_features --config configs/extract_tx.yaml \
    --model "$m" --manifest "$MANIFEST" --no-strict-filter
done
python3 -m neoropfm.train.extract_features --config configs/extract_tx.yaml \
  --model ibot_dinov2s_v1_ckpt_ep040 --ckpt outputs/ssl/ibot_dinov2s_v1/ckpt_ep040.pth \
  --base-key dinov2_vits14 --manifest "$MANIFEST" --no-strict-filter
python3 -m neoropfm.train.extract_features --config configs/extract_tx.yaml \
  --model ibot_dinov2s_heads_ckpt_ep040 --ckpt outputs/ssl/ibot_dinov2s_heads/ckpt_ep040.pth \
  --base-key dinov2_vits14 --manifest "$MANIFEST" --no-strict-filter
python3 -m neoropfm.train.extract_features --config configs/extract_tx.yaml \
  --model mae_retfound_green_v1_ckpt_ep090 --ckpt outputs/ssl/mae_retfound_green_v1/ckpt_ep090.pth \
  --base-key retfound_green --manifest "$MANIFEST" --no-strict-filter
' >> outputs/logs/e2c_tx_extract.txt 2>&1

# ---- P2 消融 + 打乱对照 ×4 ----
# 注意:config 默认 manifest 为 ssl_corpus_manifest.csv(12,074 行,含 __MACOSX 垃圾),
# 与 v1/heads 全语料实际语料(license_clean 10,656 行)不一致 → 必须显式传 --manifest。
CORPUS="data/manifests/ssl_corpus_manifest_license_clean.csv"
declare -A RUNS=(
  [pma_only]="configs/ssl_ibot_abl_pma_only.yaml"
  [cons_only]="configs/ssl_ibot_abl_cons_only.yaml"
)
for name in pma_only cons_only; do
  cfg="${RUNS[$name]}"
  dir="outputs/ssl/ibot_dinov2s_heads_${name}"
  echo "===== P2 ${name}(${cfg})====="
  MIN_FREE_MB=$GATE TIMEOUT_S=$TIMEOUT_S experiments/gpu_queue.sh \
    python3 -m neoropfm.train.ssl_pretrain --config "$cfg" \
      --manifest "$CORPUS" --output-dir "$dir" >> "outputs/logs/e2c_${name}.txt" 2>&1
  eval_ckpts "$dir"
done

for name in shuf_pma shuf_visit; do
  flag="--pma-shuffle"
  [ "$name" = "shuf_visit" ] && flag="--visit-shuffle"
  dir="outputs/ssl/ibot_dinov2s_heads_${name}"
  echo "===== P2 ${name}(heads + ${flag})====="
  MIN_FREE_MB=$GATE TIMEOUT_S=$TIMEOUT_S experiments/gpu_queue.sh \
    python3 -m neoropfm.train.ssl_pretrain --config configs/ssl_ibot_abl_heads.yaml \
      --manifest "$CORPUS" "$flag" --output-dir "$dir" >> "outputs/logs/e2c_${name}.txt" 2>&1
  eval_ckpts "$dir"
done

# ---- P1b heads 种子 ×2 ----
for s in 1 2; do
  dir="outputs/ssl/ibot_dinov2s_heads_seed${s}"
  echo "===== P1b seed ${s} ====="
  MIN_FREE_MB=$GATE TIMEOUT_S=$TIMEOUT_S experiments/gpu_queue.sh \
    python3 -m neoropfm.train.ssl_pretrain --config configs/ssl_ibot_abl_heads.yaml \
      --manifest "$CORPUS" --seed "$s" --output-dir "$dir" >> "outputs/logs/e2c_seed${s}.txt" 2>&1
  eval_ckpts "$dir"
done

echo "===== E2c 全部完成 ====="
