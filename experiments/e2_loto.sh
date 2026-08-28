#!/usr/bin/env bash
# E2 leave-one-test-out(LOTO)敏感性:逐折剔除该折数据集后重训 iBOT(50 ep,
# 与主路线同 seed/超参,仅语料不同),再评估 ep040/ep050/final 三个轮次,
# 与全语料同轮次比较——验证 SSL 增益来自可迁移表征而非"见过测试图"。
#
# 对比口径:
#   全语料:outputs/ssl/ibot_dinov2s_v1 → probe key ibot_dinov2s_v1_ckpt_ep040(0.9016)
#   LOTO  :outputs/ssl/loto_ibot_dinov2s_minus_{fold} → probe key 同名_ckpt_ep040/050/final
# 判读:若 LOTO 在该折 ≈ 全语料 → 无泄漏;若回落至基座 dinov2 水平 → 增益依赖测试图。
set -euo pipefail
cd "$(dirname "$0")/.."

FOLDS=(farfum_rop ridirp rop_vl szeh_irops)
for fold in "${FOLDS[@]}"; do
  echo "===== LOTO minus ${fold} ====="
  MIN_FREE_MB=40000 TIMEOUT_S=43200 experiments/gpu_queue.sh \
    python3 -m neoropfm.train.ssl_pretrain \
      --config configs/ssl_ibot_dinov2.yaml \
      --manifest "data/manifests/ssl_corpus_manifest_license_clean_minus_${fold}.csv" \
      --output-dir "outputs/ssl/loto_ibot_dinov2s_minus_${fold}"

  # 只评估选定的 3 个轮次(ep040=全语料最优,ep050/final=收尾参考),
  # 其余 checkpoint 移入 all_ckpts/ 保留备查
  run_dir="outputs/ssl/loto_ibot_dinov2s_minus_${fold}"
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
done
echo "===== LOTO 四折全部完成 ====="
