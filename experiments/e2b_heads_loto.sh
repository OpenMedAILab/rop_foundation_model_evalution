#!/usr/bin/env bash
# E2b metadata-assisted(heads)严格 LOTO:逐折剔除该折数据集后重训 iBOT+PMA+visit
# (50 ep,与 heads 全语料路线同 seed/超参,仅语料不同),再评估 ep040/ep050/final,
# 与 heads 全语料(transductive)比较——最高优先补实验。
#
# 对比口径:
#   heads 全语料:outputs/ssl/ibot_dinov2s_heads → iso 折均值 0.9051(transductive)
#   heads LOTO  :outputs/ssl/loto_ibot_dinov2s_heads_minus_{fold} → 归纳式
# 判读:若 heads LOTO 归纳均值仍 ≥ green 0.8990 → development-aware 叙事成立;
#       否则结论③须按 P7 分支降级。
# 注意:输出目录名带 heads,与纯 SSL 的 loto_ibot_dinov2s_minus_{fold} 不冲突;
#       checkpoint_selection_isolated.py 会按 features 池名自动发现新池。
set -euo pipefail
cd "$(dirname "$0")/.."

FOLDS=(farfum_rop ridirp rop_vl szeh_irops)
for fold in "${FOLDS[@]}"; do
  echo "===== heads LOTO minus ${fold} ====="
  MIN_FREE_MB=40000 TIMEOUT_S=43200 experiments/gpu_queue.sh \
    python3 -m neoropfm.train.ssl_pretrain \
      --config configs/ssl_ibot_abl_heads.yaml \
      --manifest "data/manifests/ssl_corpus_manifest_license_clean_minus_${fold}.csv" \
      --output-dir "outputs/ssl/loto_ibot_dinov2s_heads_minus_${fold}"

  # 只评估选定的 3 个轮次(ep040=heads 全语料最优,ep050/final=收尾参考),
  # 其余 checkpoint 移入 all_ckpts/ 保留备查
  run_dir="outputs/ssl/loto_ibot_dinov2s_heads_minus_${fold}"
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
echo "===== heads LOTO 四折全部完成 ====="
