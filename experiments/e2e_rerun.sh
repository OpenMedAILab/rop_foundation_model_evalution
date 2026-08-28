#!/usr/bin/env bash
# E2e: e2c 残局重跑(原驱动器因宿主会话重启被 SIGTERM,cons_only 止于 ep020)。
# 幂等:训练/评估齐全即跳过;残缺目录清空重训(ssl_pretrain 无 resume)。
# 完成训练+评估后直接调用 e2d_postqueue_cpu.sh 收尾(隔离选点/消融检验/种子方差)。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs
GATE=45000
TIMEOUT_S=43200
CORPUS="data/manifests/ssl_corpus_manifest_license_clean.csv"

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

train_one() {  # train_one <name> <config> [extra flags...]
  local name="$1"; local cfg="$2"; shift 2
  local dir="outputs/ssl/ibot_dinov2s_heads_${name}"

  # 训练完成判定(评估后根目录保留 ep040/ep050/final 三份)
  if [ -f "$dir/ckpt_ep040.pth" ] && [ -f "$dir/ckpt_ep050.pth" ] \
     && [ -f "$dir/ckpt_final.pth" ]; then
    echo "===== E2e skip train(已完成):${name} ====="
  else
    echo "===== E2e train ${name}(${cfg})====="
    if [ -d "$dir" ]; then
      echo "  [e2e] 清空残缺目录 $dir 后重训"
      rm -rf "$dir"
    fi
    MIN_FREE_MB=$GATE TIMEOUT_S=$TIMEOUT_S experiments/gpu_queue.sh \
      python3 -m neoropfm.train.ssl_pretrain --config "$cfg" \
        --manifest "$CORPUS" --output-dir "$dir" "$@" \
        >> "outputs/logs/e2e_${name}.txt" 2>&1
  fi

  # 评估完成判定(ssl_eval_checkpoints 会写 checkpoint_selection.csv)
  if [ -f "$dir/checkpoint_selection.csv" ]; then
    echo "===== E2e skip eval(已完成):${name} ====="
  else
    echo "===== E2e eval ${name} ====="
    eval_ckpts "$dir" >> "outputs/logs/e2e_${name}.txt" 2>&1
  fi
}

train_one cons_only configs/ssl_ibot_abl_cons_only.yaml
train_one shuf_pma configs/ssl_ibot_abl_heads.yaml --pma-shuffle
train_one shuf_visit configs/ssl_ibot_abl_heads.yaml --visit-shuffle
train_one seed1 configs/ssl_ibot_abl_heads.yaml --seed 1
train_one seed2 configs/ssl_ibot_abl_heads.yaml --seed 2

echo "===== E2e 训练+评估全部完成,进入 CPU 收尾 ====="
bash experiments/e2d_postqueue_cpu.sh
echo "===== E2e 整体完成 ====="
