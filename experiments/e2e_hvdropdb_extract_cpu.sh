#!/usr/bin/env bash
# P6b: HVDROPDB 外部折特征提取(CPU,不占 GPU;185 图 × 14 模型 ≈ 1–1.5h)
# 输出 outputs/features_hvdropdb/{model}/,由 scripts/external_fold_eval.py 消费。
# ckpt 选择口径:各 SSL 池 ckpt_final(其 4 个内部隔离折 selection_{held}.csv 一致所选);
# loto_minus_szeh 待 P1a 重训完成后单独追加。
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST=data/manifests/ext_hvdropdb_strict_manifest.csv
# 基线单权重模型:仅 --model;SSL 池:加 --ckpt/--base-key(与 outputs/features 同口径)
BASELINE_MODELS=(
  efficientnet_b0
  convnext_tiny
  dinov2_vits14
  retfound_mae_cfp
  retfound_green
  retfound_green_224
  dinov2_vitb14
  dinov2_vits14_392
)
# "key|base" — ckpt 路径 outputs/ssl/${key%%_ckpt_final}/ckpt_final.pth
SSL_POOLS=(
  "ibot_dinov2s_v1_ckpt_final|dinov2_vits14"
  "ibot_dinov2s_heads_ckpt_final|dinov2_vits14"
  "mae_retfound_green_v1_ckpt_final|retfound_green"
  "loto_ibot_dinov2s_heads_minus_farfum_rop_ckpt_final|dinov2_vits14"
  "loto_ibot_dinov2s_heads_minus_ridirp_ckpt_final|dinov2_vits14"
  "loto_ibot_dinov2s_heads_minus_rop_vl_ckpt_final|dinov2_vits14"
)

for m in "${BASELINE_MODELS[@]}"; do
  if [ -f "outputs/features_hvdropdb/$m/${m}_features.npy" ]; then
    echo "[skip] $m 已提取"
    continue
  fi
  echo "=== [$(date +%H:%M)] $m ==="
  python3 -m neoropfm.train.extract_features \
    --config configs/extract_hvdropdb.yaml \
    --model "$m" \
    --manifest "$MANIFEST" \
    --no-strict-filter
done

for entry in "${SSL_POOLS[@]}"; do
  m="${entry%%|*}"; base="${entry##*|}"
  if [ -f "outputs/features_hvdropdb/$m/${m}_features.npy" ]; then
    echo "[skip] $m 已提取"
    continue
  fi
  pool="${m%_ckpt_final}"
  echo "=== [$(date +%H:%M)] $m ==="
  python3 -m neoropfm.train.extract_features \
    --config configs/extract_hvdropdb.yaml \
    --model "$m" \
    --ckpt "outputs/ssl/$pool/ckpt_final.pth" \
    --base-key "$base" \
    --manifest "$MANIFEST" \
    --no-strict-filter
done
echo "ALL DONE $(date +%H:%M)"
