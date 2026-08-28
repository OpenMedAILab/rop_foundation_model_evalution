#!/bin/bash
# E3 batch: build longitudinal samples + run baselines for all feature models
cd "$(dirname "$0")/.."
PY=${PY:-python3}
LOG=/tmp/e3_batch.log
echo "=== E3 batch started $(date) ===" > $LOG

MODELS="dinov2_vitb14 dinov2_vits14_392 retfound_green_224 ibot_dinov2s_v1_ckpt_ep005 ibot_dinov2s_v1_ckpt_ep010 ibot_dinov2s_v1_ckpt_ep015 ibot_dinov2s_v1_ckpt_ep020 ibot_dinov2s_v1_ckpt_ep025"

for MODEL in $MODELS; do
    echo "" >> $LOG
    echo "=== $MODEL: building samples $(date) ===" >> $LOG
    $PY -m neoropfm.data.build_longitudinal_samples --feature-model $MODEL >> $LOG 2>&1

    echo "=== $MODEL: running baselines $(date) ===" >> $LOG
    $PY -m neoropfm.train.longitudinal_baselines --feature-model $MODEL >> $LOG 2>&1
done

echo "" >> $LOG
echo "=== E3 batch baselines done $(date) ===" >> $LOG

# Run Temporal Transformer for key models (SSL checkpoints + baselines)
TT_MODELS="dinov2_vits14_392 ibot_dinov2s_v1_ckpt_ep025 ibot_dinov2s_v1_ckpt_ep005 retfound_green_224"
for MODEL in $TT_MODELS; do
    echo "" >> $LOG
    echo "=== $MODEL: temporal transformer $(date) ===" >> $LOG
    $PY -m neoropfm.train.temporal_transformer --feature-model $MODEL --device cpu --n-ensemble 5 >> $LOG 2>&1
done

echo "" >> $LOG
echo "=== E3 batch ALL done $(date) ===" >> $LOG
