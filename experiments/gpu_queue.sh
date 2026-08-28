#!/usr/bin/env bash
# GPU 排队执行:等待显存空闲(≥MIN_FREE_MB)后运行命令。
#
# 用法:
#   MIN_FREE_MB=30000 experiments/gpu_queue.sh python3 -m neoropfm.train.extract_features --config configs/extract.yaml --model dinov2_vitb14
#   MIN_FREE_MB=30000 TIMEOUT_S=3600 POLL_S=60 experiments/gpu_queue.sh <任意命令>
#
# 环境变量:
#   MIN_FREE_MB  所需最小空闲显存 MiB(默认 30000)
#   TIMEOUT_S    最长等待秒数(默认 21600 = 6h;到点未等到则退出 2)
#   POLL_S       轮询间隔秒(默认 60)
set -euo pipefail

MIN_FREE_MB="${MIN_FREE_MB:-30000}"
TIMEOUT_S="${TIMEOUT_S:-21600}"
POLL_S="${POLL_S:-60}"
CMD=("$@")
if [ "${#CMD[@]}" -eq 0 ]; then
  echo "用法: $0 <command...>" >&2
  exit 1
fi

nvidia-smi >/dev/null 2>&1 || { echo "[gpu_queue] 未检测到 GPU,直接执行" >&2; exec "${CMD[@]}"; }

elapsed=0
while true; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free_mb" -ge "$MIN_FREE_MB" ]; then
    echo "[gpu_queue] GPU 空闲 ${free_mb}MB ≥ ${MIN_FREE_MB}MB,等待 ${elapsed}s 后执行:" >&2
    printf '  %q ' "${CMD[@]}" >&2; echo >&2
    exec "${CMD[@]}"
  fi
  if [ "$elapsed" -ge "$TIMEOUT_S" ]; then
    echo "[gpu_queue] 等待超时(${TIMEOUT_S}s),GPU 仍仅空闲 ${free_mb}MB,放弃。" >&2
    exit 2
  fi
  sleep "$POLL_S"
  elapsed=$((elapsed + POLL_S))
done
