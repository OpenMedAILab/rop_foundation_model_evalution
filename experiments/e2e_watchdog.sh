#!/usr/bin/env bash
# E2e 看门狗:每 300s 检查一次。
# - 训练驱动器(e2e_rerun.sh)不在 且 仍有未完成训练 → 重启驱动器(setsid 脱离会话树)
# - 训练全部完成 且 CPU 收尾产物缺失 → 补跑 e2d_postqueue_cpu.sh(幂等)
# 看门狗自身由外层 setsid 启动,宿主 Claude 会话重启也不会中断。
cd "$(dirname "$0")/.."
mkdir -p outputs/logs
RUNS="cons_only shuf_pma shuf_visit seed1 seed2"

while true; do
  DRV=$(pgrep -f "experiments/e2e_rerun.sh" | head -1 || true)
  if [ -z "$DRV" ]; then
    incomplete=0
    for n in $RUNS; do
      d="outputs/ssl/ibot_dinov2s_heads_${n}"
      if [ -f "$d/ckpt_ep040.pth" ] && [ -f "$d/ckpt_ep050.pth" ] && [ -f "$d/ckpt_final.pth" ]; then
        :
      else
        incomplete=1
      fi
    done
    if [ "$incomplete" = "1" ]; then
      echo "[watchdog] $(date '+%F %T') 驱动器不在且训练未完成 → 重启" >> outputs/logs/e2e_watchdog.txt
      setsid nohup bash experiments/e2e_rerun.sh >> outputs/logs/e2e_driver_rerun.txt 2>&1 &
    elif [ ! -f outputs/checkpoint_iso/ablation_overall.csv ] \
      || [ ! -f outputs/aggregate_e2/seed_variance.csv ]; then
      echo "[watchdog] $(date '+%F %T') 训练完成,CPU 收尾产物缺失 → 补跑 e2d" >> outputs/logs/e2e_watchdog.txt
      setsid nohup bash experiments/e2d_postqueue_cpu.sh >> outputs/logs/e2d_watcher.txt 2>&1 &
    fi
  fi
  sleep 300
done
