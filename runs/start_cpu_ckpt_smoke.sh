#!/usr/bin/env bash
set -euo pipefail

ROOT=/work/aupai
NAME=s3a_seg1
PID_FILE="$ROOT/runs/cpu_ckpt_smoke_${NAME}.pid"
LOG_FILE="$ROOT/runs/cpu_ckpt_smoke_${NAME}.daemon.log"

cd "$ROOT"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  PID="$(cat "$PID_FILE")"
  STATE="$(ps -o stat= -p "$PID" | tr -d ' ')"
  if [[ "$STATE" != Z* ]]; then
    echo "CPU_SMOKE_WATCHER_ALIVE=$PID"
    exit 0
  fi
fi

nohup env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python3 scripts/cpu_ckpt_smoke.py --name "$NAME" --watch --poll 300 --threads 4 \
  >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "CPU_SMOKE_WATCHER_STARTED=$(cat "$PID_FILE")"
