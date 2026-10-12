#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
NAME="s3a_seg1"
PID="runs/tiered_ckpt_eval_${NAME}.pid"
LOG="runs/tiered_ckpt_eval_${NAME}.daemon.log"

if [[ -f "$PID" ]]; then
  pid=$(cat "$PID")
  state=$(ps -o state= -p "$pid" 2>/dev/null | tr -d ' ' || true)
  if [[ -n "$state" && "$state" != Z* ]]; then
    echo "tiered eval watcher already running: $pid"
    exit 0
  fi
  rm -f "$PID"
fi

export CUDA_VISIBLE_DEVICES=""
nohup python3 scripts/tiered_ckpt_eval.py \
  --name "$NAME" \
  --workers 4 \
  --smoke_threads 4 \
  --code_threads 2 \
  --quick_tasks 8 \
  --poll 300 \
  --watch \
  >> "$LOG" 2>&1 &
echo $! > "$PID"
echo "tiered eval watcher started: $(cat "$PID")"
