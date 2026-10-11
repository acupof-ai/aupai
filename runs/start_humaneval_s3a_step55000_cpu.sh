#!/usr/bin/env bash
set -euo pipefail

ROOT=/work/aupai
CKPT=ckpt_s3a_seg1.pt.step55000
RUN=s3a_step55000_cpu_he
QUEUE=runs/queues/he_s3a_step55000
WORKERS=4
THREADS=4

cd "$ROOT"

for i in $(seq 0 $((WORKERS - 1))); do
  PID_FILE="runs/he_s3a_step55000.w${i}.pid"
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    STATE="$(ps -o stat= -p "$(cat "$PID_FILE")" | tr -d ' ')"
    if [[ "$STATE" != Z* ]]; then
      echo "HUMANEVAL_WORKER_ALIVE w${i} pid=$(cat "$PID_FILE")"
      exit 0
    fi
  fi
done

mkdir -p "$QUEUE"
for i in $(seq 0 $((WORKERS - 1))); do
  LOG_FILE="runs/he_s3a_step55000.w${i}.log"
  PID_FILE="runs/he_s3a_step55000.w${i}.pid"
  : > "$LOG_FILE"
  nohup env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    python3 eval/humaneval_gen.py \
      --ckpt "$CKPT" \
      --device cpu \
      --threads "$THREADS" \
      --rstrip_nl \
      --queue_dir "$QUEUE" \
      --shard_i "$i" \
      --shard_n "$WORKERS" \
      --run "$RUN" \
      --force \
      >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  echo "HUMANEVAL_WORKER_STARTED w${i} pid=$(cat "$PID_FILE")"
done
