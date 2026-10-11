#!/usr/bin/env bash
set -u
cd /work/aupai

for round in $(seq 1 40); do
  alive=0
  for i in 0 1 2 3; do
    pid=$(cat "runs/he_s3a_step55000.w${i}.pid" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  claimed=$(cat runs/queues/he_s3a_step55000/cursor 2>/dev/null || echo 0)
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) round=${round} alive=${alive} claimed=${claimed}/164"
  if [ "$alive" -eq 0 ]; then
    python3 runs/merge_humaneval_s3a_step55000.py
    exit 0
  fi
  sleep 300
done
exit 1
