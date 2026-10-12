#!/usr/bin/env bash
set -euo pipefail

ROOT=/work/aupai
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

if [ -f "$ROOT/runs/s3a_seg1.log" ]; then
  mv "$ROOT/runs/s3a_seg1.log" "$ROOT/runs/s3a_seg1.log.$STAMP.controlled-stop"
fi
if [ -f "$ROOT/runs/s3a_seg1.rc" ]; then
  mv "$ROOT/runs/s3a_seg1.rc" "$ROOT/runs/s3a_seg1.rc.$STAMP.controlled-stop"
fi
rm -f "$ROOT"/ckpt-tmp-*

setsid python3 "$ROOT/runs/launch_s3a_claimed.py" \
  >> "$ROOT/runs/s3a_seg1.bootstrap.log" 2>&1 &
echo "launch requested"
sleep 10

cd "$ROOT"
echo "---FILES---"
ls -l runs/s3a_seg1.bootstrap.pid runs/s3a_seg1.child.pid runs/s3a_seg1.rc 2>/dev/null || true
echo "---CLAIM---"
python3 scripts/card_claim.py status | head -20
echo "---LOG---"
tail -n 40 runs/s3a_seg1.log 2>/dev/null || true
echo "---GPU---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
