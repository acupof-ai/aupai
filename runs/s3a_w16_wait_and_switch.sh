#!/bin/bash
set -euo pipefail
CKPT="ckpt_s3a_seg1.pt.step60000"
cd /work/aupai
for _ in $(seq 1 1440); do
  if [ -f "$CKPT" ]; then
    size1=$(stat -c %s "$CKPT")
    sleep 10
    size2=$(stat -c %s "$CKPT")
    if [ "$size1" = "$size2" ] && [ "$size1" -gt 0 ]; then
      echo "checkpoint stable; start switch"
      bash runs/s3a_w16_switch.sh "$CKPT"
      exit $?
    fi
  fi
  sleep 30
done
echo "timeout waiting for $CKPT"
exit 1
