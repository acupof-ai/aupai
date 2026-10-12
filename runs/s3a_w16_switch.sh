#!/bin/bash
# Switch s3a from world 8 to world 16 after a complete checkpoint.
set -euo pipefail
CKPT="${1:?usage: s3a_w16_switch.sh <checkpoint-basename>}"
ROOT=/work/aupai
H20B=192.168.29.36
cd "$ROOT"
[ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT"; exit 1; }
size1=$(stat -c %s "$CKPT")
sleep 5
size2=$(stat -c %s "$CKPT")
[ "$size1" = "$size2" ] || { echo "checkpoint is still writing"; exit 1; }

echo "Copy checkpoint to H20B"
rsync -aH -e "ssh -o BatchMode=yes" "$CKPT" "$H20B:/data00/aupai_work/aupai/"

echo "Stop world-8 child"
CHILD=$(cat runs/s3a_seg1.child.pid)
BOOT=$(cat runs/s3a_seg1.bootstrap.pid)
kill -KILL -"$CHILD" 2>/dev/null || kill -KILL "$CHILD" 2>/dev/null || true
for _ in $(seq 1 30); do
  if ! kill -0 "$BOOT" 2>/dev/null; then break; fi
  sleep 1
done
if kill -0 "$BOOT" 2>/dev/null; then
  kill -KILL "$BOOT" 2>/dev/null || true
  sleep 2
fi
python3 scripts/card_claim.py release --name s3a_seg1 >/dev/null 2>&1 || true

echo "Start H20B node1"
ssh -o BatchMode=yes "$H20B" /data00/aupai_work/s3a_w16_node1.sh "$CKPT"
sleep 5

echo "Start pod node0"
rm -f runs/s3a_w16_node0.rc
setsid python3 runs/launch_s3a_w16_claimed.py "$ROOT/$CKPT" > runs/s3a_w16_node0.bootstrap.log 2>&1 &
echo $! > runs/s3a_w16_node0.bootstrap.pid
sleep 30

echo "Node1 container"
ssh -o BatchMode=yes "$H20B" docker ps --filter name=aupai_node1 --format '{{.Names}} {{.Status}}'
echo "Node0 latest log"
tail -n 30 runs/s3a_seg1.log 2>/dev/null || true
echo "Claim status"
python3 scripts/card_claim.py status
