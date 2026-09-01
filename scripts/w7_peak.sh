#!/bin/bash
# 7-card peak-memory probe: b32 then b16+accum2, serial, same tok/step (917,504).
#
# Both arms because the decision chain only closes with both numbers: if b32 does not
# fit at world 7 the fallback is b16+accum2, and that arm's peak did not exist either.
# Thirty extra steps buys the answer instead of another queue round-trip.
#
# Waits for cards by READING THE CARDS, not by watching a pid. A SIGTERMed rank held
# 72 GiB after ps had lost it, and a pid is only meaningful in the namespace that read
# it -- a guard on /proc from the wrong namespace launched a job onto a running probe's
# cards earlier today.
#
# Takes the caller's CUDA_VISIBLE_DEVICES rather than writing physical indices: an
# assignment in a child REPLACES the parent's restriction instead of indexing into it,
# so a hardcoded list escapes whatever lane the caller confined it to. Launch with
#   CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 bash scripts/w7_peak.sh
# Card 0 is another session's eval; which seven cards does not enter the measurement,
# the DDP bucket at seven ranks does.
set -u
cd /work/aupai

NGPU_WANT=$(echo "${CUDA_VISIBLE_DEVICES:-}" | awk -F, '{print NF}')
if [ "${CUDA_VISIBLE_DEVICES:-}" = "" ] || [ "$NGPU_WANT" -ne 7 ]; then
  echo "refusing: set CUDA_VISIBLE_DEVICES to exactly 7 cards; got '${CUDA_VISIBLE_DEVICES:-unset}'"
  exit 2
fi

# Wait on the cards this run was given, not a hardcoded set -- otherwise the guard
# watches one lane and the job lands in another.
busy=""
for _ in $(seq 1 120); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' -v want=",$CUDA_VISIBLE_DEVICES," \
               'index(want, ","$1",") && $2>1024 {print $1}' | tr '\n' ' ')
  [ -z "$busy" ] && break
  sleep 15
done
if [ -n "$busy" ]; then
  echo "ABORT: requested card(s) still held after 30 min: $busy"
  exit 1
fi

rm -rf /tmp/torchinductor_root

for arm in "32 1" "16 2"; do
  set -- $arm
  B=$1
  A=$2
  echo "=== ARM batch=$B accum=$A world=$NGPU_WANT cards $CUDA_VISIBLE_DEVICES $(date -u +%H:%M:%S)"
  NGPU="$NGPU_WANT" PORT=29531 ./run_ddp.sh \
    --mix data/mix_30b_stage2.json --name "w7_b${B}a${A}" \
    --dim 1024 --layers 32 --heads 8 --ffn_hidden 3072 \
    --batch "$B" --accum "$A" --grad_ckpt --lr_scale 0.85 \
    --max_steps 30 --save_every 100000 --val_every 0 \
    > "runs/w7_b${B}a${A}.log" 2>&1
  echo "=== ARM b${B}a${A} exit=$? $(date -u +%H:%M:%S)"
  grep -E '^step |peak |Error|out of memory|Traceback' "runs/w7_b${B}a${A}.log" | tail -6
done
echo W7_ALL_DONE
