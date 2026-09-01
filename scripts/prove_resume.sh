#!/bin/bash
# Prove crash-recovery on the LAUNCH RECIPE before the launch, not after (user order,
# 2026-09-02). Five people read the resume path five times tonight and produced three
# different verdicts -- fb said it was broken from a stale line number, de read the code
# and said it was fixed, b0's 26/34/92 turned out to be a derivation quoted as a
# measurement, and two separate discard branches surfaced only on the fourth read. One
# run settles all of it.
#
# SHAPE: run the launch recipe for 60 steps saving every 20, kill -9 after step 40 (a
# crash, not a clean exit -- the end-of-run save writes neither `step` nor `opt`, so a
# clean exit would test the one path we already know is broken), resume from the .step40
# checkpoint, and read the restart.
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/prove_resume.sh
#
# The card set comes from the caller's environment and is never written here. Assigning
# CUDA_VISIBLE_DEVICES in a child REPLACES the parent's restriction rather than indexing
# into it, so a script that sets it escapes whatever lane the caller confined it to --
# the same escape as a hardcoded index, and what device_set_honoured refuses.
set -u
cd /work/aupai

DEVS="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the cards the launch will use}"
NW=$(echo "$DEVS" | awk -F, '{print NF}')
NAME=proberesume
SEED=42          # ONE value, used by both launches below. See WHY THE SEED IS A VARIABLE.
MIX=data/mix_500m.json

# WHY THE SEED IS A VARIABLE, not typed twice: _sample_seed() returns Cfg.seed when
# Cfg.sample_seed is None, and sample_seed has no CLI flag, so it is ALWAYS None -- the
# corpus shuffle rides entirely on --seed. Two launches with different --seed shuffle the
# pools differently, every cursor is discarded as pointing at other rows, all nine domains
# restart at row 0, and the loss curve looks perfectly normal while it happens (de).
# A variable makes that impossible to get wrong by editing one line and not the other.

busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F', ' -v want=",$DEVS," 'index(want, ","$1",") && $2 > 1024 {print $1}' | tr '\n' ' ')
if [ -n "$busy" ]; then
  echo "refusing: card(s) $busy in use -- this test needs the cards the launch will use"
  exit 1
fi

rm -f "ckpt_${NAME}.pt" "ckpt_${NAME}.pt.step"* runs/${NAME}_*.log
rm -rf /tmp/torchinductor_root

FLAGS="--mix $MIX --dim 1024 --layers 32 --heads 8 --ffn_hidden 3072 \
--batch 32 --accum 1 --grad_ckpt --seq 4096 --warmup 300 --seed $SEED \
--max_steps 60 --save_every 20 --val_every 0"

echo "=== RUN 1: to step 40, then kill -9  $(date -u +%H:%M:%S)"
NGPU="$NW" PORT=29561 ./run_ddp.sh \
  $FLAGS --name "$NAME" > "runs/${NAME}_run1.log" 2>&1 &
RUN1=$!

# Wait for the step-40 checkpoint to exist rather than for a step-40 log line: the file is
# what run 2 consumes, and it is written after the line is printed. Killing between the two
# leaves nothing to resume from.
for _ in $(seq 1 240); do
  [ -f "ckpt_${NAME}.pt.step40" ] && break
  kill -0 $RUN1 2>/dev/null || { echo "FAIL: run 1 exited before step 40"; tail -20 "runs/${NAME}_run1.log"; exit 1; }
  sleep 15
done
[ -f "ckpt_${NAME}.pt.step40" ] || { echo "FAIL: no step40 checkpoint after 60 min"; exit 1; }

sleep 20   # let the save finish; a torch.save caught mid-write is a corrupt file, not a crash test
echo "=== killing $(date -u +%H:%M:%S)"
# kill -9 every rank. pkill by pattern is banned on this pod (it has taken out other
# sessions' jobs), so the pids come from the process group we started.
pkill -9 -P $RUN1 2>/dev/null
kill -9 $RUN1 2>/dev/null
sleep 30
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "=== RUN 2: resume from ckpt_${NAME}.pt.step40  $(date -u +%H:%M:%S)"
NGPU="$NW" PORT=29562 ./run_ddp.sh \
  $FLAGS --name "$NAME" --resume "ckpt_${NAME}.pt.step40" \
  > "runs/${NAME}_run2.log" 2>&1
echo "=== RUN 2 exit=$? $(date -u +%H:%M:%S)"

echo
echo "======== VERDICT ========"
python3 scripts/read_resume_proof.py "runs/${NAME}_run2.log" "ckpt_${NAME}.pt.step40"
