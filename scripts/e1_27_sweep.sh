#!/bin/bash
# e1-27: sweep OUR arm's SFT lr, four new points, one per card, 1024 steps each.
# restartable: launches four detached sft_math.py runs, each writing its own ckpt and log. An
# interrupt to THIS script costs nothing -- the arms are already detached; an interrupt to an ARM
# costs that arm's GPU-hours (~95 min) and it must be relaunched, since sft_math.py saves only at
# the end of the 1024 steps.
#
#
# WHY. docs/audits/control_pythia160m_vs_ours.md 5.0 records that the control reduced its own
# floor by 61.0% against our 34.8%, and states three reasons that reading cannot be called "the
# control's SFT recipe is better" -- the first being that relative reductions do not compare
# across different floors. The second is ours: the control swept five lrs and took the best,
# our arm ran ONE fixed recipe and never swept. This sweep is what separates a floor effect
# from a recipe difference, and it is the experiment 5.0 defers.
#
# THE GRID (1e's ruling) is {0.1, 0.3, 3, 10}x relative to x1, and x1 is lr_scale 0.1 --
# MEASURED, not assumed: runs/e1_27_step0/ shows 40 steps at 0.1 reproducing
# runs/control_ours.log bit-for-bit (four deltas 0.000) while lr_scale 1.0 diverges to -0.224.
# So the absolute scales are 0.01 / 0.03 / 0.3 / 1.0, and x1 = 0.1 is the FIFTH point, reused
# from ckpt_control_ours.pt rather than retrained.
#
# lr_scale 1.0 already has 40 steps as the negative control, and it was much worse there. That
# is a 40-step reading; 1024 steps can differ, so it runs as a full grid point (1e agreed).
set -uo pipefail
cd /work/aupai

OUT=runs/e1_27_sweep
mkdir -p "$OUT"

# The pod has no git: a file here is whatever the last writer left, and a pod_push --all from
# another session already rolled this flag back once mid-launch (22:20:01Z, from a63359ec),
# killing two arms in two seconds. One grep turns that into a refusal before four cards are
# claimed.
grep -q -- "--stop_after" sft_math.py || {
  echo "REFUSING: /work/aupai/sft_math.py predates --stop_after, so it is not the version"
  echo "  this sweep was designed against. Ask 1e to push current main."; exit 2; }

# The base and the pack must be the ones the reused fifth point was trained on, or the five
# points are not one sweep.
for f in ckpt_p200m_4b_0902.pt data/sft/control_sft_ours.pt; do
  [ -f "$f" ] || { echo "REFUSING: $f absent"; exit 2; }
done

# THE CARDS ARE 1e's ALLOCATION (0,1,2,7 -- b0-17 owns 3-6), so they are passed IN rather than
# chosen here: the caller sets CUDA_VISIBLE_DEVICES and each arm takes one lane out of it via
# eval/_devs.sh. Writing a physical index in a child REPLACES the caller's restriction instead of
# indexing into it, which on 2026-08-31 put a lane-card launch onto a training-block GPU (2f97e4a).
# The run this script produced was launched with CUDA_VISIBLE_DEVICES=0,1,2,7, so ${_DEVS[i]} is
# the same physical card the published arms used -- but the guard cannot know that from a literal,
# and it is right not to trust one.
: "${CUDA_VISIBLE_DEVICES:=0,1,2,7}"
source eval/_devs.sh 4 || exit 2
LRS=(0.01 0.03 0.3 1.0)
for i in "${!LRS[@]}"; do
  c=${_DEVS[$i]}
  used=$(nvidia-smi -i "$c" --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 1000 ] || { echo "REFUSING: card $c holds ${used} MiB (b0-17 owns 3-6)"; exit 2; }
done

pids=()
for i in "${!LRS[@]}"; do
  s=${LRS[$i]}
  c=${_DEVS[$i]}
  log="$OUT/sweep_lr${s}.log"
  CUDA_VISIBLE_DEVICES=${_DEVS[$i]} python3 -u sft_math.py \
    --resume ckpt_p200m_4b_0902.pt \
    --sft_path data/sft/control_sft_ours.pt \
    --out "ckpt_e1_27_lr${s}.pt" \
    --lr_scale "$s" --epochs 1 > "$log" 2>&1 &
  p=$!
  pids+=("$s:$c:$p")
  python3 scripts/card_claim.py acquire --name "e1_27_sweep_$s" --cards "$c" --pid "$p" \
    --note "e1-27: ours-arm SFT lr sweep, lr_scale $s, 1024 steps" >/dev/null 2>&1 \
    || { echo "REFUSING: could not claim card $c; killing $p"; kill "$p" 2>/dev/null; exit 2; }
  echo "launched lr_scale $s on card $c, pid $p -> $log"
done

echo "all four launched $(date -u '+%H:%M:%SZ'); ~95 min each at 55s/10steps"
for e in "${pids[@]}"; do
  s=${e%%:*}; r=${e#*:}; c=${r%%:*}; p=${r##*:}
  wait "$p"; rc=$?
  python3 scripts/card_claim.py release --name "e1_27_sweep_$s" >/dev/null 2>&1
  echo "lr_scale $s (card $c) exited $rc $(date -u '+%H:%M:%SZ')"
done
echo "SWEEP TRAINING DONE -- scoring is a separate step (eval_heldout on the shared ids)"
