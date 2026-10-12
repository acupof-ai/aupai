#!/bin/bash
# D1e: continue the WINNING run. D1b (v1 data, 360 steps, 28.7 passes/example) reached
# E2 7/10; D1d (15x data, 2.0 passes/example) reached 2/10 with the SAME final loss, so
# passes-per-example -- not loss, not data volume -- is the variable that moves the
# behaviour (docs/standards/v42_deep_plan_1009.md section 16). This resumes D1b's
# checkpoint and pushes the same v1 set to ~100 passes/example (100 epochs = 1,250 steps).
#
# Question it answers: does the C1 dataset (short answers, every one terminated by
# <|im_end|>, no echo, no second <|im_start|>) fix E2's four failed judgments, or does
# the repetition live in the base and need something else (H3: decoding/termination)?
#
# SHAPE IS SET BY THE DATA, NOT BY HABIT. format_stop_v1.pt is 100 rows of 4097 tokens
# (65,393 supervised), so world 8 x micro 2 x accum 16 -- 224 packs/step -- exceeds one
# whole epoch per step and sft_math.py refuses the schedule ("needs at least two optimizer
# steps", measured 2026-10-09). 8 rows/step gives 12 steps/epoch, 48 over 4 epochs.
#
# --no-grad_ckpt: measured on the pod the same day at micro 2 x accum 16 x 7 cards:
# grad_ckpt ON 34.7 s/step (step 20-30 window), OFF 10.4 s/step -- 3.3x. micro 1 uses less
# activation memory than either, so it is inside the verified envelope.
#
# Claim pattern: start the job, then claim ITS pid, and stop if the claim fails
# (runs/claim_pattern_selftest.sh is the control; `--pid $$` is refused by card_claim).
set -euo pipefail
cd /work/aupai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NAME=v42_d1e_format_stop_1009
OUT=ckpt_v42_d1e_format_stop_1009.pt
BASE=ckpt_v42_d1b_format_stop_1009.pt
PACK=data/sft/stage_c/format_stop/format_stop_v1.pt
[ -f "$BASE" ] || { echo REFUSING_missing_base; exit 2; }
[ -f "$PACK" ] || { echo REFUSING_missing_pack; exit 2; }
[ ! -e "$OUT" ] || { echo REFUSING_output_exists; exit 2; }
LOG=runs/${NAME}.out
: > "$LOG"
torchrun --nproc_per_node=8 --master_port=29691 sft_math.py \
  --resume "$BASE" \
  --sft_path "$PACK" \
  --out "$OUT" \
  --epochs 100 --batch 1 --accum 1 --no-grad_ckpt \
  --sft_peak_lr 2e-5 --sft_final_lr_frac 0.1 --warmup_frac 0.03 \
  --lr_decay cosine --save_every 100000 \
  > "$LOG" 2>&1 &
TRAIN_PID=$!
if ! python3 scripts/card_claim.py acquire --name "$NAME" --cards 0,1,2,3,4,5,6,7 \
     --note "D1e continue from D1b ckpt, v1 format/stop, 100 epochs (~1250 steps), micro1xaccum1, no grad_ckpt" \
     --wait 0 --pid "$TRAIN_PID"; then
  echo "REFUSING_claim_failed -- killing torchrun $TRAIN_PID, nothing runs unclaimed"
  kill -TERM "$TRAIN_PID" 2>/dev/null || true
  wait "$TRAIN_PID" 2>/dev/null || true
  exit 4
fi
trap 'python3 scripts/card_claim.py release --name "$NAME" >/dev/null 2>&1 || true' EXIT
wait "$TRAIN_PID"
