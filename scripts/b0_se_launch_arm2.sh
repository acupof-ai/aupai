#!/bin/bash
# N7 Stage E arm 2: the looped 4-7 arm at 2B tokens, on the granted cards 2+3.
#
# A SCRIPT FILE RATHER THAN AN INLINE nohup, because the inline form is what made the
# earlier test_e2e run unattributable (6e, 2026-09-04): a `bash -lc` leader with no setsid
# left a train.py on a card whose parent had been killed, and the controller had to guess
# whose it was. A file leaves the argv, the card, and the reason in one readable place.
#
# WHY 7630 STEPS: 7630 x 262,144 = 2.00016B tokens, exactly double the 1B looped arm's
# 3815, so the 1B and 2B points of THIS arm differ only in length. The gap-widens question
# needs that; a step count chosen from a token target computed some other way would move
# two things at once.
#
# WHAT DIFFERS BETWEEN THE 1B AND 2B POINTS AND IS NOT A BUG: --warmup 300 is ABSOLUTE, so
# this arm spends 3.9% of its schedule in warmup against the 1B arm's 7.9%. warmdown 0.1
# and anneal_frac 0 are FRACTIONS, so the cosine tail scales with total. Recorded rather
# than silently fixed -- changing warmup here would make the two points differ in the
# recipe as well as the length.
#
# train.py:1836 computes wd_steps from `total`, so a wrong --max_steps is not merely a
# short run: it is a different cosine tail on every step before it, and unresumable.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=b0_se_looped_2b
LOG="runs/${NAME}.log"

# THE CARDS COME FROM THE CALLER, not from a literal here. Writing CUDA_VISIBLE_DEVICES=2,3
# inside the script is what device_set_honoured refuses, and the refusal is right: an
# assignment in a child REPLACES the parent's restriction rather than indexing into it, so
# a script naming physical cards escapes whatever lane it was confined to. On 2026-08-31 a
# lane-card launch landed on GPU 0 and blocked a training run that way. Here it would also
# make the file wrong the moment the grant moves off 2,3 -- while still looking correct.
#
# Two cards, because this arm runs world 2 like every other Stage E arm.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "REFUSING: set CUDA_VISIBLE_DEVICES to the GRANTED cards, e.g." >&2
  echo "  CUDA_VISIBLE_DEVICES set to the granted pair, e.g. 2 and 3, then run this" >&2
  echo "  Ownership lives in runs/card_assignment.json and is never the nvidia-smi row." >&2
  exit 1
fi
source eval/_devs.sh 2

if [ -e "$LOG" ]; then
  echo "REFUSING: $LOG exists. A second launch under this name would append into the log of" >&2
  echo "  the first and the two runs' steps would interleave in one file. Move it or rename." >&2
  exit 1
fi

# The disk reading goes in the exp row, and it is read HERE rather than trusted from the
# row: /work sat at 96% used all evening, and the number in a row written minutes earlier
# describes the filesystem as it was then. This arm writes 7 rolling checkpoints plus a
# final at ~540 MB each.
echo "df before launch:"
df -h /work | tail -1
echo "cards from the caller: ${_DEVS[0]},${_DEVS[1]}"

nohup env CUDA_VISIBLE_DEVICES=${_DEVS[0]},${_DEVS[1]} NGPU=2 PORT=29514 ./run_ddp.sh \
  --mix data/mix_200m_8b.json --name "$NAME" \
  --dim 768 --layers 12 --heads 6 --ffn_hidden 2304 \
  --batch 16 --accum 2 --no-grad_ckpt \
  --lr_scale 1.0 --warmdown 0.1 --anneal_frac 0 --warmup 300 \
  --save_every 1000 --max_steps 7630 --seed 42 --loop 4 7 \
  > "$LOG" 2>&1 &

echo "launched, leader pid $!"
echo "the TORCHRUN pid is what card_claim needs -- card_claim refuses its own pid, and the"
echo "leader above is the nohup shell, not torchrun. Read it once the run is up:"
echo "  ps -eo pid,args | grep -- '--name $NAME' | grep torchrun | grep -v grep"
