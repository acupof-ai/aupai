#!/bin/bash
# LR probe: lr_scale 0.85 vs 1.2, 500 steps each, real shape, serial in one window.
#
# THE QUESTION, and it is narrow: does the real shape step cleanly, and is 0.85 on the
# side where gradient norms have not yet lifted? The asymmetry is the whole reason to
# run it -- too high shouts within 500 steps, too low draws a perfectly healthy curve
# and quietly underperforms for three days with no control to compare against.
#
# WHAT IT DOES NOT ANSWER: whether the recipe is right. Eight of nine domains
# (code_py_starcoder is mid-rebuild) and 500 steps are far too little. No number from
# here belongs anywhere that reads like a baseline.
#
# Both arms read data/mix_probe_lr.json, so the missing domain cancels in the
# 0.85-vs-1.2 DIFFERENCE. It would contaminate an absolute loss, which is why the
# absolute value is not the readout.
#
# Serial in one window, minutes apart, one variable. The AttnRes A/B this morning was
# invalidated twice -- once because both arms passed the same value, once because the
# arms straddled different machine load -- and neither defect is visible in a result
# that looks well-formed.
#
# Cards come from the caller (fb granted 1-7). Refuses if any is busy: an A/B whose
# arms see different load measures the load.
set -u
cd /work/aupai

source eval/_devs.sh 7 || exit 1
WANT=",$(IFS=,; echo "${_DEVS[*]}"),"
NW=${#_DEVS[@]}

busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F', ' -v want="$WANT" 'index(want, ","$1",") && $2 > 1024 {print $1}' | tr '\n' ' ')
if [ -n "$busy" ]; then
  echo "refusing: card(s) $busy are in use; an A/B whose arms see different load measures the load"
  exit 1
fi

rm -rf /tmp/torchinductor_root

for lr in 0.85 1.2; do
  echo "=== ARM lr_scale=$lr world=$NW cards ${_DEVS[*]} $(date -u +%H:%M:%S)"
  NGPU="$NW" PORT=29551 ./run_ddp.sh \
    --mix data/mix_probe_lr.json --name "lrprobe_${lr}" \
    --dim 1024 --layers 32 --heads 8 --ffn_hidden 3072 \
    --batch 32 --accum 1 --grad_ckpt \
    --warmup 20 --warmdown 0.65 --anneal_frac 0.1 \
    --lr_scale "$lr" \
    --max_steps 500 --save_every 100000 --val_every 0 \
    > "runs/lrprobe_${lr}.log" 2>&1
  echo "=== ARM lr${lr} exit=$? $(date -u +%H:%M:%S)"
  grep -E '^step |grad_norm|nan|NaN|Error|out of memory|Traceback' "runs/lrprobe_${lr}.log" | tail -5
done
echo LRPROBE_ALL_DONE
