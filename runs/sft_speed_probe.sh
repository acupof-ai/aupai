#!/bin/bash
# A1/A2 acceleration probes for the stage-D SFT: does the real SFT path run faster
# without activation checkpointing, and how much does a bigger micro-batch buy?
#
# Baseline being compared against (measured, runs/v42_mathfix_v6_1009.log):
#   micro 4 x accum 8, grad_ckpt ON  -> 43.4-45.0 s/step at 1.05M token/step
#
# A1: same shape, --no-grad_ckpt. sft_math.py defaults grad_ckpt ON because its
#     comment says FP8 backward goes NaN without it, so the NaN count is the gate.
# A2: same effective batch (4x8 == 8x4), micro 8 x accum 4, --no-grad_ckpt.
#
# 12 steps each. A NaN in any step fails that arm; the log line to read is
# "step N/.. loss .." plus the torchrun NaN report.
set -uo pipefail
cd /work/aupai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
PACK=data/sft/math_v6/math_sft_v4.pt
BASE=ckpt_v42s2.pt
STEPS=${STEPS:-12}
PORT=29601

run_arm () {
  name="$1"; shift
  log="runs/sft_probe_${name}.log"
  echo "=== arm ${name}: $(date -u '+%F %T') UTC" | tee -a runs/sft_probe_all.log
  : > "$log"
  python3 scripts/card_claim.py acquire --name "probe_${name}" --cards 0,1,2,3,4,5,6,7 \
      --note "SFT acceleration probe ${name}" --wait 0 || true
  torchrun --nproc_per_node=8 --master_port=${PORT} sft_math.py \
    --resume "$BASE" --sft_path "$PACK" --out "/tmp/probe_${name}.pt" \
    --epochs 1 --batch "$1" --accum "$2" ${3:-} \
    --sft_peak_lr 2e-5 --sft_final_lr_frac 0.1 --warmup_frac 0.03 --lr_decay cosine \
    --save_every 100000 --stop_after "$STEPS" >> "$log" 2>&1
  rc=$?
  echo "arm ${name} rc=${rc}" | tee -a runs/sft_probe_all.log
  grep -E "^step |nan|NaN|peak allocated" "$log" | tail -6 | tee -a runs/sft_probe_all.log
  python3 scripts/card_claim.py release --name "probe_${name}" >/dev/null 2>&1 || true
  PORT=$((PORT + 1))
  return $rc
}

run_arm "a1_nograd" 4 8 "--no-grad_ckpt"
run_arm "a2_micro8" 8 4 "--no-grad_ckpt"
echo "=== probes done: $(date -u '+%F %T') UTC" | tee -a runs/sft_probe_all.log
