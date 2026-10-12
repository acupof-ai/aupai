#!/bin/bash
# Operator-level A/B at a FIXED shape: micro 2 x accum 16, world 7, seq 4096, math_v6 pack.
# The only variable is activation checkpointing, so the delta prices activation RECOMPUTE
# for this model (3.9B MoE, 12 layers) -- the number eff.batch_ceiling's "25% wall clock"
# was measured for a 200M dense model, which does not transfer.
#
# 30 steps each so train.py's every-10-steps log block prints exact s/10steps; the C1 pack
# (100 rows) is too small to sustain 30 steps at 224 packs/step, hence math_v6.
set -uo pipefail
cd /work/aupai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2,3,4,5,6,7}
PACK=data/sft/math_v6/math_sft_v4.pt
BASE=ckpt_v42s2.pt
STEPS=${STEPS:-30}
PORT=29641

run_arm () {
  name="$1"; shift
  log="runs/sft_shape_${name}.log"
  echo "=== arm ${name}: $(date -u '+%F %T') UTC" | tee -a runs/sft_shape_all.log
  : > "$log"
  python3 scripts/card_claim.py acquire --name "shape_${name}" --cards 0,2,3,4,5,6,7 \
      --note "SFT shape A/B ${name}" --wait 0 || true
  torchrun --nproc_per_node=7 --master_port=${PORT} sft_math.py \
    --resume "$BASE" --sft_path "$PACK" --out "/tmp/shape_${name}.pt" \
    --epochs 2 --batch 2 --accum 16 ${1:-} \
    --sft_peak_lr 2e-5 --sft_final_lr_frac 0.1 --warmup_frac 0.03 --lr_decay cosine \
    --save_every 100000 --stop_after "$STEPS" >> "$log" 2>&1
  echo "arm ${name} rc=$?" | tee -a runs/sft_shape_all.log
  grep -E "^step |peak allocated|nan" "$log" | tail -5 | tee -a runs/sft_shape_all.log
  python3 scripts/card_claim.py release --name "shape_${name}" >/dev/null 2>&1 || true
  PORT=$((PORT + 1))
}

run_arm "gc_on"  ""
run_arm "gc_off" "--no-grad_ckpt"
echo "=== shape A/B done: $(date -u '+%F %T') UTC" | tee -a runs/sft_shape_all.log
