#!/usr/bin/env bash
# The sampled arm: is the 0.0 absent capability or a greedy decoding pathology?
# Pre-registered in docs/lessons/sampled_arm_prereg.md BEFORE any number existed.
#
#   4 cells: {step24000, 16b pin} x {math-500, code-500}
#   t=0.8, k=8, rep_stop OFF, one GPU (0), serial -- each cell is a separate artifact
#
# Serial on one card rather than sharded across seven: the checkpoints are being
# compared to each other, and a sharded run splits the holdout so a short shard changes
# the denominator. Sequential costs wall time and buys a comparison that means something.
#
# --run names every artifact after this run, so nothing collides with the greedy preds
# already on disk and both arms remain readable side by side. That is the whole reason
# --run exists rather than --force here: the greedy numbers are the control.
set -uo pipefail
cd /work/aupai

# The CARD IS THE CALLER'S. An earlier draft wrote CUDA_VISIBLE_DEVICES=0 here and
# device_set_honoured refused the commit -- correctly: a script that names a physical
# index escapes whatever lane its caller confined it to, which is the 2026-08-31
# incident. tilerl allocated GPU0 for this run and GPU7 for its ncu, but that is an
# allocation, not a property of the measurement, so it belongs in the launch line.
: "${CUDA_VISIBLE_DEVICES:?the caller sets the card, e.g. CUDA_VISIBLE_DEVICES then this script}"
export TOKENIZER=${TOKENIZER:-data/tokenizer.json}
RUN=${RUN:-sampled_t08_k8}

for CK in ckpt_pretrain_30b_s2.pt.step24000 ckpt_pretrain_30b_s2.milestone_16b_step17500.pt; do
  for EV in math code; do
    echo "=== ${EV}-500  ${CK}  t=0.8 k=8 rep_stop=OFF"
    if [ "$EV" = math ]; then
      python3 eval/math_zh.py --ckpt "$CK" --tokenizer "$TOKENIZER" \
        --k 8 --temperature 0.8 --no_rep_stop --run "$RUN" 2>&1
    else
      python3 eval/code_zh.py --ckpt "$CK" --tokenizer "$TOKENIZER" \
        --k 8 --temperature 0.8 --no_rep_stop --run "$RUN" 2>&1
    fi
    echo "   rc=$?"
  done
done
echo "=== all four cells attempted"
