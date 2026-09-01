#!/usr/bin/env bash
# One cell per card. The four cells of the sampled arm are independent measurements --
# {step24000, 16b pin} x {math-500, code-500} -- so they parallelise across cards
# perfectly. Serial on one card measured ~2.9 h per cell, 11 h for the set, with seven
# cards idle; the earlier "serial, not sharded" reasoning was about not SPLITTING a
# holdout across shards, and it does not apply to running whole cells side by side.
# Each cell still reads all 500 problems on one card, so the denominator is untouched.
#
# Usage: bash scripts/run_sampled_cell.sh <ckpt> <math|code> <run-tag>
# The caller sets the card. See run_sampled_arm.sh for why that is not set here.
set -uo pipefail
cd /work/aupai
: "${CUDA_VISIBLE_DEVICES:?the caller sets the card}"
CK=$1; EV=$2; RUN=$3
TOK=${TOKENIZER:-data/tokenizer.json}

echo "=== ${EV}-500  ${CK}  t=0.8 k=8 rep_stop=OFF  card ${CUDA_VISIBLE_DEVICES}"
if [ "$EV" = math ]; then
  python3 eval/math_zh.py --ckpt "$CK" --tokenizer "$TOK" \
    --k 8 --temperature 0.8 --no_rep_stop --run "$RUN" 2>&1
else
  python3 eval/code_zh.py --ckpt "$CK" --tokenizer "$TOK" \
    --k 8 --temperature 0.8 --no_rep_stop --run "$RUN" 2>&1
fi
echo "   rc=$? ${EV} ${CK}"
