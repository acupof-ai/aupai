#!/bin/bash
# restartable: rerun relaunches from RESUME; the pack is built separately by scripts/anneal_pack.py.
set -euo pipefail
cd "$(dirname "$0")/.."
# Continued pretraining from avg3 on rows the 30B run never read (scripts/anneal_pack.py):
# fresh optimizer, LR_SCALE x pretrain LR (0.05 = the run's own floor) decaying linearly to 0.
NAME=${NAME:-v41_anneal2}
RESUME=${RESUME:-ckpt_v41_ced_0926_avg3.pt}
PACK=${PACK:-data/sft/anneal2/anneal2.pt}
OUT=ckpt_${NAME}.pt
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
source eval/_devs.sh 8
NPROC=${#_DEVS[@]}
CARDS=$(IFS=,; echo "${_DEVS[*]}")
BATCH=${BATCH:-4}
LR_SCALE=${LR_SCALE:-0.05}
SAVE_EVERY=${SAVE_EVERY:-2000}
PREC=${PREC---no_fp8 --stochastic_round}  # set-but-empty PREC selects fp8
STEPS=${MAX_STEPS:+--max_steps $MAX_STEPS}
if [ -z "${HYPOTHESIS:-}" ]; then
  echo "REFUSING: set HYPOTHESIS='<what this run is meant to show>' before launching."
  exit 2
fi
[ -f "$RESUME" ] || { echo "REFUSING: resume ckpt $RESUME not present"; exit 2; }
[ -f "$PACK" ]   || { echo "REFUSING: pack $PACK not present (run scripts/anneal_pack.py)"; exit 2; }
python3 scripts/exp.py start --name "$NAME" \
  --cmd "torchrun n$NPROC sft_math.py --resume $RESUME --sft_path $PACK $PREC --lr_decay linear --warmup_frac 0.01 --batch $BATCH --lr_scale $LR_SCALE $STEPS (cards $CARDS)" \
  --hypothesis "$HYPOTHESIS" >/dev/null
set +e
# The pack is gate-cache rows, 13-gram decontaminated against HumanEval/MBPP upstream; it was
# never checked against holdout_hashes.txt, so it trains unstamped and says so.
torchrun --nproc_per_node="$NPROC" \
  --master_port="${PORT:-29532}" \
  sft_math.py --resume "$RESUME" --sft_path "$PACK" --out "$OUT" --allow_unstamped_pack \
  $PREC --lr_decay linear --warmup_frac 0.01 \
  --epochs 1 --batch "$BATCH" --lr_scale "$LR_SCALE" $STEPS \
  --save_every "$SAVE_EVERY" &
TORCH_PID=$!
python3 scripts/card_claim.py acquire --name "$NAME" --cards "$CARDS" \
  --note "continued pretraining $PACK" --wait 0 --wait-for-device 300 || {
  echo "REFUSING to launch: card_claim acquire refused on devices $CARDS"
  kill "$TORCH_PID" 2>/dev/null || true
  python3 scripts/exp.py done --name "$NAME" --status fail --result "card claim refused"
  exit 1
}
trap 'python3 scripts/card_claim.py release --name "$NAME" >/dev/null 2>&1 || true' EXIT
wait "$TORCH_PID"
TRAIN_RC=$?
set -e
if [ $TRAIN_RC -ne 0 ]; then
  python3 scripts/exp.py done --name "$NAME" --status fail --result "train exited $TRAIN_RC"
  exit $TRAIN_RC
fi
python3 scripts/exp.py done --name "$NAME" --status ok --result "continued pretraining done -> $OUT"
echo "$NAME done -> $OUT"
