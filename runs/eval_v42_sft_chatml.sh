#!/bin/bash
# Full ChatML evaluation of a v42 SFT checkpoint on the world-8 block.
#
#   bash runs/eval_v42_sft_chatml.sh [ckpt] [ngpu]
#
# Generative math/code use ChatML on an SFT checkpoint. The math arms auto-select it via
# score_matrix.classify reading cfg.kind="sft" (stamped by sft_math); HumanEval has no
# classifier path and needs the explicit --chatml flag. MC stays teacher-forced on raw
# prompts. Base comparison (step54000): math-500 0%, GSM8K repetition, HumanEval 28.21%.
set -uo pipefail
cd /work/aupai

CKPT=${1:-ckpt_v42_sft_run.pt}
NGPU=${2:-8}
TOK=data/tokenizer.json
TAG=${TAG:-sftchatml}
LOGDIR=runs/chatml_eval_${TAG}
mkdir -p "$LOGDIR"

[ -f "$CKPT" ] || { echo "REFUSING: $CKPT missing"; exit 2; }
bash scripts/assert_vocab.sh "$CKPT" "$TOK" || { echo "REFUSING: vocab mismatch"; exit 2; }

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/summary.log"; }

# Best-effort claim. Do NOT pass --wait-for-device here: before the first stage there is no
# GPU-holding child to follow, so it blocks a fixed timeout with cards sitting idle. A busy
# block already prevents launch; the claim is bookkeeping when the cards are free.
CARDS=$(seq 0 $((NGPU-1)) | tr '\n' ',' | sed 's/,$//')
python3 scripts/card_claim.py acquire --name "eval_${TAG}" --cards "$CARDS" \
  --pid $$ --wait 0 >/dev/null 2>&1 || say "WARN: card_claim did not bind (proceeding; cards must be free)"
trap 'python3 scripts/card_claim.py release --name "eval_${TAG}" >/dev/null 2>&1 || true' EXIT
source eval/_devs.sh "$NGPU"

# 1) math-500, sharded generative; ChatML auto via cfg.kind. Metric: TOTAL math-500.
say "=== math-500 ($NGPU shard, ChatML auto) ==="
FORCE=1 RUN="$TAG" NGPU=$NGPU TOKENIZER=$TOK bash eval/eval_math.sh "$CKPT" "$NGPU" \
  > "$LOGDIR/math500.log" 2>&1
grep -E "TOTAL math-500|FAILED|aborted" "$LOGDIR/math500.log" | tee -a "$LOGDIR/summary.log" || say "math-500 FAILED (see $LOGDIR/math500.log)"

# 2) GSM8K, single-card greedy; ChatML auto. ckpt is POSITIONAL.
say "=== GSM8K (ChatML auto) ==="
CUDA_VISIBLE_DEVICES=${_DEVS[0]} python3 eval/gsm8k.py "$CKPT" > "$LOGDIR/gsm8k.log" 2>&1
grep -E "GSM8K:" "$LOGDIR/gsm8k.log" | tee -a "$LOGDIR/summary.log" || { say "gsm8k FAILED"; tail -20 "$LOGDIR/gsm8k.log"; }

# 3) HumanEval, dynamic-queue shards, EXPLICIT --chatml, then merge+score across 164.
say "=== HumanEval ($NGPU shard, --chatml) ==="
HEQ="$LOGDIR/he_queue"; rm -rf "$HEQ"
for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=${_DEVS[$i]} OMP_NUM_THREADS=8 python3 eval/humaneval_gen.py \
    --ckpt "$CKPT" --device cuda:0 --chatml --max_new 280 --force \
    --queue_dir "$HEQ" --shard_i "$i" --shard_n "$NGPU" \
    > "$LOGDIR/he_w$i.log" 2>&1 &
done
wait
PAT="data/eval/preds_humaneval_$(basename "$CKPT").chatml.shard*of${NGPU}.jsonl"
NSH=$(ls $PAT 2>/dev/null | wc -l | tr -d ' ')
say "HumanEval shard files: $NSH/$NGPU"
if [ "$NSH" -eq "$NGPU" ]; then
  python3 eval/e0_merge_score.py --bench humaneval --n 1 --glob "$PAT" \
    --out "$LOGDIR/humaneval_chatml_merged.jsonl" --result "$LOGDIR/result.json" \
    2>&1 | tee -a "$LOGDIR/summary.log"
else
  say "HumanEval MERGE SKIPPED: expected $NGPU shard files"; grep -lE "Error|Traceback" "$LOGDIR"/he_w*.log
fi

# 4) MC likelihood suite, single card, RAW teacher-forced prompts (no ChatML).
say "=== MC suite (raw, teacher-forced) ==="
CUDA_VISIBLE_DEVICES=${_DEVS[0]} python3 eval/run_eval.py --ckpt "$CKPT" --tokenizer "$TOK" \
  --benchmarks mmlu ceval arc-easy arc-challenge boolq openbookqa winogrande \
  > "$LOGDIR/mc.log" 2>&1
grep -E "acc|SKIP" "$LOGDIR/mc.log" | tee -a "$LOGDIR/summary.log" || say "MC suite FAILED (see $LOGDIR/mc.log)"

say "=== DONE ==="
grep -E "TOTAL math-500|GSM8K:|FULL|CLEAN|acc" "$LOGDIR/summary.log"
