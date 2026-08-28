#!/usr/bin/env bash
# The full evaluation a checkpoint gets at the end of every stage: pretrain, SFT, RL.
#
#   scripts/eval_all.sh ckpt_k6_fone.pt                       # today's vocabulary
#   scripts/eval_all.sh ckpt_k5_clean_0827.pt data/tokenizer_k5.json
#   NGPU=3 scripts/eval_all.sh ckpt.pt                        # fewer shards
#
# Writes runs/evalall_<ckpt>.log and prints one summary block. Every number below
# has a known failure mode, so each is labelled with what it can and cannot say.
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:?usage: eval_all.sh <ckpt> [tokenizer.json]}
TOK=${2:-data/tokenizer.json}
NGPU=${NGPU:-6}
LOG=runs/evalall_$(basename "$CKPT" .pt).log
: > "$LOG"

# A checkpoint must be scored with the vocabulary it was trained on. data/tokenizer.json
# is rebuilt in place and ids do not survive a rebuild, so a mismatch is silent noise,
# not an error -- every tool below asserts cfg.vocab against the file.
VOCAB=$(python3 -c "
import torch, sys
ck = torch.load('$CKPT', map_location='cpu', weights_only=False)
c = ck['cfg']
print(c.get('vocab'), bool(c.get('fone')))
")
read -r CKPT_VOCAB IS_FONE <<< "$VOCAB"
TOK_VOCAB=$(python3 -c "from tokenizers import Tokenizer; print(Tokenizer.from_file('$TOK').get_vocab_size())")
if [ "$CKPT_VOCAB" != "$TOK_VOCAB" ]; then
  echo "STOP: $CKPT was trained at vocab $CKPT_VOCAB, $TOK has $TOK_VOCAB." | tee -a "$LOG"
  echo "      Pass the matching tokenizer as the second argument." | tee -a "$LOG"
  exit 1
fi
echo "ckpt $CKPT | vocab $CKPT_VOCAB | fone $IS_FONE | tokenizer $TOK" | tee -a "$LOG"

say() { echo "$*" | tee -a "$LOG"; }

# 1. math-hard -- the metric of record. 1032 problems, 899 templates, 486 solution
#    skeletons, so n_eff is 774-989 and the 95% half-width at a 3% pass rate is
#    +/-1.1pt. Differences smaller than that are not differences.
say "--- math-hard (metric of record)"
NGPU=$NGPU TOKENIZER=$TOK bash scripts/eval_hard.sh "$CKPT" "$NGPU" 2>&1 | tee -a "$LOG" | grep TOTAL || say "  FAILED"

# 2. math-500 -- saturated, and 10.2% of it has a near-duplicate carrying the same
#    answer in the Belle training data. Its absolute value is inflated; only use it
#    to compare checkpoints with equal exposure.
say "--- math-500 (inflated ~10pt by contamination; comparison only)"
NGPU=$NGPU TOKENIZER=$TOK bash scripts/eval_math.sh "$CKPT" "$NGPU" 2>&1 | tee -a "$LOG" | grep TOTAL || say "  FAILED"

# 3. English MC suite -- this is a 200M Chinese model and it sits at the 25% chance
#    line, so treat it as a regression tripwire rather than a capability measure.
say "--- MC suite (regression tripwire; chance is 25%)"
CUDA_VISIBLE_DEVICES=0 python3 eval/run_eval.py --ckpt "$CKPT" --tokenizer "$TOK" \
  --benchmarks mmlu arc-easy hellaswag piqa 2>&1 | tee -a "$LOG" | tail -8 || say "  FAILED"

# 4. FoNE digit head -- only meaningful for a --fone checkpoint. Scored against two
#    baselines because raw accuracy is unreadable without them: always-0 is 84.8%
#    per digit, and copy-the-previous-number is 16.4% whole-number.
if [ "$IS_FONE" = "True" ]; then
  say "--- FoNE digit head"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/fone_digit_acc.py --ckpt "$CKPT" --domain math \
    2>&1 | tee -a "$LOG" | tail -4 || say "  FAILED"
fi

say ""
say "summary for $CKPT"
grep -E "TOTAL|whole-number exact|^Average" "$LOG" | sed 's/^/  /'
say "log: $LOG"
