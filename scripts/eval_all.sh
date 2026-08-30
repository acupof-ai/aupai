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

# Score a checkpoint with the vocabulary it was trained on: ids do not survive a
# tokenizer rebuild, so a mismatch is silent noise.
VOCAB=$(python3 -c "
import torch, sys
ck = torch.load('$CKPT', map_location='cpu', weights_only=False)
c = ck['cfg']
print(c.get('vocab_real', c.get('vocab')), bool(c.get('fone')))
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

# 1. math-hard -- v1 retired as metric of record: our own generators contaminated it.
#    Run for continuity; numbers across the 0830v1 reset are not comparable.
say "--- math-hard (retired as metric of record; continuity only)"
NGPU=$NGPU TOKENIZER=$TOK bash scripts/eval_hard.sh "$CKPT" "$NGPU" 2>&1 | tee -a "$LOG" | grep TOTAL || say "  FAILED"

# 2. math-500 -- 0.0% contamination on the pod pretraining corpus (holdout-filtered;
#    the 10.2% figure was the local corpus). 30% of questions have a containment hit
#    in the math SFT corpus, so post-SFT absolute values are inflated; base-checkpoint
#    values are clean.
say "--- math-500 (post-SFT inflated by SFT-corpus overlap; base values clean)"
NGPU=$NGPU TOKENIZER=$TOK bash scripts/eval_math.sh "$CKPT" "$NGPU" 2>&1 | tee -a "$LOG" | grep TOTAL || say "  FAILED"

# 3. MC suite -- a 200M Chinese model at the 25% chance line; a regression tripwire,
#    not a capability measure.
say "--- MC suite (regression tripwire; chance is 25%)"
say "    ceval is the only Chinese one; the rest are English and this is a Chinese model."
CUDA_VISIBLE_DEVICES=0 python3 eval/run_eval.py --ckpt "$CKPT" --tokenizer "$TOK" \
  --benchmarks ceval mmlu arc-easy hellaswag piqa 2>&1 | tee -a "$LOG" | tail -10 || say "  FAILED"

# 4. FoNE digit head -- --fone checkpoints only; raw accuracy is unreadable without
#    the always-0 and copy-previous baselines.
if [ "$IS_FONE" = "True" ]; then
  say "--- FoNE digit head"
  CUDA_VISIBLE_DEVICES=0 python3 scripts/fone_digit_acc.py --ckpt "$CKPT" --domain math \
    2>&1 | tee -a "$LOG" | tail -4 || say "  FAILED"
fi

# 5. Arithmetic accuracy inside the generated steps: score and arithmetic are
#    different questions, and this repo measures both.
say "--- arithmetic in generated steps (eqcheck)"
python3 - "$CKPT" <<'PYEOF' 2>&1 | tee -a "$LOG"
import glob, json, os, sys
sys.path.insert(0, "scripts")
from eqcheck import check_steps
tag = os.path.basename(sys.argv[1])
files = sorted(glob.glob(f"data/eval/hard_{tag}.[0-9].jsonl")) or glob.glob(f"data/eval/hard_{tag}.jsonl")
rows = [json.loads(l) for f in files for l in open(f, encoding="utf-8")]
rows = [r for r in rows if r.get("greedy", True)]
if not rows:
    print("  no predictions on disk"); raise SystemExit
n = bad = has = 0
for r in rows:
    e, b = check_steps(r["gen"]); n += e; bad += b; has += bool(e)
print(f"  {len(rows)} generations | {n} verifiable equations | {100*bad/max(1,n):.1f}% wrong "
      f"| {100*has/len(rows):.0f}% of generations show an equation")
PYEOF

say ""
say "summary for $CKPT"
grep -E "TOTAL|whole-number exact|^Average|% wrong" "$LOG" | sed 's/^/  /'
say "log: $LOG"
