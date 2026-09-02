#!/usr/bin/env bash
# One SFT experiment end to end: log start -> train -> sharded eval -> log result -> plot.
#
#   scripts/run_sft.sh <name> <resume_ckpt> <sft_pt> [extra sft_math.py args...]
#
# Everything lands in the repo: runs/<name>.log, runs/experiments.jsonl,
# EXPERIMENTS.md, plots/<name>.png.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=$1; RESUME=$2; DATA=$3; shift 3
OUT="ckpt_${NAME}.pt"
NGPU=${NGPU:-8}
# Which cards, not just how many. `seq 0 NGPU-1` always started at card 0, so an NGPU=1 run
# took card 0 no matter what else held it -- and two single-card jobs meant to run
# concurrently (the control comparison's two arms) would have landed on the same card while
# seven sat idle. CARDS overrides; the default reproduces the old behaviour exactly.
# Built with tr, not `seq -s,`: BSD seq APPENDS the separator (GNU does not), so on this Mac
# `seq -s, 0 7` is "0,1,...,7," with a trailing comma. The original line passed that straight
# to CUDA_VISIBLE_DEVICES, where a trailing comma is tolerated -- which is why it never
# showed. Counting fields with awk -F, on the same string returns 9 for eight cards, so the
# rank count derived below would have launched nine ranks on eight cards.
CARDS=${CARDS:-$(seq 0 $((NGPU - 1)) | tr '\n' ',' | sed 's/,$//')}
# One rank per listed card, so CARDS and NGPU cannot silently disagree.
NGPU=$(printf '%s' "$CARDS" | tr ',' '\n' | grep -c .)
CMD="CUDA_VISIBLE_DEVICES=$CARDS torchrun --nproc_per_node=$NGPU sft_math.py --resume $RESUME --sft_path $DATA --out $OUT $*"

python3 scripts/exp.py start --name "$NAME" --cmd "$CMD" --notes "$(python3 - "$DATA" <<'PY'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
n = d["input_ids"].shape[0]
print(f"{n} rows x 4097 tok = {n * 4097 / 1e6:.1f}M tokens")
PY
)" >/dev/null

set +e
CUDA_VISIBLE_DEVICES=$CARDS torchrun --nproc_per_node="$NGPU" \
  --master_port="${PORT:-$((29520 + $(printf '%s' "$CARDS" | cut -d, -f1)))}" sft_math.py --resume "$RESUME" --sft_path "$DATA" --out "$OUT" "$@"
TRAIN_RC=$?
set -e
if [ $TRAIN_RC -ne 0 ]; then
  python3 scripts/exp.py done --name "$NAME" --status fail --result "train exited $TRAIN_RC"
  exit $TRAIN_RC
fi

# set -e would abort before exp.py done and strand the row as status="running"
set +e
# One evaluation path, not two. eval_all.sh runs math-hard, math-500, the MC suite
# and (for a --fone checkpoint) the digit head, and writes runs/evalall_<ckpt>.log.
# TOKENIZER travels with the checkpoint: ids do not survive a rebuild of
# data/tokenizer.json, and scoring against the wrong file yields noise, not an error.
TOKENIZER=${TOKENIZER:-data/tokenizer.json}
# EXPORT the cards, not just the count. eval_all.sh takes its devices FROM the caller's
# CUDA_VISIBLE_DEVICES (eval/_devs.sh, added because a lane-card launch once landed on
# physical GPU 0 -- a training-block card). Passing only NGPU would have sent the eval to
# cards 0..N-1 regardless of which card this run was given.
export CUDA_VISIBLE_DEVICES=$CARDS
NGPU=$NGPU bash eval/eval_all.sh "$OUT" "$TOKENIZER"
ALL_RC=$?
ALL_LOG=runs/evalall_$(basename "$OUT" .pt).log
RESULT=$(grep "TOTAL math-500" "$ALL_LOG" | tail -1)
HARD=$(grep "TOTAL math-hard" "$ALL_LOG" | tail -1)
EVAL_RC=$([ -n "$RESULT" ] && echo 0 || echo 1)
HARD_RC=$([ -n "$HARD" ] && echo 0 || echo 1)
set -e
if [ $EVAL_RC -ne 0 ] || [ $HARD_RC -ne 0 ]; then
  python3 scripts/exp.py done --name "$NAME" --status fail \
    --result "eval failed (math-500 rc=$EVAL_RC, math-hard rc=$HARD_RC)"
  exit 1
fi
python3 scripts/exp.py done --name "$NAME" --status ok --result "$RESULT | $HARD"
python3 scripts/plot_curves.py "runs/$NAME.log" || true
echo "$NAME: $RESULT | $HARD"
