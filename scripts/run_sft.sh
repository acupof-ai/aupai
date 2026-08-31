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
CMD="torchrun --nproc_per_node=$NGPU sft_math.py --resume $RESUME --sft_path $DATA --out $OUT $*"

python3 scripts/exp.py start --name "$NAME" --cmd "$CMD" --notes "$(python3 - "$DATA" <<'PY'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
n = d["input_ids"].shape[0]
print(f"{n} rows x 4097 tok = {n * 4097 / 1e6:.1f}M tokens")
PY
)" >/dev/null

set +e
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU - 1))) torchrun --nproc_per_node="$NGPU" \
  --master_port="${PORT:-29520}" sft_math.py --resume "$RESUME" --sft_path "$DATA" --out "$OUT" "$@"
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
