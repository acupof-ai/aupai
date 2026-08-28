#!/usr/bin/env bash
# Shard eval/math_zh.py across GPUs. Usage: scripts/eval_math.sh ckpt_sft_v5.pt [ngpu]
#
# A failed shard must not be reported as a lower score: bare `wait` returns 0
# regardless of child status and the merge never checked the row count, so an OOM
# in shard 3 produced "TOTAL math-500: 148/456 = 32.5%" with exit 0 and
# run_sft.sh wrote it into EXPERIMENTS.md (docs/review_2026-08-26.md #9).
set -euo pipefail
CKPT=$1; N=${2:-6}
# The vocabulary the checkpoint was trained on. Ids do not survive a rebuild of
# data/tokenizer.json, and a mismatch scores as noise rather than raising.
TOK=${TOKENIZER:-data/tokenizer.json}
cd "$(dirname "$0")/.."
LOGDIR=$(mktemp -d)                      # never reuse /tmp/evalsh_*.log across runs
trap 'rm -rf "$LOGDIR"' EXIT
EXPECTED=$(wc -l < data/eval/math_test_500.jsonl)

pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i python3 eval/math_zh.py --ckpt "$CKPT" --tokenizer "$TOK" --shards "$N" --shard "$i" \
    > "$LOGDIR/shard_$i.log" 2>&1 &
  pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "shard $i FAILED:" >&2; tail -20 "$LOGDIR/shard_$i.log" >&2; rc=1
  fi
done
[ $rc -eq 0 ] || { echo "eval aborted: $rc shard(s) failed" >&2; exit 1; }

grep -h "math-500" "$LOGDIR"/shard_*.log
python3 - "$CKPT" "$N" "$EXPECTED" <<'PY'
import json, os, sys
ck, n, expected = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
base = f"data/eval/preds_{os.path.basename(ck)}"
rows = [json.loads(l) for i in range(n) for l in open(f"{base}.{i}.jsonl", encoding="utf-8")]
assert len(rows) == expected, f"merged {len(rows)} preds, expected {expected} — a shard is short"
ok = sum(r["ok"] for r in rows)
with open(f"{base}.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"TOTAL math-500: {ok:.0f}/{len(rows)} = {ok / len(rows):.1%}")
PY
