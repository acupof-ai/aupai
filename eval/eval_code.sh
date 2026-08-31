#!/usr/bin/env bash
# Shard eval/code_zh.py across GPUs. Usage: scripts/eval_code.sh <ckpt> [ngpu]
# Same failure discipline as eval_math.sh: a failed shard aborts, never
# silently lowers the score.
set -euo pipefail
CKPT=$1; N=${2:-6}
TOK=${TOKENIZER:-data/tokenizer.json}
cd "$(dirname "$0")/.."
bash scripts/assert_vocab.sh "$CKPT" "$TOK"
LOGDIR=$(mktemp -d)
trap 'rm -rf "$LOGDIR"' EXIT
EXPECTED=$(wc -l < data/eval/code_holdout_500.jsonl)

pids=()
# Shard i maps to the i-th device the CALLER exposed; an unset caller means the
# physical first N. CUDA_VISIBLE_DEVICES=$i outright would override an outer
# restriction -- a lane-card launch (CUDA_VISIBLE_DEVICES=7) would land on
# physical GPU 0, a training-block card.
IFS=',' read -ra _DEVS <<< "${CUDA_VISIBLE_DEVICES:-}"
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=${_DEVS[$i]:-$i} python3 eval/code_zh.py --ckpt "$CKPT" --tokenizer "$TOK" --shards "$N" --shard "$i" \
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

grep -h "code-500" "$LOGDIR"/shard_*.log
python3 - "$CKPT" "$N" "$EXPECTED" <<'PY'
import json, os, sys
ck, n, expected = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
base = f"data/eval/preds_code_{os.path.basename(ck)}"
suffix = lambda i: f".{i}" if n > 1 else ""
rows = [json.loads(l) for i in range(n) for l in open(f"{base}{suffix(i)}.jsonl", encoding="utf-8")]
assert len(rows) == expected, f"merged {len(rows)} preds, expected {expected} — a shard is short"
ok = sum(r["ok"] for r in rows)
with open(f"{base}.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"TOTAL code-500: {ok:.0f}/{len(rows)} = {ok / len(rows):.1%}")
PY
