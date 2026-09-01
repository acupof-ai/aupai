#!/usr/bin/env bash
# Shard eval/code_zh.py across GPUs. Usage: eval/eval_code.sh <ckpt> [ngpu]
# Same failure discipline as eval_math.sh: a failed shard aborts, never
# silently lowers the score.
set -euo pipefail
CKPT=$1; N=${2:-6}
TOK=${TOKENIZER:-data/tokenizer.json}
HOLDOUT=${HOLDOUT:-data/eval/code_holdout_500.jsonl}
TAG=${TAG:-}
# FORCE=1 / RUN=<name>: see eval_math.sh. Same contract as eval_hard.sh, which has had
# it since 2026-08-31 while these two silently refused every rescore.
FORCE_ARG=""; [ "${FORCE:-}" = "1" ] && FORCE_ARG="--force"
RUN_ARG=""; [ -n "${RUN:-}" ] && RUN_ARG="--run ${RUN}"
cd "$(dirname "$0")/.."
bash scripts/assert_vocab.sh "$CKPT" "$TOK"
LOGDIR=$(mktemp -d)
trap 'rm -rf "$LOGDIR"' EXIT
EXPECTED=$(wc -l < "$HOLDOUT")

pids=()
# Shard i runs on the i-th device the CALLER exposed; an unset caller means the
# physical first N. See eval/_devs.sh for why a physical index is wrong here.
source eval/_devs.sh "$N"
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=${_DEVS[$i]} python3 eval/code_zh.py --ckpt "$CKPT" --tokenizer "$TOK" --shards "$N" --shard "$i" \
    ${HOLDOUT:+--data "$HOLDOUT"} ${TAG:+--tag "$TAG"} $FORCE_ARG $RUN_ARG \
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
python3 - "$CKPT" "$N" "$EXPECTED" "$TAG" <<'PY'
import json, os, sys
sys.path.insert(0, "scripts")
from eval_artifacts import open_artifact, seal  # noqa: E402

ck, n, expected, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
run = os.environ.get("RUN") or None
force = os.environ.get("FORCE") == "1"
base = f"data/eval/preds_code_{tag + '_' if tag else ''}{os.path.basename(ck)}"
# The shard writers apply --run, so the merge reads the versioned paths. Note the
# order: versioned_path appends the run AFTER the shard index. See eval_math.sh.
shard = lambda i: f"{base}{'.%d' % i if n > 1 else ''}{'.' + run if run else ''}.jsonl"
rows = [json.loads(l) for i in range(n) for l in open(shard(i), encoding="utf-8")]
assert len(rows) == expected, f"merged {len(rows)} preds, expected {expected} — a shard is short"
ok = sum(r["ok"] for r in rows)
out_path = f"{base}.{run}.jsonl" if run else f"{base}.jsonl"
if n > 1:
    # At n == 1 the single shard already wrote this exact path; see eval_math.sh.
    with open_artifact(out_path, force=force) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
sealed, msg = seal(out_path, expected, written=len(rows))
print(("  " if sealed else "  INCOMPLETE: ") + msg)
print(f"TOTAL code-500: {ok:.0f}/{len(rows)} = {ok / len(rows):.1%}")
PY
