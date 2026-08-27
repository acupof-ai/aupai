#!/usr/bin/env bash
# Shard eval/math_hard.py across GPUs. Usage: scripts/eval_math.sh ckpt_sft_v5.pt [ngpu]
#
# A failed shard must not be reported as a lower score: bare `wait` returns 0
# regardless of child status and the merge never checked the row count, so an OOM
# in shard 3 produced "TOTAL math-hard: 148/456 = 32.5%" with exit 0 and
# run_sft.sh wrote it into EXPERIMENTS.md (REVIEW_2026-08-26.md #9).
set -euo pipefail
CKPT=$1; N=${2:-6}
K=${K:-1}; TEMP=${TEMP:-0}          # K>1 turns this into a sharded pass@k run
MAXNEW=${MAXNEW:-512}               # generation budget; raise it to test reasoning-length scaling
cd "$(dirname "$0")/.."
LOGDIR=$(mktemp -d)                      # never reuse /tmp/evalsh_*.log across runs
trap 'rm -rf "$LOGDIR"' EXIT
EXPECTED=$(wc -l < data/synthetic/math_hard_eval_1k.jsonl)
# math_hard writes one prediction row per generation: the greedy one always, plus K sampled ones
# only when K > 1. Scaling by K+1 unconditionally makes the k=1 case assert 2x the real count.
ROWS=$([ "$K" -gt 1 ] && echo $((EXPECTED * (K + 1))) || echo "$EXPECTED")

pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i python3 eval/math_hard.py --ckpt "$CKPT" --shards "$N" --shard "$i" \
    --k "$K" --temperature "$TEMP" --max_new "$MAXNEW" > "$LOGDIR/shard_$i.log" 2>&1 &
  pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "shard $i FAILED:" >&2; tail -20 "$LOGDIR/shard_$i.log" >&2; rc=1
  fi
done
[ $rc -eq 0 ] || { echo "eval aborted: $rc shard(s) failed" >&2; exit 1; }

grep -h "math-hard" "$LOGDIR"/shard_*.log
python3 - "$CKPT" "$N" "$ROWS" "$K" <<'PY'
import json, os, sys
ck, n, expected, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
base = f"data/eval/hard_{os.path.basename(ck)}"
rows = [json.loads(l) for i in range(n) for l in open(f"{base}.{i}.jsonl", encoding="utf-8")]
assert len(rows) == expected, f"merged {len(rows)} preds, expected {expected} — a shard is short"
if k > 1:
    # pass@1 is the greedy row; pass@k is any-correct over that question's sampled rows. Both have
    # to be computed here rather than read off a shard line -- a shard covers 1/N of the questions.
    from collections import defaultdict

    g, sm = defaultdict(list), defaultdict(list)
    for r in rows:
        (g if r["greedy"] else sm)[r["q"]].append(r["ok"])
    n_q = len(g)
    p1 = sum(v[0] for v in g.values())
    pk = sum(1 for q in g if any(sm.get(q, [])))
    mean = sum(sum(v) / len(v) for v in sm.values()) / max(len(sm), 1)
    print(
        f"TOTAL math-hard: pass@1(greedy) {p1 / n_q:.1%} ({p1}/{n_q}) | sampled mean {mean:.1%} | "
        f"pass@{k} {pk / n_q:.1%} ({pk}/{n_q}) | gap {(pk - p1) / n_q:+.1%}"
    )
    raise SystemExit(0)
ok = sum(r["ok"] for r in rows)
by = {}
for r in rows:
    by.setdefault(r["level"], [0, 0])
    by[r["level"]][0] += int(r["ok"]); by[r["level"]][1] += 1
print("  " + ", ".join(f"{k}: {v[0]}/{v[1]}={v[0] / v[1]:.0%}" for k, v in sorted(by.items())))
with open(f"{base}.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"TOTAL math-hard: {ok:.0f}/{len(rows)} = {ok / len(rows):.1%}")
PY
