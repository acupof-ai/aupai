#!/usr/bin/env bash
# Measure this checkpoint's per-instance solve rate over the math bank, keep the 20-80% band.
#
#   scripts/probe_band.sh ckpt_sft_k4.pt [ngpu]
#
# 10,382 instances x (1 greedy + 8 sampled) is 93K generations; single-process that is hours, so it
# is sharded across the GPUs exactly like scripts/eval_hard.sh, with the same rule that a failed
# shard aborts rather than quietly lowering the measured rate.
set -euo pipefail
CKPT=$1
N=${2:-8}
cd "$(dirname "$0")/.."
PROBE=${PROBE:-data/rl/program_probe.jsonl}
GENS=data/rl/probe_gens.jsonl
BAND=data/rl/rl_band.jsonl
LOGDIR=$(mktemp -d)
trap 'rm -rf "$LOGDIR"' EXIT
EXPECTED=$(wc -l < "$PROBE")

pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i python3 eval/math_hard.py --ckpt "$CKPT" --data "$PROBE" \
    --shards "$N" --shard "$i" --k 8 --temperature 0.8 --max_new 320 \
    --dump "$GENS.$i" > "$LOGDIR/shard_$i.log" 2>&1 &
  pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "shard $i FAILED:" >&2
    tail -20 "$LOGDIR/shard_$i.log" >&2
    rc=1
  fi
done
[ $rc -eq 0 ] || { echo "probe aborted: a shard failed" >&2; exit 1; }

cat "$GENS".[0-9]* > "$GENS"
rm -f "$GENS".[0-9]*
python3 - "$GENS" "$EXPECTED" <<'PY'
import json, sys
n = sum(1 for _ in open(sys.argv[1], encoding="utf-8"))
assert n == int(sys.argv[2]), f"{n} probed, expected {sys.argv[2]} — a shard is short"
print(f"probe: {n} instances generated")
PY

python3 mathbank/program_probe.py score "$GENS"
python3 - "$BAND" <<'PY'
import json, sys
from collections import Counter
rows = [json.loads(l) for l in open("data/rl/instance_rates.jsonl", encoding="utf-8")]
band = [r for r in rows if 0.2 <= r["pass_at_k"] <= 0.8]
hist = Counter(min(int(r["pass_at_k"] * 10), 9) for r in rows)
print("solve-rate histogram (deciles): " + " ".join(f"{d / 10:.1f}:{hist[d]}" for d in range(10)))
with open(sys.argv[1], "w", encoding="utf-8") as f:
    for r in band:
        f.write(json.dumps({"instruction": r["instruction"], "answer": r["answer"]},
                           ensure_ascii=False) + "\n")
print(f"band: {len(band)}/{len(rows)} instances at 20-80% solve rate -> {sys.argv[1]}")
PY
