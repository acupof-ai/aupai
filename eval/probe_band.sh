#!/usr/bin/env bash
# Per-instance solve rate over the math bank; keep the 20-80% band.
#
#   eval/probe_band.sh ckpt_sft_k4.pt [ngpu]
#
# Sharded across the GPUs like eval/eval_hard.sh; a failed shard aborts rather than
# quietly lowering the measured rate.
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
# Shard i runs on the i-th device the CALLER exposed. See eval/_devs.sh.
source eval/_devs.sh "$N"
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=${_DEVS[$i]} python3 eval/math_hard.py --ckpt "$CKPT" --data "$PROBE" \
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
python3 scripts/select_band.py "$BAND" --min "${MIN_BAND:-800}"
