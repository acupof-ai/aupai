#!/bin/bash
# Parallel UltraData conversion: groups own disjoint shard ranges, write
# <domain>_<tag>_NNN.jsonl + stats_<tag>.json; a final aggregate pass sums
# groups into build_corpus_stats.json. L3 first (the fb-ruling exec filter),
# L2 chained after so the sandbox cores are not oversubscribed.
set -u
cd /work/aupai
RAW=/data00/aupai_raw/ultradata
PY="PYTHONPATH=/work/aupai python3 datagen/ultradata_shards.py --raw $RAW"
LOG=runs/ultra_groups
mkdir -p "$LOG"

run_level() {
  level=$1; ng=$2; nsh=$3; ew=$4
  out="data/corpus/code_ultra_${level,,}"
  size=$(( (nsh + ng - 1) / ng ))
  pids=""
  for g in $(seq 0 $((ng - 1))); do
    first=$((g * size + 1)); last=$((g * size + size)); [ $last -gt $nsh ] && last=$nsh
    [ $first -gt $nsh ] && continue
    tag=$(printf "g%02d" "$g")
    env PYTHONPATH=/work/aupai setsid python3 datagen/ultradata_shards.py \
      --level "$level" --first "$first" --last "$last" --exec-workers "$ew" \
      --tag "$tag" --stats-name "stats_$tag.json" --out "$out" \
      > "$LOG/${level,,}_$tag.log" 2>&1 &
    pids="$pids $!"
  done
  for p in $pids; do wait "$p"; done
  env PYTHONPATH=/work/aupai python3 datagen/ultradata_shards.py \
    --level "$level" --aggregate "stats_g*.json" --out "$out" \
    > "$LOG/${level,,}_aggregate.log" 2>&1
  echo "${level}_GROUPS_DONE"
}

run_level L3 10 147 16
run_level L2 6 119 1
echo ALL_GROUPS_DONE
