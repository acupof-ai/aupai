#!/bin/bash
# L2 option-2 pipeline (fb 2026-09-11): convert L2 shards in FETCH ORDER with the
# same per-group safety as L3 -- no exec filter (L2 keep = decontam + exact-dedup
# + drop CONFIG/TEST). Designed to overlap the L3 run on a small core slice.
#
# Each group owns a contiguous shard range, waits for valid parquet footers,
# converts with --raw, asserts row conservation (bucket sum == total_rows ==
# parquet num_rows), writes deleted_<tag>.manifest naming every raw path, then
# rms them. Any group failure aborts before aggregate.
#
# Token cap: the L2 run is budget-bounded (fb: keep until >= L2_STOP_TOKENS new
# tokens, then stop the fetch). Set LAST_SHARD to the highest shard fetched you
# want converted; run the aggregate over however many groups finished.
set -u
cd /work/aupai
RAW=/data00/aupai_raw/ultradata
OUT=data/corpus/code_ultra_l2
MLOG=runs/ultra_groups/manifests
NG=${NG:-4}
FIRST=${FIRST:-1}
LAST=${LAST:-119}
NSH=119
mkdir -p runs/ultra_groups "$MLOG"

# No exec on L2; the tokenizer encode is the only per-row CPU and the python
# process runs multi-threaded. NG small keeps it inside the L3-overlap slice.
CORES=$(nproc)
echo "cores=$CORES groups=$NG l2 shards $FIRST-$LAST (no exec sandbox)"

footer_ok() {
  python3 - "$1" <<'PY'
import os, sys
p = sys.argv[1]
if not os.path.exists(p) or os.path.getsize(p) < 8:
    sys.exit(1)
with open(p, "rb") as f:
    f.seek(-4, 2)
    sys.exit(0 if f.read() == b"PAR1" else 1)
PY
}

run_group() {
  g=$1
  local first=$2 last=$3
  [ $first -gt $last ] && return 0
  local tag; tag=$(printf "g%02d" "$g")
  local paths=() p
  for i in $(seq $first $last); do
    p="$RAW/UltraData-Code-L2-py-part-$(printf '%05d' $i)-of-$NSH.parquet"
    until footer_ok "$p"; do sleep 60; done
    paths+=("$p")
  done
  env PYTHONPATH=/work/aupai python3 datagen/ultradata_shards.py \
    --level L2 --first "$first" --last "$last" --exec-workers 1 \
    --raw "$RAW" \
    --tag "$tag" --stats-name "stats_$tag.json" --out "$OUT" \
    || { echo "CONVERT_FAILED $tag -- raw RETAINED"; return 1; }

  if ! env PYTHONPATH=/work/aupai python3 - "$first" "$last" "$OUT" "$tag" <<'PY'
import json, sys
import pyarrow.parquet as pq

first, last, out, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
s = json.load(open(f"{out}/stats_{tag}.json"))
assert sum(s["reasons"].values()) == s["total_rows"], (
    f"bucket sum {sum(s['reasons'].values())} != total_rows {s['total_rows']}")
expected = 0
for i in range(first, last + 1):
    p = (f"/data00/aupai_raw/ultradata/UltraData-Code-L2-py-part-{i:05d}"
         f"-of-00119.parquet")
    expected += pq.ParquetFile(p).metadata.num_rows
assert s["total_rows"] == expected, (
    f"total_rows {s['total_rows']} != parquet rows {expected} -- some input not traversed")
assert s["n_shards"] > 0
print("MANIFEST_OK", tag, "rows", expected, "kept", s["kept"], "shards", s["n_shards"])
PY
  then echo "VERIFY_FAILED $tag -- raw RETAINED"; return 1; fi

  local man="$MLOG/deleted_l2_g$tag.manifest"
  { echo "# deleted after verified conversion of group $tag (shards $first-$last)"
    for p in "${paths[@]}"; do stat -c '%s %n' "$p"; done; } > "$man"
  for p in "${paths[@]}"; do rm -f "$p"; done
  echo "GROUP_DELETED $tag $((${#paths[@]})) paths"
}
export -f run_group footer_ok
export RAW OUT MLOG NSH

pids=""
SIZE=$(( (LAST - FIRST + 1 + NG - 1) / NG ))
for k in $(seq 0 $((NG - 1))); do
  gf=$((FIRST + k * SIZE)); gl=$((FIRST + (k + 1) * SIZE - 1))
  [ $gl -gt $LAST ] && gl=$LAST
  [ $gf -gt $LAST ] && continue
  ( run_group "$k" "$gf" "$gl" ) > "runs/ultra_groups/l2_g$(printf '%02d' "$k").log" 2>&1 &
  pids="$pids $!"
done
rc=0
for p in $pids; do wait "$p" || rc=1; done
[ $rc -ne 0 ] && { echo "GROUP_FAILURE rc=$rc -- no aggregate"; exit 1; }

env PYTHONPATH=/work/aupai python3 datagen/ultradata_shards.py \
  --level L2 --aggregate "stats_g*.json" --out "$OUT" \
  2>&1 | tee runs/ultra_groups/l2_aggregate.log
echo L2_PIPELINE_DONE
