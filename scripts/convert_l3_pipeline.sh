#!/bin/bash
# Option-2 L3 pipeline (fb 2026-09-11): NG groups run CONCURRENTLY, each owning
# a disjoint shard range. Per group: wait for valid parquet footers -> convert
# -> assert row conservation -> write deleted_<tag>.manifest naming every raw
# path -> rm them. Aggregate group stats after every group finishes.
#
# Disk: the guarded fetch (fetch_l3_guarded.sh, 160G free floor) is the backstop.
# Groups only wait for shards, never fetch, so a paused fetch stalls late groups
# without corrupting early ones; each group frees its own raw after verification.
set -u
cd /work/aupai
RAW=/data00/aupai_raw/ultradata
OUT=data/corpus/code_ultra_l3
MLOG=runs/ultra_groups/manifests
EW=${EW:-0}
NG=${NG:-10}
NSH=147
mkdir -p runs/ultra_groups "$MLOG"

# NG groups x EW sandboxes must fit on the cores: oversubscription lets a
# CPU-bound candidate hit the 15s wall timeout waiting for a core, scoring
# exec_timeout/FAIL and making the histogram incomparable to the calibrated,
# un-contended exec-pass 0.45 / keep_l3 0.272 (3b #238).
CORES=$(nproc)
[ "$EW" -eq 0 ] && EW=$((CORES / NG))
if [ $((NG * EW)) -gt "$CORES" ]; then
  echo "REFUSE: NG*EW=$((NG * EW)) > nproc=$CORES (oversubscription risks wall-timeout false FAILs)"
  exit 2
fi
echo "cores=$CORES groups=$NG exec-workers/group=$EW sandboxes=$((NG * EW))"

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
  local SIZE=$(( (NSH + NG - 1) / NG ))
  local first=$((g * SIZE + 1)); local last=$((g * SIZE + SIZE))
  [ $last -gt $NSH ] && last=$NSH
  [ $first -gt $NSH ] && return 0
  local tag; tag=$(printf "g%02d" "$g")
  local paths=() p
  for i in $(seq $first $last); do
    p="$RAW/UltraData-Code-L3-py-part-$(printf '%05d' $i)-of-00147.parquet"
    until footer_ok "$p"; do sleep 60; done
    paths+=("$p")
  done
  env PYTHONPATH=/work/aupai python3 datagen/ultradata_shards.py \
    --level L3 --first "$first" --last "$last" --exec-workers "$EW" \
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
    p = (f"/data00/aupai_raw/ultradata/UltraData-Code-L3-py-part-{i:05d}"
         "-of-00147.parquet")
    expected += pq.ParquetFile(p).metadata.num_rows
assert s["total_rows"] == expected, (
    f"total_rows {s['total_rows']} != parquet rows {expected} -- some input not traversed")
assert s["n_shards"] > 0
print("MANIFEST_OK", tag, "rows", expected, "kept", s["kept"], "shards", s["n_shards"])
PY
  then echo "VERIFY_FAILED $tag -- raw RETAINED"; return 1; fi

  local man="$MLOG/deleted_l3_$tag.manifest"
  { echo "# deleted after verified conversion of group $tag (shards $first-$last)"
    for p in "${paths[@]}"; do stat -c '%s %n' "$p"; done; } > "$man"
  for p in "${paths[@]}"; do rm -f "$p"; done
  echo "GROUP_DELETED $tag $((${#paths[@]})) paths"
}
export -f run_group footer_ok
export RAW OUT MLOG EW NSH NG

pids=""
for g in $(seq 0 $((NG - 1))); do
  ( run_group "$g" ) > "runs/ultra_groups/l3_g$(printf '%02d' "$g").log" 2>&1 &
  pids="$pids $!"
done
rc=0
for p in $pids; do wait "$p" || rc=1; done
[ $rc -ne 0 ] && { echo "GROUP_FAILURE rc=$rc -- no aggregate"; exit 1; }

env PYTHONPATH=/work/aupai python3 datagen/ultradata_shards.py \
  --level L3 --aggregate "stats_g*.json" --out "$OUT" \
  2>&1 | tee runs/ultra_groups/l3_aggregate.log
echo L3_PIPELINE_DONE
