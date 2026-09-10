#!/bin/bash
# Option-2 L3 pipeline (fb 2026-09-11): one shard group at a time.
#   wait-for-shards (valid parquet footer) -> convert group -> verify stats
#   -> write deleted_<tag>.manifest naming every raw path -> rm those paths.
# The manifest is the deletion record fb requires before any raw shard is
# removed. The L3 keep predicates live in datagen/ultradata_shards.py
# (ud_solution_exec.execute AND 3b's non-triviality floor once merged).
set -u
cd /work/aupai
RAW=/data00/aupai_raw/ultradata
OUT=data/corpus/code_ultra_l3
MLOG=runs/ultra_groups/manifests
EW=${EW:-24}
mkdir -p runs/ultra_groups "$MLOG"

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

NG=10; NSH=147
SIZE=$(( (NSH + NG - 1) / NG ))
for g in $(seq 0 $((NG - 1))); do
  first=$((g * SIZE + 1)); last=$((g * SIZE + SIZE)); [ $last -gt $NSH ] && last=$NSH
  [ $first -gt $NSH ] && continue
  tag=$(printf "g%02d" "$g")
  paths=()
  for i in $(seq $first $last); do
    p="$RAW/UltraData-Code-L3-py-part-$(printf '%05d' $i)-of-00147.parquet"
    until footer_ok "$p"; do
      kb=$(df --output=avail / | tail -1 | tr -d ' ')
      [ "$kb" -lt 157286400 ] && { echo "DISK_WAIT $kb KB before shard $i"; sleep 120; } || sleep 30
    done
    paths+=("$p")
  done
  env PYTHONPATH=/work/aupai python3 datagen/ultradata_shards.py \
    --level L3 --first "$first" --last "$last" --exec-workers "$EW" \
    --tag "$tag" --stats-name "stats_$tag.json" --out "$OUT" \
    2>&1 | tee "runs/ultra_groups/l3_$tag.log"

  man="$MLOG/deleted_l3_$tag.manifest"
  { echo "# deleted after verified conversion of group $tag (shards $first-$last)"
    for p in "${paths[@]}"; do stat -c '%s %n' "$p"; done; } > "$man"
  env PYTHONPATH=/work/aupai python3 - "$first" "$last" "$OUT" "$tag" <<'PY'
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
  for p in "${paths[@]}"; do rm -f "$p"; done
  echo "GROUP_DELETED $tag $(wc -l < "$man") paths"
done
env PYTHONPATH=/work/aupai python3 datagen/ultradata_shards.py \
  --level L3 --aggregate "stats_g*.json" --out "$OUT" \
  2>&1 | tee runs/ultra_groups/l3_aggregate.log
echo L3_PIPELINE_DONE
