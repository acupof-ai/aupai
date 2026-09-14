#!/bin/bash
# L3 30-parent restart (fb ruling 2026-09-11): the per-parent executor is
# GIL/Popen-serialized (18 threads == 6 threads per parent), so 30 parents x6
# sandboxes (~2-3x the 10-parent topology) finish the UNFINISHED shards.
#
# Completed watermark from the 10-parent run: each old group finished exactly
# its first shard = 1,16,31,46,61,76,91,106,121,136. Those 10 were row-verified
# in runs/ultra_groups/done_shards.txt (rows in == kept+rejects) and are NOT
# reprocessed here. The old tagged jsonls are moved aside; their kept rows for
# the finished shards are re-derived by the shard that re-covers them is NOT
# done -- instead the finished shards are converted by dedicated single-shard
# parents so final coverage is every shard exactly once.
# Restart ALL 147 shards under 30 parents x6 with PER-SHARD verification.
# The 10-parent run's tagged jsonls merged each group's finished shard with its
# in-flight next shard, so per-shard conservation (fb step 2) could not be
# produced without re-running; redoing 10 shards at the 3x topology costs ~15m,
# far cheaper and safer than extracting mixed outputs. Old gNN outputs are
# moved aside; per-shard sNNN tags are the canonical redo artifacts.
set -u
cd /work/aupai
RAW=/data00/aupai_raw/ultradata
OUT=data/corpus/code_ultra_l3
EW=${EW:-6}
NG=${NG:-30}
mkdir -p runs/ultra_groups/redo
mkdir -p "$OUT/_abort_g_groups"
for f in "$OUT"/code_ultra_l3_g??_*.jsonl "$OUT"/stats_g??.jsonl; do
  [ -e "$f" ] && mv "$f" "$OUT/_abort_g_groups/"
done

footer_ok() { python3 - "$1" <<'PY'
import os,sys
p=sys.argv[1]
sys.exit(0 if os.path.exists(p) and os.path.getsize(p)>=8 and open(p,"rb").read()[-4:]==b"PAR1" else 1)
PY
}

conv() {
  shard=$1
  p="$RAW/UltraData-Code-L3-py-part-$(printf '%05d' "$shard")-of-00147.parquet"
  until footer_ok "$p"; do sleep 30; done
  tag="s$(printf '%03d' "$shard")"
  env PYTHONPATH=/work/aupai python3 datagen/ultradata_shards.py \
    --level L3 --first "$shard" --last "$shard" --exec-workers "$EW" \
    --raw "$RAW" --tag "$tag" --stats-name "stats_$tag.json" --out "$OUT" \
    || { echo "CONVERT_FAILED shard=$shard -- raw RETAINED"; return 1; }
  env PYTHONPATH=/work/aupai python3 - "$shard" "$OUT" "$tag" <<'PY'
import json,sys
import pyarrow.parquet as pq
shard,out,tag=int(sys.argv[1]),sys.argv[2],sys.argv[3]
s=json.load(open(f"{out}/stats_{tag}.json"))
p=f"/data00/aupai_raw/ultradata/UltraData-Code-L3-py-part-{shard:05d}-of-00147.parquet"
exp=pq.ParquetFile(p).metadata.num_rows
assert sum(s["reasons"].values())==s["total_rows"]==exp, s["reasons"]
assert s["n_shards"]>0
print("SHARD_OK",shard,"in",exp,"kept",s["kept"])
PY
}
export -f conv footer_ok
export RAW OUT EW

# All 147 shards, round-robin over NG parents for even progress.
mapfile -t todo < <(seq 1 147)
pids=""
for ((k=0; k<NG; k++)); do
  ( for ((idx=k; idx<${#todo[@]}; idx+=NG)); do conv "${todo[$idx]}" || exit 1; done;
    echo "PARENT_DONE $k" ) > "runs/ultra_groups/redo/p$k.log" 2>&1 &
  pids="$pids $!"
done
rc=0; for p in $pids; do wait "$p" || rc=1; done
[ $rc -ne 0 ] && { echo "REDO_FAILURE -- no aggregate"; exit 1; }
echo L3_REDO_DONE
