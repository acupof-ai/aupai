#!/bin/bash
# Static-only L3 conversion (USER ORDER 2026-09-11): no execute() sandbox.
# Keep = substring decontam + exact dedup + nontrivial(solution) AST floor.
# Same per-shard s### tags and row-conservation gate as the exec redo; outputs
# to code_ultra_l3_noexec (aggregate emits code_ultra_l3_noexec_dc).
set -u
ROOT=${AUPAI_ROOT:-/work/aupai}
STAGE=${AUPAI_STAGE:-}   # optional staging dir holding an uncommitted converter
CONV_PY=${CONV_PY:-${STAGE:+$STAGE/datagen/ultradata_shards.py}}
CONV_PY=${CONV_PY:-$ROOT/datagen/ultradata_shards.py}
export PYTHONPATH=${STAGE:-$ROOT}
cd "$ROOT"
RAW=/data00/aupai_raw/ultradata
OUT=data/corpus/code_ultra_l3_noexec
NG=${NG:-30}
mkdir -p runs/ultra_groups/noexec "$OUT"

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
  env PYTHONPATH="${STAGE:-$ROOT}" python3 "$CONV_PY" \
    --level L3 --first "$shard" --last "$shard" --no-exec \
    --raw "$RAW" --tag "$tag" --stats-name "stats_$tag.json" --out "$OUT" \
    || { echo "CONVERT_FAILED shard=$shard -- raw RETAINED"; return 1; }
  env PYTHONPATH="$ROOT" python3 - "$shard" "$OUT" "$tag" <<'PY'
import json,sys
import pyarrow.parquet as pq
shard,out,tag=int(sys.argv[1]),sys.argv[2],sys.argv[3]
s=json.load(open(f'{out}/stats_{tag}.json'))
p=f'/data00/aupai_raw/ultradata/UltraData-Code-L3-py-part-{shard:05d}-of-00147.parquet'
exp=pq.ParquetFile(p).metadata.num_rows
assert sum(s['reasons'].values())==s['total_rows']==exp, s['reasons']
assert s['n_shards']>0
print('SHARD_OK',shard,'in',exp,'kept',s['kept'])
PY
}
export -f conv footer_ok
export RAW OUT

mapfile -t todo < <(seq 1 147)
pids=""
for ((k=0; k<NG; k++)); do
  ( for ((idx=k; idx<${#todo[@]}; idx+=NG)); do conv "${todo[$idx]}" || exit 1; done
    echo "PARENT_DONE $k" ) > "runs/ultra_groups/noexec/p$k.log" 2>&1 &
  pids="$pids $!"
done
rc=0; for p in $pids; do wait "$p" || rc=1; done
[ $rc -ne 0 ] && { echo "NOEXEC_FAILURE -- no aggregate"; exit 1; }
echo L3_NOEXEC_DONE
