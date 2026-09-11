#!/bin/bash
# Gate the L3 noexec aggregate behind a fully VERIFIED reconvert, then choose
# the aggregate driver by the byte-identity proof (fb ruling 2026-09-11).
#
# Loop fix (defect that exited 09:06Z): gate on the live SHARD_OK count read
# from the per-shard verify logs, NOT on stats files -- stale stats from an
# earlier run can reach 147 before any shard is re-verified. Both counts must
# read 147. Intermediates are retained (no --release-intermediates) until the
# stamped stats are verified.
# OLD driver is the pre-sidecar aggregate (serial json/hash phase 2), the
# reference for the proof. Materialize it one package-deep from the commit just
# before the sidecar landed, then point AUPAI_OLD_DRIVER at it:
#   mkdir -p /tmp/oldref/datagen && git show 9aa6e4c2:datagen/ultradata_shards.py \
#     > /tmp/oldref/datagen/ultradata_shards_pre_sidecar.py
#   AUPAI_OLD_DRIVER=/tmp/oldref/datagen/ultradata_shards_pre_sidecar.py ...
# (one package-deep so its root = dirname(dirname(__file__)) resolves data/eval)
set -u
ROOT=${AUPAI_ROOT:-/work/aupai}
STAGE=${AUPAI_STAGE:-$ROOT}
N=${AUPAI_NSHARDS:-147}
cd "$ROOT"
OUT=data/corpus/code_ultra_l3_noexec
FINAL=data/corpus/code_ultra_l3_noexec_dc
LOG=runs/ultra_groups/l3_noexec_aggregate.log
SENT=runs/ultra_groups/l3_noexec_done.sentinel
PROOF=${AUPAI_PROOF:-$ROOT/scripts/proof_aggregate_identity.py}
DRIVER_NEW=${AUPAI_NEW_DRIVER:-$STAGE/datagen/ultradata_shards.py}
DRIVER_OLD=${AUPAI_OLD_DRIVER:-$STAGE/datagen/ultradata_shards_pre_sidecar.py}
rm -f "$SENT"

shard_ok() { grep -h -o '^SHARD_OK [0-9]*' runs/ultra_groups/noexec/p*.log 2>/dev/null \
  | sort -u | wc -l; }

while true; do
  g=$(shard_ok)
  n=$(ls "$OUT"/stats_s*.json 2>/dev/null | wc -l)
  if [ "$g" = "$N" ] && [ "$n" = "$N" ]; then break; fi
  ps -eo comm,args | awk -v n="$N" '$1=="python3" && /--no-exec/ {f=1} END{exit f?0:1}' || {
    echo "RECONVERT_DIED shard_ok=$g stats=$n/$N -- NO aggregate" | tee "$SENT"; exit 1; }
  sleep 30
done
echo "reconvert $N/$N verified $(date -u +%FT%TZ), running byte-identity proof" | tee -a "$SENT"

DRIVER="$DRIVER_NEW"
if timeout 900 env AUPAI_ROOT="$ROOT" AUPAI_NEW="$DRIVER_NEW" AUPAI_OLD="$DRIVER_OLD" \
    python3 "$PROOF" > /tmp/proof_aggregate_identity.out 2>&1; then
  echo "PROOF PASS -> sidecar driver" | tee -a "$SENT"
else
  DRIVER="$DRIVER_OLD"
  echo "PROOF FAIL/timeout -> old serial driver" | tee -a "$SENT"
  grep -v SyntaxWarning /tmp/proof_aggregate_identity.out | tail -15 | tee -a "$SENT"
fi

rm -rf "$FINAL"; mkdir -p "$FINAL"
env PYTHONPATH="$STAGE" python3 -u "$DRIVER" \
  --level L3 --aggregate 'stats_s*.json' \
  --out "$OUT" --final-out "$FINAL" \
  --tokenizer data/tokenizer.json --agg-workers 20 \
  > "$LOG" 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -f "$FINAL/build_corpus_stats.json" ]; then
  python3 "$ROOT/scripts/read_aggregate_stats.py" "$FINAL" > "$SENT.tmp" 2>&1
  cat "$SENT.tmp" >> "$SENT"; rm -f "$SENT.tmp"
  echo "DRIVER $DRIVER" >> "$SENT"
  echo "L3_NOEXEC_AGG_DONE $(date -u +%FT%TZ)" >> "$SENT"
else
  echo "AGGREGATE_FAILED rc=$rc" | tee -a "$SENT"
  grep -v SyntaxWarning "$LOG" | tail -20 >> "$SENT"
  exit 1
fi
