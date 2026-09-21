#!/bin/bash
# Gate the L3 noexec aggregate behind a fully VERIFIED reconvert, then choose
# the aggregate driver by the byte-identity proof (fb ruling 2026-09-11).
#
# COMPLETION IS DERIVED FROM THE PRODUCT, NOT FROM A LAYOUT (de, 2026-09-22).
# This gate used to wait for `shard_ok == N` and `n == N` with N=147, where
# `shard_ok` grepped `runs/ultra_groups/noexec/p*.log` for `SHARD_OK` and `n`
# counted `stats_s*.json`. Both readings belong to ONE builder -- the committed
# per-SHARD path (scripts/convert_l3_noexec.sh, tag="s$(printf '%03d' $shard)",
# the only emitter of SHARD_OK). The build that actually ran on the pod used a
# per-GROUP recipe (37 groups over 147 shards, tag="s$(printf '%02d' $g)", logs
# at runs/ultra_groups/l3_<tag>.log, no SHARD_OK at all), so `shard_ok` read a
# directory that does not exist (0 forever) and `n` read 37 against a target of
# 147. Neither is a corruption: the gate was addressed to a different builder.
#
# The fix is to ask the OUT directory what it holds, so the check holds for any
# recipe. What is asserted is the property this gate actually wants -- "the
# units the aggregate will consume are present, stamped, and still growing" --
# rather than "someone's file naming reached a number I chose".
#
# Still retained from the original: a stale stats file from an earlier run must
# not satisfy the gate. That is why the count is paired with a LIVENESS check on
# the produced set: units appearing while a converter is alive, and a hard stop
# when the set stops changing and no converter is left (the old loop slept every
# 30s forever in that state).
#
# The liveness predicate `ps -eo comm,args | awk '$1=="python3" && /--no-exec/'`
# is CORRECT and is NOT modified: a bare /re/ in awk matches $0 (the whole line,
# args column included), so it matches a live converter. Two independent reviews
# misread it as broken; a predicate that never matched would exit 1 on the first
# pass, not hang. The hang came from a stale count plus a predicate that was
# correctly reporting "some converter is alive". See M2 below.
#
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
# N is no longer a target count. It is kept only as the STALL BUDGET: how many
# consecutive no-progress rounds are tolerated before the gate declares the
# build dead and exits instead of sleeping forever. Default 40 rounds x 30s =
# 20 min of a set that neither grows nor has a live converter.
N=${AUPAI_STALL_ROUNDS:-40}
cd "$ROOT"
OUT=data/corpus/code_ultra_l3_noexec
FINAL=data/corpus/code_ultra_l3_noexec_dc
LOG=runs/ultra_groups/l3_noexec_aggregate.log
SENT=runs/ultra_groups/l3_noexec_done.sentinel
PROOF=${AUPAI_PROOF:-$ROOT/scripts/proof_aggregate_identity.py}
DRIVER_NEW=${AUPAI_NEW_DRIVER:-$STAGE/datagen/ultradata_shards.py}
DRIVER_OLD=${AUPAI_OLD_DRIVER:-$STAGE/datagen/ultradata_shards_pre_sidecar.py}
rm -f "$SENT"

# The units the aggregate will consume, derived from the product and using the
# SAME glob the driver does (--aggregate 'stats_s*.json', below), so gate and
# driver can never disagree about what is being counted. A unit counts when its
# stats file exists AND is non-empty: an empty file is a unit being written, not
# a finished one -- the same "0 bytes is not a finished name" rule this tree
# applies to caches and claims.
units_ready() {
  local f n=0
  for f in "$OUT"/stats_s*.json; do
    [ -s "$f" ] && n=$((n + 1))
  done
  echo "$n"
}

# Is a converter still running? UNCHANGED from the original and CORRECT: awk's
# bare /re/ matches $0, the whole line including the args column, so this
# matches a live converter. Two independent reviewers read it as "never matches"
# and both retracted; a predicate that never matched would have exited rc=1 on
# the first pass, not hung. This answers "is the build still running", never
# "is it complete".
converters_alive() {
  ps -eo comm,args | awk '$1=="python3" && /--no-exec/ {f=1} END{exit f?0:1}'
}

prev=-1
stall=0
while true; do
  u=$(units_ready)
  if [ "$u" = "$prev" ]; then
    stall=$((stall + 1))
  else
    stall=0
    prev=$u
  fi
  if converters_alive; then
    # Still building. A long run with no new unit is the one state worth
    # refusing: the old loop slept here forever and emitted nothing.
    if [ "$stall" -ge "$N" ]; then
      echo "RECONVERT_STALLED units=$u unchanged for $N round(s) with a converter alive" |
        tee "$SENT"
      exit 1
    fi
  elif [ "$u" -gt 0 ]; then
    # No converter left and the set is non-empty and stable: the build ended.
    break
  fi
  # u == 0 with no converter: nothing has started yet. Wait, as the original
  # did -- the build may be launched after this gate.
  sleep 30
done
echo "reconvert $u unit(s), no converter running $(date -u +%FT%TZ), running byte-identity proof" |
  tee -a "$SENT"

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
