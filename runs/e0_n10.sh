#!/bin/bash
# E0/ET/EC stage-2 n=10 paired eval, 8 shards. Runs BOTH benchmarks on the FULL
# datasets (HE 164, MBPP sanitized 427); the merger recomputes CLEAN 156/338 from
# the r3 contamination unions, so no clean-only data files are needed.
#
# Reused three times (E0 = r3 final, ET = textbook arm, EC = control): pass the
# checkpoint and a tag; preds/results carry the tag so the three runs never collide.
#
#   bash runs/e0_n10.sh <ckpt> <tag>
#   e.g. bash runs/e0_n10.sh ckpt_v41_r3_0914.pt e0
#
# Layout: one process per card, HE then MBPP SERIALISED on that card (one model in
# GPU memory at a time). 8 cards x 2 benchmarks = 16 processes, paired by card.
# Sharding is fixed-position (idx % 8): every task lands on exactly one card.
#
# Parameters pinned in runs/prereg.jsonl#textbook_continuation_ab_0914 (fb 2026-09-15):
#   n=10, temp 0.2, max_new 280 (both generators' default; timing calib used 280);
#   HE --rstrip_nl (FULL/164 + CLEAN/156); MBPP sig-docstring-rstrip native
#   (FULL/427 + CLEAN/338, merger excludes the union because a shard passes --no_clean).
#
# Detached launch (AGENTS.md): write this as a pod launch file, then
#   setsid nohup bash -c 'cd /work/aupai && bash runs/e0_n10.sh CK TAG > runs/e0_TAG.log 2>&1' \
#       </dev/null >/dev/null 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:?"usage: e0_n10.sh <ckpt> <tag>"}
TAG=${2:?"usage: e0_n10.sh <ckpt> <tag>  (e0 | et | ec)"}
N=10
TEMP=0.2
SHARDS=${SHARDS:-8}
# Keep the ".pt": humaneval_gen/mbpp_gen name files with os.path.basename(ckpt)
# un-stripped, so stripping here made every merge glob miss (pod-measured 2026-09-15).
CK_BASE=$(basename "$CKPT")

[ -f "$CKPT" ] || { echo "REFUSING: ckpt $CKPT not present"; exit 2; }
# Resolve granted devices through _devs.sh: never write a physical index. Caller
# exports CUDA_VISIBLE_DEVICES (the 8-card block); _DEVS[i] maps each shard onto it.
source eval/_devs.sh "$SHARDS"

run_shard() {
  local i=$1
  export CUDA_VISIBLE_DEVICES=${_DEVS[$i]}
  local dev="cuda:0"
  # HE rstrip FULL 164, this shard's fixed-position tasks.
  python3 eval/humaneval_gen.py --ckpt "$CKPT" --device "$dev" --rstrip_nl \
      --n "$N" --temperature "$TEMP" --force \
      --shard_i "$i" --shard_n "$SHARDS" \
      --run "${TAG}_he_n10_s${i}"
  # MBPP sig-rstrip; --no_clean because a shard cannot satisfy the full-427 clean
  # invariant; the merger recomputes CLEAN/338 from runs/contam_r3_mbpp_union.json.
  python3 eval/mbpp_gen.py --ckpt "$CKPT" --device "$dev" \
      --data data/eval/sanitized-mbpp.json \
      --n "$N" --temperature "$TEMP" --force --no_clean \
      --shard_i "$i" --shard_n "$SHARDS" \
      --run "${TAG}_mbpp_n10_s${i}"
}

pids=()
for ((i=0; i<SHARDS; i++)); do
  run_shard "$i" > "runs/${TAG}_shard${i}.log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=1; fi
done
[ "$fail" -eq 0 ] || { echo "REFUSING to merge: one or more shards failed; see runs/${TAG}_shard*.log"; exit 1; }

# open_artifact(run=) inserts the run tag as ".<run>" before .jsonl, so the shard
# files are ...shard{i}ofN.<TAG>_..._s{i}.jsonl. The glob must include that segment
# (pinned to TAG so it stays arm-isolated across e0/et/ec); ending shard*ofN.jsonl
# matched zero (pod-measured 2026-09-15, E0 merge).
HE_GLOB="data/eval/preds_humaneval_${CK_BASE}.rstripnl.n${N}temp${TEMP}.shard*of${SHARDS}.${TAG}_he_n10_s*.jsonl"
MB_GLOB="data/eval/preds_mbpp_${CK_BASE}.${TAG}_mbpp_n10_s*.n${N}temp${TEMP}.shard*of${SHARDS}.${TAG}_mbpp_n10_s*.jsonl"
RESULT="runs/e0_${TAG}_result.json"

python3 eval/e0_merge_score.py --bench humaneval --glob "$HE_GLOB" --n "$N" \
    --out "data/eval/${TAG}_he_merged.n${N}temp${TEMP}.jsonl" --result "$RESULT"
python3 eval/e0_merge_score.py --bench mbpp --glob "$MB_GLOB" --n "$N" \
    --out "data/eval/${TAG}_mbpp_merged.n${N}temp${TEMP}.jsonl" --result "$RESULT"

echo "E0-style eval for tag=$TAG complete: $RESULT"
cat "$RESULT"
