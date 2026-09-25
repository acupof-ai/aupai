#!/bin/bash
# HumanEval FULL/164 for one SFT-A epoch checkpoint: greedy pass@1 plus the n=8 T=0.8 hit
# rate and their gap (the RL-readiness gate: hitrate@8 - pass@1 >= 15pt). Continuation
# format (the SFT-A pack is raw continuation, not ChatML), so the in-distribution prompt is
# --rstrip_nl. One shard per card, merged by eval/e0_merge_score.py (exact-cover + per-task n).
#
#   CUDA_VISIBLE_DEVICES=<0..N-1> eval/sft_a_humaneval.sh <ckpt> [run_tag]
# Cards come ONLY from the caller's CUDA_VISIBLE_DEVICES grant through eval/_devs.sh: a shard
# never writes a physical index and so cannot escape a lane. Temp fixed 0.8 to match the
# project's pass@k gate (eval/math_hard.py --k 8 --temperature 0.8).
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
CKPT=${1:?usage: expose <cards> in CUDA_VISIBLE_DEVICES, then $0 <ckpt> [run_tag]}
TAG=${2:-$(basename "$CKPT")}
TEMP=${TEMP:-0.8}
NSAMP=${NSAMP:-8}
NGPU=$(printf '%s' "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)

# Builds the shard->device map from the caller's grant; refuses if NGPU shards do not fit.
source eval/_devs.sh "$NGPU"
BASE=$(basename "$CKPT")
mkdir -p runs/sfta_eval
echo "=== $BASE: $NGPU GPU shards, greedy + n=$NSAMP T=$TEMP, FULL/164, continuation rstrip"

pids=()
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=${_DEVS[$i]} python3 eval/humaneval_gen.py --ckpt "$CKPT" \
    --rstrip_nl --shard_i "$i" --shard_n "$NGPU" \
    --run "${TAG}_g_sh$i" --force \
    > "runs/sfta_eval/${TAG}_greedy_sh$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

pids=()
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=${_DEVS[$i]} python3 eval/humaneval_gen.py --ckpt "$CKPT" \
    --rstrip_nl --n "$NSAMP" --temperature "$TEMP" \
    --shard_i "$i" --shard_n "$NGPU" --run "${TAG}_n${NSAMP}_sh$i" --force \
    > "runs/sfta_eval/${TAG}_n${NSAMP}_sh$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

GLOB_G="data/eval/preds_humaneval_${BASE}.rstripnl.shard*of${NGPU}.${TAG}_g_sh*.jsonl"
GLOB_S="data/eval/preds_humaneval_${BASE}.rstripnl.n${NSAMP}temp${TEMP}.shard*of${NGPU}.${TAG}_n${NSAMP}_sh*.jsonl"

python3 eval/e0_merge_score.py --bench humaneval --n 1 \
  --glob "$GLOB_G" --out "runs/sfta_eval/${TAG}_greedy_merged.jsonl" \
  --result "runs/sfta_eval/${TAG}_result_greedy.json"
python3 eval/e0_merge_score.py --bench humaneval --n "$NSAMP" \
  --glob "$GLOB_S" --out "runs/sfta_eval/${TAG}_n${NSAMP}_merged.jsonl" \
  --result "runs/sfta_eval/${TAG}_result_n${NSAMP}.json"

echo "=== $BASE greedy FULL/164:"
python3 -c "import json; r=json.load(open('runs/sfta_eval/${TAG}_result_greedy.json'))['humaneval']; print(f\"pass@1 {r['full_pass']}/{r['full_denom']} = {r['full_rate']*100:.2f}%\")"
echo "=== $BASE hit rate (>=1 pass / problem):"
python3 eval/humaneval_hitrate.py "runs/sfta_eval/${TAG}_n${NSAMP}_merged.jsonl"
echo "(RL gate = hitrate@${NSAMP} - greedy pass@1 >= 15pt)"
