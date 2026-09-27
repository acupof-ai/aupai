#!/bin/bash
# Extra per-checkpoint evals for a CPU-scored run, run after heval_auto_loop has scored a step
# (its DONE file is the trigger, so the two never load the same checkpoint at once):
#   every scored step:      HumanEval gold BPB (teacher-forced, smooth) + eval/probe_cases.py
#   every MBPP_GRID steps:  MBPP sanitized 427, sharded like HumanEval, merged by e0_merge_score
# MBPP is on a coarser grid because 427 tasks on 11 CPU workers take longer than one save interval.
# restartable: a step counts as done only when every artifact it owes exists; a rerun redoes the rest.
# Single instance via flock on fd 9. Iteration cap: MAX_ITER polls.
export CUDA_VISIBLE_DEVICES=
NAME=${NAME:-v41_ced_0926}
CK=${CK:-/work/aupai/ckpt_${NAME}.pt}
HDONE=/work/aupai/runs/heval_auto_${NAME}.done
DONE=/work/aupai/runs/stage_extra_${NAME}.done
LOCK=/work/aupai/runs/stage_extra_${NAME}.lock
MBPP_GRID=${MBPP_GRID:-4000}
MAX_ITER=${MAX_ITER:-2000}
POLL=${POLL:-120}
SHARDS=11
G0="2,3,4,5,8,9,10,11";       G1="14,15,16,17,18,19,20,21"
G2="22,23,24,25,26,27,28,29"; G3="30,31,32,33,34,35,36,37"
G4="38,39,40,41,42,43,44,45"; G5="46,47,48,49,50,51,52,53"
G6="54,55,56,57,58,59,60,61"; G7="64,65,68,69,70,71,72,73"
G8="90,91,92,93,94,95,96,97"; G9="98,99,100,101,102,103,104,105"
G10="106,107,108,109,110,111,112,113"
G_NODE="0 0 0 0 0 0 0 0 1 1 1"

owes() {  # owes <step> -> prints each missing artifact path
  local s=$1
  [ -f "runs/hebpb_${NAME}_step${s}.json" ] || echo "runs/hebpb_${NAME}_step${s}.json"
  [ -f "runs/probe_${NAME}_step${s}.jsonl" ] || echo "runs/probe_${NAME}_step${s}.jsonl"
  if [ $((s % MBPP_GRID)) -eq 0 ]; then
    [ -f "runs/mbpp_merged_${NAME}_step${s}_result.json" ] || echo "runs/mbpp_merged_${NAME}_step${s}_result.json"
  fi
}

exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "another instance holds $LOCK"; exit 0; }
cd /work/aupai || exit 1
touch "$DONE"
iter=0
while [ "$iter" -lt "$MAX_ITER" ]; do
  iter=$((iter + 1))
  step=""
  for s in $(sort -n "$HDONE" 2>/dev/null | grep -E '^[0-9]+$'); do
    grep -qx "$s" "$DONE" && continue
    [ -f "$CK.step$s" ] || continue
    step=$s; break
  done
  if [ -z "$step" ]; then sleep "$POLL"; continue; fi
  ckf="$CK.step$step"
  echo "=== $(date -u +%H:%M:%SZ) extras for step $step"
  if [ ! -f "runs/hebpb_${NAME}_step${step}.json" ]; then
    OMP_NUM_THREADS=16 nice -n 5 python3 eval/humaneval_bpb.py --ckpt "$ckf" --device cpu \
      --preds "runs/hebpb_${NAME}_step${step}.preds.jsonl" --out "runs/hebpb_${NAME}_step${step}.json" \
      > "runs/hebpb_${NAME}_step${step}.log" 2>&1
  fi
  if [ ! -f "runs/probe_${NAME}_step${step}.jsonl" ]; then
    nice -n 5 python3 eval/probe_cases.py --ckpt "$ckf" --threads 16 \
      --out "runs/probe_${NAME}_step${step}.jsonl" > "runs/probe_${NAME}_step${step}.log" 2>&1
  fi
  if [ $((step % MBPP_GRID)) -eq 0 ] && [ ! -f "runs/mbpp_merged_${NAME}_step${step}_result.json" ]; then
    run="mbpp_s${step}"
    i=0
    while [ "$i" -lt "$SHARDS" ]; do
      eval "cores=\$G$i"; node=$(echo $G_NODE | cut -d" " -f$((i + 1)))
      setsid nohup numactl --physcpubind="$cores" --membind="$node" \
        nice -n 5 python3 eval/mbpp_gen.py --ckpt "$ckf" --device cpu --threads 8 --run "$run" \
        --shard_i "$i" --shard_n "$SHARDS" --no_clean --force > "runs/mbpp_${NAME}_${step}_sh$i.log" 2>&1 < /dev/null &
      i=$((i + 1))
    done
    wait
    python3 eval/e0_merge_score.py --bench mbpp --n 1 \
      --glob "data/eval/preds_mbpp_$(basename "$ckf").${run}.shard*of${SHARDS}.jsonl" \
      --out "runs/mbpp_merged_${NAME}_step${step}.jsonl" \
      --result "runs/mbpp_merged_${NAME}_step${step}_result.json" > "runs/mbpp_merge_${NAME}_${step}.log" 2>&1
  fi
  missing=$(owes "$step")
  if [ -z "$missing" ]; then
    echo "$step" >> "$DONE"; echo "=== $(date -u +%H:%M:%SZ) step $step extras DONE"
  else
    echo "=== step $step still owes: $missing (will retry)"; sleep "$POLL"
  fi
done
echo "=== STOPPED: MAX_ITER=$MAX_ITER polls"
