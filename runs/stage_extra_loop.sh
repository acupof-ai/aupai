#!/bin/bash
# Extra per-checkpoint evals for a CPU-scored run, run after heval_auto_loop has scored a step
# (its DONE file is the trigger, so the two never load the same checkpoint at once):
#   every scored step:      HumanEval gold BPB (teacher-forced, smooth) + eval/probe_cases.py
#   every MBPP_GRID steps:  MBPP sanitized 427, sharded like HumanEval, merged by e0_merge_score
# MBPP is on a coarser grid because 427 tasks on 11 CPU workers take longer than one save interval.
# restartable: a step counts as done only when every artifact it owes exists; a rerun redoes the rest.
# Single instance via flock on fd 9; the MBPP workers close fd 9 so a killed loop frees the lock.
# mbpp_gen versions its output by --run, so the shard files end in .${run}.jsonl. Iteration cap: MAX_ITER polls.
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
# MBPP workers take node1 cores 114-179, six each: heval_auto_loop pins its 11 workers to 2-113,
# and sharing those cores halved both runs (measured 2026-09-27).
MB_BASE=114; MB_W=6; MB_NODE=1

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
      lo=$((MB_BASE + i * MB_W))
      setsid nohup numactl --physcpubind="$lo-$((lo + MB_W - 1))" --membind="$MB_NODE" \
        nice -n 5 python3 eval/mbpp_gen.py --ckpt "$ckf" --device cpu --threads "$MB_W" --run "$run" \
        --shard_i "$i" --shard_n "$SHARDS" --no_clean --force > "runs/mbpp_${NAME}_${step}_sh$i.log" 2>&1 < /dev/null 9>&- &
      i=$((i + 1))
    done
    wait
    python3 eval/e0_merge_score.py --bench mbpp --n 1 \
      --glob "data/eval/preds_mbpp_$(basename "$ckf").${run}.shard*of${SHARDS}.${run}.jsonl" \
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
