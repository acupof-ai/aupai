#!/bin/bash
# Where M1's missing ~15 throughput points are: forward / backward / opt_step, five configs.
#
# The arm ran 64K tok/s/gpu at step 30 against the control's 82K -- 21% -- while the isolated
# lookup accounts for 6.5% (scripts/memory_lookup_bench.py, 51.6 ms of a 799 ms step). This
# runs profile_step_cost.py at the ARM'S EXACT SHAPE and reads the three regions inside the
# step, which nothing before it could.
#
# NOT MEASURED HERE, because it is already answered: the DDP reduction. b0's world-1 pair
# measured 30.6/38.1 = 0.80 and world 2 measured 64/82 = 0.78. The cost is the same size at
# world 1, where there is no gradient exchange at all, so a 2 GiB dense all-reduce is not
# what these points are. nccl_floor is skipped for that reason and not for cost.
#
# The five configs isolate one term each against the control:
#   off      mem_values 0                     the control's step, same launch line
#   m1       1048576, top_k 32, layers 3,6,9  the arm as it ran
#   k16      1048576, top_k 16, layers 3,6,9  halves the gather and the cross-product grid
#   l1       1048576, top_k 32, layer 6       one pooled layer instead of three
#   m2        262144, top_k 32, layers 3,6,9  quarter table, everything else M1's
#
# l1 is the discriminator that matters, because ONE pool is shared by the three layers
# (model.py:451). Per-layer terms -- the lookup, the cross-product, the touched/key_hits
# bookkeeping -- divide by three in l1. The table's own terms do not: the dense gradient,
# the Adagrad state and the optimizer read the same 1,048,576 rows whether one layer or
# three wrote them. So l1 near off means the cost is per-layer, l1 still slow means it is
# the table, and m2 separates those again by shrinking the table while keeping the layers.
# k16 halves top_k, which moves the gather and the k x k grid and nothing else.
set -euo pipefail

# The cards come from the CALLER, never written here. An assignment inside a script REPLACES
# the parent's restriction rather than indexing into it, so a literal `CUDA_VISIBLE_DEVICES=1,2`
# lands on physical 1 and 2 whatever lane the caller confined this to -- the shape that put a
# lane-card eval on a training card on 2026-08-31. Launch it as
# `CUDA_VISIBLE_DEVICES=1,2 setsid bash scripts/mem_decomp_run.sh &`.
#
# BEFORE the cd, so both refusals are reachable off the pod: with `cd /work/aupai` first, a
# laptop run dies on the missing directory and the guards below are never executed -- they
# would ship having never once refused anything.
CARDS="${CUDA_VISIBLE_DEVICES:-}"
if [ -z "$CARDS" ]; then
  echo "refusing: set CUDA_VISIBLE_DEVICES to the two cards this may use." >&2
  echo "  Unset means every card on the box, and this claims whatever it is given." >&2
  exit 1
fi
NPROC=$(printf '%s' "$CARDS" | awk -F, '{print NF}')
if [ "$NPROC" -ne 2 ]; then
  echo "refusing: world 2 is the arm's shape; CUDA_VISIBLE_DEVICES names $NPROC card(s)." >&2
  echo "  A ratio measured at another world size is not comparable to 64/82." >&2
  exit 1
fi

cd /work/aupai

# The allocator config every arm will launch with (4c, 2026-09-05), so the cells describe the
# shape that actually runs. Exported rather than set per-command: torchrun's children inherit it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=runs/mem_decomp_0905.jsonl
LOG=runs/mem_decomp_0905.log
: > "$OUT"

# The claim names THIS script's pid, which is the parent of every torchrun and lives exactly
# as long as the work does. b0_mem_m1's claim named an intermediate shell that exited while
# the run continued, so status reported a live arm as STALE and its cards as ORPHAN -- a
# claim is only worth the lifetime of the pid it names.
python3 scripts/card_claim.py acquire --name tilerl_mem_decomp --cards "$CARDS" --pid $$ \
  --note "M1 fwd/bwd/opt decomposition, 5 configs, ~45 min" >> "$LOG" 2>&1

release() {
  python3 scripts/card_claim.py release --name tilerl_mem_decomp >> "$LOG" 2>&1 || true
}
trap release EXIT

run_one() {
  local tag=$1 rc; shift
  echo "=== $tag ===" >> "$LOG"
  # NOT --peak-only: that mode returns before the JSON record is built, so the regions would
  # print to the log and never land in a row anything can tabulate. The cost is save+val at
  # the end of each config, which is outside the timed steps.
  #
  # rc is captured on the line that FOLLOWS the command with nothing between them. `echo "$tag
  # rc=$?"` reads the preceding echo's status, not torchrun's -- the shape that reported rc=0
  # over four crashed DDP arms on 2026-09-05. `|| true` would be worse than no capture at all:
  # it makes $? unconditionally 0. `rc=0; cmd || rc=$?` keeps the real status AND keeps set -e
  # from taking the other three configs down with one failure.
  rc=0
  torchrun --nproc_per_node="$NPROC" --master_port=29611 \
    scripts/profile_step_cost.py \
    --mix data/mix_200m_8b.json \
    --dim 1024 --layers 12 --heads 8 --ffn_hidden 3072 \
    --batch 16 --accum 2 --steps 20 --warmup 8 \
    --json "$OUT" "$@" >> "$LOG" 2>&1 || rc=$?
  echo "$tag rc=$rc" >> "$LOG"
}

# rc is captured per config and the loop does not stop on one failure: a config that OOMs
# still leaves the others comparable, and a missing row is visible in the table while a
# killed script would look like a machine problem.
run_one off --mem_values 0
run_one m1  --mem_values 1048576 --mem_top_k 32 --mem_layers 3,6,9 --no-mem_sparse
run_one k16 --mem_values 1048576 --mem_top_k 16 --mem_layers 3,6,9 --no-mem_sparse
run_one l1  --mem_values 1048576 --mem_top_k 32 --mem_layers 6     --no-mem_sparse
run_one m2  --mem_values 262144  --mem_top_k 32 --mem_layers 3,6,9 --no-mem_sparse

echo "ALL-DONE" >> "$LOG"
