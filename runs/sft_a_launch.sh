#!/bin/bash
# CED code-SFT plan A, post-pretraining world-8 launch (user order relayed by 1e 2026-09-25).
#
# Recipe (user-set, do not edit without a user ruling):
#   EPOCHS=2 over ALL of data/sft/sfta/sft_mixA_0924.pt, no truncation
#   LR_SCALE=0.1, first 5% of steps LINEAR warmup, then LINEAR decay to 0 over every
#     remaining step (--lr_decay linear --warmup_frac 0.05; NOT the pretraining cosine)
#   bf16 (--no_fp8 --stochastic_round) with activation checkpointing (sft_math's
#   --grad_ckpt default ON). Option B (1e 2026-09-25): weights bf16, fp32 w+delta
#   Bernoulli-rounded on write, fp32 Muon momentum; no fp32 master (that OOM'd at 93 GiB).
#   Measured card-7 single-rank 3-step probes 2026-09-25 (nvidia-smi memory.used): B4 after
#   the block-fp32 cast fix peaks 84.06 GiB (<= 85 gate, accepted; 81.66 allocator). An
#   earlier whole-group fp32 cast OOM'd B4 at 89.77 GiB. 1e ruling 2026-09-25: B4 stands.
#   seq 4096 from the pack, BATCH 4/rank (no accumulation: sft_math has none; global 32
#   rows ~131K tokens/step). The user's "default 48" was read as a GLOBAL batch; 48 was
#   actually per-rank rows, which extrapolates past the 96 GB H20 at B4x4096; the controller
#   set 4/rank 2026-09-25 after the B4 peak gate passed (step-1 peak > 85 GiB stops the run).
# Read points: .epoch1 and .epoch2 at each epoch end, each scored by
# eval/sft_a_humaneval.sh; the higher HumanEval is the kept one.
#
# Cards are the controller's: `harness launch` injects CUDA_VISIBLE_DEVICES from the grant,
# so this script never writes a device index. Launch preconditions: v41_ced_0923
# pretraining finished and the 8-card block is granted free; the pod tree carries this file
# and the merged code (pod_push after the PR merges); RESUME is the FINAL pretraining ckpt.
# The go is the controller's.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

# ── user-set hyperparameters (2026-09-25) ─────────────────────────────────────
EPOCHS=${EPOCHS:-2}
LR_SCALE=${LR_SCALE:-0.1}
WARMUP_FRAC=${WARMUP_FRAC:-0.05}
BATCH=${BATCH:-4}
NGPU=${NGPU:-8}
NAME=${NAME:-sft_a_0925}
RESUME=${RESUME:-/work/aupai/ckpt_v41_ced_0923.pt}
SFT_PT=${SFT_PT:-data/sft/sfta/sft_mixA_0924.pt}
OUT=${OUT:-/work/aupai/ckpt_${NAME}.pt}

# Per-rank step counts (sft_math shards X[rank::world], then ddp_even_len trims to a common
# multiple of BATCH). Printed before launch, as the user order requires; epoch boundary
# saves are OUT.epoch1 / OUT.epoch2 at these step numbers.
read -r ROWS STEPS_PR TOTAL WU < <(python3 - "$SFT_PT" "$NGPU" "$BATCH" "$EPOCHS" "$WARMUP_FRAC" <<'PY'
import sys, torch
pt, ngpu, batch, epochs, wuf = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
n = torch.load(pt, map_location="cpu", weights_only=True)["input_ids"].shape[0]
steps = (n // ngpu) // batch   # X[rank::world], ddp_even_len trims to a multiple of batch
total = steps * epochs
print(n, steps, total, max(1, round(wuf * total)))
PY
)
echo "pack rows $ROWS | steps/epoch/rank $STEPS_PR | total $TOTAL | warmup steps $WU"
echo "epoch boundary saves: step $STEPS_PR -> $(basename "$OUT").epoch1, step $TOTAL -> $(basename "$OUT").epoch2"

exec python3 scripts/harness.py launch "$NAME" --training --class incremental \
  --hypothesis "CED plan-A code SFT (code_if/sc2/APPS/prose, continuation, 2ep, linear-to-zero) raises HumanEval pass@1; epoch1 vs epoch2 scored, higher kept" \
  -- torchrun --nproc_per_node="$NGPU" sft_math.py \
    --resume "$RESUME" --sft_path "$SFT_PT" --out "$OUT" \
    --epochs "$EPOCHS" --batch "$BATCH" --lr_scale "$LR_SCALE" \
    --lr_decay linear --warmup_frac "$WARMUP_FRAC" \
    --no_fp8 --stochastic_round --save_every 1000000
