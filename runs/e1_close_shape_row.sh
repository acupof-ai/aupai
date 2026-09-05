#!/bin/bash
# Close e1_kda_shape_fault's ledger row on the pod.
#
# A SCRIPT, NOT AN INLINE `pod` COMMAND. ~/bin/pod passes its argv through a shell, which both
# word-splits a quoted --finding into separate arguments and dies on the parentheses inside it
# ("syntax error near unexpected token `('", then "unrecognized arguments: REFUTED. All 5..."),
# so a multi-sentence finding cannot survive the trip. Same reason runs/kda_shape_launch.sh
# exists. Quoting it correctly through two shells is a second thing to maintain for no benefit.
set -euo pipefail
cd /work/aupai
python3 /work/aupai/scripts/exp.py done --name e1_kda_shape_fault --status ok \
  --result runs/kda_shape_fault.json \
  --finding "SHAPE REFUTED. All 5 cells PASS on card 5: b1/t128, b1/full-seq, b4/4096 which is the shape that crashed, b16/4096 which is the arms per-rank shape, and b4/chunk64. chunk_kda on random tensors at the crashing shape does not fault, so the per-rank shape alone does not reproduce the misaligned-address crash. The fault needs something this probe omits by design: loaded weights, a cold-vs-warm autotune cache, or the full model call sequence. This is NOT evidence the weights are bad -- the probe can only exonerate, and what it exonerated is the shape." \
  --decision "Do not re-run arm_token_corr at batch 4 expecting a different outcome, and do not treat ckpt_b0_headmix_armA or armB as suspect on this evidence. The next discriminator is the autotune cache: re-run the failing arm_corr with TRITON_CACHE_DIR at an empty dir versus the warmed one, which separates cache state from weights without loading two checkpoints."
