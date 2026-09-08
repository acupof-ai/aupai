#!/usr/bin/env bash
# The null + seed-noise arms for the anneal reweight (docs/lessons/anneal_weight_proposal_30b_v2.md).
#
# READ N1 vs N2 BEFORE LOOKING AT R. |N1 - N2| is the noise floor; if |R - N1| falls inside it,
# the reweight had no measurable effect at this budget, which is a result and not a failed run.
#
# ALL THREE ARMS READ A MIX THAT DECLARES anneal_frac 0.10, AND THE FLAG AGREES WITH IT.
# mix_200m_4b.json declares 0.0, and train.py's _mix_anneal_frac REFUSES when the mix's declared
# value disagrees with Cfg -- so `--anneal_frac 0.10` against that file does not run at all. The
# first version of this script did exactly that and would have died at build_mix. Two new files
# carry the declaration instead: _annealN (anneal == weight, the null) and _annealR (the reweight).
#
# Why the refusal matters beyond this script: if the disagreement were resolved silently toward
# 0.0, all three arms would run with NO anneal phase and return three near-equal numbers, which
# reads exactly like "the reweight does nothing" -- the conclusion this experiment exists to test.
# A silent default in a controlled experiment manufactures the null it is meant to reject.
#
#   bash runs/anneal_arms.sh n1     # noise floor A
#   bash runs/anneal_arms.sh n2     # noise floor B, seed+1
#   bash runs/anneal_arms.sh r      # the reweight
set -euo pipefail
ARM="${1:?arm: n1 | n2 | r}"
cd /work/aupai

case "$ARM" in
  n1) MIX=data/mix_200m_4b_annealN.json; SEED=1337 ;;
  n2) MIX=data/mix_200m_4b_annealN.json; SEED=1338 ;;
  r)  MIX=data/mix_200m_4b_annealR.json; SEED=1337 ;;
  *)  echo "arm must be n1, n2 or r" >&2; exit 2 ;;
esac
NAME="anneal_${ARM}_$(date -u +%m%d)"

# batch 32 accum 1 OOMs at seq 4096 on this shape: facts/efficiency.json#eff.microbatch_32_oom
# (93.8 GiB/card, ranks 3/6 first). p200m_4b_0902 reproduced it twice. Grow the effective batch
# with accum, never with micro-batch.
# Every name in train.py's RECIPE_REQUIRED must appear or argparse exits: dim, layers, heads,
# ffn_hidden, batch, accum, lr_scale, warmdown, anneal_frac, warmup, save_every, grad_ckpt.
exec ./run_ddp.sh --mix "$MIX" --name "$NAME" \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 3072 \
  --batch 16 --accum 2 --no-grad_ckpt \
  --lr_scale 1.0 --warmup 150 --warmdown 0.65 --anneal_frac 0.10 \
  --save_every 1000 --seed "$SEED"
