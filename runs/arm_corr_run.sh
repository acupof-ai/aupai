#!/bin/bash
# Per-token arm correlation on CARD 7. Grant: runs/card_assignment.json cards[7], granted by the
# RL team (tilerl-27 via tilerl-0a) 2026-09-05T01:42Z, ONE job, ~10 min, world 1.
#
# A GPU is REQUIRED, not preferred: facts/efficiency.json#eff.model_cannot_forward_on_cpu --
# fla.ops.kda.chunk_kda is Triton-only with no CPU fallback, so this model cannot complete a
# forward pass on CPU at all.
#
# THE CLAIM IS ACQUIRED IN A RETRY LOOP against this script's own pid. card_claim
# --require-device refuses until the pid actually holds a device fd, and that only happens after
# the first checkpoint reaches the GPU -- so a single acquire before the job starts would fail,
# and one after would leave the card unclaimed for the whole load. The release runs from an EXIT
# trap so a crash frees the card too.
#
# THE CARD INDEX COMES FROM THE CALLER, not from this file. Writing CUDA_VISIBLE_DEVICES=7 here
# would REPLACE whatever set the caller confined this job to rather than index into it, which is
# how a lane-card launch landed on a training card on 2026-08-31 (eval/_devs.sh). So the launcher
# passes the grant's card and this script reads it back, meaning a moved grant moves the job.
#
# Absolute paths throughout: this runs under setsid with no cwd guarantee.
set -euo pipefail
cd /work/aupai
source eval/_devs.sh 1
JOB=$$

release() {
  python3 /work/aupai/scripts/card_claim.py release --name e1_arm_token_corr \
    --cards "${_DEVS[0]}" >&2 || true
}
trap release EXIT

(
  for i in $(seq 1 60); do
    if python3 /work/aupai/scripts/card_claim.py acquire \
         --name e1_arm_token_corr --cards "${_DEVS[0]}" --pid "$JOB" --require-device \
         --note "RL-team grant 2026-09-05T01:42Z, one job" >&2; then
      echo "claim acquired on attempt $i for card ${_DEVS[0]}" >&2
      exit 0
    fi
    sleep 5
  done
  echo "CLAIM NEVER ACQUIRED after 60 attempts -- the job is running UNCLAIMED" >&2
) &

CUDA_VISIBLE_DEVICES=${_DEVS[0]} python3 /work/aupai/probes/arm_token_corr.py \
  --ckpt_a /work/aupai/ckpt_b0_headmix_armA.pt \
  --ckpt_b /work/aupai/ckpt_b0_headmix_armB.pt \
  --domain code_py_starcoder \
  --cache /data00/tokens_code_py_starcoder.pt \
  --rows 64 --batch 4 --device cuda:0 --allow_cuda \
  --out /work/aupai/runs/arm_corr_64rows.json
