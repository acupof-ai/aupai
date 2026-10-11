#!/bin/bash
# Stage 3 WSD continuation: 8 GPUs, 786,432 tokens per optimizer step, target 100B total tokens.
set -euo pipefail
cd /work/aupai || exit 1
mkdir -p runs
export ALLOW_UNSYNCED=1 ALLOW_DIRECT_RUN=1 AUPAI_TOKEN_CACHE_DIR=/data00 NGPU=8
# External watchdogs send SIGTERM to torchrun only. Keep the run alive.
# Use kill -9 for a controlled hard stop.
trap '' TERM HUP QUIT
export TORCHELASTIC_SIGNALS_TO_HANDLE=""
exec bash run_ddp.sh \
  --mix data/mix_v42_stage3.json --resume ckpt_v42s2.pt \
  --lr_origin_step 54051 --lr_peak_mult 0.10 --lr_origin_stable \
  --name v42_stage3 --max_steps 127157 \
  --batch 4 --accum 6 --warmup 500 --warmdown 0.20 --anneal_frac 0.20 --lr_scale 1.0 \
  --arch v42 --moe_arm v42b --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 \
  "--v42_impl=attn_impl=fused,rope_impl=real,moe_stacked=1,hc_impl=liger,norm_impl=liger,attn_logit_softcap=50" \
  --v42_lr 1e-3 --no-grad_ckpt --save_every 5000 \
  --allow_corpus_drift --allow_pod_drift --allow_env_drift
