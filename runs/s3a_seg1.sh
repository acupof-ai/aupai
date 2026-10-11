#!/bin/bash
# Stage 3 segment 1: Engram continuation under the full 100B LR plan; stop after the next 10B tokens.
set -euo pipefail
cd /work/aupai || exit 1
mkdir -p runs
export ALLOW_UNSYNCED=1 AUPAI_TOKEN_CACHE_DIR=/data00 NGPU=8
trap '' TERM HUP QUIT
export TORCHELASTIC_SIGNALS_TO_HANDLE=""
exec bash run_ddp.sh \
  --mix data/mix_s3a.json --resume ckpt_v42s2.pt \
  --v42_engram 1,5,9,13,17,21 \
  --lr_origin_step 54051 --lr_peak_mult 0.10 --lr_origin_stable \
  --name s3a_seg1 --max_steps 127157 --stop_at_step 66767 \
  --batch 4 --accum 6 --warmup 500 --warmdown 0.20 --anneal_frac 0.20 --lr_scale 1.0 \
  --arch v42 --moe_arm v42b --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 \
  "--v42_impl=attn_impl=fused,rope_impl=real,moe_stacked=1,hc_impl=liger,norm_impl=liger,attn_logit_softcap=50" \
  --v42_lr 1e-3 --no-grad_ckpt --save_every 5000 \
  --allow_corpus_drift --allow_pod_drift --allow_env_drift
