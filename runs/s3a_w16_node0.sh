#!/bin/bash
# s3a segment 1, world-16 node 0: pod node, ranks 0-7.
# Start node 1 first. Keep tokens/step at 786,432: batch 4, accum 3, world 16.
set -euo pipefail
RESUME="${1:?usage: s3a_w16_node0.sh <checkpoint-path>}"
cd /work/aupai || exit 1
[ -f "$RESUME" ] || { echo "resume checkpoint not found: $RESUME"; exit 1; }
mkdir -p runs /var/run/nvidia-topologyd
cp -n runs/virtualTopology.xml /var/run/nvidia-topologyd/ 2>/dev/null || true
export ALLOW_UNSYNCED=1 ALLOW_DIRECT_RUN=1 AUPAI_TOKEN_CACHE_DIR=/data00
export NGPU=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.29.37 PORT=29500
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA='^mlx5_0,mlx5_5'
export NCCL_IB_GID_INDEX=3
trap '' TERM HUP QUIT
export TORCHELASTIC_SIGNALS_TO_HANDLE=""
exec bash run_ddp.sh \
  --mix data/mix_s3a.json --resume "$RESUME" \
  --v42_engram 1,5,9,13,17,21 \
  --lr_origin_step 54051 --lr_peak_mult 0.10 --lr_origin_stable \
  --name s3a_seg1 --max_steps 127157 --stop_at_step 66767 \
  --batch 4 --accum 3 --warmup 500 --warmdown 0.20 --anneal_frac 0.20 --lr_scale 1.0 \
  --arch v42 --moe_arm v42b --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 \
  "--v42_impl=attn_impl=fused,rope_impl=real,moe_stacked=1,hc_impl=liger,norm_impl=liger,attn_logit_softcap=50" \
  --v42_lr 1e-3 --no-grad_ckpt --save_every 4000 \
  --allow_corpus_drift --allow_pod_drift --allow_env_drift
