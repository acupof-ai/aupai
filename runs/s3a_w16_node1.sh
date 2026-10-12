#!/bin/bash
# s3a segment 1, world-16 node 1: H20B node, ranks 8-15.
# Start this container before node 0.
set -euo pipefail
RESUME="${1:?usage: s3a_w16_node1.sh <checkpoint-basename>}"
ROOT=/data00/aupai_work/aupai
[ -f "$ROOT/$RESUME" ] || { echo "resume checkpoint not on H20B: $RESUME"; exit 1; }
IMAGE="${IMAGE:-iaas-gpu-cn-beijing.cr.volces.com/serving/sglang:v0.5.13.post1.iaas.nightly.202606171156-cu129}"
docker rm -f aupai_node1 2>/dev/null || true
docker run -d --name aupai_node1 \
  --gpus all --network host --ipc host --shm-size 32g \
  --device /dev/infiniband --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /run/nvidia-topologyd:/var/run/nvidia-topologyd:ro \
  -v /data00:/data00 \
  -v "$ROOT:/work/aupai" \
  -e RESUME="$RESUME" \
  -e ALLOW_UNSYNCED=1 \
  -e ALLOW_DIRECT_RUN=1 \
  -e AUPAI_TOKEN_CACHE_DIR=/data00 \
  -e NGPU=8 \
  -e NNODES=2 \
  -e NODE_RANK=1 \
  -e MASTER_ADDR=192.168.29.37 \
  -e PORT=29500 \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_HCA='^mlx5_0,mlx5_5' \
  -e NCCL_IB_GID_INDEX=3 \
  -e TORCHELASTIC_SIGNALS_TO_HANDLE='' \
  "$IMAGE" \
  bash -c 'cd /work/aupai && exec bash run_ddp.sh \
    --mix data/mix_s3a.json --resume "$RESUME" \
    --v42_engram 1,5,9,13,17,21 \
    --lr_origin_step 54051 --lr_peak_mult 0.10 --lr_origin_stable \
    --name s3a_seg1 --max_steps 127157 --stop_at_step 66767 \
    --batch 4 --accum 3 --warmup 500 --warmdown 0.20 --anneal_frac 0.20 --lr_scale 1.0 \
    --arch v42 --moe_arm v42b --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 \
    "--v42_impl=attn_impl=fused,rope_impl=real,moe_stacked=1,hc_impl=liger,norm_impl=liger,attn_logit_softcap=50" \
    --v42_lr 1e-3 --no-grad_ckpt --save_every 4000 \
    --allow_corpus_drift --allow_pod_drift --allow_env_drift'
echo "node1 container up; logs: docker logs -f aupai_node1"
