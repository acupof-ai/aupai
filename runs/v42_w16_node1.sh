#!/bin/bash
# v42 gate, world-16 resume: node 1 of 2 -- runs ON H20B (192.168.29.36, borrowed box).
# Starts the training container holding ranks 8-15. Start this BEFORE node0's script;
# torchrun waits on the master's rendezvous. h20b's host /data00 mirrors the pod's:
# tokens_*.pt caches at /data00, the code tree at /data00/aupai_work/aupai, so the two
# mounts below reproduce the pod container's view exactly. Nothing irreplaceable lives
# on h20b; the pod stays the box of record and writes every checkpoint (rank 0 is there).
set -e
RESUME="${1:?usage: v42_w16_node1.sh <interrupt-ckpt-basename>}"
[ -f "/data00/aupai_work/aupai/$RESUME" ] || { echo "resume checkpoint not on h20b: $RESUME"; exit 1; }
docker rm -f aupai_node1 2>/dev/null || true
docker run -d --name aupai_node1 \
  --gpus all --network host --ipc host --shm-size 32g \
  --device /dev/infiniband --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /run/nvidia-topologyd:/var/run/nvidia-topologyd:ro \
  -v /data00:/data00 \
  -v /data00/aupai_work/aupai:/work/aupai \
  aupai/train:pod-copy-1002 \
  bash -c 'cd /work/aupai && \
    ALLOW_DIRECT_RUN=1 \
    PREC_FLAG=--bf16 NGPU=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=192.168.29.37 PORT=29500 \
    NCCL_SOCKET_IFNAME=eth0 NCCL_IB_HCA=^mlx5_0,mlx5_5 NCCL_IB_GID_INDEX=3 \
    ./run_ddp.sh --mix data/mix_v41_gate.json --name v42_gate_1001r \
    --arch v42 --moe_arm v42b \
    --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 2 --accum 6 \
    --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.10 --warmup 500 --save_every 2000 --no-grad_ckpt \
    --v42_impl attn_impl=fused,rope_impl=real,moe_stacked=1,hc_impl=liger,norm_impl=liger,attn_logit_softcap=50 \
    --v42_lr 1e-3 --v42_record --allow_env_drift \
    --resume '"$RESUME"' > /work/aupai/runs/v42_w16_node1.log 2>&1'
echo "node1 container up; log: /data00/aupai_work/aupai/runs/v42_w16_node1.log"
