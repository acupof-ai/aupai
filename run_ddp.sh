#!/bin/bash
# DDP pretraining on all 8 GPUs. Flags: see `python train.py --help`.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
cd "$(dirname "$0")"
torchrun --nproc_per_node="${NGPU:-8}" --master_port="${PORT:-29500}" train.py --fp8 "$@"
