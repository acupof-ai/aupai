#!/bin/bash
# DDP pretraining on all 8 GPUs. Flags: see `python train.py --help`.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
cd "$(dirname "$0")"
torchrun --nproc_per_node="${NGPU:-8}" --master_port="${PORT:-29500}" train.py --fp8 "$@"
rc=$?
# A training run without a score-matrix record is what the score_matrix_present
# check catches; score here so the record exists by construction.
NAME=
for arg in "$@"; do
  case "$arg" in --name=*) NAME=${arg#--name=};; --name) ;; *) [ "${prev:-}" = "--name" ] && NAME=$arg;; esac
  prev=$arg
done
if [ $rc -eq 0 ] && [ -n "$NAME" ] && [ -f "ckpt_${NAME}.pt" ]; then
  python scripts/score_matrix.py --ckpt "ckpt_${NAME}.pt" --json runs/score_matrix.jsonl \
    || echo "WARN: score_matrix failed for ckpt_${NAME}.pt -- the harness check will flag the missing record" >&2
fi
exit $rc
