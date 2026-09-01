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
  # Not the block. This scoring runs inside the training shell, where
  # CUDA_VISIBLE_DEVICES is still the seven-card block, so it used to take whatever
  # card 0 was doing -- on 2026-09-01 a process holding 14.37 GiB, and the scorer
  # died asking for 96 MiB. Measure a free lane card, and queue rather than force.
  CARD=$(python scripts/harness.py free-card --wait 1800)
  if [ -n "$CARD" ]; then
    CUDA_VISIBLE_DEVICES="$CARD" python eval/score_matrix.py --ckpt "ckpt_${NAME}.pt" --json runs/score_matrix.jsonl \
      || echo "WARN: score_matrix failed for ckpt_${NAME}.pt -- the harness check will flag the missing record" >&2
  else
    echo "WARN: no free lane card in 30min; ckpt_${NAME}.pt unscored. Run: python eval/score_matrix.py --ckpt ckpt_${NAME}.pt --json runs/score_matrix.jsonl" >&2
  fi
fi
exit $rc
