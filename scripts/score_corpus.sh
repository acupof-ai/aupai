#!/usr/bin/env bash
# Score every web shard with the distilled quality head, one worker per GPU.
#
#   scripts/score_corpus.sh [ngpu] [glob]
#
# The 27B teacher answers 0.76/s -- four days for 1.97M documents; the distilled head
# does 231/s. That gap is the reason for the two-stage design.
#
# Writes data/web_scores.<i>.npy per worker plus the shard list each covered, so
# datagen/clean_web.py can line scores up with documents in glob order.
set -euo pipefail
cd "$(dirname "$0")/.."

NGPU=${1:-6}
GLOB=${2:-data/corpus/web/*.jsonl}
CKPT=${CKPT:-ckpt_k5_clean_0827.pt}
TOK=${TOKENIZER:-data/tokenizer_k5.json}
HEAD=${HEAD:-data/quality_head.pt}

mapfile -t SHARDS < <(ls $GLOB | sort)
N=${#SHARDS[@]}
echo "$N shards over $NGPU gpus"

for ((g = 0; g < NGPU; g++)); do
  # Contiguous blocks, not a stride: clean_web.py concatenates the per-worker
  # score arrays in worker order, so worker g must own a contiguous run of the
  # sorted shard list or every score lands on the wrong document.
  lo=$((g * N / NGPU))
  hi=$(((g + 1) * N / NGPU))
  [ "$lo" -ge "$hi" ] && continue
  printf '%s\n' "${SHARDS[@]:lo:hi-lo}" > "runs/shards_$g.txt"
  CUDA_VISIBLE_DEVICES=$((g + 1)) setsid nohup python3 -u datagen/train_quality_head.py \
    --score "@runs/shards_$g.txt" --head "$HEAD" --ckpt "$CKPT" --tokenizer "$TOK" \
    --out "data/web_scores.$g.npy" --device cuda:0 > "runs/score_$g.log" 2>&1 &
  echo "  gpu $((g + 1)): shards $lo..$((hi - 1)) -> data/web_scores.$g.npy"
done
wait
echo "done; concatenate in worker order:"
echo "  python -c \"import numpy as np,glob;np.save('data/web_scores.npy',np.concatenate([np.load(f) for f in sorted(glob.glob('data/web_scores.[0-9].npy'))]))\""
