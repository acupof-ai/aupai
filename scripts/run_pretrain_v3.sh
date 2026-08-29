#!/usr/bin/env bash
# Pretrain on corpus v3. Every precondition is checked before a GPU is touched,
# because each of them fails silently: the run trains, the loss looks ordinary,
# and the result is a model built on the wrong data.
#
#   scripts/run_pretrain_v3.sh [name]
#
# Preconditions, in the order a mistake would bite:
#   1. the mix names web_hq, not web -- "web" is the UNFILTERED 2.99M-document
#      corpus and would train perfectly well while discarding every filter
#   2. every domain in the mix has a directory with shards in it
#   3. the tokenizer carries the ChatML specials as single tokens
#   4. the token caches, if present, were built by THIS vocabulary
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=${1:-k7_v3}
MIX=${MIX:-data/mix_v3.json}
NGPU=${NGPU:-8}

# The mix guard lives in train.py, on the path main() takes, so run_ddp.sh and a bare
# `python train.py` are covered too. A copy here would be a second thing to keep in sync.

echo "checking the epoch caps against the real pools (this tokenizes if needed)..."
python3 scripts/check_mix.py --mix "$MIX"

echo
echo "launching: $NAME on $NGPU gpus"
NGPU=$NGPU bash run_ddp.sh --mix "$MIX" --name "$NAME" \
  --fp8 --attn_res --attn_res_blocks 4 --warmup 150 --lr_scale 0.5
