#!/usr/bin/env bash
# STOP (exit 1) when the checkpoint's trained vocabulary != the tokenizer's.
# A tokenizer rebuild changes ids; a mismatch scores as silent noise.
# Usage: scripts/assert_vocab.sh <ckpt> <tokenizer.json>
set -euo pipefail
CKPT=$1; TOK=$2
VOCAB=$(python3 -c "
import torch
c = torch.load('$CKPT', map_location='cpu', weights_only=False)['cfg']
print(c.get('vocab_real', c.get('vocab')), bool(c.get('fone')))
")
read -r CKPT_VOCAB IS_FONE <<< "$VOCAB"
TOK_VOCAB=$(python3 -c "from tokenizers import Tokenizer; print(Tokenizer.from_file('$TOK').get_vocab_size())")
if [ "$CKPT_VOCAB" != "$TOK_VOCAB" ]; then
  echo "STOP: $CKPT was trained at vocab $CKPT_VOCAB, $TOK has $TOK_VOCAB." >&2
  echo "      Pass the matching tokenizer via TOKENIZER=<path>." >&2
  exit 1
fi
echo "vocab ok: $CKPT_VOCAB (fone=$IS_FONE) tokenizer=$TOK" >&2
