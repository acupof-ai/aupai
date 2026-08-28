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

python3 - "$MIX" <<'PY'
import json
import os
import sys

mix = json.load(open(sys.argv[1], encoding="utf-8"))
doms = list(mix["domains"])
assert "web" not in doms, (
    "the mix names 'web', which is the UNFILTERED corpus. train.py globs "
    "data/corpus/<domain>/, so this would train on 2,991,648 unfiltered documents "
    "and silently discard every filter. Use web_hq."
)
for d in doms:
    p = os.path.join("data", "corpus", d)
    shards = [f for f in os.listdir(p)] if os.path.isdir(p) else []
    assert any(f.endswith(".jsonl") for f in shards), f"data/corpus/{d} has no .jsonl shards"
    print(f"  {d:<9} {sum(1 for f in shards if f.endswith('.jsonl')):>4} shards")

from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file("data/tokenizer.json")
for sp in ("<|im_start|>", "<|im_end|>", "<eos>"):
    ids = tok.encode(sp, add_special_tokens=False).ids
    assert len(ids) == 1, f"{sp} is {len(ids)} tokens, not 1 -- rebuild with scripts/build_tokenizer.py"
print(f"  tokenizer vocab {tok.get_vocab_size()}, ChatML specials are single tokens")

w = sum(v["weight"] for v in mix["domains"].values())
a = sum(v["anneal"] for v in mix["domains"].values())
assert abs(w - 1) < 1e-6 and abs(a - 1) < 1e-6, f"weights sum {w}, anneal sum {a}; both must be 1"
print(f"  mix total {mix['total_tokens'] / 1e9:.2f}B tokens over {len(doms)} domains")
PY

echo "checking the epoch caps against the real pools (this tokenizes if needed)..."
python3 scripts/check_mix.py --mix "$MIX"

echo
echo "launching: $NAME on $NGPU gpus"
NGPU=$NGPU bash run_ddp.sh --mix "$MIX" --name "$NAME" \
  --fp8 --attn_res --attn_res_blocks 4 --warmup 150 --lr_scale 0.5
