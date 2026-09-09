"""Fertility of one vocabulary on a fixed sample of OUR corpus.

THE SAMPLE IS A FILE, NOT A DRAW. Both tokenizers must read the same bytes or the
comparison measures the sample; scripts/_mksample-style drawing happens once on the pod
(the corpus lives there) and the result is carried as a gzip+base64 blob. Pass the blob
and a tokenizer:

    python3 scripts/fertility_vs_external.py runs/fert_sample.b64 data/tokenizer.json

facts/tokenizer.json#tok.fertility_ours_vs_minicpm5_own_corpus is this script's output for
data/tokenizer.json and for MiniCPM5-2B's tokenizer.json on one blob (153 docs/domain from
code_rp1t_dd09, math_owm_stage2, zh_web; 3 shards, 1200-char clip, seed 7).
"""

import base64
import gzip
import json
import re
import sys

from tokenizers import Tokenizer

with open(sys.argv[1]) as fh:
    blob = fh.read()
S = json.loads(gzip.decompress(base64.b64decode(blob)).decode())
tok = Tokenizer.from_file(sys.argv[2])

WORD = re.compile(r"\S+")
HANZI = re.compile(r"[一-鿿]")
print(f"vocab {tok.get_vocab_size()}")
for tag in ("code", "math", "zh"):
    docs = S[tag]
    n_tok = sum(len(tok.encode(d).ids) for d in docs)
    n_chr = sum(len(d) for d in docs)
    n_byt = sum(len(d.encode("utf-8")) for d in docs)
    n_wrd = sum(len(WORD.findall(d)) for d in docs)
    n_han = sum(len(HANZI.findall(d)) for d in docs)
    print(
        json.dumps(
            {
                "domain": tag,
                "docs": len(docs),
                "tokens": n_tok,
                "chars": n_chr,
                "bytes": n_byt,
                "ws_words": n_wrd,
                "hanzi": n_han,
                "chars_per_token": round(n_chr / n_tok, 4),
                "bytes_per_token": round(n_byt / n_tok, 4),
                "tokens_per_ws_word": round(n_tok / n_wrd, 4),
            },
            ensure_ascii=False,
        )
    )
