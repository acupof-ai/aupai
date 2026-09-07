#!/bin/bash
# 58's caution: a recount that only adds the terminator agrees with the extrapolation and
# reads as confirmation. Run the SHARED counter over the same shards, so what is verified
# is the convention rather than the arithmetic.
cd /work/aupai || exit 1
python3 - <<'PY'
import glob, json, sys, time
sys.path.insert(0, "scripts")
from count_tokens import count_shards, CONVENTION
from tokenizers import Tokenizer

t0 = time.perf_counter()
tok = Tokenizer.from_file("data/tokenizer.json")
ps = sorted(glob.glob("data/corpus/en_c4_30b/*.jsonl"))
t, b = count_shards(ps, tok)
print(json.dumps({
    "domain": "en_c4_30b", "shards": len(ps), "tokens": t, "bytes": b,
    "convention": CONVENTION, "sampling": "none (full population)",
    "seconds": round(time.perf_counter() - t0),
}, indent=1))
PY
