#!/usr/bin/env bash
# Turn the scored web corpus into data/corpus/web_hq, then report what it cost.
#
#   scripts/build_web_hq.sh [keep_fraction]
#
# Runs after scripts/score_corpus.sh. Concatenates the per-worker score arrays in
# WORKER order -- each worker owned a contiguous block of the sorted shard list,
# and clean_web.py matches score[i] to document i across the same sorted glob, so
# any other order silently attaches every score to a different document.
# clean_web.py asserts the counts match rather than trusting that.
set -euo pipefail
cd "$(dirname "$0")/.."

KEEP=${1:-0.40}
OUT=${OUT:-data/corpus/web_hq}

python3 - <<'PY'
import glob
import numpy as np

parts = sorted(glob.glob("data/web_scores.[0-9].npy"))
assert parts, "no data/web_scores.<i>.npy -- run scripts/score_corpus.sh first"
a = np.concatenate([np.load(f) for f in parts])
np.save("data/web_scores.npy", a)
q = np.percentile(a, [1, 10, 25, 50, 75, 90, 99])
print(f"{len(parts)} workers -> {len(a):,} scores")
print("  percentiles 1/10/25/50/75/90/99: " + " ".join(f"{v:.2f}" for v in q))
PY

rm -rf "$OUT"
python3 datagen/clean_web.py --scores data/web_scores.npy --keep "$KEEP" --out "$OUT"

python3 - "$OUT" <<'PY'
import json
import os
import random
import sys

from tokenizers import Tokenizer

out = sys.argv[1]
files = sorted(f for f in os.listdir(out) if f.endswith(".jsonl"))
n = sum(1 for f in files for line in open(os.path.join(out, f), encoding="utf-8") if line.strip())
tok = Tokenizer.from_file("data/tokenizer_k5.json")
rng = random.Random(0)
sample = []
for f in rng.sample(files, min(6, len(files))):
    with open(os.path.join(out, f), encoding="utf-8") as fh:
        lines = fh.readlines()
    sample += [json.loads(x).get("content", "") for x in rng.sample(lines, min(300, len(lines)))]
toks = sum(len(e.ids) for e in tok.encode_batch(sample))
per = toks / len(sample)
print(f"\n{out}: {len(files)} shards, {n:,} documents, ~{n * per / 1e9:.2f}B tokens ({per:.0f} tok/doc)")
print("Now: python3 scripts/check_mix.py --mix data/mix_v3.json  (it names any domain the epoch cap truncates)")
PY
