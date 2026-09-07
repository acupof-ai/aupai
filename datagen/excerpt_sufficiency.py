#!/usr/bin/env python3
# restartable: a sampled read with one print at the end; the draw is seeded (Random(11)) so a
# re-run reproduces the same sample rather than a different one.
"""Can code_dedup_handread's 400-char excerpt decide "same file"?

The sheet truncates rep_text_excerpt and member_text_excerpt to 400 chars
(code_dedup_handread.py:67-68), and the reader fills same_file from those excerpts. So the
excerpt length is part of the hand-read's definition, and this measures what it can resolve:
how often two DIFFERENT documents share a whitespace-normalised 400-char prefix. That is the
instrument's false-"same" rate, and #70's conclusion rests on it being small.

Normalised by collapsing whitespace, because the criterion the reader applies is "same code
modulo whitespace, identifiers, or comments" -- comparing raw bytes would understate the
collision rate the reader actually faces.

Measured 2026-09-08: 15 of 4200 (0.36%), median doc 1883 chars, 14.5% of docs under 400 chars
and therefore shown whole. Recorded as facts/data_quality.json#dq.handread_excerpt_sufficient.

    python3 datagen/excerpt_sufficiency.py

It does NOT bound the opposite error: two excerpts of the same file that read as different
because the difference falls inside the first 400 chars.
"""

import glob
import hashlib
import json
import os
import random
import statistics

# Can a 400-char excerpt decide "same file"? Measure how often two DIFFERENT starcoder docs
# share their first 400 chars: that is the sheet's false-"same" rate, and it bounds what the
# hand read can conclude. Whitespace-normalised, because the criterion is "modulo whitespace".
ROOT = os.environ.get("AUPAI_ROOT", "/work/aupai")
DOMAIN = os.environ.get("AUPAI_DOMAIN", "code_py_starcoder")
shards = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", DOMAIN, "*.jsonl")))
if not shards:
    raise SystemExit(f"no shards under {ROOT}/data/corpus/{DOMAIN} -- this reads the pod corpus")
rng = random.Random(11)
docs = []
for sh in rng.sample(shards, 6):
    with open(sh, encoding="utf-8") as f:
        lines = f.readlines()
    for ln in rng.sample(lines, min(700, len(lines))):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        t = d.get("content") or d.get("text") or ""
        if t:
            docs.append(t)
print("docs sampled", len(docs), flush=True)


def norm(s):
    return " ".join(s.split())


pre, coll = {}, 0
for t in docs:
    k = hashlib.sha1(norm(t[:400]).encode()).hexdigest()
    full = hashlib.sha1(norm(t).encode()).hexdigest()
    if k in pre and pre[k] != full:
        coll += 1
    pre.setdefault(k, full)
print("distinct 400-char prefixes", len(pre))
print(f"DIFFERENT docs sharing a 400-char prefix: {coll} ({coll / max(1, len(docs)):.2%})")
L = sorted(len(t) for t in docs)
print(
    "doc chars: median",
    int(statistics.median(L)),
    "p10",
    L[len(L) // 10],
    f"frac under 400: {sum(1 for x in L if x < 400) / len(L):.1%}",
)
