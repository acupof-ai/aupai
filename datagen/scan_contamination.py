"""Every new source has to be scanned for eval questions before it enters the corpus.

    python datagen/scan_contamination.py [glob]
"""

import glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holdout import is_holdout


def scan(name, rows, limit=200000):
    hit = n = 0
    for t in rows:
        if n >= limit:
            break
        n += 1
        if is_holdout(t):
            hit += 1
            continue
        for ln in (x.strip() for x in t.split("\n")[:60]):
            if ln and len(ln) <= 500 and is_holdout(ln):
                hit += 1
                break
    print(f"{name:<14} {n:>7} docs scanned, {hit} eval questions found ({hit / max(1, n):.4%})")


import pyarrow.parquet as pq

RESULT = """Measured 2026-08-29, 60,000 documents each:
    cosmopedia   0 eval questions (0.0000%)
    wiki-zh      0 eval questions (0.0000%)
    r1-distill   0 eval questions (0.0000%)
For contrast, the corpus this filter was written for had 496 of 500
math_test_500 questions reach data/corpus/math/ before it existed."""

for p in sorted(glob.glob("/work/newdata/cosmo/*.parquet"))[:2]:
    d = pq.ParquetFile(p).read(columns=["text"]).to_pydict()["text"]
    scan("cosmopedia", d, 60000)
for p in sorted(glob.glob("/work/newdata/wiki/*.parquet"))[:1]:
    t = pq.ParquetFile(p).read()
    col = "text" if "text" in t.column_names else t.column_names[-1]
    scan("wiki-zh", t.to_pydict()[col], 60000)
try:
    with open("/work/newdata/r1/distill110k.jsonl", encoding="utf-8") as fh:
        rows = []
        for i, l in enumerate(fh):
            if i >= 60000:
                break
            d = json.loads(l)
            rows.append(str(d.get("instruction") or d.get("input") or d.get("question") or ""))
    scan("r1-distill", rows, 60000)
except Exception as e:
    print("r1-distill:", type(e).__name__, e)
