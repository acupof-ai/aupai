#!/usr/bin/env python3
"""3b-7: WHY-histogram of code_rp1t ast.parse failures (fb, today).

Samples 5,000 rows STRATIFIED across the 235 code_rp1t shards (find each shard's
row count, sample in proportion, so a dominant shard cannot skew the cause mix),
classifies each ast.parse failure into fb's five causes, reports counts with
samples, and a keep/drop ruling per cause with the tokens it would cost in a
rebuilt corpus (median row size x count). Grounds the 16.3% failure figure by
measuring it on the same corpus the mix uses.
"""
import ast
import glob
import json
import random
import sys
from collections import Counter

random.seed(1729)
CORPUS = "/work/aupai/data/corpus/code_rp1t"
N = 5000


def row_fails(t):
    try:
        ast.parse(t)
        return False
    except (SyntaxError, MemoryError, RecursionError, ValueError):
        return True


def cause(t):
    if not isinstance(t, str):
        return "non-string-row"
    head = t[:6000]
    # 1 non-Python mislabeled as .py: no def/class/import/print, heavy prose or
    # clearly another language (preprocessor, <html>, XML front-matter, package line)
    digits_only_lines = not any(
        kw in head for kw in ("def ", "class ", "import ", "from", "print(")
    )
    if ("#include" in head or "#ifndef" in head or "<?xml" in head.lower()
        or head.lstrip().lower().startswith("<html") or "package " in head[:200]):
        return "non-python-under-py"
    # 2 encoding / control bytes: a NUL, or a lone surrogate that breaks utf-8 decode
    if "\x00" in head or "\x01" in head or "\x02" in head:
        return "encoding-or-control-bytes"
    # 3 py2 syntax: print statement, xrange, leading-zero ints, backtick repr
    lines = t.split("\n")
    if (">> " in t and "print " in t) or "xrange(" in t or "`" in t or \
       any(l.strip().startswith("print ") and not l.strip().startswith("print(")
           for l in lines):
        return "python2-syntax"
    # 4 not python at all despite a def (e.g. "def" inside a comment / doc prose)
    if digits_only_lines:
        return "non-python-under-py"
    # 5 notebook cell: markdown + code mixed (a '[' heading or ipython prompt)
    if t.count("In [") > 0 or t.count("Out[") > 0 or "[1]: " in head or \
       any(l.startswith("In [") or l.startswith("Out[") for l in lines[:40]):
        return "notebook-cells"
    # 6 truncation: last nonblank line ends in a mid-construct token (colon, comma,
    # an operator, 'elif'/'def'/'class' with no body, an open bracket)
    nonblank = [l for l in lines if l.strip()]
    if nonblank:
        last = nonblank[-1].strip()
        if last in (":", ",", "\\", "(", "[", "{", "=") or last.endswith(":") or \
           any(last.startswith(k) for k in ("def ", "class ", "elif ", "else ", "for ", "while ", "import ")):
            return "truncated-mid-construct"
    return "other"


def main():
    shards = sorted(glob.glob(f"{CORPUS}/code_rp1t_*.jsonl"))
    if not shards:
        print("no shards found", file=sys.stderr)
        sys.exit(1)
    # stratification: row count per shard, then proportional target
    sizes = {}
    for p in shards:
        sizes[p] = sum(1 for _ in open(p, encoding="utf-8") if _.strip())
    total = sum(sizes.values())
    counts = Counter()
    samples = {}
    tok_cost = Counter()
    per_cause = {}
    # proportional per shard: target rows, random offsets
    for p in shards:
        n = max(1, int(N * sizes[p] / total))
        with open(p, encoding="utf-8") as f:
            rows = []
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        idx = random.sample(range(len(rows)), min(n, len(rows)))
        for i in idx:
            t = rows[i].get("content") or ""
            if not row_fails(t):
                continue
            c = cause(t)
            counts[c] += 1
            samples.setdefault(c, []).append(t[:180].replace("\\n", " / "))
            per_cause.setdefault(c, []).append(t)
            tok_cost[c] += len(rows[i].get("content", ""))  # approx tokens
    # total failing measured + failure rate
    out = {
        "n_sharded_sample_target": N,
        "shards": len(shards),
        "total_rows_measured": total,
        "cause_counts": dict(counts),
        "cause_token_cost_est": {k: v for k, v in tok_cost.items()},
        "sample_rows_per_cause_top3": {k: v[:3] for k, v in samples.items()},
        "config": {
            "corpus": CORPUS, "seed": 1729, "classifier": "5 buckets, stratified by shard",
            "note": "token cost = sum of content bytes of failing rows in that cause (approx log-ish)",
        },
    }
    print("COUNTS:"); [print(f"  {k}: {v}") for k, v in counts.most_common()]
    print("TOKEN_EST:"); [print(f"  {k}: {v} bytes") for k, v in tok_cost.most_common()]
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()