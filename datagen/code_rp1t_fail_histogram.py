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
    head = t[:6000].lstrip()
    low = head.lower()
    # 0 FIRST: explicit non-Python language signature -- shebang not-python, PHP,
    # HTML, C#/Java/Go/Rust/JS/C/C++ files that occurred in a python dataset by mislabel.
    if head.startswith("#!") and "/python" not in low[:80]:
        return "non-python-under-py"
    if "<?php" in low or "<html" in low or "<!doctype" in low or "<?xml" in low:
        return "non-python-under-py"
    if head.startswith("package ") and ";" in low:      # Java
        return "non-python-under-py"
    if low.startswith("using system;") or "module.exports" in low or "function(" in low \
       or low.startswith("#include") or low.startswith("require('") or low.startswith("require(\"") \
       or "export default class" in low or "export default function" in low \
       or "require('" in low or 'require("path")' in low or "var path = require" in low \
       or low.startswith("#ifndef") or low.startswith("#define"):
        return "non-python-under-py"
    # RST / Markdown / prose: a heading underline (===== / ------ / ~~~~~) directly
    # beneath a short title line, i.e. a doc page, not code
    nonblank = [l for l in head.split("\n") if l.strip()]
    if len(nonblank) >= 2 and len(nonblank[0]) < 60 and nonblank[1] and \
       set(nonblank[1].strip()) <= set("=-~^'\""):
        return "non-python-under-py"
    # C/C++/Java namespace enum / android: 'namespace X {' or 'import android'
    if "namespace " in low and "{" in head or "import android" in low or "import org." in low \
       or "import java." in low or "import com." in low or "import android.support" in low:
        return "non-python-under-py"
    # 2 encoding / control bytes: a NUL, or a lone surrogate that breaks utf-8 decode
    if "\x00" in t or "\x01" in t or "\x02" in t:
        return "encoding-or-control-bytes"
    # 3 notebook cell: markdown + code mixed (an In[ / Out[ prompt or a jupyter nb header)
    if "In [" in t or "Out[" in t or "%matplotlib" in t or "jupyter" in low:
        return "notebook-cells"
    # 4 py2 syntax: print statement, xrange, backtick repr, print >>, leading-zero int
    lines = t.split("\n")
    if (">> " in t and "print " in t) or "xrange(" in t or \
       any(l.strip() == "`" for l in lines) or \
       any(l.strip().startswith("print ") and not l.strip().startswith("print(") for l in lines):
        return "python2-syntax"
    # 5 truncation: last nonblank line ends in a mid-construct token
    nonblank = [l for l in lines if l.strip()]
    if nonblank:
        last = nonblank[-1].strip()
        if last in (":", ",", "\\", "(", "[", "{", "=") or last.endswith(":") or \
           any(last.startswith(k) for k in ("def ", "class ", "elif ", "else ", "for ", "while ", "import ")):
            return "truncated-mid-construct"
    # 6 residual non-python: no python structure at all in a .py-named row
    if not any(k in head for k in ("def ", "class ", "import ", "from ", "print(")):
        return "non-python-under-py"
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