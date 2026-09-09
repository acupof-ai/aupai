#!/usr/bin/env python3
"""Tokenizer gates on a proxy of the p1 composition: ~3:1 prose:code by chars.

p1's final composition (docs/standards/p1_data_recipe.md, PR #154):
~20B synthetic textbooks + ~6B filtered code + ~0.18B exercises ~= 3.3:1 prose:code.
The synthetic textbooks do not exist yet, so the prose half is a proxy. TWO proxies
are measured because the recipe does not state the textbooks' language and the
answer flips on it:

  --prose-domain textbook       76% hanzi by chars (the existing synthetic tutorials);
                                reads as the Chinese-prose branch
  --prose-domain en_c4_stage2  English web; reads as the phi-1-shape English-prose branch

When b0 ships the first batch of real synthetic textbooks, re-run on them: 4c's
standing instruction is that if the proxy and the real sample disagree, the real
sample wins and the vocab is rebuilt against it.

The frozen vocabulary was fitted on a Chinese-web-heavy mix, so the two branches
are not both "the same measurement with noise": they are the two populations the
sizing decision is a bet between. never_used is the decision metric -- the hanzi
gate measures corpus composition, not vocab quality (tokenizer_report.py:144-154),
and on the English branch its 600-row sample catches whatever Chinese lives in
code comments, which is why its three seeds span 0.001..0.384.

    python scripts/tokenizer_p1_proxy.py --prose-domain en_c4_stage2
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import tokenizer_report as R
from tokenizer_eval import collect

TOK = os.path.join(ROOT, "data", "tokenizer.json")
SEEDS = (7, 13, 21)


def sample_domain(domain, want_chars, rng):
    fs = R.shard_paths(domain)
    if not fs:
        sys.exit("no shards for " + domain)
    rows, got = [], 0
    while got < want_chars:
        for f in rng.sample(fs, min(8, len(fs))):
            with open(f, encoding="utf-8") as fh:
                lines = fh.readlines()
            for x in rng.sample(lines, min(len(lines), 4000)):
                try:
                    c = json.loads(x).get("content", "")
                except Exception:
                    continue
                if c:
                    rows.append(c)
                    got += len(c)
                    if got >= want_chars:
                        return rows
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prose-domain", default="en_c4_stage2")
    ap.add_argument("--code-domains", default="code_py_starcoder,code_py_rp1t")
    ap.add_argument("--target-chars", type=int, default=30_000_000)
    ap.add_argument("--ratio", type=float, default=3.0, help="prose chars per code char")
    ap.add_argument("--tokenizer", default=TOK)
    a = ap.parse_args()

    code_want = a.target_chars / (a.ratio + 1)
    code_doms = [d for d in a.code_domains.split(",") if d]
    results = []
    for seed in SEEDS:
        rng = random.Random(seed)
        prose = sample_domain(a.prose_domain, a.target_chars - code_want, rng)
        # split the code half across the code domains, 95/5 like the pure-code fact
        code = []
        for i, d in enumerate(code_doms):
            share = code_want * (0.05 if i else 0.95)
            code += sample_domain(d, share, rng)
        pc, cc = sum(map(len, prose)), sum(map(len, code))
        mix = prose + code
        rng.shuffle(mix)  # so _even_rows' stride does not land on one domain
        tok, m, g = collect(a.tokenizer, {"p1_proxy": mix}, None, None, False)
        row = {"seed": seed, "ratio": round(pc / max(1, cc), 2),
               "prose_chars_M": round(pc / 1e6, 2), "code_chars_M": round(cc / 1e6, 2)}
        for k in ("chars/token", "ref fertility", "hanzi whole-char", "never used frac",
                  "en fertility", "utilised", "byte-fragment tokens"):
            if k in m:
                row[k] = round(m[k], 4)
        row["gates"] = g
        results.append(row)
        print(json.dumps(row), flush=True)

    nu = [r["never used frac"] for r in results]
    print("never_used across seeds: min %.4f max %.4f range %.4f"
          % (min(nu), max(nu), max(nu) - min(nu)))


if __name__ == "__main__":
    main()
