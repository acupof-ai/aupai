#!/usr/bin/env python3
"""Measure a vocabulary on more than compression.

chars/token is the number this repo has been optimising, and the literature is
clear that it is a weak proxy: "Beyond Text Compression" (arXiv 2506.03101) finds
its correlation with downstream performance swings from rho=-0.77 on translation
to rho=-0.09 on summarisation, and reports that **deviation from a Zipfian power
law** is the single most informative cheap predictor. Cognetta et al. (2024) give
counterexamples where raising Renyi efficiency *hurts*, and Dagan et al. (2024)
find higher Renyi entropy correlates with worse code generation.

The same paper reports that small models expose tokenizer differences most
sharply -- a 350M model with a good tokenizer beat a 2.7B model with a bad one.
This model is 200M.

So this prints the cheap metrics together, with their known reliability, and
never reduces them to one score. The only trustworthy comparison is still two
runs that differ in nothing but the vocabulary.

    python scripts/tokenizer_report.py --tokenizer data/tokenizer.json
    python scripts/tokenizer_report.py --tokenizer a.json --compare b.json
"""

import argparse
import collections
import glob
import json
import math
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAN = re.compile(r"[一-鿿]")
DIGIT = re.compile(r"\d")


def sample_corpus(domains, per_domain=400, seed=7):
    rng = random.Random(seed)
    out = {}
    for d in domains:
        fs = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", d, "*.jsonl")))
        if not fs:
            continue
        rows = []
        for f in rng.sample(fs, min(3, len(fs))):
            with open(f, encoding="utf-8") as fh:
                lines = fh.readlines()
            for x in rng.sample(lines, min(per_domain // 3 + 1, len(lines))):
                rows.append(json.loads(x).get("content", "")[:2000])
        out[d] = rows
    return out


def zipf_deviation(counts):
    """RMS deviation of log-frequency from the Zipf line, over the ranked tokens.

    Reported by arXiv 2506.03101 as the strongest cheap predictor of downstream
    performance. Lower is closer to the power law natural language follows.
    """
    freqs = sorted(counts.values(), reverse=True)
    freqs = [f for f in freqs if f > 0]
    if len(freqs) < 50:
        return float("nan")
    n = len(freqs)
    xs = [math.log(i + 1) for i in range(n)]
    ys = [math.log(f) for f in freqs]
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sxy / sxx
    b = my - slope * mx
    rms = math.sqrt(sum((y - (slope * x + b)) ** 2 for x, y in zip(xs, ys, strict=True)) / n)
    return rms, slope


def renyi_efficiency(counts, alpha=2.5):
    """Renyi entropy over token frequencies, normalised by log|V|.

    Included WITH the caveat: Cognetta et al. (2024) construct cases where raising
    this hurts downstream, and Dagan et al. (2024) find it anti-correlates with
    code generation. It is reported, not optimised.
    """
    tot = sum(counts.values())
    if not tot:
        return float("nan")
    ps = [c / tot for c in counts.values() if c]
    h = math.log(sum(p**alpha for p in ps)) / (1 - alpha)
    return h / math.log(len(ps))


def report(tok, corpus, name):
    from tokenizers import Tokenizer  # noqa: F401

    print(f"\n=== {name}  (vocab {tok.get_vocab_size()})")
    counts = collections.Counter()
    tot_c = tot_t = tot_han = 0
    print(f"  {'domain':<10}{'chars/tok':>10}{'han/tok':>9}")
    for dom, rows in corpus.items():
        encs = tok.encode_batch(rows)
        c = sum(len(r) for r in rows)
        t = sum(len(e.ids) for e in encs)
        han = sum(len(HAN.findall(r)) for r in rows)
        for e in encs:
            counts.update(e.ids)
        tot_c += c
        tot_t += t
        tot_han += han
        print(f"  {dom:<10}{c / t:>10.3f}{han / t:>9.3f}")
    print(f"  {'ALL':<10}{tot_c / tot_t:>10.3f}{tot_han / tot_t:>9.3f}")

    v = tok.get_vocab()
    used = len(counts)
    print(f"\n  vocabulary actually used : {used}/{len(v)} = {100 * used / len(v):.1f}%")
    print(f"  tokens seen once or less : {sum(1 for c in counts.values() if c <= 1)}")
    rms, slope = zipf_deviation(counts)
    print(
        f"  Zipf deviation (RMS)     : {rms:.4f}   slope {slope:.3f}   [lower is better; best cheap predictor]"
    )
    print(f"  Renyi efficiency (a=2.5) : {renyi_efficiency(counts):.4f}   [reported, NOT to be optimised]")

    # where the slots went
    digits = [s for s in v if s.strip("Ġ").isdigit()]
    han_only = [s for s in v if s.strip("Ġ") and all(HAN.match(ch) for ch in s.strip("Ġ"))]
    multi_digit = [s for s in digits if len(s.strip("Ġ")) > 1]
    print(f"\n  slots: {len(han_only)} pure-hanzi, {len(digits)} digit ({len(multi_digit)} multi-digit)")

    # the number question, concretely
    print("  same number, different contexts:")
    for ctx in ["63", " 63", "= 63", "共63个"]:
        print(f"    {ctx!r:>10} -> {tok.encode(ctx, add_special_tokens=False).tokens}")
    return {"chars_per_tok": tot_c / tot_t, "zipf": rms, "used": used / len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--compare", help="second tokenizer to print alongside")
    ap.add_argument("--domains", default="web_hq,textbook,wiki,math,chat,code,en")
    a = ap.parse_args()

    from tokenizers import Tokenizer

    corpus = sample_corpus([d for d in a.domains.split(",") if d])
    if not corpus:
        sys.exit("no corpus shards found under data/corpus/")
    print(f"sampled {sum(len(v) for v in corpus.values())} documents from {len(corpus)} domains")
    a_stats = report(Tokenizer.from_file(a.tokenizer), corpus, os.path.basename(a.tokenizer))
    if a.compare:
        b_stats = report(Tokenizer.from_file(a.compare), corpus, os.path.basename(a.compare))
        print("\n=== difference (compare - tokenizer)")
        for k in a_stats:
            print(f"  {k:<16}{b_stats[k] - a_stats[k]:+.4f}")
        print(
            "\n  None of these settles it. Compression correlates with downstream\n"
            "  performance only on some task types, and Renyi has documented\n"
            "  counterexamples. Two training runs differing only in the vocabulary do."
        )


if __name__ == "__main__":
    main()
