#!/usr/bin/env python3
"""A tokenizer harness: compression, distribution, structure, and integrity.

chars/token is the number this repo optimised and it is a weak proxy. "Beyond Text
Compression" (arXiv 2506.03101) reports its correlation with downstream
performance swinging from rho=-0.77 on translation to rho=-0.09 on summarisation,
and finds **deviation from a Zipfian power law** the strongest cheap predictor.
TokEval (arXiv 2608.18062) adds that information-theoretic metrics predict
language-modelling ability at rho up to 0.80, and that **structure-sensitive**
checks -- digit place-value alignment, line-break handling -- correlate with task
accuracy. Cognetta et al. (2024) and Dagan et al. (2024) give counterexamples
where raising Renyi efficiency *hurts*, so it is reported and never optimised.

Small models expose tokenizer differences most sharply: arXiv 2506.03101 has a
350M model with a good tokenizer beating a 2.7B model with a bad one. This model
is 200M, which is exactly where this matters.

Four groups, in increasing order of how much they should be trusted:

  COMPRESSION  cheap, weak, task-dependent
  DISTRIBUTION Zipf deviation, utilisation, undertrained tokens
  STRUCTURE    digit place-value, UTF-8 boundaries, whitespace, round-trip
  EXTRINSIC    two training runs differing only in the vocabulary -- the only
               one that settles anything, and the only one this cannot do

    python scripts/tokenizer_report.py
    python scripts/tokenizer_report.py --tokenizer new.json --compare old.json
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
DEFAULT_DOMAINS = "web_hq,textbook,wiki,math,chat,code,en"


# ---------------------------------------------------------------- corpus
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
        if rows:
            out[d] = rows
    return out


# ---------------------------------------------------------------- distribution
def zipf_deviation(counts):
    """RMS deviation of log-frequency from the fitted Zipf line, and its slope.

    arXiv 2506.03101 reports this as the single most informative cheap predictor
    of downstream performance. Natural language follows a power law; a vocabulary
    whose token frequencies depart from one is spending slots badly."""
    freqs = sorted((f for f in counts.values() if f > 0), reverse=True)
    if len(freqs) < 50:
        return float("nan"), float("nan")
    n = len(freqs)
    xs = [math.log(i + 1) for i in range(n)]
    ys = [math.log(f) for f in freqs]
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum((x - mx) ** 2 for x in xs)
    b = my - slope * mx
    rms = math.sqrt(sum((y - (slope * x + b)) ** 2 for x, y in zip(xs, ys, strict=True)) / n)
    return rms, slope


def renyi_efficiency(counts, alpha=2.5):
    """Reported WITH its counterexamples: Cognetta et al. (2024) raise it and hurt
    downstream; Dagan et al. (2024) find it anti-correlates with code generation."""
    tot = sum(counts.values())
    ps = [c / tot for c in counts.values() if c]
    if not ps:
        return float("nan")
    return (math.log(sum(p**alpha for p in ps)) / (1 - alpha)) / math.log(len(ps))


# ---------------------------------------------------------------- structure
def digit_consistency(tok):
    """Does the same number tokenise the same way in different contexts?

    BPE merges digits by frequency, so ' 63' and '63' can be different token
    sequences and '1640' can become '16|40', which has nothing to do with its
    value. Every arithmetic result in the literature assumes the model can see
    place value; this measures whether it can.
    """
    ctxs = ["{n}", " {n}", "= {n}", "共{n}个", "第{n}章", "({n})"]
    nums = ["7", "63", "122", "1640", "2024", "10000"]
    inconsistent, splits = 0, []
    for n in nums:
        seen = set()
        for c in ctxs:
            ids = tuple(tok.encode(c.format(n=n), add_special_tokens=False).tokens)
            core = tuple(t for t in ids if any(ch.isdigit() for ch in t))
            seen.add(core)
        if len(seen) > 1:
            inconsistent += 1
        splits.append((n, len(tok.encode(n, add_special_tokens=False).ids)))
    place_aligned = sum(1 for n, k in splits if k == len(n) or k == 1)
    return {
        "numbers tested": len(nums),
        "context-inconsistent": inconsistent,
        "place-value aligned": f"{place_aligned}/{len(nums)}",
        "splits": splits,
    }


def utf8_integrity(tok, corpus):
    """Fraction of hanzi that survive as part of a whole-character token rather
    than being emitted as ByteLevel fragments.

    CLAUDE.md records this exact failure: a vocabulary trained on unstratified
    text had no whole token for common traditional characters, split them into
    byte pieces, and scored web at 1.04 chars/token -- worse than one token per
    character."""
    rows = [r for v in corpus.values() for r in v][:600]
    frag = whole = 0
    for e in tok.encode_batch(rows):
        for t in e.tokens:
            s = t.replace("Ġ", "")
            if not s:
                continue
            # a ByteLevel fragment of a hanzi decodes to mojibake, not to a hanzi
            if HAN.search(s):
                whole += len(HAN.findall(s))
            elif len(s) <= 2 and all(ord(ch) > 127 for ch in s):
                frag += 1
    tot = whole + frag
    return {"hanzi in whole-char tokens": f"{100 * whole / max(1, tot):.2f}%", "byte fragments": frag}


def roundtrip(tok, corpus):
    """encode -> decode must return the input. A vocabulary trained without the
    full 256-byte alphabet silently drops bytes; CLAUDE.md records NUL being lost
    that way, which breaks every fast tokenizer library."""
    rows = [r for v in corpus.values() for r in v][:400]
    extra = ["NUL\x00byte", "emoji 🚀 ok", "tab\tnewline\n", "混合 mixed 123", "  双空格  "]
    bad = []
    for s in rows + extra:
        if tok.decode(tok.encode(s, add_special_tokens=False).ids) != s:
            bad.append(s[:40])
    return {"lossless": f"{len(rows) + len(extra) - len(bad)}/{len(rows) + len(extra)}", "failures": bad[:3]}


WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

#: Words whose morpheme boundaries are not in doubt. A vocabulary that cuts inside
#: a morpheme forces the model to relearn the same prefix or suffix in every word
#: that carries it; one that cuts AT the boundary shares it.
MORPH = [
    ("unhappiness", ["un", "happi", "ness"]),
    ("rebuilding", ["re", "build", "ing"]),
    ("nationalise", ["nation", "al", "ise"]),
    ("teacher", ["teach", "er"]),
    ("disagreement", ["dis", "agree", "ment"]),
    ("faster", ["fast", "er"]),
]


def english_metrics(tok, corpus):
    """Fertility and word-splitting, which are the standard measures for an
    alphabetic script and are meaningless for Chinese -- Chinese has no word
    boundary to divide by, so chars/token stands in for it there.

    This matters here beyond the `en` domain: every multiple-choice benchmark in
    eval/ except C-Eval is English, so English tokenisation quality shows up
    directly in the MC suite.
    """
    rows = corpus.get("en") or [r for v in corpus.values() for r in v]
    rows = rows[:400]
    words = n_tok = split = 0
    for r in rows:
        for w in WORD.findall(r):
            k = len(tok.encode(" " + w, add_special_tokens=False).ids)
            words += 1
            n_tok += k
            split += k > 1
    if not words:
        return None
    out = {
        "fertility (tokens/word)": n_tok / words,
        "words split (>1 token)": f"{100 * split / words:.1f}%",
    }
    # case and leading-space consistency: "the" and " the" and "The" should not be
    # three unrelated things
    incons = 0
    for w in ["the", "model", "number", "answer", "question"]:
        forms = {tuple(tok.encode(f, add_special_tokens=False).tokens) for f in (w, " " + w)}
        incons += len(forms) > 1
    out["leading-space inconsistent"] = f"{incons}/5"
    # morphology
    hit = 0
    for w, morphs in MORPH:
        toks = [t.replace("Ġ", "") for t in tok.encode(" " + w, add_special_tokens=False).tokens]
        cuts, pos = set(), 0
        for t in toks:
            pos += len(t)
            cuts.add(pos)
        true, pos = set(), 0
        for m in morphs:
            pos += len(m)
            true.add(pos)
        hit += len(cuts & true) > 1 or toks == [w]
    out["morpheme-aligned"] = f"{hit}/{len(MORPH)}"
    return out


def parity(tok, corpus):
    """Bytes per token in each domain, relative to the best domain.

    A vocabulary that serves one language far better than another spends the
    model's context budget unevenly. Perfect parity is 1.00 everywhere."""
    per = {}
    for dom, rows in corpus.items():
        encs = tok.encode_batch(rows)
        b = sum(len(r.encode()) for r in rows)
        t = sum(len(e.ids) for e in encs)
        per[dom] = b / t
    best = max(per.values())
    return {d: v / best for d, v in sorted(per.items(), key=lambda kv: -kv[1])}


def whitespace_handling(tok):
    """Line breaks and indentation, which TokEval finds correlate with task
    accuracy -- a vocabulary that spends a token per newline wastes budget on
    every structured document."""
    out = {}
    for s in ["\n", "\n\n", "    ", "\n    ", "。\n"]:
        out[repr(s)] = len(tok.encode(s, add_special_tokens=False).ids)
    return out


# ---------------------------------------------------------------- report
def report(tok, corpus, name):
    print(f"\n{'=' * 62}\n{name}   vocab {tok.get_vocab_size()}")

    print("\n-- COMPRESSION  (cheap, weak, task-dependent: rho -0.77 .. -0.09)")
    counts = collections.Counter()
    tot_c = tot_t = tot_b = 0
    print(f"   {'domain':<10}{'chars/tok':>10}{'bytes/tok':>10}")
    for dom, rows in corpus.items():
        encs = tok.encode_batch(rows)
        c = sum(len(r) for r in rows)
        b = sum(len(r.encode()) for r in rows)
        t = sum(len(e.ids) for e in encs)
        for e in encs:
            counts.update(e.ids)
        tot_c, tot_t, tot_b = tot_c + c, tot_t + t, tot_b + b
        print(f"   {dom:<10}{c / t:>10.3f}{b / t:>10.3f}")
    print(f"   {'ALL':<10}{tot_c / tot_t:>10.3f}{tot_b / tot_t:>10.3f}")

    print("\n-- DISTRIBUTION")
    v = tok.get_vocab()
    rms, slope = zipf_deviation(counts)
    once = sum(1 for c in counts.values() if c <= 1)
    print(f"   Zipf deviation (RMS)   {rms:>8.4f}   slope {slope:.3f}   [lower better; best cheap predictor]")
    print(
        f"   vocabulary utilised    {100 * len(counts) / len(v):>7.1f}%   ({len(v) - len(counts)} slots never used)"
    )
    print(f"   undertrained (<=1 use) {once:>8}   [glitch-token risk]")
    print(f"   Renyi efficiency       {renyi_efficiency(counts):>8.4f}   [reported, has counterexamples]")

    print("\n-- STRUCTURE")
    d = digit_consistency(tok)
    print(f"   digits: {d['context-inconsistent']}/{d['numbers tested']} tokenise DIFFERENTLY by context")
    print(f"   digits: place-value aligned {d['place-value aligned']}")
    for n, k in d["splits"]:
        print(f"      {n:>6} -> {k} token(s)")
    u = utf8_integrity(tok, corpus)
    print(
        f"   hanzi in whole-char tokens {u['hanzi in whole-char tokens']}  (byte fragments {u['byte fragments']})"
    )
    r = roundtrip(tok, corpus)
    print(
        f"   round-trip lossless {r['lossless']}" + (f"  FAILURES {r['failures']}" if r["failures"] else "")
    )
    print(f"   whitespace {whitespace_handling(tok)}")

    print("\n-- ENGLISH  (fertility is the alphabetic-script analogue of chars/token;")
    print("             every MC benchmark in eval/ except C-Eval is English)")
    em = english_metrics(tok, corpus)
    if em:
        for k, val in em.items():
            print(f"   {k:<28}{val if isinstance(val, str) else format(val, '.3f')}")

    par = parity(tok, corpus)
    print("\n-- PARITY  (bytes/token relative to the best-served domain; 1.00 is even)")
    for dd, vv in par.items():
        print(f"   {dd:<10}{vv:>7.3f}")

    return {
        "chars/tok": tot_c / tot_t,
        "zipf_dev": rms,
        "utilised": len(counts) / len(v),
        "undertrained": once,
        "digit_inconsistent": d["context-inconsistent"],
        "en fertility": (em or {}).get("fertility (tokens/word)", float("nan")),
        "parity spread": max(par.values()) - min(par.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--compare")
    ap.add_argument("--domains", default=DEFAULT_DOMAINS)
    a = ap.parse_args()

    from tokenizers import Tokenizer

    corpus = sample_corpus([d for d in a.domains.split(",") if d])
    if not corpus:
        sys.exit("no shards under data/corpus/ -- nothing to measure on")
    print(f"{sum(len(v) for v in corpus.values())} documents from {len(corpus)} domains")

    x = report(Tokenizer.from_file(a.tokenizer), corpus, os.path.basename(a.tokenizer))
    if a.compare:
        y = report(Tokenizer.from_file(a.compare), corpus, os.path.basename(a.compare))
        print(
            f"\n{'=' * 62}\nDIFFERENCE  ({os.path.basename(a.compare)} minus {os.path.basename(a.tokenizer)})"
        )
        better = {
            "chars/tok": +1,
            "zipf_dev": -1,
            "utilised": +1,
            "undertrained": -1,
            "digit_inconsistent": -1,
            "en fertility": -1,
            "parity spread": -1,
        }
        for k in x:
            d = y[k] - x[k]
            mark = "" if abs(d) < 1e-9 else ("  better" if d * better[k] > 0 else "  worse")
            print(f"   {k:<20}{d:+.4f}{mark}")
    print(
        "\nNone of the above settles it. Compression predicts downstream performance\n"
        "only on some task types and Renyi has documented counterexamples. The\n"
        "EXTRINSIC test -- two runs identical except for the vocabulary -- does."
    )


if __name__ == "__main__":
    main()
