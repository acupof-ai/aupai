#!/usr/bin/env python3
"""Rank candidate vocabularies by bits per CHARACTER under an n-gram model.

Why this metric and not chars/token. Compression alone is a weak proxy -- arXiv
2506.03101 measures its correlation with downstream performance swinging from
rho=-0.77 to rho=-0.09 by task. Bits per character is different in kind: it is
the cost of encoding held-out text under a model trained on the token stream that
vocabulary produces, divided by the CHARACTERS in that text. So it is directly
comparable across vocabularies of different sizes and segmentations, and it
rewards two things at once --

    fewer tokens per character   (compression, what chars/token measures)
    more predictable tokens      (what chars/token cannot see)

A vocabulary that compresses well into an unpredictable stream and one that
compresses poorly into a trivial stream both score badly, which is the behaviour
wanted. An n-gram is enough: it is the same quantity a transformer minimises, at
a lower ceiling, and it runs on CPU in seconds.

This is a proxy, not a verdict. The verdict is two training runs differing only
in the vocabulary. But it is a proxy that can rank twenty candidates in minutes,
where the verdict costs a day each.

    python scripts/tokenizer_sweep.py --tokenizers data/tokenizer.json,data/tokenizer_k5.json
    python scripts/tokenizer_sweep.py --sweep       # train and rank variants
"""

import argparse
import collections
import glob
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS = "web_hq,textbook,wiki,math,chat,code,en"


def load_text(domains, n_train=3000, n_eval=800, seed=11):
    """Held-out characters are identical for every candidate; only the
    segmentation of them changes."""
    rng = random.Random(seed)
    rows = []
    for d in domains:
        fs = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", d, "*.jsonl")))
        if not fs:
            continue
        for f in rng.sample(fs, min(2, len(fs))):
            with open(f, encoding="utf-8") as fh:
                lines = fh.readlines()
            k = (n_train + n_eval) // (len(domains) * 2) + 1
            for x in rng.sample(lines, min(k, len(lines))):
                rows.append(json.loads(x).get("content", "")[:1500])
    rng.shuffle(rows)
    return rows[:n_train], rows[n_train : n_train + n_eval]


def bits_per_char(tok, train_rows, eval_rows, order=3, lam=(0.55, 0.30, 0.15)):
    """Interpolated n-gram over the token stream; returns bits per character.

    Linear interpolation of orders 3/2/1 with fixed weights, add-k on the
    unigram so nothing is zero. Fixed weights rather than tuned ones on purpose:
    tuning per candidate would let a vocabulary win by being easier to tune."""
    V = tok.get_vocab_size()
    c1 = collections.Counter()
    c2 = collections.Counter()
    c3 = collections.Counter()
    ctx2 = collections.Counter()
    ctx1 = collections.Counter()
    for e in tok.encode_batch(train_rows):
        ids = e.ids
        c1.update(ids)
        for i in range(1, len(ids)):
            c2[(ids[i - 1], ids[i])] += 1
            ctx1[ids[i - 1]] += 1
        for i in range(2, len(ids)):
            c3[(ids[i - 2], ids[i - 1], ids[i])] += 1
            ctx2[(ids[i - 2], ids[i - 1])] += 1
    n1 = sum(c1.values())
    k = 0.1

    total_bits = 0.0
    n_chars = sum(len(r) for r in eval_rows)
    n_toks = 0
    for e, raw in zip(tok.encode_batch(eval_rows), eval_rows, strict=True):
        ids = e.ids
        n_toks += len(ids)
        for i, t in enumerate(ids):
            p1 = (c1[t] + k) / (n1 + k * V)
            p2 = c2[(ids[i - 1], t)] / ctx1[ids[i - 1]] if i >= 1 and ctx1[ids[i - 1]] else 0.0
            p3 = 0.0
            if i >= 2:
                cc = ctx2[(ids[i - 2], ids[i - 1])]
                if cc:
                    p3 = c3[(ids[i - 2], ids[i - 1], t)] / cc
            p = lam[0] * p3 + lam[1] * p2 + lam[2] * p1
            total_bits += -math.log2(max(p, 1e-12))
    return {
        "bits/char": total_bits / n_chars,
        "bits/token": total_bits / n_toks,
        "chars/token": n_chars / n_toks,
        "tokens": n_toks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers", default=os.path.join(ROOT, "data", "tokenizer.json"))
    ap.add_argument("--domains", default=DOMAINS)
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_eval", type=int, default=800)
    a = ap.parse_args()

    from tokenizers import Tokenizer

    doms = [d for d in a.domains.split(",") if d]
    train, ev = load_text(doms, a.n_train, a.n_eval)
    if not ev:
        sys.exit("no corpus found under data/corpus/")
    print(f"train {len(train)} docs, held-out {len(ev)} docs / {sum(len(r) for r in ev):,} chars")
    print(f"\n{'vocabulary':<28}{'vocab':>7}{'chars/tok':>11}{'bits/tok':>10}{'BITS/CHAR':>11}")
    rows = []
    for path in [p.strip() for p in a.tokenizers.split(",") if p.strip()]:
        tok = Tokenizer.from_file(path)
        m = bits_per_char(tok, train, ev)
        rows.append((os.path.basename(path), tok.get_vocab_size(), m))
        print(
            f"{os.path.basename(path):<28}{tok.get_vocab_size():>7}"
            f"{m['chars/token']:>11.3f}{m['bits/token']:>10.3f}{m['bits/char']:>11.4f}"
        )
    if len(rows) > 1:
        best = min(rows, key=lambda r: r[2]["bits/char"])
        print(f"\nlowest bits/char: {best[0]}")
        print(
            "  bits/char is comparable across vocabularies because the denominator is\n"
            "  characters, which the segmentation cannot change. chars/token alone can\n"
            "  prefer a vocabulary that packs more into each token and makes the stream\n"
            "  harder to predict; this cannot."
        )


if __name__ == "__main__":
    main()
