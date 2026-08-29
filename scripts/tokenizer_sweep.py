#!/usr/bin/env python3
"""Rank candidate vocabularies by bits per CHARACTER under an n-gram model.

Held-out text encoded under a model trained on the token stream a vocabulary
produces, divided by the CHARACTERS in it -- so it is comparable across
segmentations and rewards compression AND predictability at once, where
chars/token sees only the first (arXiv 2506.03101 puts chars/token's correlation
with downstream performance anywhere from rho=-0.77 to -0.09 by task). A proxy;
the verdict is two pretrains differing only in the vocabulary.

**IT CANNOT RANK VOCABULARY SIZE.** Measured 2026-08-29: a trigram over 32K types
has 8x the parameters of one over 16K, so at equal token counts it is
data-starved and smaller always wins -- an estimator artifact, not a property of
the vocabulary. Two signatures: the ordering is strictly monotone in size with no
interior optimum (16K/32K/49K/65K = 4.5947/4.6546/4.7042/4.7479 at n_train=30K,
where a real optimum would be U-shaped), and the 16K-vs-32K gap shrinks toward
zero with data (0.1070 -> 0.0896 -> 0.0599 at n_train 3K -> 12K -> 30K).

Use it only for decisions that HOLD SIZE FIXED. Digit splitting is one, and costs
-0.08% / +0.10% / +0.18% bits/char at those three sizes -- growing with data.

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
    """Held-out characters are identical for every candidate; only segmentation differs."""
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
    """Interpolated 3/2/1-gram over the token stream; returns bits per character.

    Weights are fixed, not tuned per candidate: tuning would let a vocabulary win
    by being easier to tune."""
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
