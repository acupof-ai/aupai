#!/usr/bin/env python3
"""Rank candidate vocabularies by bits per CHARACTER under an n-gram model.

Held-out text encoded under a model trained on the token stream a vocabulary produces, divided by
the CHARACTERS in it -- comparable across segmentations, rewarding compression AND predictability
at once, where chars/token sees only the first. A proxy; the verdict is two pretrains differing
only in the vocabulary.

**IT CANNOT RANK VOCABULARY SIZE.** A trigram over 32K types has 8x the parameters of one over
16K, so at equal token counts it is data-starved and smaller always wins -- an estimator artifact,
not a property of the vocabulary. Signatures: the ordering is strictly monotone in size with no
interior optimum, and the size gap shrinks toward zero with more data.

Use it only for decisions that HOLD SIZE FIXED. Digit splitting is one.

    python scripts/tokenizer_sweep.py --tokenizers data/tokenizer.json,data/tokenizer_k5.json

`--sweep` stood here until 2026-09-08 and the parser has never had it: this script scores
the tokenizers named by --tokenizers and has no train-and-rank mode. doc_flags_parse caught
it.
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
    """Held-out characters are identical for every candidate; only segmentation differs.

    REFUSES a short held-out set rather than returning one. Until 2026-09-08 a corpus dir's
    non-shard artifacts were sampled as if they were data: `holdout_slice_<domain>.jsonl` is
    one row of `{phase, rule_fp, n}` with no `content` key, and `.get("content", "")` made it
    an empty string instead of an error. Because this function takes 2 files per domain, a
    small domain draws the slice often, and the shortfall lands on the TAIL of the shuffled
    list -- which is `eval_rows`. Measured on the pod: the v2 composition got 0 of 800 eval
    rows and bits/char died on ZeroDivisionError; the resume-1 composition got 105 of 800 and
    reported a number. tokenizer_report.shard_paths is the fix; this refusal is the check that
    the next such artifact cannot quietly shrink the held-out set again.
    """
    from tokenizer_report import shard_paths

    rng = random.Random(seed)
    rows = []
    for d in domains:
        fs = shard_paths(d)
        if not fs:
            continue
        for f in rng.sample(fs, min(2, len(fs))):
            with open(f, encoding="utf-8") as fh:
                lines = fh.readlines()
            k = (n_train + n_eval) // (len(domains) * 2) + 1
            for x in rng.sample(lines, min(k, len(lines))):
                rows.append(json.loads(x).get("content", "")[:1500])
    rng.shuffle(rows)
    train, ev = rows[:n_train], rows[n_train : n_train + n_eval]
    if len(ev) < n_eval:
        raise SystemExit(
            f"REFUSING: {len(rows)} rows over {len(domains)} domains gives {len(ev)} held-out "
            f"rows, not {n_eval}. bits/char divides by the characters in this set, so a short "
            f"one is a quieter version of the ZeroDivisionError at zero. Each domain yields at "
            f"most 2 x {(n_train + n_eval) // (len(domains) * 2) + 1} rows: either raise "
            f"--n_train's domain count, lower n_train, or find the domain that came up short."
        )
    return train, ev


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


def _demo():
    """load_text's two properties. This file had NO selftest until 2026-09-08, which is why
    the non-shard artifact reached the held-out set for as long as it did: three mutations of
    tokenizer_report's samplers went red on its selftest and the two aimed at THIS file
    survived, because nothing here ran."""
    import tempfile

    _root = ROOT
    try:
        with tempfile.TemporaryDirectory() as td:
            # FOUR domains, each with exactly ONE real shard beside its artifact. Both
            # numbers are forced by load_text's own arithmetic, not chosen for looks:
            #   - 2 files per dir, so `rng.sample(fs, min(2, len(fs)))` draws BOTH every
            #     time. With a third file the artifact is drawn only 2/3 of the time and
            #     the case would pass on the broken version by luck.
            #   - k = (n_train + n_eval) // (2 * D) + 1 = 2 at D=4 and 6+2 rows, so four
            #     domains supply exactly the 8 rows asked for. Fewer domains cannot: the
            #     formula hands each one about half of what a 2-shard domain would give.
            doms = [f"dd{i}" for i in range(4)]
            for dom in doms:
                d = os.path.join(td, "data", "corpus", dom)
                os.makedirs(d)
                with open(os.path.join(d, f"{dom}_000.jsonl"), "w", encoding="utf-8") as fh:
                    for j in range(10):
                        fh.write(json.dumps({"content": f"{dom} row {j}\n"}) + "\n")
                with open(os.path.join(d, f"holdout_slice_{dom}.jsonl"), "w", encoding="utf-8") as fh:
                    fh.write(json.dumps({"phase": dom, "rule_fp": "0" * 16, "n": "0"}) + "\n")
            globals()["ROOT"] = td
            sys.modules.pop("tokenizer_report", None)
            import tokenizer_report as R

            R.ROOT = td

            # 1. The artifact is not in the held-out set, and the request is satisfiable, so
            #    a short return would mean a file was misread rather than that the fixture
            #    is too small.
            train, ev = load_text(doms, 6, 2)
            assert len(train) == 6 and len(ev) == 2, (len(train), len(ev))
            assert not any(r == "" for r in train + ev), (
                f"{sum(1 for r in train + ev if r == '')} empty rows: the non-shard artifact "
                f"is still being sampled"
            )
            # NEGATIVE CONTROL: the same world read the old way DOES yield empty rows.
            # Without this the assertion above would also hold on a fixture containing no
            # defect at all.
            old = sorted(glob.glob(os.path.join(td, "data", "corpus", "*", "*.jsonl")))
            empties = sum(
                1 for f in old
                if json.loads(next(open(f, encoding="utf-8"))).get("content", "") == ""
            )
            assert len(old) == 8 and empties == 4, (
                f"the fixture does not contain the defect ({empties} of {len(old)} files "
                f"contentless); this case would prove nothing"
            )

            # 2. A short held-out set REFUSES rather than returning quietly. 208 rows cannot
            #    come out of 40, and the shortfall lands on `ev` -- the silent form of the
            #    ZeroDivisionError the v2 composition hit.
            try:
                load_text(doms, 200, 8)
            except SystemExit as e:
                assert "held-out" in str(e), e
            else:
                raise AssertionError("load_text returned a short held-out set instead of refusing")
    finally:
        globals()["ROOT"] = _root
        sys.modules.pop("tokenizer_report", None)

    print("tokenizer_sweep self-test OK (shard selection with a negative control, short held-out refuses)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _demo()
    else:
        main()
