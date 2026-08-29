#!/usr/bin/env python3
"""Rank vocabularies across dimensions at once, with the weights written down.

`tokenizer_report.py` prints the numbers; this file combines them. Correctness is
a gate, not a scored metric -- k5's vocabulary dropped NUL and tab, and any
aggregate trading correctness against compression would have let it win.
Correlated metrics are averaged within a dimension first, so adding another
compression metric cannot outvote another dimension.

Gaps, so they are not mistaken for zeros: no Chinese word-boundary alignment
(needs jieba); bits/char is scored only on a size-matched field (see
tokenizer_sweep.py); every dimension is a proxy.

    python scripts/tokenizer_eval.py --tokenizers data/tokenizer.json,data/vocab_sweep/v16384.json
"""

import argparse
import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# CORRECTNESS is not here: it is a gate. Everything below is a preference.
DIMENSIONS = {
    "compression": 0.30,
    "predictability": 0.25,  # bits/char -- compression the model can actually use
    "distribution": 0.20,  # dead slots and glitch tokens are wasted parameters
    "structure": 0.15,
    "equity": 0.10,  # no domain starved relative to the best-served one
}


def gates(tok, corpus):
    """Disqualifiers: correctness properties, not preferences."""
    import tokenizer_report as R

    out = {}
    r = R.roundtrip(tok, corpus)
    out["round-trip lossless"] = bool(r["lossless"])
    # A vocabulary missing byte-alphabet entries silently drops those bytes and
    # breaks every fast tokenizer library. Measured at 193/256 on an early build.
    v = tok.get_vocab()
    have = sum(1 for b in range(256) if _byte_token(b) in v)
    out["all 256 bytes present"] = have == 256
    out["_bytes"] = have
    return out


def _byte_token(b):
    """ByteLevel's printable remapping of byte b (the GPT-2 table)."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for x in range(256):
        if x not in bs:
            bs.append(x)
            cs.append(256 + n)
            n += 1
    return chr(cs[bs.index(b)])


def robustness(tok, corpus):
    """Fraction of tokens that are byte fragments rather than whole characters.

    Missing whole-character hanzi tokens is what made web score 1.04 chars/token
    -- worse than one token per character -- which chars/token alone underreports."""
    frag = tot = 0
    for rows in corpus.values():
        for e in tok.encode_batch(rows):
            for t in e.tokens:
                tot += 1
                if len(t) == 1 and ord(t) >= 256:
                    frag += 1
    return {"byte-fragment tokens": frag / max(tot, 1)}


def cost(tok, d_model=1024):
    """Vocabulary size is compute, not just parameters: tying halves the params
    and none of the output matmul's per-forward FLOPs."""
    V = tok.get_vocab_size()
    return {"embed params (M)": V * d_model / 1e6, "output FLOPs/token (M)": 2 * V * d_model / 1e6}


def collect(path, corpus, train_rows, eval_rows, score_bits):
    from tokenizers import Tokenizer

    import tokenizer_report as R

    tok = Tokenizer.from_file(path)
    counts = collections.Counter()
    tot_c = tot_t = 0
    for rows in corpus.values():
        for e in tok.encode_batch(rows):
            counts.update(e.ids)
            tot_t += len(e.ids)
        tot_c += sum(len(r) for r in rows)

    rms, _ = R.zipf_deviation(counts)
    dg = R.digit_consistency(tok)
    par = R.parity(tok, corpus)
    m = {
        "chars/token": tot_c / tot_t,
        "zipf deviation": rms,
        "utilised": len(counts) / len(tok.get_vocab()),
        "undertrained frac": sum(1 for c in counts.values() if c <= 1) / len(tok.get_vocab()),
        "digit consistent": 1 - dg["context-inconsistent"] / dg["numbers tested"],
        "parity spread": max(par.values()) - min(par.values()),
    }
    m.update(robustness(tok, corpus))
    m.update(cost(tok))
    if score_bits:
        from tokenizer_sweep import bits_per_char

        m["bits/char"] = bits_per_char(tok, train_rows, eval_rows)["bits/char"]
    return tok, m, gates(tok, corpus)


# metric -> (dimension, higher_is_better)
METRICS = {
    "chars/token": ("compression", True),
    "byte-fragment tokens": ("compression", False),
    "bits/char": ("predictability", False),
    "zipf deviation": ("distribution", False),
    "utilised": ("distribution", True),
    "undertrained frac": ("distribution", False),
    "digit consistent": ("structure", True),
    "parity spread": ("equity", False),
}


# Below this relative spread a metric is a tie, not normalised: min-max stretches
# any range onto 0..1, so 93.39/93.17/93.43% utilisation (0.28% spread) scored
# 0.611/0.333/0.723 before this existed.
TIE_SPREAD = 0.01


def normalise(rows, key, higher_better):
    """Min-max across the field; relative, since chars/token has no absolute scale."""
    vals = [r[1][key] for r in rows if key in r[1]]
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    scale = max(abs(sum(vals) / len(vals)), 1e-12)
    if hi - lo < 1e-12 or (hi - lo) / scale < TIE_SPREAD:
        return {r[0]: 0.5 for r in rows if key in r[1]}
    return {
        r[0]: ((r[1][key] - lo) / (hi - lo)) if higher_better else (1 - (r[1][key] - lo) / (hi - lo))
        for r in rows
        if key in r[1]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers", required=True)
    ap.add_argument("--domains", default="web_hq,textbook,wiki,math,chat,code,en")
    ap.add_argument("--per_domain", type=int, default=400)
    ap.add_argument("--n_train", type=int, default=12000)
    a = ap.parse_args()

    import tokenizer_report as R

    doms = [d for d in a.domains.split(",") if d]
    corpus = R.sample_corpus(doms, a.per_domain)
    if not corpus:
        sys.exit("no corpus under data/corpus/")
    paths = [p.strip() for p in a.tokenizers.split(",") if p.strip()]

    from tokenizer_sweep import load_text

    train_rows, eval_rows = load_text(doms, a.n_train, 800)

    # bits/char cannot rank across sizes (tokenizer_sweep documents why), so it is
    # scored only when the field is size-matched. 5% tolerance: a few reserved
    # specials (32,773 vs 32,768) are not a size sweep; the artifact needs ~2x.
    from tokenizers import Tokenizer

    sizes = [Tokenizer.from_file(p).get_vocab_size() for p in paths]
    score_bits = (max(sizes) - min(sizes)) / max(sizes) < 0.05
    if not score_bits:
        print(f"! vocabulary sizes span {min(sizes)}..{max(sizes)} -- bits/char REPORTED, NOT SCORED")
        print("  (it is strictly monotone in size; see tokenizer_sweep.py)")

    rows = []
    for p in paths:
        tok, m, g = collect(p, corpus, train_rows, eval_rows, True)
        rows.append((os.path.basename(p), m, g, tok.get_vocab_size()))

    print(f"\n{'=' * 78}\nGATES  (a failure disqualifies; correctness is not traded against compression)")
    print(f"  {'vocabulary':<24}{'round-trip':>12}{'256 bytes':>12}{'verdict':>10}")
    ok = []
    for name, m, g, V in rows:
        passed = g["round-trip lossless"] and g["all 256 bytes present"]
        print(
            f"  {name:<24}{'pass' if g['round-trip lossless'] else 'LOSSY':>12}"
            f"{str(g['_bytes']) + '/256':>12}{'ok' if passed else 'DISQUALIFIED':>10}"
        )
        if passed:
            ok.append((name, m))
    if not ok:
        sys.exit("\nevery candidate failed a gate; nothing to rank")

    print(f"\n{'=' * 78}\nRAW")
    keys = [k for k in METRICS if any(k in m for _, m in ok)] + ["embed params (M)"]
    print(f"  {'vocabulary':<24}" + "".join(f"{k.split()[0][:11]:>12}" for k in keys))
    for name, m in ok:
        print(f"  {name:<24}" + "".join(f"{m.get(k, float('nan')):>12.4f}" for k in keys))

    norm = {k: normalise(ok, k, hb) for k, (dim, hb) in METRICS.items()}
    if not score_bits:
        norm["bits/char"] = {}

    print(f"\n{'=' * 78}\nSCORE  weights " + " ".join(f"{d}={w}" for d, w in DIMENSIONS.items()))
    print(f"  {'vocabulary':<24}" + "".join(f"{d[:11]:>13}" for d in DIMENSIONS) + f"{'TOTAL':>9}")
    ranked = []
    for name, m in ok:
        per = {}
        for dim in DIMENSIONS:
            ms = [k for k, (d, _) in METRICS.items() if d == dim and name in norm.get(k, {})]
            per[dim] = sum(norm[k][name] for k in ms) / len(ms) if ms else float("nan")
        live = {d: w for d, w in DIMENSIONS.items() if per[d] == per[d]}
        total = sum(per[d] * w for d, w in live.items()) / sum(live.values())
        ranked.append((total, name, per))
        print(
            f"  {name:<24}"
            + "".join(f"{per[d]:>13.3f}" if per[d] == per[d] else f"{'--':>13}" for d in DIMENSIONS)
            + f"{total:>9.3f}"
        )

    ranked.sort(reverse=True)
    print(f"\n  highest: {ranked[0][1]} ({ranked[0][0]:.3f})")
    print(
        "\n  The scalar is a summary of the row, not a verdict. Two vocabularies at\n"
        "  the same total can be opposite vocabularies -- read the dimensions. And\n"
        "  every column here is a proxy: the verdict is two pretrains differing\n"
        "  only in the vocabulary."
    )


def _demo():
    """The scoring scale is logic and has been wrong once."""
    rows = [("a", {"x": 0.9339}), ("b", {"x": 0.9317}), ("c", {"x": 0.9343})]
    n = normalise(rows, "x", True)
    assert set(n.values()) == {0.5}, f"0.28% spread must be a tie, got {n}"

    rows = [("a", {"x": 1.0}), ("b", {"x": 2.0})]
    n = normalise(rows, "x", True)
    assert n == {"a": 0.0, "b": 1.0}, n
    n = normalise(rows, "x", False)
    assert n == {"a": 1.0, "b": 0.0}, n

    # a spread just above the threshold must NOT collapse
    rows = [("a", {"x": 1.0}), ("b", {"x": 1.02})]
    assert set(normalise(rows, "x", True).values()) == {0.0, 1.0}

    assert abs(sum(DIMENSIONS.values()) - 1.0) < 1e-9, "weights must sum to 1"
    ts = [_byte_token(b) for b in range(256)]
    assert len(set(ts)) == 256, "byte table collides"
    print("tokenizer_eval self-test OK (tie threshold, direction, weights, byte table)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _demo()
    else:
        main()
