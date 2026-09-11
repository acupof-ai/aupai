#!/usr/bin/env python3
"""Rank vocabularies across dimensions at once, with the weights written down.

`tokenizer_report.py` prints the numbers; this file combines them. Correctness is a gate, not a
scored metric -- an aggregate trading correctness against compression would let a lossy vocabulary
win. Correlated metrics are averaged within a dimension first, so adding another compression metric
cannot outvote another dimension.

Gaps, so they are not mistaken for zeros: no Chinese word-boundary alignment (needs jieba);
bits/char is scored only on a size-matched field (see tokenizer_sweep.py); every dimension is a proxy.

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


# Hard thresholds, chosen by what is EXPENSIVE TO CHANGE LATER: a vocabulary is used for the
# life of every checkpoint trained on it, and ids do not survive a rebuild.
#
# TokEval (arXiv 2608.18062): intrinsic metrics SCREEN but do not RANK, so this file gates on
# correctness and the two failures no training can repair, and leaves ranking to the weighted
# dimensions and, finally, to two pretrains differing only in the vocabulary.
#
# The corpus and the normalisation unit are part of every metric's definition -- always gate on
# the full mix, not a slice.
GATES = {
    # name: (threshold, higher_is_better, why it is a veto and not a preference)
    # On REF_EN, not on whatever English the corpus sample holds: the same vocabulary reads
    # 1.429 on REF_EN and 1.870 on our own `en` domain.
    # 1.55 is the price of bilingual-at-32K -- the bilingual frontier (DeepSeek-V3 1.104,
    # Qwen3/GLM-4.5 1.130) buys 1.13 with 128K-200K slots, and the fitted scaling law puts our
    # optimum at 12-20K. A ceiling that must not DRIFT, not a defect to fix.
    "ref fertility": (1.55, False, "regression guard: English must not get worse than it is"),
    # hanzi whole-char is a veto ONLY for a corpus that contains Chinese: see
    # corpus_has_hanzi() and the 2026-09-10 V4.1 pivot (zero Chinese domains). On an
    # English/math/code mix the measured ratio is undefined-as-a-veto (incidental CJK in
    # comments is too sparse to be a real distribution) and the gate prints N/A rather than
    # failing -- scoped, not deleted: a corpus that actually carries hanzi still FAILs here.
    "hanzi whole-char": (0.95, True, "byte-fragmented hanzi is worse than one token per character"),
}

#: A corpus is "Chinese-bearing" only when a meaningful FRACTION of its documents carry
#: hanzi. An absolute count cannot discriminate: a 7-domain x 600-row English/code sample
#: finds ~20 docs with incidental CJK in comments/identifiers (measured on the gate mix:
#: per-domain hanzi-doc share 0.2%-3.8%, code highest), while real Chinese prose is ~100%.
#: The gate arms at a share well above incidental CJK and far below a real Chinese
#: distribution. Measured gate-mix max 0.038; threshold 0.10 gives 2.6x headroom.
HANZI_DOC_FRACTION = 0.10
_HANZI_SAMPLE_PER_DOMAIN = 600


def corpus_has_hanzi(corpus):
    """True iff >= HANZI_DOC_FRACTION of the evenly-sampled docs carry hanzi.

    Scopes the hanzi veto (3b, PR #233): on the V4.1 English/math/code mix the share is
    <=0.038 so this returns False and the hanzi gate is N/A
    (facts/tokenizer.json#tok.gate_tokenizer_choice_0910). On a corpus with real Chinese
    (share ~1.0) it returns True and byte-fragmented hanzi still FAILs. The discriminator
    is the fraction, not the count -- incidental CJK identifiers exist in code comments."""
    import tokenizer_report as R

    sample = R._even_rows(corpus, _HANZI_SAMPLE_PER_DOMAIN * max(1, len(corpus)))
    if not sample:
        return False
    with_hanzi = sum(1 for s in sample if R.HAN.search(s))
    return with_hanzi / len(sample) >= HANZI_DOC_FRACTION

#: REPORTED, NOT GATED, and the reason is a measurement rather than a preference.
#:
#: `never used frac` was a regression guard at 0.01 until 2026-09-08. It is not one now
#: because the reading is not decidable at that threshold: on the v2 composition, three
#: seeds at a FIXED (per_domain, shards) setting span
#:
#:     9000 x 16   0.0061 / 0.0103 / 0.0073   range 0.0042
#:     20000 x 24  0.0156 / 0.0046 / 0.0066   range 0.0110
#:
#: and varying only `shards` at a fixed seed spans 0.0043 / 0.0156 / 0.0057, range 0.0113.
#: Two of those three ranges exceed the whole 0.01 threshold, so the seed alone decides pass
#: or fail on the same corpus at the same setting. Fixing the setting as a constant does not
#: help -- neither axis dominates, and the larger setting is further from decidable than the
#: smaller one.
#:
#: THE THRESHOLD WAS SET INSIDE ITS OWN NOISE. 143f5d4a (2026-08-29) records "never used
#: 0.0070 <=0.01 ok (regression guard: 234K params, 0.1% of the model)" over "106 tokens with
#: zero occurrences in 142M" -- a deliberate guard 0.003 above a measured value, at
#: essentially the scale where the range measured above is 0.0110. One draw could not have
#: shown that, and the comment below it already knew the metric was scale-bound.
#:
#: THE METRIC IS SOUND; only the threshold is not. On strictly nested prefixes of one draw it
#: is monotone decreasing (0.5890 / 0.5705 / 0.0495 / 0.0156 at 24.9M / 38.5M / 71.1M /
#: 174.5M tokens), reproduced on a second draw by 3b. It stays in the RAW table, where a
#: reader can watch the fragment population across runs at a fixed setting without a
#: threshold asserting a precision the sampling does not have.
#:
#: NEVER used, not "<=1 use": the <=1 rate is a function of how much text you counted (4.0%
#: at 1.6M tokens, 0.43% at 142M, same vocabulary). Zero occurrences in 142M tokens of the
#: training distribution is a glitch token by the Fishing-for-Magikarp definition.
#:
#: facts/tokenizer.json#tok.never_used_not_decidable carries the data. To restore the gate,
#: find a setting whose three-seed range is under a third of the threshold and record it
#: there; do not re-add the entry on a single passing draw.
REPORTED_NOT_GATED = {
    "never used frac": "unreachable slots; reported, not gated -- see the note above",
}



def gates(tok, corpus):
    """Disqualifiers: correctness first, then the two properties a rebuild is the only
    remedy for. Everything else is a preference and is scored, not gated."""
    import tokenizer_report as R

    out = {}
    r = R.roundtrip(tok, corpus)
    out["round-trip lossless"] = bool(r["lossless"])
    # A vocabulary missing byte-alphabet entries silently drops those bytes and breaks
    # every fast tokenizer library.
    v = tok.get_vocab()
    have = sum(1 for b in range(256) if _byte_token(b) in v)
    out["all 256 bytes present"] = have == 256
    out["_bytes"] = have
    return out


def threshold_gates(metrics, hanzi_applies=True):
    """(name, value, threshold, ok) for each GATES entry present in `metrics`.

    A name in REPORTED_NOT_GATED is NOT here by construction, whatever its value: it is
    printed from the RAW table and never reaches `failed`. The hanzi veto is skipped
    (returned as None-ok via the caller's N/A path) when the corpus has no Chinese --
    hanzi_applies=False from corpus_has_hanzi(); the bilingual guard stays armed on any
    corpus that does."""
    rows = []
    for k, (thr, higher, _why) in GATES.items():
        if k not in metrics:
            continue
        if k == "hanzi whole-char" and not hanzi_applies:
            continue
        val = metrics[k]
        rows.append((k, val, thr, (val >= thr) if higher else (val <= thr)))
    return rows


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


def never_used(tok, counts):
    """Fraction of the vocabulary that no corpus text reaches, EXCLUDING the entries that
    are unreachable by design.

    The 256-entry ByteLevel alphabet is seeded on purpose (without it only the bytes the
    corpus happens to contain survive), and it is a byte-FALLBACK net, so normal text not
    reaching it is the design working. The four chat specials and [NUM] never appear in raw
    corpus text either. What is left -- ASCII shards and incomplete UTF-8 -- is the real
    Fishing-for-Magikarp population."""
    v = tok.get_vocab()
    reserved = {v[_byte_token(b)] for b in range(256) if _byte_token(b) in v}
    reserved |= {i for t, i in v.items() if t.startswith("<|") or t == "[NUM]" or t in ("<unk>", "<eos>")}
    live = [i for i in v.values() if i not in reserved]
    return sum(1 for i in live if i not in counts) / max(len(live), 1)


def robustness(tok, corpus):
    """Fraction of tokens that are byte fragments rather than whole characters.

    Missing whole-character hanzi tokens scores worse than one token per character,
    which chars/token alone underreports."""
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
        "never used frac": never_used(tok, counts),
        "digit consistent": 1 - dg["context-inconsistent"] / dg["numbers tested"],
        "parity spread": max(par.values()) - min(par.values()),
    }
    m.update(robustness(tok, corpus))
    m.update(cost(tok))
    # Rényi is TokEval's single strongest predictor of BPB (rho = -0.80), stronger than
    # compression rate (-0.51); it is scored rather than gated because its counterexamples
    # (arXiv 2402.14614) are real.
    m["renyi"] = R.renyi_efficiency(counts)
    m.update(R.ref_fertility(tok))
    em = R.english_metrics(tok, corpus)
    if em:
        m["en fertility"] = em["fertility (tokens/word)"]
    u = R.utf8_integrity(tok, corpus)
    m["hanzi whole-char"] = float(u["hanzi in whole-char tokens"].rstrip("%")) / 100
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
    "never used frac": ("distribution", False),
    "undertrained frac": ("distribution", False),
    "digit consistent": ("structure", True),
    "renyi": ("predictability", True),
    "en fertility": ("compression", False),
    "ref fertility": ("compression", False),
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
    ap.add_argument("--per_domain", type=int, default=3000)
    ap.add_argument("--shards", type=int, default=8, help="shards per domain; part of the metric definition")
    ap.add_argument("--clip", type=int, default=0, help="chars per doc, 0 = whole document")
    ap.add_argument("--n_train", type=int, default=12000)
    a = ap.parse_args()

    import tokenizer_report as R

    doms = [d for d in a.domains.split(",") if d]
    corpus = R.sample_corpus(doms, a.per_domain, shards=a.shards, clip=a.clip or 10**9)
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
    hanzi_applies = corpus_has_hanzi(corpus)
    if not hanzi_applies:
        print("! hanzi whole-char gate is N/A: the sampled corpus is not Chinese-bearing "
              f"(hanzi-doc share < {HANZI_DOC_FRACTION:.0%}). V4.1 pivot 2026-09-10 trains "
              "English/math/code with zero Chinese domains; incidental CJK in code comments "
              "(measured <=3.8% of docs) does not arm the bilingual guard. A corpus WITH Chinese "
              "still FAILs on byte-fragmented hanzi -- the gate is scoped, not removed.")
    if not score_bits:
        print(f"! vocabulary sizes span {min(sizes)}..{max(sizes)} -- bits/char REPORTED, NOT SCORED")
        print("  (it is strictly monotone in size; see tokenizer_sweep.py)")

    rows = []
    for p in paths:
        tok, m, g = collect(p, corpus, train_rows, eval_rows, True)
        rows.append((os.path.basename(p), m, g, tok.get_vocab_size()))

    print(f"\n{'=' * 78}\nGATES  (a failure disqualifies; correctness is not traded against compression)")
    print(f"  {'vocabulary':<24}{'round-trip':>12}{'256 bytes':>12}{'verdict':>10}")
    ok, failed = [], []
    for name, m, g, V in rows:
        passed = g["round-trip lossless"] and g["all 256 bytes present"]
        print(
            f"  {name:<24}{'pass' if g['round-trip lossless'] else 'LOSSY':>12}"
            f"{str(g['_bytes']) + '/256':>12}{'ok' if passed else 'DISQUALIFIED':>10}"
        )
        if passed:
            ok.append((name, m))
        else:
            failed.append(name)
    if not ok:
        sys.exit("\nevery candidate failed a gate; nothing to rank")

    print(f"\n{'=' * 78}\nTHRESHOLDS  (what a rebuild is the only remedy for)")
    print(f"  {'vocabulary':<24}{'metric':<20}{'value':>10}{'needs':>10}{'':>8}")
    for name, m in ok:
        for k, val, thr, good in threshold_gates(m, hanzi_applies=hanzi_applies):
            arrow = "<=" if not GATES[k][1] else ">="
            print(f"  {name:<24}{k:<20}{val:>10.4f}{arrow + f'{thr:g}':>10}{'ok' if good else '  FAIL':>8}")
            if not good:
                failed.append(f"{name}:{k}")
                print(f"  {'':<24}  -> {GATES[k][2]}")
        if not hanzi_applies and "hanzi whole-char" in m:
            print(f"  {name:<24}{'hanzi whole-char':<20}{m['hanzi whole-char']:>10.4f}{'N/A':>10}"
                  f"{'no-zh':>8}  (no Chinese in corpus; reported, not vetoed)")
    for k, why in REPORTED_NOT_GATED.items():
        if any(k in m for _, m in ok):
            vals = " ".join(f"{m[k]:.4f}" for _, m in ok if k in m)
            print(f"  {'(not gated)':<24}{k:<20}{vals:>10}   {why}")

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

    if failed:
        print(f"\n  {len(failed)} threshold gate(s) FAILED: {', '.join(failed)}")
        print("  A gate is a REBUILD trigger, not a preference: ids do not survive a rebuild,")
        print("  so every checkpoint trained on this vocabulary inherits the defect for life.")
        return 1
    print("\n  all threshold gates pass")
    return 0


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

    # never-used is REPORTED, not gated (2026-09-08). 0.9 is 90x the retired 0.01 threshold,
    # so this world crosses any threshold anyone could re-add; the positive control below is
    # what keeps that from being vacuous.
    m = {"never used frac": 0.9, "ref fertility": 1.4286, "hanzi whole-char": 0.9890}
    rows = threshold_gates(m)
    assert not any(k == "never used frac" for k, *_ in rows), f"never-used must not be gated: {rows}"
    assert all(good for _, _, _, good in rows), f"the other two must pass at these values: {rows}"
    # positive control: the same call DOES fail a metric that is still a gate, so the
    # assertion above is about never-used and not about threshold_gates returning nothing.
    bad = threshold_gates({**m, "ref fertility": 9.9})
    assert [(k, good) for k, _, _, good in bad if not good] == [("ref fertility", False)], bad
    # and it is reported somewhere: a name in neither table is silently dropped from the run.
    assert "never used frac" in REPORTED_NOT_GATED and "never used frac" in METRICS
    assert not (set(GATES) & set(REPORTED_NOT_GATED)), "a metric is gated or reported, not both"

    # hanzi veto is SCOPED (3b, PR #233): it fails a Chinese corpus and is N/A on English.
    # A byte-fragmenting vocab (whole-char 0.3) ...
    bad_hanzi = {"hanzi whole-char": 0.30, "ref fertility": 1.0}
    # ... FAILs when the corpus carries Chinese ...
    zh = threshold_gates(bad_hanzi, hanzi_applies=True)
    assert [(k, good) for k, _, _, good in zh if k == "hanzi whole-char"] == [
        ("hanzi whole-char", False)], zh
    # ... and is absent (N/A, not a pass, not a fail) when it does not.
    en = threshold_gates(bad_hanzi, hanzi_applies=False)
    assert not any(k == "hanzi whole-char" for k, *_ in en), en
    # the detector itself: real Chinese across the corpus arms; pure English does not;
    # incidental CJK in a few percent of code comments does not arm a prose guard.
    zhdoc = "函数返回值"
    zh_corpus = {"d": [zhdoc + str(i) for i in range(60)]}
    en_corpus = {"d": ["def f(x): return x + %d" % i for i in range(200)]}
    # 5% hanzi docs, below the 10% arm threshold but far above zero
    sparse_cjk = {"d": [zhdoc + str(i) for i in range(10)] + ["return x = %d" % i for i in range(190)]}
    assert corpus_has_hanzi(zh_corpus), "a Chinese corpus must arm the hanzi gate"
    assert not corpus_has_hanzi(en_corpus), "English must not arm it"
    assert not corpus_has_hanzi(sparse_cjk), "5% incidental CJK docs must not arm it"

    print("tokenizer_eval self-test OK (tie threshold, direction, weights, byte table, "
          "never-used ungated, hanzi veto scoped: FAIL on Chinese, N/A on English)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _demo()
    else:
        sys.exit(main() or 0)
