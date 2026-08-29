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


# Hard thresholds. A vocabulary is used for the life of every checkpoint trained on
# it -- ids do not survive a rebuild -- so these are chosen by what is EXPENSIVE TO
# CHANGE LATER, not by what scores well today.
#
# TokEval (arXiv 2608.18062) is the field's suite: 14 metrics, 6 categories, and its
# own conclusion is that intrinsic metrics SCREEN but do not RANK -- "adjacent rows of
# results tables mostly differ by less than seed retraining would move a single model".
# So this file gates on correctness and on the two failures that no amount of training
# can repair, and leaves ranking to the weighted dimensions below and, finally, to two
# pretrains differing only in the vocabulary.
#
# It also follows TokEval's protocol requirement: the corpus and the normalisation unit
# are part of the metric's definition. The same vocabulary reads 6.4% utilisation on 402
# math documents and 94.5% on 2,814 across seven domains -- always gate on the full mix.
GATES = {
    # name: (threshold, higher_is_better, why it is a veto and not a preference)
    # NEVER used, not "<=1 use": the <=1 rate is a function of how much text you counted
    # (4.0% at 1.6M tokens, 0.43% at 142M, same vocabulary), so it cannot carry a fixed
    # threshold. A token with zero occurrences in 142M tokens of the training distribution
    # is a glitch token by the Fishing-for-Magikarp definition and no amount of training
    # reaches it.
    "never used frac": (0.005, False, "glitch tokens: never trained, and training cannot fix them"),
    # On REF_EN, not on whatever English happens to be in the corpus sample: the same
    # vocabulary reads 1.429 on REF_EN and 1.870 on our own `en` domain.
    #
    # The threshold was 1.35 and the REFERENCE CLASS WAS WRONG: it came from English-only
    # vocabularies (bert 1.182, gpt2 1.156), which is not what we are trying to be. Against
    # the bilingual frontier -- DeepSeek-V3 1.104, Qwen3 and GLM-4.5 1.130, Phi-4-mini 1.143
    # -- every one of them buys that with 128K-200K slots, and our Chinese chars/token
    # (1.693) TIES DeepSeek-V3 and BEATS Qwen3 and GLM-4.5 on a quarter of the vocabulary.
    # 1.55 is the price of bilingual-at-32K, recorded as a ceiling that must not DRIFT
    # rather than as a defect to fix: the field's fix is a bigger vocabulary, and the fitted
    # scaling law puts our optimum at 12-20K, not 128K.
    "ref fertility": (1.55, False, "regression guard: English must not get worse than it is"),
    "hanzi whole-char": (0.95, True, "byte-fragmented hanzi is worse than one token per character"),
}


def gates(tok, corpus):
    """Disqualifiers: correctness first, then the two properties a rebuild is the only
    remedy for. Everything else is a preference and is scored, not gated."""
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


def threshold_gates(metrics):
    """(name, value, threshold, ok) for each GATES entry present in `metrics`."""
    rows = []
    for k, (thr, higher, _why) in GATES.items():
        if k not in metrics:
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
        "never used frac": 1 - len(counts) / len(tok.get_vocab()),
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
        for k, val, thr, good in threshold_gates(m):
            arrow = "<=" if not GATES[k][1] else ">="
            print(f"  {name:<24}{k:<20}{val:>10.4f}{arrow + f'{thr:g}':>10}{'ok' if good else '  FAIL':>8}")
            if not good:
                failed.append(f"{name}:{k}")
                print(f"  {'':<24}  -> {GATES[k][2]}")

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
    print("tokenizer_eval self-test OK (tie threshold, direction, weights, byte table)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _demo()
    else:
        sys.exit(main() or 0)
