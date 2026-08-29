#!/usr/bin/env python3
"""A tokenizer harness: compression, distribution, structure, and integrity.

chars/token is the number this repo optimised and it is a weak proxy: arXiv
2506.03101 puts its correlation with downstream performance anywhere from
rho=-0.77 (translation) to -0.09 (summarisation), and finds Zipf deviation the
strongest cheap predictor; TokEval (arXiv 2608.18062) adds structure-sensitive
checks. Groups are printed in increasing order of trustworthiness, and the one
that would settle it -- two runs differing only in the vocabulary -- is extrinsic
and not here.

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
def sample_corpus(domains, per_domain=400, seed=7, shards=3, clip=2000):
    """Documents per domain. `shards` and `clip` are part of every metric's DEFINITION,
    not tuning knobs: the same vocabulary reads 4.0% undertrained on the 1.6M-token
    default and 0.43% on 142M, because a token of true frequency 1e-6 appears 1.6 times
    in 1.6M tokens and a healthy Zipf tail therefore MUST put percent of the vocabulary
    at <=1 use. Any frequency-tail threshold has to name the corpus it was measured on."""
    rng = random.Random(seed)
    out = {}
    for d in domains:
        fs = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", d, "*.jsonl")))
        if not fs:
            continue
        rows = []
        for f in rng.sample(fs, min(shards, len(fs))):
            with open(f, encoding="utf-8") as fh:
                lines = fh.readlines()
            for x in rng.sample(lines, min(per_domain // shards + 1, len(lines))):
                rows.append(json.loads(x).get("content", "")[:clip])
        if rows:
            out[d] = rows
    return out


# ---------------------------------------------------------------- distribution
def zipf_deviation(counts):
    """RMS deviation of log-frequency from the fitted Zipf line, and its slope."""
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
    """Reported, never optimised: Cognetta et al. (2024) raise it and hurt
    downstream; Dagan et al. (2024) find it anti-correlates with code generation."""
    tot = sum(counts.values())
    ps = [c / tot for c in counts.values() if c]
    if not ps:
        return float("nan")
    return (math.log(sum(p**alpha for p in ps)) / (1 - alpha)) / math.log(len(ps))


# ---------------------------------------------------------------- structure
def digit_consistency(tok):
    """Does the same number tokenise the same way in different contexts?

    BPE merges digits by frequency, so ' 63' and '63' can differ and '1640'
    becomes '16|40', which has nothing to do with its value."""
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
    """Fraction of hanzi in whole-character tokens rather than ByteLevel fragments.

    A vocabulary without whole tokens for common characters scores worse than one
    token per character, which chars/token alone underreports."""
    # ByteLevel BPE token strings are byte-mapped (今天 stored as 'ä»Ĭå¤©'), so a
    # literal-hanzi search fires on every correct vocabulary. Decode each token first.
    rows = [r for v in corpus.values() for r in v][:600]
    frag = whole = 0
    for e in tok.encode_batch(rows):
        for tid in e.ids:
            s = tok.decode([tid]).strip()
            if not s:
                continue
            n = len(HAN.findall(s))
            if n:
                whole += n
            elif "\ufffd" in s:
                frag += 1  # an incomplete UTF-8 sequence: a genuine byte fragment
    tot = whole + frag
    return {"hanzi in whole-char tokens": f"{100 * whole / max(1, tot):.2f}%", "byte fragments": frag}


def roundtrip(tok, corpus):
    """encode -> decode must return the input. A vocabulary trained without the
    full 256-byte alphabet silently drops bytes (NUL and tab)."""
    rows = [r for v in corpus.values() for r in v][:400]
    extra = ["NUL\x00byte", "emoji 🚀 ok", "tab\tnewline\n", "混合 mixed 123", "  双空格  "]
    bad = []
    for s in rows + extra:
        if tok.decode(tok.encode(s, add_special_tokens=False).ids) != s:
            bad.append(s[:40])
    return {"lossless": f"{len(rows) + len(extra) - len(bad)}/{len(rows) + len(extra)}", "failures": bad[:3]}


WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

#: Words whose morpheme boundaries are not in doubt. Cutting inside a morpheme
#: makes the model relearn the same affix in every word carrying it.
MORPH = [
    ("unhappiness", ["un", "happi", "ness"]),
    ("rebuilding", ["re", "build", "ing"]),
    ("nationalise", ["nation", "al", "ise"]),
    ("teacher", ["teach", "er"]),
    ("disagreement", ["dis", "agree", "ment"]),
    ("faster", ["fast", "er"]),
]


def english_metrics(tok, corpus):
    """Fertility and word-splitting: chars/token's analogue for an alphabetic script.

    Matters beyond `en`: every MC benchmark in eval/ except C-Eval is English."""
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
    # "the" and " the" should not be unrelated token sequences
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


# A FIXED English passage, so `en fertility` names the text it is measured on: the same
# vocabulary reads 1.429 here and 1.870 on our own `en` domain, and a threshold that does
# not say which is not a threshold. Reproducible with no corpus and no network.
#
# The English gap is the price of bilingual-at-32K, not a defect: every frontier model buys
# 1.13 with a 128K-200K vocabulary, and a fitted scaling law (arXiv 2407.13623) puts this
# 166M non-embedding model's optimum at 12-20K. A measured sweep showed 32K->64K buys +2.8%
# compression for +33.6M parameters and +14% compute per character. Revisit if the model grows.
REF_EN = (
    "The transformer architecture has become the dominant approach for natural language "
    "processing. Researchers demonstrated that self-attention mechanisms could replace "
    "recurrence entirely, enabling parallelization across sequence positions. Subsequent "
    "investigations established scaling relationships between parameters, dataset size, and "
    "computational budget. Practitioners increasingly emphasize data quality over raw "
    "quantity, particularly for smaller models where memorization capacity is constrained. "
    "Tokenization remains an underappreciated design decision: vocabulary construction "
    "determines compression efficiency, downstream generalization, and the granularity at "
    "which numerical reasoning operates."
) * 8
# Bilingual frontier tokenizers, the right reference class -- an English-ONLY vocabulary
# is not what we are trying to be.
EN_REFERENCE = {"DeepSeek-V3": 1.104, "Qwen3": 1.130, "GLM-4.5": 1.130, "Phi-4-mini": 1.143}
ZH_REFERENCE = {"DeepSeek-V3": 1.693, "GLM-4.5": 1.608, "Qwen3": 1.494, "SmolLM3": 1.134}


def ref_fertility(tok):
    """Tokens per word on REF_EN -- the same text for every vocabulary, forever."""
    n = s = 0
    words = WORD.findall(REF_EN)
    for w in words:
        k = len(tok.encode(" " + w, add_special_tokens=False).ids)
        n += k
        s += k > 1
    return {"ref fertility": n / len(words), "ref split": s / len(words)}


def parity(tok, corpus):
    """Bytes per token per domain, relative to the best-served one; 1.00 is even."""
    per = {}
    for dom, rows in corpus.items():
        encs = tok.encode_batch(rows)
        b = sum(len(r.encode()) for r in rows)
        t = sum(len(e.ids) for e in encs)
        per[dom] = b / t
    best = max(per.values())
    return {d: v / best for d, v in sorted(per.items(), key=lambda kv: -kv[1])}


def whitespace_handling(tok):
    """Line breaks and indentation; TokEval finds these correlate with task accuracy."""
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


# ---------------------------------------------------------------- self-test
#
# A METRIC WITHOUT A KNOWN-ANSWER CASE IS NOT A METRIC, AND A METRIC WHOSE VALUE MOVES
# WITH THE SAMPLE MUST CARRY THE SAMPLE IN ITS DEFINITION. This file once reported four
# wrong numbers in one day, all of one class -- a value that depends on the measurement
# configuration, printed without it.

SCALE_STABLE = ("chars/token", "fertility", "hanzi whole-char")  # must not move with sample size
SCALE_BOUND = ("utilised", "never used", "undertrained")  # meaningless without the corpus size


def _tiny_tokenizer(merges=200):
    """A real ByteLevel BPE over a known corpus, so a metric can be checked against an
    answer computed by hand rather than against the metric's own output."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as BLD
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel as BLP
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = BLP(add_prefix_space=False)
    tok.decoder = BLD()
    # A REAL frequency tail: 40 copies of one sentence has none, so the <=1-use metric
    # cannot move with sample size and the scale-stability case below proves nothing.
    import random as _r

    rng = _r.Random(0)
    zh = "今天天气很好我们去公园散步他昨天买了三本书这条河很长学校在山的南边"
    en = "the quick brown fox jumps over lazy dog and then walks home slowly again"
    text = [
        "".join(rng.sample(zh, 12)) + "。" + " ".join(rng.sample(en.split(), 6)) + f". {rng.randrange(10000)}"
        for _ in range(400)
    ]
    tok.train_from_iterator(
        text, BpeTrainer(vocab_size=merges, initial_alphabet=BLP.alphabet(), show_progress=False)
    )
    return tok, text


def _demo():
    tok, text = _tiny_tokenizer()
    corpus = {"t": text}

    # 1. KNOWN ANSWERS, A PAIR. One number cannot show a metric discriminates: a single
    #    low-answer case passes a broken reader. A 200-merge vocabulary genuinely cannot
    #    form whole hanzi out of 3-byte UTF-8 and must score LOW; a 3000-merge one over the
    #    same text must score HIGH.
    hz = lambda t: float(utf8_integrity(t, corpus)["hanzi in whole-char tokens"].rstrip("%"))
    lo = hz(tok)
    big, _bigtext = _tiny_tokenizer(merges=3000)
    hi = hz(big)
    assert lo < 20, f"a 200-merge vocabulary reports {lo}% whole-char hanzi; it cannot form them"
    assert hi > 80, f"a 3000-merge vocabulary reports {hi}% whole-char hanzi (should be near 100)"
    assert hi - lo > 60, f"utf8_integrity does not discriminate: {lo}% vs {hi}%"

    # 2. KNOWN ANSWER. Round-trip on text this vocabulary was trained on must be lossless.
    assert roundtrip(tok, corpus)["lossless"], "round-trip lost bytes on its own training text"

    # 3. SCALE STABILITY. Ten times the text, same characters: a per-character or per-word
    #    ratio must not move. `utilised` and the frequency tail MUST move, which is why
    #    they are SCALE_BOUND and carry their corpus size instead of a bare threshold.
    import collections

    # `big`, not `tok`: a 200-target vocabulary is just the 256-byte alphabet with no
    # merges, so it has no frequency tail at all and the assertion below would pass by
    # being vacuous rather than by demonstrating anything.
    def counts_for(rows):
        c = collections.Counter()
        for e in big.encode_batch(rows):
            c.update(e.ids)
        return c

    few, many = text[:40], text
    cs, cb = counts_for(few), counts_for(many)
    ratio = lambda c, rows: sum(c.values()) / sum(len(r) for r in rows)
    rs, rb = ratio(cs, few), ratio(cb, many)
    assert abs(rs - rb) / rb < 0.02, f"tokens/char moved {rs:.4f} -> {rb:.4f} with sample size"

    V = len(big.get_vocab())
    us, ub = len(cs) / V, len(cb) / V
    assert ub >= us, "utilisation fell with more text"
    tail_s = sum(1 for c in cs.values() if c <= 1) / V
    tail_b = sum(1 for c in cb.values() if c <= 1) / V
    assert tail_s != tail_b, (
        "the <=1-use tail did not move with a 10x sample, so this self-test cannot "
        "demonstrate why that metric needs its corpus size stated"
    )

    # 4. The sample knobs are part of the definition, so they must actually bind.
    import sys as _sys

    if os.path.isdir(os.path.join(ROOT, "data", "corpus")):
        a = sample_corpus(["sample"], 60, shards=1, clip=200)
        b = sample_corpus(["sample"], 60, shards=1, clip=2000)
        if a and b:
            la = sum(len(r) for r in a["sample"])
            lb = sum(len(r) for r in b["sample"])
            assert lb > la, f"clip= did not bind: {la} vs {lb} chars"
        print("   sample_corpus: shards/clip bind", file=_sys.stderr)

    # 5. REF_EN is a fixed string, so ref_fertility must be reproducible to the digit --
    #    it is the anchor the English gate's threshold was derived from, and a silent edit
    #    to the passage would move the threshold's meaning without moving the threshold.
    r1, r2 = ref_fertility(big)["ref fertility"], ref_fertility(big)["ref fertility"]
    assert r1 == r2, "ref_fertility is not deterministic"
    assert len(WORD.findall(REF_EN)) == 616, (
        f"REF_EN changed ({len(WORD.findall(REF_EN))} words, was 616): the English gate's "
        "threshold was measured on the old passage and no longer means what it says"
    )

    print(
        f"tokenizer_report self-test OK ({len(SCALE_STABLE)} scale-stable metrics checked, "
        f"{len(SCALE_BOUND)} declared scale-bound, 2 known-answer cases, REF_EN pinned)"
    )


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
    if "--selftest" in sys.argv:
        _demo()
    else:
        main()
