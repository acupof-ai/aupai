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
#: A shard is NAMED like one: `<prefix>_<NNN>.jsonl`, the only thing ShardWriter emits.
#: Same pattern as train.py's SHARD_RE, and it must stay the same: a tokenizer gate that
#: samples files training never reads is not measuring the training distribution.
#:
#: WHY A WHITELIST AND NOT A `holdout_slice_` BLACKLIST. Every corpus dir may hold artifacts
#: written beside its shards. Today that is five one-row `holdout_slice_<domain>.jsonl`
#: headers -- `{phase, rule_fp, n}`, no `content` key -- and a blacklist would have to name
#: each new artifact type before it can be excluded. train.py chose the whitelist direction
#: for this exact reason (its own comment: a blacklist reads an unknown new file as DATA).
#:
#: WHAT IT COST HERE, measured on the pod 2026-09-08. `load_text` samples 2 files per domain
#: out of `sorted(glob(*.jsonl))`, so a domain with few files draws the slice with high
#: probability, and `.get("content", "")` turns it into an empty string rather than an error:
#:
#:     v2 composition       train 11394/12000   eval    0/800   -> bits/char ZeroDivisionError
#:     resume-1 composition train 12000/12000   eval  105/800   -> ran, on 13% of its eval set
#:
#: With this filter both compositions read 12000/800. So the failure is not new to v2; v2 is
#: where it stopped being silent. Every recorded bits/char figure was computed against a
#: truncated held-out set, and the resume-1 gate readings in
#: facts/tokenizer.json#tok.gates_v2_composition were taken with 105 of 800 eval rows.
#: `sample_corpus` was hit more weakly -- one empty row each in chatml and chat_qa -- because
#: it samples 8 files per domain, not 2.
SHARD_RE = re.compile(r"_\d{3,}\.jsonl$")


def shard_paths(domain):
    """Every real shard of `domain`, in sorted order. The one place either sampler asks
    the filesystem what a domain contains."""
    fs = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", domain, "*.jsonl")))
    return [f for f in fs if SHARD_RE.search(os.path.basename(f))]


def sample_corpus(domains, per_domain=400, seed=7, shards=3, clip=2000):
    """Documents per domain. `shards` and `clip` are part of every metric's DEFINITION,
    not tuning knobs: the same vocabulary reads 4.0% undertrained on the 1.6M-token
    default and 0.43% on 142M, because a token of true frequency 1e-6 appears 1.6 times
    in 1.6M tokens and a healthy Zipf tail therefore MUST put percent of the vocabulary
    at <=1 use. Any frequency-tail threshold has to name the corpus it was measured on."""
    rng = random.Random(seed)
    out = {}
    for d in domains:
        fs = shard_paths(d)
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


def _even_rows(corpus, total):
    """`total` rows drawn EVENLY over the domains present, not off a flattened prefix.

    THE PREFIX FORM WAS A DEFECT IN A VETO. Every caller below used to write
    `[r for v in corpus.values() for r in v][:N]`, which reads whichever domains happen to
    come first in dict order and calls the result "the corpus". With the nine-domain v2
    composition at ~3000 rows each, the 600 rows utf8_integrity took were ENTIRELY
    math_owm_stage2 -- an English/LaTeX corpus -- so `hanzi whole-char` measured 0.1594 and
    FAILED its 0.95 veto. Measured on the same tokenizer over zh_web alone: 0.9895.

    The gate had been right only by luck: the old default domain list happened to put a
    Chinese directory inside the first 600 rows. Changing the composition, which is exactly
    what this gate exists to evaluate, silently changed what the metric measured. A veto
    whose answer depends on dict order can fire a rebuild that invalidates every checkpoint.

    Quota, not proportion: each domain contributes total//len(corpus) rows (remainder to the
    first domains), so the metric is defined by the composition's DOMAIN SET and not by how
    many rows each directory happened to yield. A domain with fewer rows than its quota
    contributes all of them and the shortfall is not redistributed -- redistribution would
    make the answer depend on row counts again.
    """
    doms = [d for d in corpus if corpus[d]]
    if not doms:
        return []
    per, extra = divmod(total, len(doms))
    out = []
    for i, d in enumerate(doms):
        out.extend(corpus[d][: per + (1 if i < extra else 0)])
    return out


def utf8_integrity(tok, corpus):
    """Fraction of hanzi in whole-character tokens rather than ByteLevel fragments.

    A vocabulary without whole tokens for common characters scores worse than one
    token per character, which chars/token alone underreports."""
    # ByteLevel BPE token strings are byte-mapped (今天 stored as 'ä»Ĭå¤©'), so a
    # literal-hanzi search fires on every correct vocabulary. Decode each token first.
    rows = _even_rows(corpus, 600)
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
    full 256-byte alphabet silently drops bytes (NUL and tab).

    DECODE WITH skip_special_tokens=False, which is not the default. The default strips
    the four chat specials, so any corpus row containing a literal `<|im_start|>` fails a
    round-trip that is working exactly as designed. Measured when _even_rows first let
    this reader reach chatml: 361/405, with all 44 failures in chatml and 0 in the other
    eight domains, and `<|im_start|>user\\n...` decoding to `user\\n...`. That is the
    stripping, not a lost byte -- with skip_special_tokens=False the same rows are exact.
    A gate that fails on a domain merely because that domain uses the chat format would
    fire a rebuild for a formatting convention.
    """
    rows = _even_rows(corpus, 400)
    extra = ["NUL\x00byte", "emoji 🚀 ok", "tab\tnewline\n", "混合 mixed 123", "  双空格  "]
    bad = []
    for s in rows + extra:
        if tok.decode(tok.encode(s, add_special_tokens=False).ids, skip_special_tokens=False) != s:
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
    # THE FALLBACK MATTERS MORE THAN IT LOOKS. `en` is a literal directory name, and the
    # v2 composition's English domain is `en_c4_stage2`, so this falls through on every
    # real mix and the flattened prefix used to make it math_owm_stage2's fertility.
    # ref_fertility (the gated one) reads a fixed text and was never affected; this
    # ungated `en fertility` column was.
    rows = corpus.get("en") or _even_rows(corpus, 400)
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

    # 2a. AND A ROW CARRYING A LITERAL CHAT SPECIAL MUST STILL ROUND-TRIP. tok.decode's
    #     default strips them, so before this the gate failed every chatml row -- 44 of 405
    #     when _even_rows first let it reach that domain, 0 failures in the other eight.
    #     A lost byte and a stripped special are different findings and only the first is
    #     what this gate exists for.
    #
    #     THE SPECIALS MUST BE ADDED TO THE FIXTURE OR THIS CASE IS VACUOUS. _tiny_tokenizer
    #     has no chat specials, so `<|im_start|>` is ordinary text to it, nothing is ever
    #     stripped, and the assertion passes on the broken decode too -- measured: the
    #     mutation reverting to the stripping default SURVIVED until these two lines existed.
    _sp_tok, _ = _tiny_tokenizer(merges=3000)
    _sp_tok.add_special_tokens(["<|im_start|>", "<|im_end|>"])
    _sp = {"s": ["<|im_start|>user\nhi<|im_end|>\n"]}
    assert "<|im_start|>" in _sp_tok.get_vocab(), (
        "the fixture has no chat special, so this case cannot distinguish a stripping "
        "decode from a correct one"
    )
    assert roundtrip(_sp_tok, _sp)["failures"] == [], (
        f"a row containing a literal chat special fails round-trip: "
        f"{roundtrip(_sp_tok, _sp)['failures']}; decode must pass skip_special_tokens=False"
    )


    # 2b. THE SAMPLE MUST COVER EVERY DOMAIN, NOT THE FIRST ONES IN DICT ORDER. Every case
    #     above uses a ONE-domain corpus, which is why none of them could see the defect
    #     this checks: the readers took `[r for v in corpus.values() for r in v][:N]`, and
    #     with one domain a flattened prefix and an even draw are the same thing.
    #
    #     The world here is two domains, sized so the flattened prefix cannot reach the
    #     second: 600 rows of latin first, then the hanzi text. Under the old prefix form
    #     utf8_integrity saw no hanzi at all; under _even_rows it sees 300 rows of each.
    #     Measured on the real corpus, the same defect read `hanzi whole-char` 0.1594
    #     against 0.9895 on zh_web alone and FAILED a veto that invalidates checkpoints.
    _latin = ["the quick brown fox jumps over the lazy dog\n"] * 600
    _two = {"a_latin": _latin, "z_hanzi": text if isinstance(text, list) else [text]}

    _prefix = [r for v in _two.values() for r in v][:600]
    assert not any(HAN.search(r) for r in _prefix), (
        "the world does not establish the defect: the flattened 600-row prefix already "
        "contains hanzi, so the old form and the fixed form cannot be told apart here"
    )
    _even = _even_rows(_two, 600)
    assert any(HAN.search(r) for r in _even), (
        "_even_rows drew 600 rows from a two-domain corpus and reached no hanzi; the "
        "quota is not covering every domain"
    )
    assert len(_even) <= 600, f"_even_rows returned {len(_even)} rows for a 600 quota"
    _hz_two = float(utf8_integrity(big, _two)["hanzi in whole-char tokens"].rstrip("%"))
    assert _hz_two > 80, (
        f"utf8_integrity reports {_hz_two}% whole-char hanzi on a corpus whose second "
        f"domain is entirely hanzi; the sample is reading domain order, not the corpus"
    )

    # EVERY reader that samples must cover every domain, not just utf8_integrity. roundtrip
    # and english_metrics take their own draws, and reverting either to a flattened prefix
    # survived the case above -- it only watches the hanzi metric. These two assert on the
    # DRAW each reader takes, so the coverage property is checked where each one reads.
    #
    # A domain the prefix cannot reach is given text only that domain can answer for:
    # `_two`'s second domain is the hanzi text, so a reader that misses it sees no hanzi.
    #
    # roundtrip's OWN COVERAGE IS NOT ASSERTED, AND THAT IS DELIBERATE. Two fixtures were
    # tried and the mutation reverting its line to a flattened prefix survived both:
    # a ByteLevel BPE is lossless on ANY text by construction, so this metric returns
    # 405/405 whichever domains it reads and no world can make its verdict depend on
    # coverage. The line is still changed to _even_rows -- the `failures` list it prints
    # should quote whatever domain is failing, not whichever sorts first -- but the
    # protection is utf8_integrity's case above and _even_rows' own two cases, not a
    # roundtrip assertion that cannot fail. Do not add one; check it did not become
    # coverage-sensitive first.
    assert any(HAN.search(r) for r in _even_rows(_two, 400)), (
        "roundtrip's draw size reaches no hanzi on this world; if roundtrip ever becomes "
        "coverage-sensitive, this is the world its case would need"
    )



    # english_metrics falls back to the same draw whenever no domain is literally named
    # `en` -- which is every real composition, since the English dir is en_c4_stage2.
    # ITS WORLD CANNOT BE THE HANZI ONE: WORD is [A-Za-z]+, so hanzi rows contribute no
    # words and both draws return the same fertility whether or not the reader got there.
    # The second domain here is long latin words the first domain does not contain, so
    # reaching it must move tokens/word.
    assert "en" not in _two, "the fallback path is what this checks; do not name a domain `en`"
    _long = ["antidisestablishmentarianism incomprehensibilities\n"] * 600
    _w2 = {"a_short": ["a b c\n"] * 600, "z_long": _long}
    assert not any(len(w) > 6 for r in [x for v in _w2.values() for x in v][:400] for w in WORD.findall(r)), (
        "the world does not establish the defect: the flattened 400-row prefix already "
        "reaches the long-word domain"
    )
    _em_two = english_metrics(big, _w2)["fertility (tokens/word)"]
    _em_first = english_metrics(big, {"a_short": _w2["a_short"]})["fertility (tokens/word)"]
    assert _em_two > _em_first + 0.1, (
        f"english_metrics returns {_em_two:.4f} on a two-domain corpus and {_em_first:.4f} "
        f"on the first domain alone; a long-word second domain must raise tokens/word, so "
        f"equality means the reader never got past the flattened prefix"
    )


    # A domain shorter than its quota contributes all it has, and the shortfall is NOT
    # redistributed -- redistribution would put row counts back into the metric's answer.
    _short = _even_rows({"a": ["x"] * 2, "b": ["y"] * 500}, 100)
    assert _short.count("x") == 2 and _short.count("y") == 50, (
        f"short-domain handling changed: got {_short.count('x')} x and {_short.count('y')} y, "
        f"want 2 and 50 (quota 50 each, `a` has only 2, no redistribution)"
    )

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

    # 6. SHARD SELECTION. A corpus dir holds artifacts beside its shards, and both samplers
    #    used to read them as data: holdout_slice_<domain>.jsonl is one row of
    #    {phase, rule_fp, n} with no `content`, which .get("content", "") turns into an empty
    #    string instead of an error. Built on disk rather than asserted from the pattern:
    #    the defect is about which FILES the sampler reaches, and a regex assertion would
    #    hold on a version that still globs *.jsonl.
    import tempfile

    _root = ROOT
    try:
        with tempfile.TemporaryDirectory() as td:
            d = os.path.join(td, "data", "corpus", "dd")
            os.makedirs(d)
            for i in range(3):
                with open(os.path.join(d, f"dd_{i:03d}.jsonl"), "w", encoding="utf-8") as fh:
                    for j in range(50):
                        fh.write(json.dumps({"content": f"row {i} {j}\n"}) + "\n")
            # the real artifact, byte for byte: one row, no `content` key
            with open(os.path.join(d, "holdout_slice_dd.jsonl"), "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"phase": "dd", "rule_fp": "0" * 16, "n": "0"}) + "\n")
            # globals(), not `global ROOT`: this function reads ROOT above, and a global
            # declaration has to come first in the body.
            globals()["ROOT"] = td
            got = [os.path.basename(p) for p in shard_paths("dd")]
            assert got == ["dd_000.jsonl", "dd_001.jsonl", "dd_002.jsonl"], got
            # and the sampler that consumes it yields no empty row from any of the 4 files
            rows = sample_corpus(["dd"], per_domain=120, shards=4)["dd"]
            assert rows and not any(r == "" for r in rows), (
                f"sample_corpus returned {sum(1 for r in rows if r == '')} empty rows of "
                f"{len(rows)}: it is still reading the non-shard artifact"
            )
            # NEGATIVE CONTROL: the same world, globbed the old way, DOES produce the empty
            # row -- otherwise the assertion above would pass on a fixture with no defect
            # in it (three of my fixtures this week proved nothing that way).
            old = sorted(glob.glob(os.path.join(d, "*.jsonl")))
            assert len(old) == 4 and any(
                json.loads(next(open(f, encoding="utf-8"))).get("content", "") == ""
                for f in old
            ), "the fixture does not contain the defect; this case would prove nothing"
    finally:
        globals()["ROOT"] = _root

    print(
        f"tokenizer_report self-test OK ({len(SCALE_STABLE)} scale-stable metrics checked, "
        f"{len(SCALE_BOUND)} declared scale-bound, 3 known-answer cases including the "
        f"two-domain sample-coverage world, REF_EN pinned, shard selection on a built dir)"
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
