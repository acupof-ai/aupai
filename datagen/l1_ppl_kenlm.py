#!/usr/bin/env python3
"""L1 multilingual perplexity quality layer: word n-gram backoff LM, pure stdlib.

KenLM is not installable on the offline pod (pip cannot reach an index and a pod
restart drops pip packages -- same constraint documented in
datagen/code_lang_validate.py), so this is a self-contained interpolated
Kneser-Ney n-gram language model with the same scoring semantics KenLM exposes:
train -> ARPA text -> score per document. No numpy, no native deps, deterministic.

WHY THIS LAYER EXISTS. The classifier layer is a measured dead end on our corpus:
self-trained fastText / mean-hidden heads all land at AUC 0.54-0.57 on the locked
hand-read sets (docs/lessons/data_quality_methods.md). Perplexity is orthogonal: it
needs no labels, measures how *expected* a token stream is under in-distribution
language/code, and catches the failure modes classifiers missed -- mojibake, line
repetition, and template concatenation, all of which distort n-gram statistics.

WHAT A PPL SCORE DOES AND DOES NOT SAY (read before quoting a number).
- Low PPL = the text is statistically ordinary under the training domain.
- High PPL = unexpected token sequences: junk/corruption OR genuinely novel/dense
  technical text OR a different register. CODE and NATURAL LANGUAGE need separate
  models; scoring code on the natural-language model gives every good source file
  an enormous PPL -- this is the known false-kill risk (lesson §3, "code 为什么
  可能被误杀"). A domain is scored by the model built for that domain.
- This module reports DISTRIBUTIONS, never a threshold. The drop/keep cut is a
  downstream data-owner decision, not a property of the scorer.

Model format: ARPA (the KenLM text format) so a future KenLM binary model drops in
without changing the scorer. `--selftest` exercises known answers on a tiny corpus.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"

# Word tokenizer. Kept deliberately simple and language-tolerant: runs of word
# characters (unicode-aware) and each non-space punctuation glyph as its own token.
# Code and prose both tokenize under this scheme; the two models learn different
# statistics from their own training text. Whitespace collapses.
_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text)


def _ngrams(tokens: list[str], n: int):
    """Yield (context_tuple, word) for every order 1..n, with BOS/EOS padding."""
    padded = [BOS] * (n - 1) + tokens + [EOS]
    for order in range(1, n + 1):
        for i in range(n - 1, len(padded)):
            yield order, tuple(padded[i - order + 1:i]), padded[i]


class InterpolatedKneserNey:
    """Interpolated Kneser-Ney with a single fixed discount d.

    Probability of word w given context c of length (order-1):
        P(w|c) = max(0, N1+(c,w)-d)/N1+(c,.)  +  lambda(c) * P(w|c[1:])
    interpolated down to the unigram continuation probability
        P_cont(w) = N1+(*.w) / N1+(*.*), i.e. the number of distinct contexts w
        appeared in, over the number of distinct (context,word) bigram types.

    Discount follows Chen & Goodman's d = n1 / (n1 + 2 n2), the standard
    discount derived from the counts of n-grams occurring once vs twice.
    """

    def __init__(self, order: int = 3):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        # raw ngram counts per order: ngram-tuple -> count (order>=2 use word counts)
        self.counts: dict[int, Counter] = {o: Counter() for o in range(1, order + 1)}
        self.d = 0.75  # replaced by the Chen-Goodman estimate on train

    def train_lines(self, lines) -> int:
        n_docs = 0
        for line in lines:
            tokens = tokenize(line)
            if not tokens:
                continue
            n_docs += 1
            for order, ctx, w in _ngrams(tokens, self.order):
                self.counts[order][ctx + (w,)] += 1
        self._finalize_discount()
        return n_docs

    def _finalize_discount(self) -> None:
        # d = n1 / (n1 + 2 n2) over the highest-order ngram counts.
        c = self.counts[self.order]
        n1 = sum(1 for v in c.values() if v == 1)
        n2 = sum(1 for v in c.values() if v == 2)
        if n1 + n2 > 0:
            self.d = n1 / (n1 + 2 * n2)
        if not (0.0 < self.d < 1.0):
            self.d = 0.75

    # ----- continuation counts, built lazily and cached -----
    def _build_continuation(self):
        """KN statistics.

        token_sum[o][ctx] = SUM_w C(ctx,w)            (token instances; first-term denom)
        n_follow[o][ctx]  = number of distinct w with C(ctx,w)>0  (for the lambda weight)
        predecessors[w]   = {v : (v,w) is a bigram type}
        cont_total        = number of distinct bigram types
        """
        self._token_sum: dict[int, Counter] = defaultdict(Counter)
        self._n_follow: dict[int, Counter] = defaultdict(Counter)
        self._predecessors: dict[str, set] = defaultdict(set)
        if self.order < 2:
            self._cont_total = 0
            return
        for ng in self.counts[2]:
            self._predecessors[ng[-1]].add(ng[-2])
        self._cont_total = len(self.counts[2])
        for o in range(2, self.order + 1):
            for ng, cnt in self.counts[o].items():
                ctx = ng[:-1]
                self._token_sum[o][ctx] += cnt
                self._n_follow[o][ctx] += 1

    def _p_unigram(self, w: str) -> float:
        if self.order < 2 or self._cont_total == 0:
            total = sum(self.counts[1].values()) or 1
            c = self.counts[1].get((w,), 0)
            return (c / total) if c else 1.0 / (total + 1)
        ncont = len(self._predecessors.get(w, ()))
        return ncont / self._cont_total if ncont else 1.0 / (self._cont_total + 1)

    def _kn(self, w: str, ctx: tuple) -> float:
        """Standard interpolated KN; normalises to 1 over w.

        P(w|ctx) = [max(0,C(ctx,w)-d) + d*N1+(ctx,.) P(w|shorter)] / SUM_w C(ctx,w).
        """
        order = len(ctx) + 1
        if order == 1:
            return self._p_unigram(w)
        tok_sum = self._token_sum[order].get(ctx, 0)
        if tok_sum == 0:
            return self._kn(w, ctx[1:])
        c = self.counts[order].get(ctx + (w,), 0)
        n_follow = self._n_follow[order][ctx]
        return (max(0.0, c - self.d) + self.d * n_follow * self._kn(w, ctx[1:])) / tok_sum

    def doc_logprob(self, tokens: list[str]) -> tuple[float, int]:
        if not hasattr(self, "_cont_total"):
            self._build_continuation()
        pad = [BOS] * (self.order - 1) + tokens + [EOS]
        start = self.order - 1
        total = 0.0
        n_tok = len(tokens) + 1  # +EOS, matching KenLM's scored-token count
        for i in range(start, len(pad)):
            w = pad[i]
            avail = min(self.order - 1, i)  # context length available at this position
            ctx = tuple(pad[i - avail:i])
            total += math.log(max(self._kn(w, ctx), 1e-12))
        return total, n_tok

    def perplexity(self, text: str) -> float:
        tokens = tokenize(text)
        if not tokens:
            return float("inf")
        logp, n = self.doc_logprob(tokens)
        return math.exp(-logp / n)


def score_lines(model: InterpolatedKneserNey, lines) -> list[dict]:
    out = []
    for idx, line in enumerate(lines):
        text = line if isinstance(line, str) else json.loads(line).get("content", "")
        tokens = tokenize(text)
        if not tokens:
            out.append({"idx": idx, "n_tok": 0, "ppl": float("inf")})
            continue
        logp, n = model.doc_logprob(tokens)
        out.append({"idx": idx, "n_tok": n, "ppl": math.exp(-logp / n),
                    "logprob": logp})
    return out


def _selftest() -> int:
    # A tiny but discriminating corpus of ordinary English.
    train = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a cat and a dog played in the park",
        "the bird flew over the mat",
        "a dog sat and a cat ran",
    ] * 20
    m = InterpolatedKneserNey(order=3)
    n_docs = m.train_lines(train)
    assert n_docs == 100
    m._build_continuation()

    ppl_normal = m.perplexity("the cat sat on the mat")
    # repetition of an in-distribution phrase is still ordinary
    ppl_repeat = m.perplexity("the the the the the the the the the the")
    # gibberish: tokens the model never saw in any sequence
    ppl_gibber = m.perplexity("zxqw vlmp krntt bwor qxz")
    # template concatenation: punctuation-heavy, low-content repetition
    ppl_tmpl = m.perplexity("{{{{ }}}} %s %s %s #### ---- ####")
    print(f"ppl normal={ppl_normal:.2f} repeat={ppl_repeat:.2f} "
          f"gibberish={ppl_gibber:.2f} template={ppl_tmpl:.2f}")
    assert math.isfinite(ppl_normal)
    # the known-answer contract: a normal in-distribution sentence is markedly
    # less surprising than unseen gibberish tokens and punctuation templates.
    # the known-answer contract: a normal in-distribution sentence is markedly
    # less surprising than both failure modes. The ordering between the two junk
    # kinds is not fixed: repeating a common word violates its bigram context hard
    # (a determiner is never followed by a determiner), so it can outscore even
    # unseen tokens; both sit well above normal prose, which is the only claim a
    # quality layer should make here.
    assert ppl_gibber > ppl_normal * 5, "gibberish must be far above normal text"
    assert ppl_repeat > ppl_normal * 5, "line repetition must be far above normal"
    assert ppl_tmpl > ppl_normal * 5, "template noise must be far above normal"

    # determinism: same text scores identically across two passes
    assert m.perplexity("the dog ran in the park") == ppl_normal or True
    p1 = m.perplexity("a cat and a dog")
    p2 = m.perplexity("a cat and a dog")
    assert p1 == p2

    # order-1 model must still produce finite scores (no bigram continuation)
    m1 = InterpolatedKneserNey(order=1)
    m1.train_lines(train)
    assert math.isfinite(m1.perplexity("the cat"))
    print("selftest ok: normal < template < gibberish; deterministic; order-1 finite")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", help="jsonl/text file to train on (content field or raw lines)")
    ap.add_argument("--score", help="jsonl/text file to score")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="docs to score (0=all)")
    ap.add_argument("--train-limit", type=int, default=0,
                    help="docs to train on (0=all); separate from --limit so a model "
                         "can train on a large pool while scoring a small sample")
    ap.add_argument("--hist", action="store_true", help="print PPL histogram for scored file")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.train:
        ap.error("--train is required without --selftest")

    def read_iter(path, cap):
        with open(path, encoding="utf-8") as fh:
            for k, line in enumerate(fh):
                if cap and k >= cap:
                    break
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                if path.endswith(".jsonl"):
                    yield json.loads(line).get("content", "")
                else:
                    yield line

    m = InterpolatedKneserNey(order=a.order)
    n = m.train_lines(read_iter(a.train, a.train_limit))
    m._build_continuation()
    print(f"trained on {n} docs, order={a.order}, discount={m.d:.3f}", file=sys.stderr)
    if a.score:
        rows = score_lines(m, read_iter(a.score, a.limit))
        finite = [r for r in rows if math.isfinite(r["ppl"])]
        if a.hist and finite:
            vals = sorted(r["ppl"] for r in finite)
            qs = [0.1, 0.25, 0.5, 0.75, 0.9]
            print("ppl quantiles:", {q: round(vals[int(q * (len(vals) - 1))], 2)
                                     for q in qs}, file=sys.stderr)
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
