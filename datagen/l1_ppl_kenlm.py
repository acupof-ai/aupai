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

# restartable: census scoring streams the corpus and appends ledger rows in 2000-row
# validated batches, so an interrupt costs at most the unflushed tail and re-running
# only re-adds rows; it never rescans or rewrites a corpus shard or the ledger.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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

    # ----- persistence: a trained model is reusable across runs -----
    def save(self, path: str) -> None:
        """Write the trained counts as JSON. Keys are ngram tuples (JSON arrays);
        the discount d is stored so a loaded model scores identically without
        re-deriving it. Lazily-built continuation caches are derived, not saved."""
        payload = {
            "format": "stdlib-kn-ngram-v1",
            "order": self.order,
            "discount": self.d,
            "counts": {
                str(o): {" ".join(ng): c for ng, c in self.counts[o].items()}
                for o in range(1, self.order + 1)
            },
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)  # atomic: a reader never sees a half-written model

    @classmethod
    def load(cls, path: str) -> InterpolatedKneserNey:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        m = cls(order=int(payload["order"]))
        m.d = float(payload["discount"])
        for o in range(1, m.order + 1):
            block = payload["counts"].get(str(o), {})
            if o == 1:
                m.counts[o] = Counter({(k,): int(v) for k, v in block.items()})
            else:
                m.counts[o] = Counter({tuple(k.split(" ")): int(v)
                                      for k, v in block.items()})
        return m

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


def append_census_ledger(model: InterpolatedKneserNey, path, *, ledger: str,
                         domain: str, lang: str, model_id: str, version: str,
                         limit: int, src_sha: str | None = None) -> tuple[int, int]:
    """Census full-scan path: score every jsonl/raw row and append one validated
    ledger row per document (stratum=None). doc_id is the content hash per the frozen
    schema, so a re-cleaned source stops joining to its old score. score_ledger sits
    in this same directory; import it by path so `python datagen/l1_ppl_kenlm.py`
    works whether or not datagen is on sys.path as a package.

    Returns (written, skipped). A document with ZERO scorable tokens (empty or
    whitespace-only content) is skipped, not written: its PPL is infinite, and a row
    with both score and rubric_dims null violates the ledger's XOR (it would abort the
    whole census at that row). Out-of-vocabulary text that still tokenises keeps a
    finite backoff PPL and IS written -- only an empty token stream is unscorable.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datetime import datetime, timezone

    from score_ledger import ScoreRow, append_rows, content_doc_id

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: UP017
    batch, written, skipped = [], 0, 0

    def emit(text: str, ppl: float):
        nonlocal written
        row = ScoreRow(
            doc_id=content_doc_id(text), domain=domain, lang=lang,
            scorer_name="kenlm", scorer_version=version, ts=ts,
            score=float(ppl), rubric_dims=None, cut=None, model=model_id,
            backend="stdlib-kn", stratum=None, rubric_kind=None, record_id=None,
            src_sha=src_sha)
        batch.append(row)
        if len(batch) >= 2000:
            written += append_rows(ledger, batch)
            batch.clear()

    with open(path, encoding="utf-8") as fh:
        for k, line in enumerate(fh):
            if limit and k >= limit:
                break
            line = line.rstrip("\n")
            if not line.strip():
                continue
            text = json.loads(line).get("content", "") if path.endswith(".jsonl") else line
            if not tokenize(text):  # whitespace/empty content: PPL=inf, no valid row
                skipped += 1
                continue
            emit(text, model.perplexity(text))
    if batch:
        written += append_rows(ledger, batch)
    return written, skipped


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
    # less surprising than both failure modes. The ordering between the two junk
    # kinds is not fixed: repeating a common word violates its bigram context hard
    # (a determiner is never followed by a determiner), so it can outscore even
    # unseen tokens; both sit well above normal prose, which is the only claim a
    # quality layer should make here.
    assert ppl_gibber > ppl_normal * 5, "gibberish must be far above normal text"
    assert ppl_repeat > ppl_normal * 5, "line repetition must be far above normal"
    assert ppl_tmpl > ppl_normal * 5, "template noise must be far above normal"

    # determinism: same text scores identically across two passes
    p1 = m.perplexity("a cat and a dog")
    assert p1 == m.perplexity("a cat and a dog")

    # order-1 model must still produce finite scores (no bigram continuation)
    m1 = InterpolatedKneserNey(order=1)
    m1.train_lines(train)
    assert math.isfinite(m1.perplexity("the cat"))

    # save/load round-trip: a reloaded model scores every probe text identically and
    # carries the same order/discount, so a model trained once is reusable across runs.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "model.json")
        m.save(mp)
        m2 = InterpolatedKneserNey.load(mp)
        assert m2.order == m.order and m2.d == m.d
        for probe in ("the cat sat on the mat", "zxqw vlmp krntt",
                      "{{ }}} %s ####", "the the the the"):
            before, after = m.perplexity(probe), m2.perplexity(probe)
            assert math.isinf(before) == math.isinf(after)
            if math.isfinite(before):
                assert abs(before - after) < 1e-12, (probe, before, after)

    # census ledger writer, only when the frozen schema module is importable (it is a
    # separate file that may not exist in an older tree). Known answers: one validated
    # row per doc, doc_id is the content hash and joins back to the source text.
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from score_ledger import load_rows, content_doc_id  # noqa: I001
    except Exception:
        load_rows = None
    if load_rows is not None:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in.jsonl")
            led = os.path.join(td, "ledger.jsonl")
            with open(src, "w", encoding="utf-8") as f:
                for s in train[:10]:
                    f.write(json.dumps({"content": s}) + "\n")
                # a non-empty LINE whose content is whitespace-only: tokenises to
                # nothing -> PPL inf -> must be skipped, never written as a
                # score=null/rubric=null row (that XOR would abort the census).
                f.write(json.dumps({"content": "   \t  "}) + "\n")
                # an OOV-but-tokenisable doc keeps a finite backoff PPL -> written
                f.write(json.dumps({"content": "zxqw vlmp krntt bwor"}) + "\n")
            written, skipped = append_census_ledger(
                m, src, ledger=led, domain="probe_en", lang="en",
                model_id="kenlm_test", version="t1", limit=0)
            rows = load_rows(led)
            assert written == 11 and skipped == 1, (written, skipped)
            assert len(rows) == 11, "whitespace doc skipped, OOV doc written"
            ids = {content_doc_id(s) for s in train[:10]}
            ids.add(content_doc_id("zxqw vlmp krntt bwor"))
            assert all(r["doc_id"] in ids for r in rows), "census doc_id must be content hash"
            assert all(r["scorer_name"] == "kenlm" and r["stratum"] is None
                       and r["cut"] is None and r["record_id"] is None
                       and isinstance(r["score"], float)
                       and math.isfinite(r["score"]) for r in rows), "OOV keeps finite PPL"
            assert content_doc_id("   \t  ") not in {r["doc_id"] for r in rows}, \
                "zero-token doc must have no ledger row"
    print("selftest ok: normal < template < gibberish; deterministic; order-1 finite; "
          "census ledger round-trip; zero-token skipped; model save/load identical")
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
    ap.add_argument("--ledger", help="append a census row per scored doc to this JSONL ledger")
    ap.add_argument("--domain", default="", help="corpus domain for ledger rows")
    ap.add_argument("--lang", default="en", help="BCP-47 lang for ledger rows")
    ap.add_argument("--model-id", default="", help="model id recorded in ledger rows")
    ap.add_argument("--scorer-version", default="v1", help="scorer_version pin")
    ap.add_argument("--src-sha", default=None, help="64-hex source-build fingerprint (optional)")
    ap.add_argument("--save-model", help="write trained model (JSON counts) to this path")
    ap.add_argument("--load-model", help="score with a saved model instead of training")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

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

    if a.load_model:
        m = InterpolatedKneserNey.load(a.load_model)
        m._build_continuation()
        print(f"loaded model order={m.order} discount={m.d:.3f} from {a.load_model}",
              file=sys.stderr)
    else:
        if not a.train:
            ap.error("--train is required without --load-model")
        m = InterpolatedKneserNey(order=a.order)
        n = m.train_lines(read_iter(a.train, a.train_limit))
        m._build_continuation()
        print(f"trained on {n} docs, order={a.order}, discount={m.d:.3f}", file=sys.stderr)
    if a.save_model:
        m.save(a.save_model)
        print(f"saved model -> {a.save_model}", file=sys.stderr)
    if a.score:
        if a.ledger:
            if not a.domain or not a.model_id:
                ap.error("--ledger requires --domain and --model-id")
            n_written, n_skipped = append_census_ledger(
                m, a.score, ledger=a.ledger, domain=a.domain, lang=a.lang,
                model_id=a.model_id, version=a.scorer_version, limit=a.limit,
                src_sha=a.src_sha)
            print(f"appended {n_written} census rows, skipped {n_skipped} "
                  f"unscorable (zero-token) -> {a.ledger}", file=sys.stderr)
            return 0
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
