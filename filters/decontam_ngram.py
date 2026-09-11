#!/usr/bin/env python3
"""13-gram decontamination against the HumanEval / MBPP gate (ae-7, fb ruling 2026-09-11).

One module shared by the six non-Ultra corpus filter runs and by 0e's UltraData
L2/L3 aggregate step. DROP a training row on ANY 13-token n-gram overlap with a
gate problem. There is no allow-list for "common" snippets: a verbatim recursive
fibonacci (HumanEval/55) is exactly what a 30% pass@1 number would be accused of
memorising (BigCode preprocessing convention).

What is keyed per benchmark:
  HumanEval: prompt (signature/docstring) + canonical_solution + test (asserts)
  MBPP holdouts: text (prompt) + code (reference solution). The holdout file has
  no separate test_list field; MBPP reference code frequently embeds its asserts,
  which are then part of the code key.

Normalisation is ONE function applied identically to the benchmark side and the
corpus side (de review): full-line comments are stripped, whitespace runs
collapsed, then a fixed word-level split -- not the BPE tokenizer, so the
decontam fingerprint never depends on a vocabulary. n-grams are word tokens:
the gate is a verbatim TEXT property that must survive a vocabulary rebuild.

Performance for the aggregate stream (0e): build the gram set ONCE
(Decontaminator.load_default()), then per row it is set membership over the row's
13-grams -- linear in row length, CPU-only, no subprocess.

    d = Decontaminator.load_default()
    hit = d.hit(content)            # -> None or dict(problem=, part=)
    keep = d.keeps(content)          # -> bool
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUMANEVAL = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
MBPP = os.path.join(ROOT, "data", "eval", "mbpp_holdouts.jsonl")
N = 13

_WS = re.compile(r"\s+")
# a full-line code comment (Python # and // for the few non-py shards); inline
# trailing comments are left in place so code structure is not rewritten.
_COMMENT_LINE = re.compile(r"^\s*(?:#|//)\s?.*$", re.M)


# ONE normalisation applied identically to the benchmark side and the corpus side
# before shingling (de review 2026-09-11): strip full-line comments, collapse all
# whitespace runs, then split ON WHITESPACE into word tokens. This is a fixed,
# vocabulary-independent split (not the BPE tokenizer), so the decontam fingerprint
# never depends on a vocab. We deliberately do NOT split punctuation into its own
# tokens: that makes "13 tokens" span only 4-9 whitespace tokens of code (x=[1,2,3]
# is 17 punct-tokens vs 9 whitespace tokens) and over-fires on generic idioms
# (measured: py_rp1t 3709 rows / 184 generic problems vs 7 / 9 under whitespace) --
# the no-allowlist gate must drop verbatim GATE ANSWERS, not every short idiom.
def normalise(s):
    """The single normaliser: full-line comment strip, whitespace collapse."""
    s = _COMMENT_LINE.sub("", s or "")
    return _WS.sub(" ", s).strip()


def tokenize(s):
    """Fixed word-level split on whitespace, vocab-independent."""
    s = (s or "").strip()
    return _WS.split(s) if s else []


def ngrams(text, n=N):
    """Set of n-word-token windows; normalise() is applied to both sides identically."""
    toks = tokenize(normalise(text))
    if len(toks) < n:
        return set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


class Decontaminator:
    """Holds the gate 13-gram sets and answers row membership.

    parts[problem_id] = {part_name: set_of_grams}; hit() returns the first match.
    """

    def __init__(self, parts):
        # flat list of (problem_id, part, grams) kept in a stable order for the
        # deterministic selftest and for "which problem dropped this row".
        self.parts = parts

    @classmethod
    def from_benchmarks(cls, humaneval=HUMANEVAL, mbpp=MBPP):
        if not os.path.exists(humaneval):
            raise SystemExit(f"decontam benchmark missing: {humaneval}")
        if not os.path.exists(mbpp):
            raise SystemExit(f"decontam benchmark missing: {mbpp}")
        parts = {}

        def add(pid, part, text):
            g = ngrams(text)
            if g:
                parts.setdefault(pid, {})[part] = g

        with open(humaneval, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                tid = f"humaneval:{r['task_id']}"
                add(tid, "prompt", r.get("prompt", ""))
                add(tid, "solution", r.get("canonical_solution", ""))
                add(tid, "test", r.get("test", ""))
        with open(mbpp, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                tid = f"mbpp:{r['task_id']}"
                add(tid, "prompt", r.get("text", ""))
                add(tid, "solution", r.get("code", ""))
        return cls(parts)

    @classmethod
    def load_default(cls, root=None):
        """Build from the repo benchmark paths (or an explicit root)."""
        if root is None:
            return cls.from_benchmarks()
        return cls.from_benchmarks(
            os.path.join(root, "data", "eval", "humaneval", "humaneval_164.jsonl"),
            os.path.join(root, "data", "eval", "mbpp_holdouts.jsonl"),
        )

    def hit(self, content):
        """Return {problem, part} for the first gate 13-gram in content, else None.

        One normalise() is applied to BOTH sides, so this is symmetric by
        construction. Shingles prompts, canonical solutions and tests -- a row holding
        only the reference body (or only the problem statement) still drops.
        """
        if not content:
            return None
        g = ngrams(content)
        if not g:
            return None
        for pid, pmap in self.parts.items():
            for part, grams in pmap.items():
                if grams and (g & grams):
                    return {"problem": pid, "part": part}
        return None

    def keeps(self, content):
        return self.hit(content) is None


# convenience function boundary for callers that want one assembled-doc boolean
def keeps(content, decon=None):
    decon = decon or _DEFAULT
    return decon.keeps(content)


_DEFAULT = None


def default():
    """Lazy process-wide Decontaminator (build the gram set once)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Decontaminator.load_default()
    return _DEFAULT


def decontam_fp(*paths):
    """Content fingerprint of this module + the gate files, for the build stamp."""
    import hashlib

    h = hashlib.sha256()
    for p in [os.path.abspath(__file__), *paths]:
        if os.path.exists(p):
            with open(p, "rb") as bf:
                h.update(bf.read())
    return h.hexdigest()[:16]


def _selftest():
    # Pure-logic known answers use a SYNTHETIC gate so the selftest runs in CI where the
    # gitignored benchmark files are absent; the real fib/HumanEval check runs where the
    # data exists (the pod) as end-to-end confirmation.
    sol = ("if n == 0:\n        return 0\n    if n == 1:\n        return 1\n"
           "    return fib(n - 1) + fib(n - 2)\n")
    prompt = ("Return the nth Fibonacci number using a recursive implementation for a "
              "positive integer value n.")
    d = Decontaminator({
        "synth:fib": {"solution": ngrams(sol), "prompt": ngrams(prompt), "test": ngrams("assert fib(0) == 0 assert fib(1) == 1 assert fib(2) == 1 assert fib(9) == 34 assert fib(10) == 55")},
    })
    # (a) verbatim reference BODY drops even with no prompt present
    h = d.hit("def fib(n):\n" + sol + "print(fib(9))")
    assert h == {"problem": "synth:fib", "part": "solution"}, h
    # (b) a row carrying only the problem statement (as a docstring) drops
    h2 = d.hit('def f():\n    """' + prompt + '"""\n    pass')
    assert h2 is not None and h2["part"] == "prompt", h2
    # (c) a row carrying only the TEST asserts drops
    h3 = d.hit("def check():\n    assert fib(0) == 0\n    assert fib(1) == 1\n    assert fib(2) == 1\n    assert fib(9) == 34\n    assert fib(10) == 55")
    assert h3 is not None and h3["part"] == "test", h3
    clean = ("def load_config(path):\n    with open(path) as fh:\n"
             "        return dict(line.strip().split('=') for line in fh if '=' in line)")
    assert d.keeps(clean), "unrelated code must keep"
    assert d.keeps("x = 1") and d.keeps("")
    # ONE normalisation on both sides: stripping a full-line comment cannot create a match
    assert ngrams("# a comment line here\nx y z w\n") == ngrams("x y z w")
    # word split is on whitespace (punctuation stays attached to its word)
    assert tokenize("a, b\nc  d") == ["a,", "b", "c", "d"]
    assert keeps(clean, d) and not keeps("def fib(n):\n" + sol, d)

    if os.path.exists(HUMANEVAL) and os.path.exists(MBPP):
        real = Decontaminator.load_default()
        hf = real.hit("def fibonacci(n):\n"
                      "    if n == 0:\n        return 0\n    if n == 1:\n        return 1\n"
                      "    return fibonacci(n-1) + fibonacci(n-2)")
        assert hf is not None and hf["problem"] == "humaneval:HumanEval/55", hf
        print("decontam_ngram selftest OK (synthetic + real HumanEval/55)")
    else:
        print("decontam_ngram selftest OK (synthetic only; benchmark files absent)")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit("filters/decontam_ngram.py is a library; run with --selftest")
