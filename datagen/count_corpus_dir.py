#!/usr/bin/env python3
"""Exact frozen-tokenizer token count over one corpus dir -- every shard, no sampling.

The counter behind cs.code_rp1t_landed and cs.code_dedup08_landed. It exists as a tracked
file because facts_well_formed refuses a fact whose source names a path that is not in the
tree: a number nobody can recount is not a measurement, and the /tmp script that produced
the first version of these counts was exactly that.

Why not datagen/count_code_dirs.py, which already does this: its SHARDS constant is bound
to code_rp1t at import time through count_cleaned_code, so it cannot be pointed at
code_dedup08 or at the ten other domains whose stamps carry a 3-shard extrapolation. Same
method (frozen tokenizer, ids over content||text, multiprocessing Pool), one argument.

    python3 datagen/count_corpus_dir.py code_dedup08 [--root /work/aupai] [--pool 16]

Prints one json line: domain, shards, docs, tokens, text_bytes, tok_per_byte.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from count_tokens import CONVENTION  # noqa: E402

_TOK = None


def _init(tok_path):
    global _TOK
    from tokenizers import Tokenizer

    _TOK = Tokenizer.from_file(tok_path)


def _count_shard(shard):
    from count_tokens import count_docs

    docs = tokens = nbytes = 0
    texts = []
    with open(shard, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("content") or d.get("text") or ""
            if not t:
                continue
            docs += 1
            nbytes += len(t.encode("utf-8"))
            texts.append(t)
            if len(texts) >= 2000:
                tokens += count_docs(texts, _TOK)
                texts = []
    tokens += count_docs(texts, _TOK)
    return docs, tokens, nbytes


def count(domain, root=ROOT, pool=16):
    tok_path = os.path.join(root, "data", "tokenizer.json")
    shards = sorted(glob.glob(os.path.join(root, "data", "corpus", domain, "*.jsonl")))
    if not shards:
        raise SystemExit(f"no shards under {root}/data/corpus/{domain}/")
    with mp.Pool(pool, initializer=_init, initargs=(tok_path,)) as p:
        counts = p.map(_count_shard, shards)
    docs = sum(c[0] for c in counts)
    tokens = sum(c[1] for c in counts)
    nbytes = sum(c[2] for c in counts)
    return {
        "domain": domain,
        "shards": len(shards),
        "docs": docs,
        "tokens": tokens,
        "text_bytes": nbytes,
        "tok_per_byte": round(tokens / nbytes, 5) if nbytes else None,
        "convention": CONVENTION,
        "counter": "datagen/count_corpus_dir.py -> scripts/count_tokens.count_docs",
    }


def _selftest():
    """A two-shard world with a known token count, and two traps it must not fall into.

    COVERAGE, NOT JUST INDEPENDENCE. The first version of this test asserted only that a
    full-directory count equals the sum over shards, with an independent oracle and a real
    negative control -- and it stayed green for a week while the counter omitted the
    terminator, because both sides used "ids, no eos" as their shared definition of a token
    and no assertion named which definition was correct. An oracle can be independent and
    still be silent on the property that later matters (58, 2026-09-08). So the expected
    total is written as `ids + one per document`, arithmetic the subject never performs:

      trap 1  a counter that skips a shard          -> the first-shard-only control differs
      trap 2  a counter that drops the terminator   -> want_tokens is short by exactly N

    Shard 1 is deliberately much larger than shard 0, so a first-shard-only count cannot
    land on the right total by luck.
    """
    import shutil
    import tempfile

    fails = []
    d = tempfile.mkdtemp()
    try:
        dom = os.path.join(d, "data", "corpus", "tiny")
        os.makedirs(dom)
        shutil.copy(os.path.join(ROOT, "data", "tokenizer.json"), os.path.join(d, "data", "tokenizer.json"))
        rows0 = [{"text": "hello world"}]
        rows1 = [{"content": "def f(x):\n    return x + 1\n"} for _ in range(9)]
        for i, rows in ((0, rows0), (1, rows1)):
            with open(os.path.join(dom, f"tiny_{i}.jsonl"), "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(r) for r in rows) + "\n")
        # a row with neither key, and a blank line: both skipped, neither crashes
        with open(os.path.join(dom, "tiny_2.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"other": "x"}) + "\n\n")

        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(os.path.join(d, "data", "tokenizer.json"))
        want_docs = 10
        bare = len(tok.encode("hello world").ids) + 9 * len(tok.encode("def f(x):\n    return x + 1\n").ids)
        want_tokens = bare + want_docs

        got = count("tiny", root=d, pool=2)
        if got["docs"] != want_docs:
            fails.append(f"docs {got['docs']} != {want_docs}")
        if got["tokens"] != want_tokens:
            fails.append(
                f"tokens {got['tokens']} != {bare} ids + {want_docs} <eos> = {want_tokens}. "
                f"A difference of exactly {want_docs} is the terminator; anything else is a "
                f"skipped shard or a changed tokenizer"
            )
        if got["shards"] != 3:
            fails.append(f"shards {got['shards']} != 3")
        if got["convention"] != CONVENTION:
            fails.append(f"convention {got['convention']!r} != {CONVENTION!r}")
        only_first = _first_shard_only("tiny", d)
        if only_first == want_tokens:
            fails.append(
                "the first-shard-only control equals the full count; the world cannot detect sampling"
            )
    finally:
        shutil.rmtree(d, ignore_errors=True)

    for f in fails:
        print(f"  FAIL {f}", file=sys.stderr)
    if fails:
        print(f"count_corpus_dir selftest: {len(fails)} failure(s)", file=sys.stderr)
        return 1
    print(
        f"count_corpus_dir selftest OK: 10 docs over 3 shards, {bare} ids + {want_docs} <eos> "
        f"= {want_tokens} counted exactly; rows with no text key and blank lines skipped; the "
        f"first-shard-only control differs from the full count"
    )
    return 0


def _first_shard_only(domain, root):
    """The negative control: what a counter that sampled shard 0 would report."""
    _init(os.path.join(root, "data", "tokenizer.json"))
    shards = sorted(glob.glob(os.path.join(root, "data", "corpus", domain, "*.jsonl")))
    return _count_shard(shards[0])[1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", nargs="?")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if not a.domain:
        ap.error("domain is required")
    print(json.dumps(count(a.domain, a.root, a.pool)))
