#!/usr/bin/env python3
# restartable: a pure counter -- reads shards, writes one json, holds no lock. An interrupt
# costs the tokenizing done so far and nothing else.
"""Exact token, document and packed-row count over a corpus directory. Full population.

Why this exists beside scripts/count_tokens.py rather than inside it: count_shards is the
single-process definition of the convention and is called from build_corpus at stamp time,
where a 32-worker pool would be wrong. This is the batch reader for a whole directory --
same convention, same count_docs, a process pool and a fingerprint. The convention lives in
one place and both call it.

It was pod-only and untracked until 2026-09-08. That is the shape that produced the
en_c4_30b defect: its stamp cited /tmp/count_30b.py, so nobody could read the counter to
find the missing <eos>. A counter whose path cannot be re-read is a counter whose output
cannot be audited, so this one is in the tree.

    python3 scripts/count_dir.py data/corpus/<domain> [workers]
    python3 scripts/count_dir.py --selftest
"""

import glob
import json
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "datagen"))
TOK = os.path.join(ROOT, "data", "tokenizer.json")
SEQ = 4097
_tok = None


def _init():
    global _tok
    from tokenizers import Tokenizer

    _tok = Tokenizer.from_file(TOK)


def _one(p):
    from count_tokens import count_docs

    texts = []
    with open(p, "rb") as f:
        raw = f.read()
    # split("\n"), never splitlines(): splitlines breaks on U+2028/U+2029, which occur
    # INSIDE json string values in this corpus, so a valid row becomes 2+ unparseable
    # fragments and its document is silently dropped from the count. Measured on
    # code_rp1t_dd09 2026-09-07: 1,312 fragments from 288 documents.
    for line in raw.decode("utf-8", "replace").split("\n"):
        if not line.strip():
            continue
        try:
            texts.append(json.loads(line)["content"])
        except (json.JSONDecodeError, KeyError):
            continue
    docs = len(texts)
    toks = 0
    for i in range(0, docs, 2000):
        toks += count_docs(texts[i : i + 2000], _tok)
    return p, toks, docs, len(raw)


def shard_paths(d):
    """The shards a supply count must read.

    Shard prefix is the SOURCE domain, not the directory: code_rp1t_dd09/ holds
    code_rp1t_*.jsonl, so glob every non-stats .jsonl rather than the directory name.
    holdout_slice_<phase>.jsonl is a frozen slice record, not corpus: it carries no
    `content` key, so it adds 0 tokens and 0 docs but WOULD add a shard to n_shards and its
    bytes to the total. fp_dir hashes it (it is part of the stamped domain); the supply
    count must not.
    """
    return sorted(
        p
        for p in glob.glob(os.path.join(d, "*.jsonl"))
        if os.path.basename(p) != "build_corpus_stats.json"
        and not os.path.basename(p).startswith("holdout_slice_")
        and not os.path.basename(p).startswith(".")
    )


def count_dir(d, nw=24, quiet=False):
    shards = shard_paths(d)
    assert shards, f"no shards in {d}"
    t0 = time.time()
    tot = docs = nbytes = 0
    with mp.Pool(nw, initializer=_init) as pool:
        for i, (_p, t, n, b) in enumerate(pool.imap_unordered(_one, shards), 1):
            tot += t
            docs += n
            nbytes += b
            if not quiet and (i % 20 == 0 or i == len(shards)):
                el = time.time() - t0
                print(
                    f"  {i}/{len(shards)} | {tot:,} tok | {docs:,} docs | "
                    f"{nbytes / 1e6 / el:.1f} MB/s | {el:.0f}s",
                    flush=True,
                )
    from corpus_fingerprint import fp_dir
    from count_tokens import CONVENTION

    return {
        "dir": d,
        "n_shards": len(shards),
        "docs": docs,
        "bytes": nbytes,
        "tokens": tot,
        "packed_rows": tot // SEQ,
        "fingerprint": fp_dir(d),
        "convention": CONVENTION,
        "counter": f"scripts/count_dir.py -> {os.path.basename(TOK)}",
        "wall_s": round(time.time() - t0, 1),
    }


def _selftest():
    """Known answer plus the two exclusions, on a real directory shape.

    The tokens assertion is the same shape count_tokens' selftest makes and for the same
    reason -- N documents exceed the no-terminator count by exactly N -- because that is
    the defect this counter's output is used to correct.
    """
    import tempfile

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOK)
    rows = ["hello world", "def f(x):\n    return x + 1", "中文测试", "a b"]
    with tempfile.TemporaryDirectory() as d:
        for i in range(2):
            with open(os.path.join(d, f"src_{i:03d}.jsonl"), "w", encoding="utf-8") as f:
                for t in rows:
                    f.write(json.dumps({"content": t}, ensure_ascii=False) + "\n")
        # Both exclusions, each written as the real thing is: a slice record with no
        # `content`, and the stats file. Either one counted would move n_shards and bytes.
        with open(os.path.join(d, "holdout_slice_p1.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"ids": [1, 2, 3]}) + "\n")
        with open(os.path.join(d, "build_corpus_stats.json"), "w") as f:
            json.dump({"domain": "x"}, f)

        got = count_dir(d, nw=2, quiet=True)
        bare = 2 * sum(len(e.ids) for e in tok.encode_batch(rows))
        want = bare + 2 * len(rows)
        assert got["n_shards"] == 2, f"n_shards {got['n_shards']} != 2 (exclusions leaked)"
        assert got["docs"] == 2 * len(rows), f"docs {got['docs']} != {2 * len(rows)}"
        assert got["tokens"] == want, f"tokens {got['tokens']} != {bare} + {2 * len(rows)}"

        # Negative control in the direction that actually failed on the pod: counting the
        # slice record must change the answer, so the exclusion is load-bearing and not a
        # no-op the assertion above would pass either way.
        leaked = sorted(glob.glob(os.path.join(d, "*.jsonl")))
        assert len(leaked) == 3, "fixture must hold 3 .jsonl for the control to mean anything"
        assert sum(os.path.getsize(p) for p in leaked) > got["bytes"], (
            "negative control: including holdout_slice_* must inflate bytes, it did not"
        )
    print(
        f"count_dir selftest OK: 2 shards, {got['docs']} docs, {bare} ids + "
        f"{2 * len(rows)} <eos> = {got['tokens']}; holdout_slice_* and stats excluded"
    )
    return 0


def main():
    if "--selftest" in sys.argv:
        return _selftest()
    d = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    out = count_dir(d, nw)
    print(json.dumps(out, indent=1), flush=True)
    with open(os.path.join(ROOT, "runs", f"count_{os.path.basename(d)}.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
