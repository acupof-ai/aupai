#!/usr/bin/env python3
# restartable: hardlinks are idempotent (skip-if-exists) and the count is a pure read, so an
# interrupt costs at most the linking done so far and a re-run resumes by skipping.
"""3b-17: one directory expressing dd09 + b2v2_dd as a single supply, by hardlink.

train.py globs one directory per domain (the `glob.glob(... "corpus", domain, "*.jsonl")` in
its shard scan) and write_mix_500m._rp1t_tokens reads one stamp, so a mix cannot name two
directories as one domain. Hardlinks give the union a directory of its own without copying
30 GB.

WHAT MAKES THIS SAFE, asserted rather than assumed:
  - no name collision: dd09 holds code_rp1t_*.jsonl, b2v2_dd holds code_rp1t_b2v2_dd_*.jsonl.
    A collision would silently drop a shard, so it refuses instead.
  - sources untouched: their fingerprints are read BEFORE and AFTER and must match. A hardlink
    cannot modify its target, but the assertion is cheap and the claim is what matters.
  - st_ino equality per link, so "hardlink" is verified and not just intended -- a copy would
    pass every other check here while costing 30 GB on a filesystem at 93%.

The stamp's tokens are COUNTED over the union, never summed from the two source stamps: a
sum cannot see a shard the link pass dropped, which is the one failure this script can have.
"""

import json
import os
import sys

# ROOT is where THIS FILE lives, so --selftest runs on a laptop; POD_ROOT is where the corpus
# is. They differ on purpose: the modules are tracked, the 30 GB is not.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POD_ROOT = "/work/aupai"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "datagen"))

SOURCES = ("code_rp1t_dd09", "code_rp1t_b2v2_dd")
OUT = "code_rp1t_dd09_full"
EXPECT_TOKENS = 9837521903
EXPECT_DOCS = 3434322 + 2103485


def shards(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".jsonl") and not f.startswith("."))


def build(corpus, sources=SOURCES, out_name=OUT, expect_tokens=None, expect_docs=None, nw=32):
    from corpus_fingerprint import fp_dir

    src_dirs = [os.path.join(corpus, s) for s in sources]
    before = {s: fp_dir(d) for s, d in zip(sources, src_dirs, strict=True)}
    print(json.dumps({"source_fingerprints_before": before}, indent=1), flush=True)

    plan, seen = [], {}
    for s, d in zip(sources, src_dirs, strict=True):
        for f in shards(d):
            if f in seen:
                raise SystemExit(
                    f"REFUSE: shard name {f!r} is in both {seen[f]} and {s}. Linking both into "
                    f"one directory would drop one of them silently."
                )
            seen[f] = s
            plan.append((os.path.join(d, f), f))
    print(f"{len(plan)} shards, no name collisions", flush=True)

    out = os.path.join(corpus, out_name)
    os.makedirs(out, exist_ok=True)
    linked = already = 0
    for src, name in plan:
        dst = os.path.join(out, name)
        if os.path.exists(dst):
            if os.stat(dst).st_ino != os.stat(src).st_ino:
                raise SystemExit(f"REFUSE: {name} exists in {out_name} and is NOT a link to {src}")
            already += 1
            continue
        os.link(src, dst)
        if os.stat(dst).st_ino != os.stat(src).st_ino:
            raise SystemExit(f"REFUSE: {name} did not hardlink (st_ino differs) -- a copy?")
        linked += 1
    print(f"linked {linked}, already present {already}", flush=True)

    after = {s: fp_dir(d) for s, d in zip(sources, src_dirs, strict=True)}
    if after != before:
        raise SystemExit(f"REFUSE: a source directory changed: {before} -> {after}")
    print("source fingerprints unchanged", flush=True)

    from count_dir import count_dir
    from count_tokens import CONVENTION

    got = count_dir(out, nw=nw, quiet=True)
    tok, docs = got["tokens"], got["docs"]
    if expect_tokens is not None:
        print(
            json.dumps({"counted": got, "expected_tokens": expect_tokens}, indent=1),
            flush=True,
        )
        if tok != expect_tokens or docs != expect_docs:
            raise SystemExit(
                f"REFUSE: counted {tok:,} tok / {docs:,} docs, expected {expect_tokens:,} / "
                f"{expect_docs:,} (tokens {tok - expect_tokens:+,}, docs {docs - expect_docs:+,})."
                f" The union is NOT the two sources; do not stamp it."
            )

    sha = os.environ.get("COUNT_TOKENS_SHA", "")
    stats = {
        "domain": out_name,
        "producer": "datagen/build_dd09_full.py (task 3b-17), hardlink union of " + " + ".join(sources),
        "producer_note": "NOT a build_corpus product: no clean ran here. Every shard is a "
        "hardlink to a shard in one of the two sources, which are unmodified "
        "and keep their own stamps. Delete this directory and nothing is lost. "
        "reasons/kept/kept_chars/workers do not exist for it.",
        "inputs": {
            s: {
                "srcfp": before[s],
                "n_shards": len(shards(os.path.join(corpus, s))),
                "role": "hardlinked in whole; not rewritten, not modified",
            }
            for s in sources
        },
        "n_shards": got["n_shards"],
        "docs": docs,
        "bytes": got["bytes"],
        "tokens": tok,
        "tokens_status": "measured",
        "tokens_config": (
            f"{CONVENTION}; scripts/count_dir.py"
            + (f"@{sha}" if sha else "")
            + f" over all {got['n_shards']} shards of the union, full population, no sampling. "
            f"COUNTED over the union, not summed from the source stamps."
        ),
        "packed_rows": got["packed_rows"],
        "fingerprint": got["fingerprint"],
        "filters": "near-dedup-th0.9 (dd09) + near-dedup-th0.9-vs-dd09 (b2v2_dd)",
        "filters_note": "the two sources were deduped separately and against each other; this "
        "directory adds no filtering of its own",
    }
    p = os.path.join(out, "build_corpus_stats.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    return stats


def _selftest():
    """Two real source directories, a union, and the three refusals.

    The count assertion is the one that matters and it is a KNOWN ANSWER, not a sum: the
    union's tokens must equal count_docs over all six documents, computed here from the text
    rather than by adding the two source stamps -- because a sum cannot see a shard the link
    pass dropped, which is this script's only real failure mode.
    """
    import shutil
    import tempfile

    from tokenizers import Tokenizer

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from count_tokens import count_docs

    tok_path = os.path.join(ROOT, "data", "tokenizer.json")
    if not os.path.isfile(tok_path):
        raise SystemExit(f"tokenizer missing at {tok_path} (gitignored; copy it in)")
    tk = Tokenizer.from_file(tok_path)
    A = ["alpha one", "def f(x):\n    return x"]
    B = ["beta two", "中文", "gamma three", "d"]

    d = tempfile.mkdtemp()
    try:
        corpus = os.path.join(d, "data", "corpus")
        for name, rows in (("src_a", A), ("src_b", B)):
            os.makedirs(os.path.join(corpus, name))
            for i, t in enumerate(rows):
                p = os.path.join(corpus, name, f"{name}_{i:03d}.jsonl")
                with open(p, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"content": t}, ensure_ascii=False) + "\n")
            with open(os.path.join(corpus, name, "build_corpus_stats.json"), "w") as f:
                json.dump({"domain": name, "tokens": 0}, f)

        want_tokens = count_docs(A + B, tk)
        st = build(corpus, ("src_a", "src_b"), "u", want_tokens, len(A) + len(B), nw=2)
        assert st["docs"] == 6, st["docs"]
        assert st["n_shards"] == 6, st["n_shards"]
        assert st["tokens"] == want_tokens, f"{st['tokens']} != {want_tokens}"

        # every shard is a link, not a copy -- the check that separates 0 bytes from 30 GB
        u = os.path.join(corpus, "u")
        for f in shards(u):
            src = os.path.join(corpus, "src_a" if f.startswith("src_a") else "src_b", f)
            assert os.stat(os.path.join(u, f)).st_ino == os.stat(src).st_ino, f
        assert os.stat(os.path.join(u, shards(u)[0])).st_nlink >= 2

        # idempotent: a second run links nothing and still stamps the same numbers
        st2 = build(corpus, ("src_a", "src_b"), "u", want_tokens, 6, nw=2)
        assert st2["tokens"] == st["tokens"] and st2["fingerprint"] == st["fingerprint"]

        # REFUSAL 1: a name in both sources. Written as the real risk is -- same basename,
        # different content -- because that is what silently drops a shard.
        os.makedirs(os.path.join(corpus, "src_c"))
        clash = "src_a_000.jsonl"
        with open(os.path.join(corpus, "src_c", clash), "w", encoding="utf-8") as f:
            f.write(json.dumps({"content": "different bytes entirely"}) + "\n")
        try:
            build(corpus, ("src_a", "src_c"), "u2", None, None, nw=2)
        except SystemExit as e:
            assert "is in both" in str(e), e
        else:
            raise AssertionError("a colliding shard name must refuse, it did not")

        # REFUSAL 2: an existing entry that is a COPY rather than a link
        u3 = os.path.join(corpus, "u3")
        os.makedirs(u3)
        shutil.copy(os.path.join(corpus, "src_a", "src_a_000.jsonl"), u3)
        try:
            build(corpus, ("src_a", "src_b"), "u3", None, None, nw=2)
        except SystemExit as e:
            assert "NOT a link" in str(e), e
        else:
            raise AssertionError("a copied shard must refuse, it did not")

        # REFUSAL 3: the count not matching. This is the acceptance gate, so it gets a world:
        # one shard is left out of the union by hand, and the wrong total must refuse.
        u4 = os.path.join(corpus, "u4")
        os.makedirs(u4)
        for f in shards(os.path.join(corpus, "src_a")):
            os.link(os.path.join(corpus, "src_a", f), os.path.join(u4, f))
        try:
            build(corpus, ("src_a",), "u4", want_tokens, 6, nw=2)
        except SystemExit as e:
            assert "REFUSE: counted" in str(e) and "do not stamp it" in str(e), e
        else:
            raise AssertionError("a short union must refuse, it did not")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print(
        f"build_dd09_full selftest OK: 6 docs over 6 shards, {want_tokens} tokens counted over "
        f"the union (not summed), every shard verified st_ino-identical to its source, "
        f"idempotent on re-run; refuses a name collision, a copy, and a short count"
    )
    return 0


def main():
    if "--selftest" in sys.argv:
        return _selftest()
    st = build(os.path.join(POD_ROOT, "data", "corpus"), SOURCES, OUT, EXPECT_TOKENS, EXPECT_DOCS)
    print(json.dumps(st, indent=1), flush=True)
    print("3b-17 OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
