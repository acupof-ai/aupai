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
  - the union's file set equals the plan, so a stray .jsonl left in the directory is refused
    rather than counted as supply. No count can catch that: a count over whatever is there is
    a true count of whatever is there.
  - every source inode is distinct, so two sources that are already hardlinks of each other
    are refused rather than double-counted. b2v2_dd was deduped against dd09, so today's two
    sources are disjoint by how they were built -- a third source would not inherit that.

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
    """The .jsonl shards in d, REFUSING a case variant rather than filtering it away.

    b0's review of 3b-17 (2026-09-08) found the hole this closes, and it is not the obvious
    one. The file-set comparison in build() cannot catch a `.JSONL` shard, because BOTH sides
    of that comparison come from this function: `want` is built from the plan, which came from
    shards(), and `got` is set(shards(out)). A file this filter cannot see is absent from both
    sets, so the sets agree about a plan that never contained it. Measured on a source holding
    src_a_000.jsonl + src_a_001.JSONL: the build printed "file set == plan, 2 shards" and
    passed, having silently dropped one shard's documents from the supply.

    Refusing beats lowercasing the test. Every other reader is case-sensitive too --
    count_dir.py globs *.jsonl and train.py's shard scan globs *.jsonl, 719 such sites across
    190 files measured 2026-09-08 -- so accepting a variant here would put a shard into a
    union that the trainer still cannot read. The name has to be fixed on disk, once, and a
    refusal says so where a filter says nothing.

    NOT live today: zero case variants exist under data/corpus or data/raw on the pod, and the
    RedPajama manifest names are all lowercase. The reachable entry point is
    fetch_rp1t_batch.py:39, `dst = os.path.join(a.out, f)`, which takes the filename verbatim
    from the manifest list -- so a remote name's casing lands on disk unchanged.
    """
    names = sorted(f for f in os.listdir(d) if not f.startswith("."))
    variants = [f for f in names if f.lower().endswith(".jsonl") and not f.endswith(".jsonl")]
    if variants:
        raise SystemExit(
            f"REFUSE: {d} holds {len(variants)} shard(s) whose extension is not lowercase "
            f".jsonl: {variants[:5]}. Every reader here is case-sensitive (count_dir globs "
            f"*.jsonl, train.py's shard scan globs *.jsonl), so such a file is counted by "
            f"nobody and its documents are silent under-supply. The file-set check below "
            f"cannot catch it either: both sides of that comparison come from this function, "
            f"so a name it cannot see is missing from both. Rename it on disk."
        )
    return [f for f in names if f.endswith(".jsonl")]


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

    # THE FILE SET, which no count can check. A count over whatever the link pass produced is
    # self-consistent no matter what that pass dropped OR what was already sitting in the
    # directory: count_dir globs *.jsonl, so a stray shard from an earlier run, or one left by
    # a different source, is counted as supply and the total still looks well-formed. Only
    # comparing the sets catches it. b0 ran exactly this comparison by hand when reviewing
    # 3b-17, because the script did not (2026-09-08).
    #
    # It compares NAMES, and that is all it can compare: the link loop above already refuses
    # any planned name whose st_ino differs from its source, so by the time this runs every
    # name in the plan is inode-verified and only an UNPLANNED name can be present. Carrying
    # inodes in these sets looked stronger and was unreachable -- a name-only mutant survived
    # the selftest (2026-09-08), which is what proved the inode half dead rather than strict.
    #
    # For the same reason `want <= got` always holds -- the loop creates every planned name --
    # so `missing` is empty on every reachable path and a mutant comparing len() instead of
    # membership also survives. Both are kept: the loop is what makes them unreachable, and a
    # refactor there should not silently turn this into a cardinality check.
    want = {f for _, f in plan}
    got = set(shards(out))
    if got != want:
        extra = sorted(got - want)
        missing = sorted(want - got)
        raise SystemExit(
            f"REFUSE: {out_name}'s file set is not the plan. "
            f"{len(extra)} came from no source: {extra[:5]}; "
            f"{len(missing)} planned but absent: {missing[:5]}. Any count over this directory "
            f"would be self-consistent and wrong."
        )
    # DISTINCT INODES, which the name comparison cannot see: two sources may hold shards that
    # are already hardlinks of each other, and then two names in the union are one file. Every
    # name is present, every link verified, and the count doubles that file's tokens.
    inos = [os.stat(src).st_ino for src, _ in plan]
    if len(set(inos)) != len(inos):
        dup = sorted(f for src, f in plan if inos.count(os.stat(src).st_ino) > 1)
        raise SystemExit(
            f"REFUSE: {out_name} would link one inode under two names -- {len(dup)} shards "
            f"share an inode with another ({dup[:6]}), so a count over the union charges that "
            f"file's tokens twice. The sources are not disjoint."
        )
    print(f"file set == plan, {len(got)} shards, {len(set(inos))} distinct inodes", flush=True)

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

        # REFUSAL 4: a stray shard already in the output directory. The world b0's review had
        # to build by hand: a .jsonl that came from no source, which count_dir globs and counts
        # as supply. The count stays self-consistent -- it is a true count of what is there --
        # so only the file-set comparison catches it, and the token gate does NOT (the stray's
        # tokens are simply included). Asserted in that order: set check first, then that a
        # count-only build would have accepted it.
        u5 = os.path.join(corpus, "u5")
        os.makedirs(u5)
        stray = os.path.join(u5, "src_a_999.jsonl")
        with open(stray, "w", encoding="utf-8") as f:
            f.write(json.dumps({"content": "not from any source"}) + "\n")
        try:
            build(corpus, ("src_a", "src_b"), "u5", None, None, nw=2)
        except SystemExit as e:
            assert "file set is not the plan" in str(e), e
            assert "src_a_999.jsonl" in str(e), e
        else:
            raise AssertionError("a stray shard in the output must refuse, it did not")
        # and the control that shows why the set check is needed at all: the stray is real
        # supply to a counter, so a build gated only on tokens would have passed it.
        from count_dir import count_dir as _cd

        assert _cd(u5, nw=2, quiet=True)["docs"] == 7, "the stray must be counted, else no defect"

        # REFUSAL 6: a case-variant extension. b0's review found this and it is the one world
        # where the file-set check is structurally blind rather than merely silent: `want` and
        # `got` are BOTH derived from shards(), so a name shards() cannot see is absent from both
        # and the sets agree about a plan that never held it. The control below is the whole
        # point -- with the old filtering shards(), this world PASSED at "file set == plan,
        # 2 shards" while one shard's documents were dropped from the supply.
        u7src = os.path.join(corpus, "src_case")
        os.makedirs(u7src)
        for nm, body in (("src_case_000.jsonl", "lower"), ("src_case_001.JSONL", "upper")):
            with open(os.path.join(u7src, nm), "w", encoding="utf-8") as f:
                f.write(json.dumps({"content": body}) + "\n")
        with open(os.path.join(u7src, "build_corpus_stats.json"), "w") as f:
            json.dump({"domain": "src_case", "tokens": 0}, f)
        try:
            build(corpus, ("src_case",), "u7", None, None, nw=2)
        except SystemExit as e:
            assert "not lowercase" in str(e), e
            assert "src_case_001.JSONL" in str(e), e
        else:
            raise AssertionError("a case-variant shard must refuse, it did not")
        # the control: the variant holds a real document, so what was dropped was supply. Read it
        # directly rather than through shards(), which is the function under test.
        assert sum(1 for _ in open(os.path.join(u7src, "src_case_001.JSONL"))) == 1, (
            "the case-variant shard must hold a document, else nothing was under-counted"
        )

        # REFUSAL 5: two sources that are already hardlinks of each other. The world for the
        # distinct-inode check, and the only one it can have: every planned name is present,
        # every link is st_ino-verified against its source, the file set equals the plan -- and
        # one file is counted under two names. No name comparison can see this; the token gate
        # cannot either, because the doubled tokens ARE in the directory. b2v2_dd was deduped
        # against dd09 so the real sources are disjoint, but that is a property of how they
        # were built, not of this script, and a third source would not inherit it.
        os.makedirs(os.path.join(corpus, "src_d"))
        os.link(
            os.path.join(corpus, "src_a", "src_a_000.jsonl"),
            os.path.join(corpus, "src_d", "src_d_000.jsonl"),
        )
        try:
            build(corpus, ("src_a", "src_d"), "u6", None, None, nw=2)
        except SystemExit as e:
            assert "one inode under two names" in str(e), e
            assert "src_a_000.jsonl" in str(e) and "src_d_000.jsonl" in str(e), e
        else:
            raise AssertionError("two sources sharing an inode must refuse, it did not")
        # the control: the union really does double-count, so the refusal is load-bearing. The
        # failed build left both links in place, and a counter charges 3 documents for the 2
        # distinct files behind them -- src_a_000 and src_d_000 are one inode.
        u6 = os.path.join(corpus, "u6")
        n6 = _cd(u6, nw=2, quiet=True)["docs"]
        n_ino = len({os.stat(os.path.join(u6, f)).st_ino for f in shards(u6)})
        assert (n6, n_ino) == (3, 2), (
            f"the shared inode must be counted under both names: {n6} docs over {n_ino} "
            f"distinct inodes, expected 3 over 2 -- else nothing was double-counted"
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print(
        f"build_dd09_full selftest OK: 6 docs over 6 shards, {want_tokens} tokens counted over "
        f"the union (not summed), every shard verified st_ino-identical to its source, file "
        f"set asserted equal to the plan and every source inode distinct, idempotent on "
        f"re-run; refuses a name collision, a copy, a short count, a stray shard a count "
        f"alone would accept, two sources that are already links of each other, and a "
        f"case-variant extension that both sides of the file-set check are blind to"
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
