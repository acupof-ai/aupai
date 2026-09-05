#!/usr/bin/env python3
"""Build the experiment-1 injection corpus: n exposures of a fixed 1,000-document S set.

WHAT THIS PRODUCES. One shard directory per arm, data/corpus/s_inject_n<N>/s_inject_000.jsonl,
holding the SAME 1,000 S_pool documents repeated N times, plus data/corpus/p_format/ holding the
1,000 P documents ONCE. The shards are ordinary corpus shards, so train.py tokenizes and caches
them through the normal path and no training code changes.

WHY A SHARD AND NOT A CODE CHANGE. The alternative is a hook in build_mix that splices rows into
the plan. That would be a second scheduling path used by one experiment, and the plan is where
the shuffle (train.py:2195) and the rank stripe (:2241) live -- the two things this experiment's
exposure schedule depends on. A shard reaches those through the same code the control ran.

THE UNIT IS A ROW OF seq+1 TOKENS, NOT A DOCUMENT (train.py:1899-1903 reshapes the
<eos>-separated stream to [-1, seq+1]). S documents are 104.0 tokens mean, so ~39 share a
4,097-token row and 1,000 documents pack into 25.4 rows. Two consequences the recipe has to
answer rather than ignore:

  * "n exposures" is n passes over the 1,000-document SET, so the shard holds N*1,000 documents
    and the mix reads it at epochs 1. Putting 1,000 documents in the shard and asking for N
    epochs would work too, but the epoch cap interacts with used[] bookkeeping in build_mix
    (:2181-2190) and a capped want silently reduces exposures -- writing the repeats into the
    shard makes the count a property of the data instead of the scheduler.

  * Documents sharing a row see each other unless masked. Cfg.doc_mask is True by default and
    resets KDA state and SWA per <eos>, but conv_doc_isolated is False in the control
    (train.py:319, and the control's launch line does not pass it), so the short_conv still
    convolves across the boundary -- eff.kda_document_isolation_violated, measured 48.88 at the
    block-0 output. The arms MUST match the control here: passing --conv_doc_isolated would
    change the topology relative to the checkpoint being continued. So the leak stays, identically
    in every arm and in the no-injection control arm, and the row states it as a shared condition
    rather than a confound between arms.

SHUFFLE. The repeats are interleaved by a seeded permutation of the whole N*1,000 list, not
written as N contiguous blocks. Contiguous blocks would put every exposure of a document inside a
handful of adjacent rows, so the model would see document d N times in a few hundred steps and
never again -- that measures massed repetition, and the human comparison ("how many exposures
does it take") is about exposures distributed over training. The permutation is seeded 20260905
and pinned in the shard's own header line.

REFUSES rather than overwrite: an existing shard directory is an error, because a silent rebuild
at a different seed would change the exposure schedule while every stamp still matched.

    python3 datagen/build_s_inject.py --n 1 8 64 256 --out data/corpus
    python3 datagen/build_s_inject.py --selftest
"""
import argparse
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SEED = 20260905
N_DOCS = 1000
S_POOL = os.path.join(ROOT, "data", "probes", "novel_ops", "S_pool.jsonl")
P_POOL = os.path.join(ROOT, "data", "probes", "novel_ops", "P_pool.jsonl")
# The frozen sets, facts/contamination.json#cont.novel_ops_frozen_sets.
S_POOL_SHA = "bf7e609f0672e07d0800ccaf7d431b0a11ef188fd0c36ab9e538b08b98d04c85"
P_POOL_SHA = "d5a85a05ecb03e28da3cd60cfc5c8219e0292587f9dff0dfcb1f4f1e62dd4f59"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pool(path, want_sha):
    """The pool's items, refusing a file whose sha is not the frozen one.

    Checked on CONTENT, not on the commit: the sets were regenerated twice on 2026-09-05 (once for
    the carry-rule ambiguity, once for truncated operands) and an arm built against a superseded
    build would carry labels nobody can reproduce.
    """
    got = sha256(path)
    if got != want_sha:
        raise SystemExit(
            f"REFUSING: {os.path.relpath(path, ROOT)} is sha256 {got[:16]}, not the frozen "
            f"{want_sha[:16]} (facts/contamination.json#cont.novel_ops_frozen_sets). The sets were "
            f"regenerated twice on 2026-09-05; an arm built against a superseded build carries "
            f"labels that cannot be reproduced."
        )
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("_header"):
            continue
        out.append(d)
    return out


def doc_text(item):
    """One training document: the instruction, the worked solution, the answer.

    THE SOLUTION IS INCLUDED. A document holding only "question -> 42" teaches the answer to that
    instance; the skill is the procedure, and the procedure is what the solution lines spell out
    (3 * 12 - 2 * 17 + 1 = 3, then the carry line when it fires). Excluding them would inject the
    skill's outputs while withholding the skill, and then the curve measures memorisation of 1,000
    answers -- which S_test's disjointness makes score zero at every n.
    """
    return item["instruction"] + "\n" + "\n".join(item["solution"]) + "\n" + str(item["answer"])


def build_docs(items, n_exposures, n_docs=N_DOCS, seed=SEED):
    """The document list for one arm: n_exposures interleaved passes over items[:n_docs].

    Returns [(text, source_index, exposure_index)] so the shard can record which exposure each
    line is, which is what makes an off-by-one in the count visible in the artifact.
    """
    base = items[:n_docs]
    if len(base) < n_docs:
        raise SystemExit(f"REFUSING: pool holds {len(base)} items, need {n_docs}")
    seq = [(i, e) for e in range(n_exposures) for i in range(n_docs)]
    random.Random(seed + n_exposures).shuffle(seq)
    return [(doc_text(base[i]), i, e) for i, e in seq]


#: The keys train._jsonl_content requires of every corpus line, and the one it forbids. Named
#: here rather than assumed, because it was assumed once and cost the launch: this builder wrote
#: {"text": ...} doc lines with a {"_header": true} line at the top, and train.py:1361 is
#: `json.loads(ln)["content"]` with NO header skip -- so all five injection domains raised
#: KeyError: 'content' when the caches were built on the pod, and the arm could not start.
#: Both halves were independently fatal (measured 2026-09-05: header+text, header+content, and
#: content-with-no-header -- only the third reads).
SHARD_DOC_KEY = "content"
SHARD_FORBIDDEN_KEY = "_header"


def shard_contract(path, meta_path=None):
    """Assert `path` is a corpus shard train._jsonl_content can read. Returns the doc count.

    RUN BY THE BUILDER ON ITS OWN OUTPUT, so a format drift fails at build time rather than on
    the pod an hour before a launch. The contract is not written down anywhere else: every
    natural corpus satisfies it by accident of having been produced by ShardWriter, and the
    reader states it only as a subscript. A third writer would break it the same way -- 4c's
    ruling 2026-09-05, and de lifts this same assertion into a harness check over every corpus
    directory.

    The reader's rule, exactly: EVERY non-blank line must parse as JSON and have "content".
    There is no header line and no skip, so a header in the file is a defect however well
    formed it is.
    """
    import re as _re
    problems, n = [], 0
    if not _re.search(r"_\d{3,}\.jsonl$", os.path.basename(path)):
        problems.append(f"{os.path.basename(path)} does not match train.py's SHARD_RE "
                        f"r'_\\d{{3,}}\\.jsonl$' -- _domain_seqs refuses the directory")
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"line {i} is not JSON: {e}")
                break
            if SHARD_FORBIDDEN_KEY in r:
                problems.append(
                    f"line {i} carries {SHARD_FORBIDDEN_KEY!r}. train._jsonl_content reads "
                    f"[\"{SHARD_DOC_KEY}\"] on EVERY line with no header skip, so a header line "
                    f"raises KeyError and the domain's cache cannot be built. Put it in "
                    f"<shard>.meta.json instead.")
                break
            if SHARD_DOC_KEY not in r:
                problems.append(
                    f"line {i} has no {SHARD_DOC_KEY!r} key (has {sorted(r)[:6]}). "
                    f"train._jsonl_content reads exactly that key; 'text' is what every natural "
                    f"shard does NOT use and what broke this build.")
                break
            n += 1
    if not n and not problems:
        problems.append("no document lines")
    if meta_path is not None and not os.path.exists(meta_path):
        problems.append(f"sidecar {os.path.basename(meta_path)} is absent -- the header has to "
                        f"live somewhere or the shard's provenance is lost")
    if problems:
        raise SystemExit("REFUSING: " + os.path.relpath(path, ROOT) + " violates the shard "
                         "contract:\n  " + "\n  ".join(problems))
    return n


def write_shard(out_dir, name, docs, header):
    """<out>/<name>/<name>_000.jsonl, one {"content": ...} per line, header in a sidecar.

    The shard pattern is `_\\d{3,}\\.jsonl$` (train.py SHARD_RE); a file that does not match is
    REFUSED by _domain_seqs rather than skipped, so the name matters.

    THE HEADER GOES TO <shard>.meta.json, not into the shard. train._jsonl_content reads
    ["content"] on every line with no skip, so an in-band header is a KeyError at cache-build
    time. The sidecar is not a shard by SHARD_RE, so _domain_seqs ignores it; _s_index and
    _exposure stay as extra keys on the doc lines, where the plan check reads them.
    """
    d = os.path.join(out_dir, name)
    if os.path.exists(d):
        raise SystemExit(
            f"REFUSING: {os.path.relpath(d, ROOT)} already exists. A rebuild at a different seed "
            f"changes the exposure schedule while the vocab and source stamps still match, so the "
            f"cache would look fresh. Remove it deliberately if that is what you mean."
        )
    os.makedirs(d)
    p = os.path.join(d, f"{name}_000.jsonl")
    with open(p, "w", encoding="utf-8") as fh:
        for text, si, ei in docs:
            fh.write(json.dumps({SHARD_DOC_KEY: text, "_s_index": si, "_exposure": ei},
                                ensure_ascii=False) + "\n")
    meta = os.path.join(d, f"{name}_000.jsonl.meta.json")
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in header.items() if k != "_header"}, fh,
                  ensure_ascii=False, indent=1)
    # THE BUILDER CHECKS ITS OWN OUTPUT. Not the caller's job: this is the write that has to be
    # readable, and the reader lives in another file whose contract nobody states.
    shard_contract(p, meta)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[1, 8, 64, 256],
                    help="exposure counts, one shard per value")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "corpus"))
    ap.add_argument("--n_docs", type=int, default=N_DOCS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

    s_items = load_pool(S_POOL, S_POOL_SHA)
    p_items = load_pool(P_POOL, P_POOL_SHA)
    made = []
    for n in a.n:
        docs = build_docs(s_items, n, a.n_docs, a.seed)
        header = {"_header": True, "family": "S", "exposures": n, "n_docs": a.n_docs,
                  "documents": len(docs), "seed": a.seed + n,
                  "source": "data/probes/novel_ops/S_pool.jsonl", "source_sha256": S_POOL_SHA,
                  "interleaved": "seeded permutation of all exposures, not contiguous blocks"}
        made.append(write_shard(a.out, f"s_inject_n{n}", docs, header))
    # P ONCE, identically in every arm: it is the format control, so its exposure count must not
    # vary with n or a P movement could not be read as "format learned" (the stop rule).
    p_docs = build_docs(p_items, 1, a.n_docs, a.seed)
    made.append(write_shard(a.out, "p_format", p_docs,
                            {"_header": True, "family": "P", "exposures": 1,
                             "n_docs": a.n_docs, "documents": len(p_docs), "seed": a.seed + 1,
                             "source": "data/probes/novel_ops/P_pool.jsonl",
                             "source_sha256": P_POOL_SHA}))
    for p in made:
        # No `- 1`: the header is a sidecar now, so every line in the shard is a document.
        print(f"{os.path.relpath(p, ROOT)}  {sum(1 for _ in open(p, encoding='utf-8'))} docs")
    return 0


def _selftest():
    items = [{"instruction": f"q{i}", "solution": [f"s{i}a", f"s{i}b"], "answer": i}
             for i in range(N_DOCS)]

    # 1. EXPOSURE COUNT IS EXACT AND EVERY DOCUMENT APPEARS EQUALLY OFTEN. A shard that holds
    #    n*1000 lines but distributes them unevenly measures a different n per document, and the
    #    curve's x-axis would be a mean rather than a count.
    import collections

    for n in (1, 8, 64):
        docs = build_docs(items, n)
        assert len(docs) == n * N_DOCS, (n, len(docs))
        per = collections.Counter(si for _t, si, _e in docs)
        assert set(per.values()) == {n}, (n, collections.Counter(per.values()))
        assert len(per) == N_DOCS, len(per)

    # 2. THE REPEATS ARE INTERLEAVED, NOT CONTIGUOUS. Measured as the mean gap between
    #    consecutive appearances of one document: contiguous blocks give a gap of ~1000 (one per
    #    block), and a permutation of n*1000 slots gives ~1000 too -- so the discriminating
    #    statistic is the SPREAD of the first appearance's position, not the gap. Under contiguous
    #    blocks every document first appears in the first 1,000 slots; under a permutation the
    #    first appearances spread over the whole list.
    docs = build_docs(items, 8)
    first = {}
    for pos, (_t, si, _e) in enumerate(docs):
        first.setdefault(si, pos)
    assert max(first.values()) > 2 * N_DOCS, (
        f"first appearances all land within {max(first.values())} of the start, so the exposures "
        f"are massed rather than distributed -- that measures a different thing than 'n exposures "
        f"over training'"
    )

    # 3. THE SOLUTION IS IN THE DOCUMENT. Injecting question->answer without the worked steps
    #    teaches 1,000 answers, and S_test shares no instance with S_pool, so the curve would read
    #    zero at every n for a reason that has nothing to do with exposures.
    t = doc_text(items[3])
    assert "s3a" in t and "s3b" in t and t.rstrip().endswith("3"), t

    # 4. THE FROZEN SHA IS CHECKED ON CONTENT. A wrong-build pool must be refused, not read.
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(json.dumps({"instruction": "q", "solution": ["s"], "answer": 1}) + "\n")
        bad = fh.name
    try:
        try:
            load_pool(bad, S_POOL_SHA)
            raise AssertionError("a pool whose sha does not match the frozen one was accepted")
        except SystemExit as e:
            assert "not the frozen" in str(e), str(e)
    finally:
        os.unlink(bad)

    # 5. AN EXISTING SHARD DIR IS REFUSED BY THE GUARD, not by os.makedirs. The first version of
    #    this case pre-created the directory and then called write_shard, so removing the guard
    #    gave FileExistsError from makedirs -- red, but for the wrong reason: a version that
    #    reached open() would have overwritten the shard and this case would still have "passed".
    #    So the world is a directory that exists and is WRITABLE-INTO (exist_ok), and the
    #    assertion is that the file's content is unchanged.
    d = tempfile.mkdtemp()
    try:
        sd = os.path.join(d, "s_inject_n1")
        os.makedirs(sd, exist_ok=True)
        marker = os.path.join(sd, "s_inject_n1_000.jsonl")
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write("PRE-EXISTING\n")
        try:
            write_shard(d, "s_inject_n1", [("t", 0, 0)], {"_header": True})
        except SystemExit as e:
            assert "already exists" in str(e), str(e)
        except FileExistsError as e:  # noqa: PERF203 -- the distinction IS the assertion
            raise AssertionError(
                f"the refusal came from os.makedirs, not from the guard: {e}. A build that "
                f"reached open() would have overwritten the shard and this case would still "
                f"have gone red."
            ) from e
        else:
            raise AssertionError("an existing shard directory was not refused")
        assert open(marker, encoding="utf-8").read() == "PRE-EXISTING\n", (
            "the pre-existing shard was overwritten; a rebuild at a different seed changes the "
            "exposure schedule while the vocab and source stamps still match, so the cache reads "
            "fresh and the arm trains on a schedule nobody recorded"
        )
    finally:
        import shutil

        shutil.rmtree(d, ignore_errors=True)

    # 6. THE SHARD NAME MATCHES train.py's SHARD_RE, read from train.py rather than restated:
    #    a file that does not match is REFUSED by _domain_seqs, so a rename there breaks this
    #    silently at the next build.
    import re

    src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    m = re.search(r'SHARD_RE\s*=\s*re\.compile\(\s*r?"([^"]+)"', src)
    assert m, "SHARD_RE not found in train.py -- the shard-name contract cannot be checked"
    assert re.search(m.group(1), "s_inject_n1_000.jsonl"), (
        f"the shard name this script writes does not match train.py's SHARD_RE {m.group(1)!r}, so "
        f"_domain_seqs would refuse the directory"
    )

    # 7. THE SHARD CONTRACT REFUSES BOTH HALVES OF THE DEFECT THAT COST THE LAUNCH, and accepts
    #    what this builder now writes. Each world is a file, checked through shard_contract
    #    itself -- not through a re-implementation of its conditions, which would assert on a copy
    #    of the rule. The reader's rule is read from train.py so a change there breaks this rather
    #    than the pod.
    with open(os.path.join(ROOT, "train.py"), encoding="utf-8") as _fh:
        src2 = _fh.read()
    assert f'["{SHARD_DOC_KEY}"]' in src2, (
        f'train.py no longer reads ["{SHARD_DOC_KEY}"] -- SHARD_DOC_KEY is stale and this '
        f'contract is checking the wrong key')
    d2 = tempfile.mkdtemp()
    try:
        def _w(nm, lines):
            q = os.path.join(d2, nm)
            with open(q, "w", encoding="utf-8") as fh:
                for ln in lines:
                    fh.write(json.dumps(ln, ensure_ascii=False) + "\n")
            return q

        # (a) the OLD format, both faults: an in-band header and "text" doc lines.
        old = _w("old_000.jsonl", [{"_header": True, "family": "S"},
                                   {"text": "hi", "_s_index": 0, "_exposure": 0}])
        try:
            shard_contract(old)
            raise AssertionError("the old shard format was accepted; this is the exact file shape "
                                 "whose cache build raised KeyError: 'content' on the pod")
        except SystemExit as e:
            assert "_header" in str(e), str(e)

        # (b) HEADER ALONE is fatal, even with the key fixed -- so the sidecar move is load
        #     bearing and not cosmetic. THE HEADER LINE HERE CARRIES "content" TOO, which is the
        #     only way this world tests the header rule: with a bare {"_header": true} the line
        #     also lacks "content", so deleting the header check entirely left this green (the
        #     missing-key branch caught it) and the mutant survived. A world has to fail for the
        #     rule it names -- see red-for-the-wrong-reason.
        hdr = _w("hdr_000.jsonl", [{"_header": True, "family": "S", SHARD_DOC_KEY: "meta"},
                                   {SHARD_DOC_KEY: "hi"}])
        try:
            shard_contract(hdr)
            raise AssertionError(
                "a header line was accepted alongside correct doc lines. It parses and it has "
                f"{SHARD_DOC_KEY!r}, so the ONLY thing that can refuse it is the "
                f"{SHARD_FORBIDDEN_KEY!r} rule -- and a header read as a document injects the "
                f"provenance blob into the corpus as training text")
        except SystemExit as e:
            assert SHARD_FORBIDDEN_KEY in str(e), str(e)

        # (c) "text" ALONE is fatal, with no header -- so the key rename is load bearing too.
        txt = _w("txt_000.jsonl", [{"text": "hi", "_s_index": 0, "_exposure": 0}])
        try:
            shard_contract(txt)
            raise AssertionError("a 'text'-keyed shard was accepted")
        except SystemExit as e:
            assert SHARD_DOC_KEY in str(e), str(e)

        # (d) A NAME THAT IS NOT A SHARD is refused, because _domain_seqs refuses the directory.
        bad_name = _w("notashard.jsonl", [{SHARD_DOC_KEY: "hi"}])
        try:
            shard_contract(bad_name)
            raise AssertionError("a non-shard filename was accepted")
        except SystemExit as e:
            assert "SHARD_RE" in str(e), str(e)

        # (e) AN EMPTY SHARD is refused rather than reported as 0 docs.
        empty = _w("empty_000.jsonl", [])
        try:
            shard_contract(empty)
            raise AssertionError("an empty shard was accepted")
        except SystemExit as e:
            assert "no document lines" in str(e), str(e)

        # (f) WHAT write_shard NOW PRODUCES PASSES, and read back through train's own reader.
        wp = write_shard(d2, "s_inject_n1", [("t0", 0, 0), ("t1", 1, 0)], {"_header": True, "a": 1})
        assert shard_contract(wp, wp + ".meta.json") == 2
        sys.path.insert(0, ROOT)
        import train as _t
        assert _t._jsonl_content(wp) == ["t0", "t1"], _t._jsonl_content(wp)
        with open(wp + ".meta.json", encoding="utf-8") as _fh:
            assert json.load(_fh) == {"a": 1}, "sidecar content"
        # The sidecar must not itself look like a shard, or _domain_seqs would read it as data.
        assert not re.search(m.group(1), os.path.basename(wp + ".meta.json")), (
            "the sidecar name matches SHARD_RE, so _domain_seqs would glob it as a shard")

        # (g) A MISSING SIDECAR is refused when one is expected: the header is provenance, and
        #     moving it out of band must not mean losing it.
        os.unlink(wp + ".meta.json")
        try:
            shard_contract(wp, wp + ".meta.json")
            raise AssertionError("a missing sidecar was accepted")
        except SystemExit as e:
            assert "sidecar" in str(e), str(e)
    finally:
        shutil.rmtree(d2, ignore_errors=True)

    print("build_s_inject selftest OK: exposure counts exact and equal per document (1/8/64), "
          "repeats interleaved so first appearances spread past 2x the set size rather than "
          "massing in the first block, the worked solution is inside each document, a pool whose "
          "sha256 is not the frozen one is refused, an existing shard directory is refused rather "
          "than rebuilt at a new seed, the shard name satisfies train.py's own SHARD_RE, and the "
          "shard contract refuses each half of the launch defect independently -- an in-band "
          "_header line (even one carrying \"content\"), a \"text\"-keyed doc line, a non-shard "
          "filename, an empty shard and a missing sidecar -- while what write_shard now emits "
          "reads back through train._jsonl_content itself")
    return 0


if __name__ == "__main__":
    sys.exit(main())
