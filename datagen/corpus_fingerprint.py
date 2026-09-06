#!/usr/bin/env python3
"""Per-domain corpus fingerprint: hash of (shard name, size, sha256 of the first and
last 64KB).

The incident: two corpora both called data/corpus/math, one 0.0% contaminated and
one 30.0%, and nothing but an audit distinguished them. A checkpoint needs to say
which corpus it trained on the way it already says which tokenizer (vocab_id).

Content-based, not mtime-based: a copy, podput, rsync or mv changes mtime without
touching a byte, and the 2026-08-30 sample-domain drift was exactly that -- a
transfer that red the guard with no editor to trace. Head+tail 64KB also catches
same-size edits, which mtime-only missed: a same-size rewrite almost necessarily
moves the head or the tail. Cost stays O(shards): 128KB read per shard,
milliseconds per domain on 108GB.

    python datagen/corpus_fingerprint.py [mix.json]   # print {domain: fp} for a mix
    python datagen/corpus_fingerprint.py --self-check # mutate/utime/parity on a real shard

build_corpus.py stamps the fingerprint into build_corpus_stats.json at build time;
harness check corpus_fp_matches compares that stamp to the live directory.
train.py carries an inline copy (it imports nothing from scripts/); --self-check
asserts the two agree bit-for-bit."""

import argparse
import ast
import glob
import hashlib
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _shard_line(name, path):
    """One shard's contribution: name, size, sha256 of the first and last 64KB.
    A shard <= 64KB is hashed whole via its head."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(65536)
        if size > 65536:
            f.seek(-65536, os.SEEK_END)
            tail = f.read(65536)
        else:
            tail = b""
    return f"{name}:{size}:{hashlib.sha256(head).hexdigest()}:{hashlib.sha256(tail).hexdigest()}\n".encode()


# The filters build_corpus.py actually loads, named there as a literal tuple at
# datagen/build_corpus.py:62 and exec'd into the GARBAGE pattern. Kept in sync by
# fp_filters' own assertion below rather than by this comment.
PIPELINE_FILTERS = ("pass1_garbage.py", "pass2_garbage.py", "pass3_garbage.py")


def fp_filters(root=ROOT):
    """Hash of the filter sources that produced a corpus. Content-based, not the git sha: an
    uncommitted edit to filters/ changes what a build keeps, and a commit sha would not see it.

    The gap this closes: PROVENANCE records the Build COMMAND, and the same command run before
    and after a filter change produces different corpora that nothing distinguishes.
    corpus_fingerprint says the content changed; this says what produced it.

    SCOPED TO THE PIPELINE, not to the directory listing. It used to hash every *.py in
    filters/, which made the fingerprint a property of what the directory CONTAINS rather than
    of what produced the shards: adding filters/secrets.py on 2026-09-06 -- a corpus-row
    credential redactor that build_corpus.py does not import -- moved the fingerprint from
    33462c13868a2194 to 9c0c1fd3160ebd27 and turned four stage-2 domains stale, though not one
    byte of any shard would differ if they were rebuilt. A fingerprint that changes when the
    output cannot is a false positive, and the corpus_filters_fp red it raised is a red nobody
    can act on except by rebuilding nine domains for nothing.

    The scoping is verified, not asserted: build_corpus.py:62 iterates a literal tuple of three
    names and exec's each, so those three are the whole input, and the assertion below fails if
    that tuple and this one drift apart."""
    d = os.path.join(root, "filters")
    if not os.path.isdir(d):
        return None
    h = hashlib.sha1()
    for name in PIPELINE_FILTERS:
        p = os.path.join(d, name)
        if not os.path.exists(p):
            # build_corpus.py raises FileNotFoundError on the same condition. Hashing "absent"
            # would let a build whose filter file vanished carry a valid-looking fingerprint.
            raise FileNotFoundError(f"{p} missing; fp_filters cannot describe a build without it")
        with open(p, "rb") as f:
            h.update(name.encode() + b"\0" + hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:16]


def _assert_pipeline_filters_current(root=ROOT):
    """PIPELINE_FILTERS must equal the tuple build_corpus.py loads, read from its source.

    Without this the scoping above degrades silently in the direction that matters: a fourth
    filter added to build_corpus.py's tuple would change what the shards contain while
    fp_filters kept reporting the old three-file hash -- a fingerprint that cannot see a real
    change, which is worse than the false positive this scoping removed."""
    src = os.path.join(root, "datagen", "build_corpus.py")
    with open(src, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.For) and isinstance(node.iter, (ast.Tuple, ast.List))):
            continue
        try:
            names = ast.literal_eval(node.iter)
        except ValueError:
            continue
        if not (names and all(isinstance(x, str) and x.endswith("_garbage") for x in names)):
            continue
        want = tuple(f"{x}.py" for x in names)
        if want != PIPELINE_FILTERS:
            raise AssertionError(
                f"build_corpus.py loads {want}, PIPELINE_FILTERS says {PIPELINE_FILTERS} -- "
                f"fp_filters would not see a change to the filters that actually run"
            )
        return want
    raise AssertionError(
        "no *_garbage loader tuple found in build_corpus.py; fp_filters' "
        "scope can no longer be verified against the code it describes"
    )


def fp_dir(d):
    """Hash of sorted shard lines for one domain directory. The workhorse: fp_domain
    and build_corpus.py both call this, so the stamper cannot diverge from the guard."""
    h = hashlib.sha1()
    for name in sorted(os.listdir(d)):
        if name == "build_corpus_stats.json" or name.startswith("."):
            continue
        h.update(_shard_line(name, os.path.join(d, name)))
    return h.hexdigest()[:16]


def fp_domain(domain, corpus_dir=None):
    """Fingerprint of data/corpus/<domain>; None if the domain is absent."""
    d = os.path.join(corpus_dir or os.path.join(ROOT, "data", "corpus"), domain)
    if not os.path.isdir(d):
        return None
    return fp_dir(d)


def fp_mix(mix_path):
    """{domain: fp} for every domain named in a mix file."""
    mix = json.load(open(mix_path, encoding="utf-8"))
    return {dom: fp_domain(dom) for dom in mix["domains"]}


def self_check():
    """Known answers on a REAL shard: mutation changes the fp, mtime-only change does
    not, deletion changes it, and train.py's inline copy agrees bit-for-bit. Uses the
    first corpus domain with shards (math on the pod, sample on a fresh checkout), so
    the parity assertion runs in CI too, not only where the full corpus lives."""
    real = []
    for dom in sorted(os.listdir(os.path.join(ROOT, "data", "corpus"))):
        real = sorted(glob.glob(os.path.join(ROOT, "data", "corpus", dom, "*.jsonl")))
        if real:
            break
    if not real:
        print("self-check SKIP: no corpus shards to copy")
        return 0
    with tempfile.TemporaryDirectory() as d:
        dom = os.path.join(d, "dom")
        os.makedirs(dom)
        shard = os.path.join(dom, "real_shard.jsonl")
        with open(real[0], "rb") as f, open(shard, "wb") as g:
            g.write(f.read())
        fp1 = fp_dir(dom)
        with open(shard, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({"question": "指纹自检：改一行必须变", "output": "1"}, ensure_ascii=False) + "\n"
            )
        fp2 = fp_dir(dom)
        assert fp1 and fp2 and fp1 != fp2, f"mutation did not change fingerprint: {fp1} -> {fp2}"
        # Transfer invariance: copy/rsync/podput change mtime only -- the fp must not move.
        os.utime(shard, (0, 0))
        assert fp_dir(dom) == fp2, "mtime-only change moved the fingerprint"
        os.unlink(shard)
        fp3 = fp_dir(dom)
        assert fp3 != fp1, "deleting the only shard must change the fingerprint"
        # train.py's inline copy must agree bit-for-bit: a divergent inline copy would
        # stamp checkpoints with a corpus id the guard never recognizes.
        sys.path.insert(0, ROOT)
        from train import _corpus_fp as _inline_fp  # noqa: E402

        with open(real[0], "rb") as src, open(shard, "wb") as g:
            g.write(src.read())
        assert _inline_fp(dom) == fp_dir(dom), "train.py _corpus_fp diverged from canonical"
    # fp_filters is scoped to the three filters build_corpus.py loads. Both directions are
    # asserted, because each failure mode is the other's cure taken too far.
    want = _assert_pipeline_filters_current()
    base = fp_filters()
    with tempfile.TemporaryDirectory() as d:
        froot = os.path.join(d, "root")
        os.makedirs(os.path.join(froot, "filters"))
        os.makedirs(os.path.join(froot, "datagen"))
        for name in PIPELINE_FILTERS:
            with (
                open(os.path.join(ROOT, "filters", name), "rb") as f,
                open(os.path.join(froot, "filters", name), "wb") as g,
            ):
                g.write(f.read())
        # 1. A NON-pipeline file in filters/ must NOT move the fingerprint. This is the false
        #    positive that turned four stage-2 domains stale for a redactor no build imports.
        only_pipeline = fp_filters(froot)
        assert only_pipeline == base, (
            f"fp_filters over the 3 pipeline files ({only_pipeline}) disagrees with the live "
            f"tree ({base}) -- the live filters/ holds a non-pipeline file that still counts"
        )
        with open(os.path.join(froot, "filters", "zz_not_in_pipeline.py"), "w") as f:
            f.write("PATTERNS = ['this file is not loaded by build_corpus']\n")
        assert fp_filters(froot) == only_pipeline, (
            "adding a non-pipeline .py to filters/ moved the fingerprint; the scoping is not "
            "in effect and every corpus goes stale on an unrelated file"
        )
        # 2. A PIPELINE file's content MUST move it, or the fingerprint is blind to real change.
        with open(os.path.join(froot, "filters", PIPELINE_FILTERS[0]), "a") as f:
            f.write("\nPATTERNS.append('a real filter edit')\n")
        assert fp_filters(froot) != only_pipeline, (
            f"editing {PIPELINE_FILTERS[0]} did not move the fingerprint"
        )
    print(
        f"self-check OK (mutate {fp1} -> {fp2}, utime invariant, delete -> {fp3}, train.py "
        f"parity, fp_filters scoped to {len(want)} pipeline filters: non-pipeline file inert, "
        f"pipeline edit caught)"
    )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mix", nargs="?", help="mix json; default: the live Cfg.mix")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        sys.exit(self_check())
    mix = args.mix
    if not mix:
        import ast

        src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ClassDef) and node.name == "Cfg":
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign) and stmt.targets[0].id == "mix":
                        mix = ast.literal_eval(stmt.value)
        assert mix, "no mix arg and no Cfg.mix"
    print(json.dumps(fp_mix(os.path.join(ROOT, mix) if not os.path.isabs(mix) else mix), indent=1))


if __name__ == "__main__":
    main()
