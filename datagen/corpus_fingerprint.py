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


def _patterns_of(path, name):
    """The PATTERNS list a filter file contributes, read by AST rather than exec.

    This is EXACTLY what build_corpus.load_garbage_patterns consumes (it execs the file and
    takes ns["PATTERNS"], concatenating across the tuple in order), minus the exec: the list is
    a module-level literal of string constants in every pipeline filter, and the assertion below
    keeps that true. exec would import nothing, but it would also run whatever else the file
    holds; AST reads only the value the pipeline actually concatenates into GARBAGE."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Name) and tgt.id == "PATTERNS"):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            raise AssertionError(
                f"{path}: PATTERNS is not a literal list, so its value cannot be read without "
                f"exec and fp_filters cannot describe what the pipeline compiles"
            )
        pats = []
        for elt in node.value.elts:
            if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                raise AssertionError(
                    f"{path}: PATTERNS holds a non-string-constant element ({ast.dump(elt)[:60]}); "
                    f"a computed pattern would enter GARBAGE without entering the fingerprint"
                )
            pats.append(elt.value)
        return pats
    raise AssertionError(
        f"{path}: no module-level PATTERNS assignment found; build_corpus.load_garbage_patterns "
        f"would contribute nothing from {name} and fp_filters would describe a build that "
        f"does not exist"
    )


def fp_filters(root=ROOT):
    """Fingerprint of the RULES that produced a corpus: sha1 over the flattened PATTERNS lists of
    the pipeline filters, in the order build_corpus concatenates them. Content-based, not the git
    sha: an uncommitted edit to filters/ changes what a build keeps, and a commit sha would not
    see it.

    The gap this closes: PROVENANCE records the Build COMMAND, and the same command run before
    and after a filter change produces different corpora that nothing distinguishes.
    corpus_fingerprint says the content changed; this says what produced it.

    HASHES THE PATTERNS, NOT THE FILE BYTES. It hashed bytes until 2026-09-17, and `filters/secrets.py`
    was not the only false positive that admits: a semantics-preserving edit to a pipeline filter
    moved it while the drop decision could not change. Measured: commit 3a972f57 hoisted
    `COMPILED = [re.compile(p) for p in PATTERNS]` and added a shared `drops()` -- same regexes,
    same order, same behavior -- and the byte hash went 88ee503b -> 9bbed36b. Every corpus built
    before that commit then read as stale for a refactor that removes no document. A fingerprint
    that changes when the output cannot is a false positive, and the red it raises is a red
    nobody can act on except by rebuilding nine domains for nothing. The earlier scoping fix
    (hash only the pipeline files, not the directory) removed the same class one level up;
    this removes it for real.

    WHAT IT CANNOT SEE, stated because a narrower fingerprint must not be mistaken for a wider
    one: it covers the PATTERNS lists of the three pipeline filters and nothing else. Code in
    those files that changes the drop decision WITHOUT changing a pattern -- a different
    normalization, a threshold read from the environment, an early return -- is outside it. No
    such code exists today (the files are a PATTERNS literal plus a predicate that iterates it,
    and the predicate is asserted by filters/test_l0_garbage_known.py), and the boundary is
    recorded in the PR and in facts/corpus_supply.json rather than left to be discovered.

    The scoping is verified, not asserted: build_corpus.py:62 iterates a literal tuple of three
    names and exec's each, so those three files' PATTERNS are the whole input, and
    _assert_pipeline_filters_current() fails if that tuple and this one drift apart."""
    d = os.path.join(root, "filters")
    if not os.path.isdir(d):
        return None
    pats = []
    for name in PIPELINE_FILTERS:
        p = os.path.join(d, name)
        if not os.path.exists(p):
            # build_corpus.py raises FileNotFoundError on the same condition. Hashing "absent"
            # would let a build whose filter file vanished carry a valid-looking fingerprint.
            raise FileNotFoundError(f"{p} missing; fp_filters cannot describe a build without it")
        pats.extend(_patterns_of(p, name))
    # ONE hash over the FLATTENED list, not one per file. build_corpus compiles a single
    # alternation -- re.compile("|".join(f"(?:{p})" for p in pats)) -- so the file a rule came
    # from is not part of the drop decision, and mixing it in would report a change for a move
    # that cannot alter a byte of output. json.dumps, not a join: an unambiguous encoding, so a
    # pattern containing whatever separator a join would use cannot alias a different list.
    # List order is NOT sorted, so the value keeps whatever stability the load order has; the
    # compiled alternation's hit-set is order-independent under search(), so this is a choice
    # about readability, not a claim that order changes the drop decision.
    #
    # THE "p1-" PREFIX IS THE MIGRATION MARKER, not decoration. Until 2026-09-17 this returned a
    # bare 16-hex sha1 over the filter FILES' bytes; both generations are 16 hex characters, so
    # without a marker an old stamp and a new one are indistinguishable by value and a stale
    # domain would compare equal-looking-but-meaningless values. Every stamp written by that
    # byte hash carries no prefix and is therefore visibly UNMIGRATED: it must be re-stamped by a
    # rebuild, never compared against a live pattern hash. check_corpus_filters_fp treats an
    # unprefixed stamp as the baselined debt it is.
    return "p1-" + hashlib.sha1(json.dumps(pats, ensure_ascii=False).encode()).hexdigest()[:16]


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
        # 2. A PIPELINE file's PATTERNS MUST move it, or the fingerprint is blind to real change.
        #    Four mutations per file, each changing what GARBAGE matches: append a rule, edit a
        #    rule in place, remove a rule, and empty the list. EVERY pipeline file is mutated --
        #    a version that hashed only the first would pass an all-files-mutated test only by
        #    accident, and would read green while the other two files' edits went unseen.
        f0 = os.path.join(froot, "filters", PIPELINE_FILTERS[0])
        with open(f0, encoding="utf-8") as f:
            orig = f.read()
        mutants = {
            "append": orig.replace("\n]\n", "\n    r'a real filter edit',\n]\n", 1),
            "edit": orig.replace("PATTERNS = [", "PATTERNS = [\n    r'edited rule',", 1),
            "remove": orig.replace("\n    r", "\n    # r", 1),
            "empty": orig.split("PATTERNS = [")[0] + "PATTERNS = []\n",
        }
        for label, body in mutants.items():
            assert body != orig, f"mutation {label} did not change the file; the test is dead"
            with open(f0, "w", encoding="utf-8") as f:
                f.write(body)
            assert fp_filters(froot) != only_pipeline, (
                f"PATTERNS mutation '{label}' on {PIPELINE_FILTERS[0]} did not move the "
                f"fingerprint -- a real change to the rules would read as the same build"
            )
        with open(f0, "w", encoding="utf-8") as f:
            f.write(orig)
        for extra in PIPELINE_FILTERS[1:]:
            fe = os.path.join(froot, "filters", extra)
            with open(fe, encoding="utf-8") as f:
                eorig = f.read()
            with open(fe, "w", encoding="utf-8") as f:
                f.write(eorig.replace("PATTERNS = [", "PATTERNS = [\n    r'extra-file rule',", 1))
            assert fp_filters(froot) != only_pipeline, (
                f"a PATTERNS edit in {extra} did not move the fingerprint: the hash covers only "
                f"{PIPELINE_FILTERS[0]}, so {extra}'s rules enter GARBAGE unseen"
            )
            with open(fe, "w", encoding="utf-8") as f:
                f.write(eorig)
        # 2b. The flattened list must be what build_corpus actually compiles. It execs each file
        #     and concatenates ns['PATTERNS'] in tuple order; the AST reader above must agree with
        #     that, or the fingerprint describes a different rule set than the pipeline runs.
        ns_pats = []
        for name in PIPELINE_FILTERS:
            ns = {}
            with open(os.path.join(ROOT, "filters", name), encoding="utf-8") as f:
                exec(compile(f.read(), name, "exec"), ns)
            ns_pats.extend(ns.get("PATTERNS", []))
        ast_pats = []
        for name in PIPELINE_FILTERS:
            ast_pats.extend(_patterns_of(os.path.join(ROOT, "filters", name), name))
        assert ast_pats == ns_pats, (
            f"the AST reader and build_corpus's exec disagree: {len(ast_pats)} vs {len(ns_pats)} "
            f"patterns; fp_filters would describe rules the pipeline does not compile"
        )
        # 2c. A pipeline file whose PATTERNS cannot be read as a literal list must RAISE, not
        #     contribute nothing. build_corpus's exec takes ns.get('PATTERNS', []), so a quiet []
        #     here would describe a reduced build as the same build -- and the reduction is
        #     invisible in the shards' content too. Three shapes: renamed list, computed list,
        #     and a list holding a non-constant element.
        for label, body in (
            ("no-PATTERNS", orig.replace("PATTERNS = [", "RULES = [", 1)),
            ("non-literal", orig.replace("PATTERNS = [", "PATTERNS = _load() or [", 1)),
            ("computed-element", orig.replace("PATTERNS = [", "PATTERNS = [\n    'x' + 'y',", 1)),
        ):
            with open(f0, "w", encoding="utf-8") as f:
                f.write(body)
            try:
                fp_filters(froot)
            except AssertionError:
                continue
            raise AssertionError(
                f"a pipeline file in the '{label}' shape did not raise: fp_filters returned a "
                f"value for a rule set it cannot read, so a reduced build fingerprints as the "
                f"same build"
            )
        with open(f0, "w", encoding="utf-8") as f:
            f.write(orig)
        assert fp_filters(froot) == only_pipeline, "restoring the file must restore the value"
        # 3. SEMANTICS-PRESERVING EDITS MUST NOT MOVE IT. This is the whole point of hashing
        #    patterns rather than bytes: 3a972f57 hoisted COMPILED, added a shared drops(), and
        #    the byte hash moved 88ee503b -> 9bbed36b with the same regexes in the same order, so
        #    nine domains read stale for a refactor that removes no document. Each variant below
        #    keeps PATTERNS identical and changes only the file's bytes around it.
        sem = {
            "hoisted-compiled": orig.replace(
                "COMPILED = [re.compile(p) for p in PATTERNS]",
                "COMPILED = tuple(re.compile(p) for p in PATTERNS)  # hoist rewritten"),
            "comment": orig.replace("#!/usr/bin/env python3", "#!/usr/bin/env python3\n# a comment"),
            "import-order": orig.replace("import json\nimport re", "import re\nimport json"),
            "blank-lines": orig.replace("\n\n\n", "\n\n\n\n\n"),
        }
        for label, body in sem.items():
            if body == orig:
                continue  # this file's shape does not hold that construct; nothing to assert
            with open(f0, "w", encoding="utf-8") as f:
                f.write(body)
            assert fp_filters(froot) == only_pipeline, (
                f"semantics-preserving edit '{label}' on {PIPELINE_FILTERS[0]} MOVED the "
                f"fingerprint ({only_pipeline} -> {fp_filters(froot)}): the fingerprint is still "
                f"reading bytes, so a refactor that drops no document reds every corpus"
            )
        with open(f0, "w", encoding="utf-8") as f:
            f.write(orig)
        assert fp_filters(froot) == only_pipeline, "restoring the file must restore the value"
    print(
        f"self-check OK (mutate {fp1} -> {fp2}, utime invariant, delete -> {fp3}, train.py "
        f"parity, fp_filters scoped to {len(want)} pipeline filters: non-pipeline file inert, "
        f"{len(mutants)} PATTERNS mutations x {len(PIPELINE_FILTERS)} files caught, AST==exec, "
        f"{len(sem)} semantics-preserving edits inert)"
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
