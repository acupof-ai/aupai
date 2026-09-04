#!/usr/bin/env python3
"""Audit instrument: does every facts/*.json entry's cited source resolve to a real artifact?

WHAT THIS ANSWERS, and the boundary matters because the gate next door answers something
else. scripts/harness.py's facts_well_formed already checks that a fact's `source` names
paths that exist -- but it was satisfied by prose (`A/B`, `batch16/seq4096`) and by
brace-expanded globs until they were found by hand. This asks the narrower, checkable
question: of the tokens in a `source` field that ARE path-shaped, how many resolve, and
which do not.

THE TOKENISER IS THE WHOLE PROBLEM. A source field is prose with paths embedded:

    "runs/b0_sd_looped.log and runs/b0_sd_unlooped.log paired by eval/block_paired.py"
    "A/B at batch16/seq4096, 3 seeds (runs/warmup_smoke*.log)"
    "probes/lm_head_gemm.py@bcc43c4"

A naive `'/' in token` splitter calls `A/B` and `batch16/seq4096` paths and reports them
missing, which is a false positive that buries the real ones. So a token counts as
path-shaped only if it has a known file extension, or names a directory that exists in the
repo. `@sha` suffixes are stripped and the path checked at that commit, because a probe
deleted after it ran is cited honestly by sha.

BROKEN-WORLD TEST (principle 4): --selftest builds three fixtures whose answers are known
-- a real path, a path-shaped string that does not exist, and prose that must not be
called a path -- and asserts this script reports exactly one missing. An instrument that
has not been shown to fail on a known defect has not run.
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Extensions that make a token a file citation. Deliberately a whitelist: the repo's source
# fields also carry version strings (torchao/float8/config.py:353-379 is a path, 2609.01343
# is not) and a blacklist would keep growing.
EXTS = (".py", ".sh", ".json", ".jsonl", ".log", ".md", ".txt", ".pt", ".html", ".tsv",
        ".csv", ".xml", ".yaml", ".yml", ".c", ".cu", ".h")


def path_shaped(tok):
    """Is this token a citation of a file, rather than prose that happens to hold a slash?

    Two accepts: a known extension anywhere in the token, or a leading directory that
    exists in the repo. The second catches extensionless citations (`data/corpus/sample`)
    without accepting `A/B`, since there is no directory named `A`.
    """
    t = tok.strip("()[[]{},;:'\"").rstrip(".")
    if not t or t.startswith("#"):
        return None
    base = t.split("@")[0].split("#")[0]
    base = re.sub(r":\d+(-\d+)?$", "", base)      # file.py:353-379 -> file.py
    if any(e in base for e in EXTS):
        return t
    if "/" in base:
        head = base.split("/")[0]
        if head and os.path.isdir(os.path.join(ROOT, head)):
            return t
    return None


def resolves(tok, root=ROOT):
    """Does the citation name something readable? Returns (ok, how).

    A `@sha` citation is checked at that commit with `git cat-file`, because a probe that
    was deleted after it ran is honestly cited by sha and demanding it on disk would
    report an honest citation as a broken one.
    """
    t = tok.strip("()[]{},;:'\"").rstrip(".")
    base = t.split("@")[0].split("#")[0]
    base = re.sub(r":\d+(-\d+)?$", "", base)
    sha = t.split("@")[1] if "@" in t else None
    if "*" in base or "?" in base:
        import glob
        hits = glob.glob(os.path.join(root, base))
        return (bool(hits), f"glob {len(hits)} hit(s)")
    if sha:
        p = subprocess.run(["git", "-C", root, "cat-file", "-e", f"{sha}:{base}"],
                           capture_output=True)
        if p.returncode == 0:
            return True, f"exists at {sha}"
        # A sha citation whose path is absent at that sha may still be on disk now.
        if os.path.exists(os.path.join(root, base)):
            return True, "on disk (not at cited sha)"
        return False, f"absent at {sha} and on disk"
    return (os.path.exists(os.path.join(root, base)), "on disk")


def audit(files, root=ROOT):
    rows, n_facts, n_tok = [], 0, 0
    for f in files:
        p = os.path.join(root, f)
        for fact in json.load(open(p, encoding="utf-8"))["facts"]:
            n_facts += 1
            src = str(fact.get("source") or "")
            for raw in src.replace(";", " ").replace(",", " ").split():
                tok = path_shaped(raw)
                if tok is None:
                    continue
                n_tok += 1
                ok, how = resolves(tok, root)
                if not ok:
                    rows.append((f, fact.get("id"), tok, how))
    return rows, n_facts, n_tok


def selftest():
    """Three fixtures, one known-missing. Fails loudly if the instrument is blind."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "runs"))
        open(os.path.join(d, "runs", "real.log"), "w").write("x")
        os.makedirs(os.path.join(d, "facts"))
        json.dump({"facts": [
            {"id": "a.real", "source": "runs/real.log"},
            {"id": "a.missing", "source": "runs/nope.log"},
            {"id": "a.prose", "source": "A/B at batch16/seq4096, 3 seeds"},
        ]}, open(os.path.join(d, "facts", "t.json"), "w"))
        rows, nf, nt = audit(["facts/t.json"], root=d)
        assert nf == 3, nf
        # the prose fact must contribute NO tokens: that is the false positive this guards
        assert nt == 2, f"tokeniser counted prose as a path: {nt} tokens, expected 2"
        assert len(rows) == 1 and rows[0][1] == "a.missing", rows
        # and it must actually catch a real absence -- delete the real one and re-run
        os.remove(os.path.join(d, "runs", "real.log"))
        rows2, _, _ = audit(["facts/t.json"], root=d)
        assert len(rows2) == 2, f"blind to a newly-missing artifact: {rows2}"
    print("selftest OK: catches a missing citation, ignores prose, and fails when a "
          "previously-present artifact is removed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        raise SystemExit(0)
    FILES = ["facts/efficiency.json", "facts/smelt_deeploop.json"]
    rows, n_facts, n_tok = audit(FILES)
    print(f"{n_facts} facts, {n_tok} path-shaped source tokens, {len(rows)} unresolved")
    for f, fid, tok, how in rows:
        print(f"  {f}#{fid}\n      {tok}   [{how}]")
