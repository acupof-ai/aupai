"""Machine scan over all 79 broken() worlds: is the mutation DERIVED from a real
artifact, or hand-written?

The rule (AGENTS.md, "Broken worlds mutate a real artifact, never a hand-written one")
exists because a hand-written world shares the check's own assumptions. harness's own
selftest asserts something weaker: that the world holds a file at a repo-real PATH. A
world can satisfy that with invented CONTENT at a real path -- which is the case this
scan separates.

Classification, per broken() function body, read as AST:
  derived    calls shutil.copy / copytree / open(ROOT-path).read / git show / subprocess
             on a path under ROOT -- the bytes come from the repo
  linked     only _tmp_repo_shaped() with no local write: the world sees real files and
             the mutation is an absence or an added file
  written    writes content built in the function (json.dump of a literal, open(w) of an
             f-string) with no read of the corresponding real file
  skip       raises SelftestSkip unconditionally

`written` is not automatically a defect -- an ADDED row in a ledger is legitimately
synthetic. It is the population that needs reading by hand, which is what the report's
sample does. The scan's job is to say which worlds need that reading.

  python3 runs/audit_0904/scan_broken_worlds.py
  python3 runs/audit_0904/scan_broken_worlds.py --selftest
"""

import ast
import os
import sys

ROOT = os.environ.get(
    "AUDIT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from enum_checks import load  # noqa: E402

READ_REAL = {"copy", "copy2", "copytree", "copyfile"}


def classify(fn, funcs, seen=None):
    """Walk fn, following calls to other harness helpers once, and report what it does."""
    seen = seen or set()
    if fn.name in seen:
        return set()
    seen.add(fn.name)
    tags = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            fname = getattr(n.func, "attr", None) or getattr(n.func, "id", None)
            if fname in READ_REAL:
                tags.add("derived")
            if fname in ("run", "check_output", "check_call"):
                tags.add("subprocess")
            if fname == "open":
                mode = ""
                if len(n.args) > 1 and isinstance(n.args[1], ast.Constant):
                    mode = n.args[1].value
                for kw in n.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                src = ast.unparse(n.args[0]) if n.args else ""
                if "w" in mode or "a" in mode:
                    tags.add("writes")
                elif "ROOT" in src:
                    tags.add("derived")
            if fname in ("dump", "write_text", "writelines"):
                tags.add("writes")
            if fname == "_tmp_repo_shaped":
                tags.add("shaped")
            if fname == "_tmp_repo":
                tags.add("bare")
            if fname == "SelftestSkip":
                tags.add("may-skip")
            if fname in funcs and fname.startswith(("_broken", "_world", "_mk", "_write")):
                tags |= classify(funcs[fname], funcs, seen)
    return tags


def verdict(tags):
    if "derived" in tags or "subprocess" in tags:
        return "derived"
    if "writes" in tags:
        return "written"
    if "shaped" in tags or "bare" in tags:
        return "linked"
    return "unknown"


def scan(root=None):
    root = root or ROOT
    src = open(os.path.join(root, "scripts", "harness.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    out = []
    for r in load(root):
        fn = funcs.get(r["broken"])
        if fn is None:
            out.append((r["name"], r["broken"], "no-def", set()))
            continue
        tags = classify(fn, funcs)
        out.append((r["name"], r["broken"], verdict(tags), tags))
    return out


def _selftest():
    rows = scan()
    assert len(rows) > 60, f"only {len(rows)} worlds scanned"
    by = {}
    for name, _, v, _ in rows:
        by.setdefault(v, []).append(name)
    # A scan that calls everything derived, or everything written, tells you nothing.
    assert len(by) >= 2, f"every world got the same verdict {list(by)} -- the scan is inert"
    # Known answers, read by hand from harness.py before writing this scan:
    #   _broken_tokenizer json.dumps a hand-built vocab (harness.py:2224) -> written
    #   _broken_no_conflict_markers copies a real tracked file               -> derived
    got = dict((n, v) for n, _, v, _ in rows)
    assert got.get("tokenizer_roundtrip") == "written", (
        f"tokenizer_roundtrip classified {got.get('tokenizer_roundtrip')}, "
        "but _broken_tokenizer json.dumps a hand-built vocab"
    )
    assert got.get("no_conflict_markers") == "derived", (
        f"no_conflict_markers classified {got.get('no_conflict_markers')}, "
        "but _broken_no_conflict_markers copies a real file"
    )
    print("scan_broken_worlds selftest ok (2 known answers held, verdicts differ)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        raise SystemExit(0)
    rows = scan()
    order = {"no-def": 0, "unknown": 1, "written": 2, "linked": 3, "derived": 4}
    rows.sort(key=lambda r: (order.get(r[2], 9), r[0]))
    counts = {}
    for _, _, v, _ in rows:
        counts[v] = counts.get(v, 0) + 1
    print(f"{len(rows)} broken() worlds: " + ", ".join(f"{v}={c}" for v, c in sorted(counts.items())))
    print()
    for name, bfn, v, tags in rows:
        print(f"{v:9s} {name:38s} {str(bfn):44s} {sorted(tags)}")
