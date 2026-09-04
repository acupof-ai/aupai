"""Enumerate CHECKS by AST, and draw the fixed-seed population sample.

Regex over the CHECKS literal missed 4 of 79 entries (multi-line asserts/incident
strings), so this reads the tuple structure instead of matching its formatting.
The sample is drawn with random.Random(904) over the names sorted alphabetically,
so the pair can redraw the identical 30 from this file alone.

  python3 runs/audit_0904/enum_checks.py            # counts + name/function mismatches
  python3 runs/audit_0904/enum_checks.py --sample   # the 30 sampled and the rest
  python3 runs/audit_0904/enum_checks.py --json     # every field
  python3 runs/audit_0904/enum_checks.py --selftest
"""

import ast
import os
import random
import sys

ROOT = os.environ.get(
    "AUDIT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
SEED = 904
N = 30


def _text(node):
    """asserts/incident are normally literals, but at least one is an f-string, so a
    literal_eval reader dies on the real file. Return the source segment for those."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return "<computed> " + ast.unparse(node)


def load(root=None):
    root = root or ROOT
    src = open(os.path.join(root, "scripts", "harness.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    checks = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "CHECKS" for t in n.targets):
            checks = n.value
    if not isinstance(checks, (ast.List, ast.Tuple)):
        raise SystemExit("CHECKS is not a literal list -- this reader cannot see it")

    rows = []
    for el in checks.elts:
        if not (isinstance(el, ast.Tuple) and len(el.elts) == 5):
            raise SystemExit(f"CHECKS entry at harness.py:{el.lineno} is not a 5-tuple")
        name, asserts, incident, run, broken = el.elts
        rows.append(
            {
                "name": ast.literal_eval(name),
                "asserts": _text(asserts),
                "incident": _text(incident),
                "run": getattr(run, "id", None),
                "broken": getattr(broken, "id", None),
                "run_line": funcs[run.id].lineno if getattr(run, "id", None) in funcs else None,
                "broken_line": funcs[broken.id].lineno if getattr(broken, "id", None) in funcs else None,
                "entry_line": el.lineno,
            }
        )
    return rows


def sample(rows, n=N, seed=SEED):
    names = sorted(r["name"] for r in rows)
    return sorted(random.Random(seed).sample(names, min(n, len(names))))


def _selftest():
    rows = load()
    assert len(rows) > 60, f"only {len(rows)} entries parsed; the reader is not seeing CHECKS"
    # Drawing twice must give the identical set, or the pair cannot redraw it.
    assert sample(rows) == sample(rows), "the sample is not reproducible"
    # The draw must depend on the seed, or "fixed-seed" means nothing.
    assert sample(rows, seed=905) != sample(rows), "changing the seed changed nothing"
    # And it must be a subset of the real names, not invented ones.
    real = {r["name"] for r in rows}
    assert set(sample(rows)) <= real, "the sample names checks that do not exist"
    # Broken world: a CHECKS entry with 4 elements must be refused, not silently skipped.
    import shutil
    import tempfile

    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "scripts"))
    p = os.path.join(d, "scripts", "harness.py")
    shutil.copy(os.path.join(ROOT, "scripts", "harness.py"), p)
    src = open(p, encoding="utf-8").read()
    first = load()[0]
    bad = src.replace(f'        {first["broken"]},\n', "", 1)
    assert bad != src, "the mutation changed nothing"
    open(p, "w", encoding="utf-8").write(bad)
    try:
        load(d)
    except SystemExit as e:
        assert "not a 5-tuple" in str(e), f"refused for the wrong reason: {e}"
    else:
        raise AssertionError("a 4-element CHECKS entry parsed without complaint")
    print("enum_checks selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        raise SystemExit(0)

    rows = load()
    if "--json" in sys.argv:
        import json

        print(json.dumps(rows, indent=1))
        raise SystemExit(0)

    if "--sample" in sys.argv:
        s = sample(rows)
        print(f"seed={SEED} n={len(s)} of {len(rows)}")
        print("\nSAMPLED (population read against the rule by hand):")
        for n in s:
            print(f"  {n}")
        print("\nNOT SAMPLED (mechanical scan only, population vs rule unread):")
        for n in sorted(r["name"] for r in rows if r["name"] not in set(s)):
            print(f"  {n}")
        raise SystemExit(0)

    print(f"{len(rows)} CHECKS entries")
    missing = [r for r in rows if r["run_line"] is None or r["broken_line"] is None]
    print(f"run/broken not a module-level def: {len(missing)} {[r['name'] for r in missing]}")
    mismatch = [r for r in rows if r["run"] != "check_" + r["name"]]
    print(f"\nname != check_<name> ({len(mismatch)}):")
    for r in mismatch:
        print(f"  {r['name']:38s} runs {r['run']}  (harness.py:{r['run_line']})")
    dup = {}
    for r in rows:
        dup.setdefault(r["name"], []).append(r["entry_line"])
    print("\nduplicate check names:")
    for k, v in dup.items():
        if len(v) > 1:
            print(f"  {k} at lines {v}")
