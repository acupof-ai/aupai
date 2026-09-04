"""Per check, print the calls that DEFINE its population -- the enumeration, not the test.

Population before property (audit_0904 principle 3). To read a population against its
rule you first have to see it, and the population of a harness check is wherever it
enumerates: os.listdir / os.walk / glob / _tracked_files / git ls-files / a hardcoded
tuple of paths / a single open(). This prints those lines per check so the reading is
against the code, not against the docstring.

  python3 runs/audit_0904/populations.py <check> [<check> ...]
  python3 runs/audit_0904/populations.py --sample     # the 30 sampled
  python3 runs/audit_0904/populations.py --selftest
"""

import ast
import os
import re
import sys

ROOT = os.environ.get(
    "AUDIT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enum_checks import load, sample  # noqa: E402

ENUM = re.compile(
    r"\b(os\.listdir|os\.walk|os\.scandir|glob\.glob|glob\.iglob|_tracked_files|"
    r"walk_tracked|tracked_files|read_jsonl|_read_jsonl|iter_rows|json\.load|open\(|"
    r"for\s+\w+\s+in\s+\(|REGISTRY|CHECKS|_RULE_CHECKS|subprocess\.run)\b|ls-files"
)


def sources(root=None):
    root = root or ROOT
    src = open(os.path.join(root, "scripts", "harness.py"), encoding="utf-8").read()
    lines = src.splitlines()
    tree = ast.parse(src)
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    return lines, funcs


def population_lines(name, lines, funcs, rows):
    r = next((x for x in rows if x["name"] == name), None)
    if r is None:
        return None, []
    fn = funcs.get(r["run"])
    if fn is None:
        return r, []
    body = lines[fn.lineno - 1 : fn.end_lineno]
    out = []
    for i, ln in enumerate(body, start=fn.lineno):
        if ENUM.search(ln):
            out.append((i, ln.rstrip()))
    return r, out


def _selftest():
    lines, funcs = sources()
    rows = load()
    # Known answer: selftests_are_gated enumerates by walking, so it must show an
    # enumeration line; timestamps_are_utc reads ledgers, so it must too. A regex that
    # finds nothing in either is inert.
    for n in ("selftests_are_gated", "timestamps_are_utc"):
        r, pl = population_lines(n, lines, funcs, rows)
        assert r is not None, f"{n} not in CHECKS"
        assert pl, f"{n}: no enumeration line found -- the pattern is inert"
    # And it must not match every line: a check with no enumeration should show few.
    r, pl = population_lines("no_shared_stash", lines, funcs, rows)
    body = funcs[r["run"]].end_lineno - funcs[r["run"]].lineno
    assert len(pl) < body / 2, f"no_shared_stash matched {len(pl)} of {body} lines -- too broad"
    print("populations selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        raise SystemExit(0)
    lines, funcs = sources()
    rows = load()
    names = sys.argv[1:]
    if "--sample" in names:
        names = sample(rows)
    for n in names:
        r, pl = population_lines(n, lines, funcs, rows)
        if r is None:
            print(f"== {n}: NOT A CHECK")
            continue
        print(f"== {n}  ({r['run']} at harness.py:{r['run_line']}, {'broken=' + str(r['broken'])})")
        print(f"   asserts: {r['asserts']}")
        for i, ln in pl:
            print(f"   {i}: {ln.strip()[:150]}")
        print()
