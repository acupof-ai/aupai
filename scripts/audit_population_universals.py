#!/usr/bin/env python3
"""Which checks build their own population and then report a universal over it?

fb's sweep, 2026-09-01, from the selftests_are_gated defect. That check reported
"27 selftest-carrying file(s), all gated by the hook" while the real population was
36: nine files dispatch --selftest on sys.argv and its matcher required argparse.

The sharp form, and it is why the vacuous-PASS meta-check did not fire: **the defect
is not "found nothing and said all", it is "found some and reported a universal"**.
N=27 looks like a measurement. A universal quantifier over a self-constructed
population is only as true as the construction, and nothing checks the construction.

So this enumerates every check whose evidence string asserts a universal ("all", "every",
"each") over a population it discovered by regex or glob, and reports what it would
take to test each one's completeness. It does not decide -- the test is mechanical and
per-check: narrow the matcher, and see whether the check still claims a universal over
the smaller population instead of failing.

    python3 scripts/audit_population_universals.py
    python3 scripts/audit_population_universals.py --selftest
"""

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: An evidence string claiming something holds for a whole population.
_UNIVERSAL = re.compile(r"\ball\b|\bevery\b|\beach\b|\bno\s+\w+\s+(?:is|are|has|have)\b", re.I)
#: The population was DISCOVERED rather than enumerated: a directory walk, a glob, or
#: a regex over source text. These are the constructions nothing checks.
_DISCOVERY = re.compile(r"os\.listdir|os\.walk|glob\.|re\.findall|re\.finditer|\.rglob")


def check_bodies(src):
    """{name: body} for every `def check_*` in harness.py, by indentation."""
    out, cur, buf = {}, None, []
    for line in src.split("\n"):
        m = re.match(r"^def (check_\w+)\(", line)
        if m:
            if cur:
                out[cur] = "\n".join(buf)
            cur, buf = m.group(1), [line]
        elif cur:
            if line and not line[0].isspace() and not line.startswith(")"):
                out[cur] = "\n".join(buf)
                cur, buf = None, []
            else:
                buf.append(line)
    if cur:
        out[cur] = "\n".join(buf)
    return out


def scan(src):
    """[(name, discovers, claims_universal)] for each check."""
    rows = []
    for name, body in sorted(check_bodies(src).items()):
        discovers = bool(_DISCOVERY.search(body))
        # only the strings the check RETURNS as evidence, not its prose
        evidence = " ".join(re.findall(r"return\s+(?:PASS|FAIL|WARN|SKIP)\s*,\s*\(?(.{0,300})",
                                       body, re.S))
        rows.append((name, discovers, bool(_UNIVERSAL.search(evidence))))
    return rows


def selftest():
    """Known answers. A scan whose success condition is a list needs a case that must
    appear in it and one that must not -- otherwise a broken matcher returns [] and
    reads as 'no exposure anywhere', the finding it was written to look for."""
    src = '''
def check_a(root):
    """doc"""
    have = set()
    for nm in os.listdir(d):
        have.add(nm)
    return PASS, f"{len(have)} file(s), all gated by the hook"


def check_b(root):
    """doc mentioning every and all, which must NOT count -- prose is not evidence"""
    n = read_one_known_path()
    return PASS, f"{n} rows"


def check_c(root):
    x = glob.glob("*.py")
    return PASS, f"{len(x)} found"
'''
    rows = dict((n, (d, u)) for n, d, u in scan(src))
    assert rows["check_a"] == (True, True), f"discovery + universal must be flagged: {rows}"
    assert rows["check_b"] == (False, False), (
        f"a universal in the DOCSTRING is not an evidence claim, and a check that reads "
        f"a known path builds no population: {rows}")
    assert rows["check_c"] == (True, False), f"discovery without a universal claim: {rows}"
    print("selftest OK: flags discovery+universal, ignores docstring prose, and "
          "distinguishes discovery alone")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    path = os.path.join(ROOT, "scripts", "harness.py")
    with open(path, encoding="utf-8") as f:
        rows = scan(f.read())
    exposed = [n for n, d, u in rows if d and u]
    disc_only = [n for n, d, u in rows if d and not u]

    print(f"{len(rows)} checks scanned in scripts/harness.py\n")
    print(f"EXPOSED -- discovers its own population AND reports a universal ({len(exposed)}):")
    for n in exposed:
        print(f"  {n}")
    print(f"\ndiscovers a population, claims no universal ({len(disc_only)}):")
    for n in disc_only:
        print(f"  {n}")
    print("\nThe test for each exposed check is mechanical: narrow its matcher and see")
    print("whether it still reports a universal over the smaller population instead of")
    print("failing. 'Found some and reported a universal' is the defect; N looking")
    print("healthy is not evidence the population is complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
