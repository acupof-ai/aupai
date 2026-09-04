#!/usr/bin/env python3
"""Mutation sweep for gate_cards' lane path: break it, the selftests must go red.

A green selftest proves nothing until a defect makes it red. Six mutations, one per property
the fix claims, run against BOTH selftests (launch_gate's own world built from the real grant,
and card_claim's, which asserts the writer and the gate agree). A mutation no selftest catches
is printed as SURVIVED and is a hole in the tests, not in the mutant.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MUTANTS = [
    ("gate ignores the request: always answer about the block",
     "scripts/launch_gate.py",
     "    if LAUNCH_CARDS is None:",
     "    if True:"),
    ("gate skips the ungranted-card refusal",
     "scripts/launch_gate.py",
     "    if ungranted:",
     "    if False and ungranted:"),
    ("gate skips the lane-ownership refusal",
     "scripts/launch_gate.py",
     "    if in_lane and lane_to != who:",
     "    if False and in_lane and lane_to != who:"),
    ("gate guesses the lane owner from the prose instead of lane_to",
     "scripts/launch_gate.py",
     '    lane_to = str(a.get("lane_to") or "").strip()',
     '    lane_to = _launch_owner(root) if _launch_owner(root).lower() in '
     '(str(a.get("lane_note") or "") + str(a.get("cards", {}).get(next(iter(sorted(want)), ""), '
     '""))).lower() else str(a.get("lane_to") or "").strip()'),
    ("writer leaves lane_to behind",
     "scripts/card_claim.py",
     '    obj["lane_to"] = to',
     "    pass"),
    ("writer grants onto a foreign card",
     "scripts/card_claim.py",
     "    if foreign:",
     "    if False and foreign:"),
]

SELFTESTS = [
    ("launch_gate", [sys.executable, "scripts/launch_gate.py", "--selftest"]),
    ("card_claim", [sys.executable, "scripts/card_claim.py", "--selftest"]),
]


def run_selftests(d):
    """(caught_by, output) -- the first selftest that goes red, or None if all pass."""
    for name, cmd in SELFTESTS:
        p = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            return name, (p.stdout + p.stderr)[-400:]
    return None, ""


def main():
    base = tempfile.mkdtemp(prefix="mutate_cards_")
    try:
        # A real tree: the mutants must run against the real grant file, which is what the
        # launch_gate world is built from. Copy tracked files plus the untracked grant if any.
        tracked = subprocess.run(["git", "-C", ROOT, "ls-files"],
                                 capture_output=True, text=True).stdout.split("\n")
        d = os.path.join(base, "tree")
        for rel in (p for p in tracked if p.strip()):
            src = os.path.join(ROOT, rel)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(d, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        # .git, because both selftests call git ls-files and SKIP without it
        subprocess.run(["git", "init", "-q"], cwd=d, check=True)
        subprocess.run(["git", "add", "-A"], cwd=d, check=True)
        subprocess.run(["git", "-c", "user.email=x@y", "-c", "user.name=x",
                        "commit", "-qm", "base"], cwd=d, check=True)

        caught, out = run_selftests(d)
        if caught:
            print(f"REFUSING: the UNMUTATED tree is already red ({caught}), so every mutation "
                  f"below would 'be caught' for free:\n{out}")
            return 1
        print("baseline: both selftests green on the unmutated tree\n")

        survived = []
        for desc, rel, old, new in MUTANTS:
            p = os.path.join(d, rel)
            orig = open(p, encoding="utf-8").read()
            if orig.count(old) != 1:
                print(f"  SKIP  {desc}\n        anchor appears {orig.count(old)}x in {rel}, "
                      f"not once -- the mutation is not the one described")
                survived.append(desc + " (anchor not found)")
                continue
            try:
                open(p, "w", encoding="utf-8").write(orig.replace(old, new, 1))
                caught, out = run_selftests(d)
            finally:
                open(p, "w", encoding="utf-8").write(orig)
            if caught:
                first = next((ln for ln in out.split("\n") if "FAIL" in ln or "still" in ln
                              or "must" in ln or "does not" in ln), out.split("\n")[0])
                print(f"  caught by {caught:<12} {desc}\n        {first.strip()[:150]}")
            else:
                print(f"  SURVIVED   {desc}  <-- no selftest sees this defect")
                survived.append(desc)
        print()
        if survived:
            print(f"{len(survived)} of {len(MUTANTS)} mutations SURVIVED:")
            for s in survived:
                print(f"  - {s}")
            return 1
        print(f"all {len(MUTANTS)} mutations caught")
        return 0
    finally:
        shutil.rmtree(base, ignore_errors=True)


def selftest():
    """The sweep must report SURVIVED for a mutation nothing checks, or its own green is empty.

    Known answer: mutating a comment changes no behaviour, so no selftest can catch it and the
    sweep must say so. If this reported 'caught' the sweep would be measuring noise.
    """
    bad = []
    src = open(os.path.join(ROOT, "scripts", "launch_gate.py"), encoding="utf-8").read()
    for _desc, rel, old, _new in MUTANTS:
        n = open(os.path.join(ROOT, rel), encoding="utf-8").read().count(old)
        if n != 1:
            bad.append(f"anchor {old[:40]!r} appears {n}x in {rel}, not once")
    if not re.search(r"^def gate_cards", src, re.M):
        bad.append("gate_cards is gone from launch_gate.py; the sweep targets nothing")
    for b in bad:
        print(f"  FAIL {b}")
    print(f"mutate_gate_cards selftest: {len(MUTANTS) - len(bad)}/{len(MUTANTS)} anchors resolve")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
