#!/usr/bin/env python3
# restartable: a read-only grep; an interrupt costs nothing.
"""scripts/retract.py <phrase> [<phrase>...]: every site a retracted claim still circulates.

A retraction's last step is a run of this tool, not recall. §102/§107: retracted sentences
averaged 3-4 propagation sites on 2026-09-03 and were found by hand grep each time; the
rule "a retraction ends with a whole-repo grep" had no tool, so it was prose people
followed when they remembered.

Literal substring, case-sensitive: the point is the exact retracted sentence, so a
near-miss (one number or word changed) is NOT a hit -- it is a different claim. Prints
file:line for every site and a count; exit 0 either way, because zero hits means the
retraction is complete. Git history is not searched: a claim living only in a commit
message circulates only when someone re-quotes it, and re-quotes land in the tree.

Searches text files repo-wide (docs/, facts/, AGENTS.md, runs/*.jsonl, scripts/, *.md),
skipping .git and data/ (corpus and binaries). --root points at another tree.

    python3 scripts/retract.py "warmdown's start from step 359 to step 14" "18.7x"
    python3 scripts/retract.py --selftest
"""
import argparse
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".py", ".txt"}
PRUNE_DIRS = {".git", "data", "__pycache__", "node_modules"}


def find_sites(root, phrases):
    """Every (relpath, lineno, phrase, line) where a phrase occurs, repo-wide."""
    sites = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1] not in TEXT_SUFFIXES:
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        for ph in phrases:
                            if ph in line:
                                sites.append((os.path.relpath(path, root), i, ph,
                                              line.strip()[:160]))
            except (IsADirectoryError, PermissionError, OSError):
                continue
    return sites


def _selftest():
    """Four planted sites must all be found; a near-miss must not; zero hits is a valid answer."""
    phrase = "warmdown moves 359 to 14 at 18.7x"
    near = "warmdown moves 359 to 14 at 19.2x"  # one number changed: a different claim
    with tempfile.TemporaryDirectory() as td:
        def put(rel, text):
            p = os.path.join(td, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)

        # the four site kinds a retraction must reach
        put("docs/lessons/x.md", f"# a doc\n{phrase} -- stated as fact\n")
        put("facts/x.json", json.dumps({"id": "x.fact", "claim": f"once said {phrase}"}) + "\n")
        put("runs/tasks.jsonl", json.dumps({"id": "t1", "task": phrase}) + "\n")
        put("scripts/x.py", f'"""a tool\n{phrase}\n"""\nx = 1\n')
        # the near-miss: same shape, one number -- must NOT be reported as the retracted claim
        put("scripts/near.py", f'"""{near}"""\n')

        sites = find_sites(td, [phrase])
        files = {s[0] for s in sites}
        want = {"docs/lessons/x.md", "facts/x.json", "runs/tasks.jsonl", "scripts/x.py"}
        if len(sites) != 4 or files != want:
            print(f"FAIL: planted 4 sites, found {len(sites)}: {sorted(files)}", file=sys.stderr)
            return 1
        if "scripts/near.py" in files:
            print("FAIL: a near-miss (one number changed) was reported as the retracted claim",
                  file=sys.stderr)
            return 1
        # a phrase that appears nowhere: zero sites is a complete retraction, not an error
        if find_sites(td, ["no such phrase anywhere 9q7w4e"]):
            print("FAIL: a nonexistent phrase returned sites", file=sys.stderr)
            return 1
    print("retract selftest OK: 4/4 planted sites found (doc, fact, ledger row, docstring); "
          "a one-number near-miss did not hit; a nonexistent phrase returns zero")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phrases", nargs="*", help="literal key phrases of the retracted claim")
    ap.add_argument("--root", default=ROOT, help="tree to search (default: the repo)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    if not args.phrases:
        ap.error("at least one phrase is required (or --selftest)")
    sites = find_sites(args.root, args.phrases)
    cur = None
    for path, lineno, ph, line in sites:
        if path != cur:
            print(f"\n{path}")
            cur = path
        print(f"  {lineno}: [{ph}] {line}")
    print(f"\n{len(sites)} site(s) in {len({s[0] for s in sites})} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
