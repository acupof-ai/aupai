"""de-5 classification: the 45 files reachability reports as unreached are 45 UNCLASSIFIED,
not 45 deletion candidates (6e, 2026-09-03).

Four kinds of liveness evidence, three of which the citation graph cannot see by construction:
  hook        registered in scripts/hooks/pre-commit's SELFTEST_FILES -- runs on every commit
  glob        loaded by a runtime glob or importlib somewhere in the tree
  terminal    an operational tool whose only caller is a person at a shell (progress_feed.py's
              row in AGENTS.md is the precedent; a2a_bandwidth.py is one I just added)
  ci          named by .github/workflows
  unclear     none of the above found -- these are the ones worth running before judging

This script only CLASSIFIES. It deletes nothing and rules nothing.
"""

import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def unreached():
    out = subprocess.run(["python3", os.path.join(ROOT, "scripts/reachability.py")],
                         capture_output=True, text=True, cwd=ROOT).stdout
    files = []
    for ln in out.splitlines():
        if re.search(r"\s+none\s*$", ln):
            files.append(ln.split()[0])
    return files


def _read(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return ""
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def classify(files):
    hook = _read("scripts/hooks/pre-commit")
    ci = ""
    wf = os.path.join(ROOT, ".github/workflows")
    if os.path.isdir(wf):
        for n in os.listdir(wf):
            ci += _read(f".github/workflows/{n}")
    agents = _read("AGENTS.md")

    # Every glob/importlib pattern in the tree, so a file matched by one is found by its
    # PATTERN rather than by its name -- the vet_programs.py:37 case.
    globs = []
    for dp, dn, fns in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in {".git", "data", "runs", "__pycache__", ".venv"}]
        for f in fns:
            if not f.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dp, f), ROOT)
            for m in re.finditer(r"glob\.glob\(([^)]*)\)|importlib[.\w]*\(([^)]*)\)", _read(rel)):
                globs.append((rel, (m.group(1) or m.group(2) or "").strip()))

    out = {}
    for rel in files:
        base = os.path.basename(rel)
        stem = base.rsplit(".", 1)[0]
        why = []
        if f'"{rel}"' in hook or f"'{rel}'" in hook:
            why.append("hook: SELFTEST_FILES")
        if base in ci or rel in ci or stem in ci:
            why.append("ci: named by a workflow")
        for src, pat in globs:
            # a pattern whose literal prefix matches this file's name is a candidate edge
            lit = re.split(r"[*?\[]", pat.strip("\"'"))[0]
            lit = os.path.basename(lit)
            if lit and len(lit) >= 4 and base.startswith(lit) and src != rel:
                why.append(f"glob: {src} {pat[:40]}")
                break
        if rel in agents or base in agents:
            why.append("terminal: has a row in AGENTS.md")
        out[rel] = why or ["unclear"]
    return out


def main():
    files = unreached()
    got = classify(files)
    order = ["hook", "ci", "glob", "terminal", "unclear"]
    groups = {k: [] for k in order}
    for rel, why in got.items():
        key = next((k for k in order if any(w.startswith(k) for w in why)), "unclear")
        groups[key].append((rel, why))
    print(f"{len(files)} files reported unreached, classified by liveness evidence:\n")
    for k in order:
        rows = sorted(groups[k])
        if not rows:
            continue
        label = {
            "hook": "ALIVE via the pre-commit hook -- invisible to a citation graph",
            "ci": "ALIVE via CI",
            "glob": "ALIVE via a runtime glob -- invisible to a citation graph",
            "terminal": "OPERATIONAL, cited in AGENTS.md -- the progress_feed.py precedent",
            "unclear": "NO evidence found -- RUN each before judging, do not delete on this listing",
        }[k]
        print(f"[{k}] {len(rows)}: {label}")
        for rel, why in rows:
            print(f"    {rel:52s} {'; '.join(why)[:70]}")
        print()


if __name__ == "__main__":
    main()
