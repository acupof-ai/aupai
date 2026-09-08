#!/usr/bin/env python3
"""Guards reachability.py's edge kinds against the two ways they go quietly wrong.

Read by the pre-commit hook whenever reachability.py is staged. Its output is a PASS/FAIL
line per case; a failure means the deletion-candidate list in runs/reachability.txt is
answering a different question than its header claims.

Both cases below are regressions of defects that were LIVE, not hypothetical:

  1. THE INSTRUMENT IN ITS OWN SEARCH SPACE. comment_edges() reads every file's comments
     for path citations. reachability.py's own FATE dict names 47 paths in prose, so on the
     first run it rescued 12 files from the candidate list -- every path it had previously
     ruled on -- purely because it had annotated them. A past verdict is not a citation,
     and a tool that vouches for whatever it has judged can never list those files again.
     Measured: 56 unreachable with the defect, 70 without it.

  2. ONE LEDGER OF 32. The function this replaced read runs/experiments.jsonl's `cmd` field
     and nothing else, reaching 53 files, while the report's conclusion covered the whole
     repo. review.jsonl alone cites 128 and tasks.jsonl 115.

Each case asserts the PROPERTY, not a count: counts move as the repo does, and a test that
pins them fails on every unrelated commit until someone reflexively updates the number.
"""
import importlib.util
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    """Import reachability.py without running its main()."""
    spec = importlib.util.spec_from_file_location(
        "_reach", os.path.join(ROOT, "scripts", "reachability.py"))
    mod = importlib.util.module_from_spec(spec)
    buf, real = io.StringIO(), sys.stdout
    sys.stdout = buf                      # module level does no printing today; cheap insurance
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.stdout = real
    return mod


fails = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        fails.append(name)
        if detail:
            print(f"       {detail}")


m = _load()

# --- 1. reachability.py must not cite anything through its own comments ---------------
edges = m.comment_edges()
self_sourced = sorted(f for f, (_kind, src) in edges.items() if src == m.SELF_PATH)
check("no edge is sourced from reachability.py's own comments",
      not self_sourced,
      f"{len(self_sourced)} file(s) rescued by this tool's own annotations, "
      f"e.g. {self_sourced[:3]}")

# THE NEGATIVE CONTROL. Without it the case above passes when comment_edges() is broken,
# returns {}, or stops reading FATE-style prose at all -- three worlds where "no self-edge"
# is true because there are no edges. So prove the exclusion is doing work: this file's own
# comments DO name paths (the docstring above names reachability.py and experiments.jsonl),
# and a source that is not excluded must still produce edges.
check("comment_edges still finds edges from other files",
      len(edges) > 0,
      "comment_edges() returned nothing; the case above would pass vacuously")

# And prove the exclusion is what suppresses them, not an empty scan: reading this file's
# prose through the same resolver must yield paths, so SELF_PATH's prose would too.
_self_text = m._read(m.SELF_PATH)
_self_prose = "\n".join(L for L in _self_text.splitlines() if m._COMMENT_RE.match(L))
check("reachability.py's own prose does contain resolvable paths (so the exclusion matters)",
      len(m._resolve_script(_self_prose)) > 0,
      "its comments name no known path; the exclusion would be a no-op and case 1 vacuous")

# --- 2. the ledger edge must read every ledger, not one -------------------------------
import glob  # noqa: E402  (kept beside its only use)

ledger_files = glob.glob(os.path.join(ROOT, "runs", "*.jsonl"))
sources = {src for _kind, src in m.ledger_edges().values()}
check("ledger edges come from more than one runs/*.jsonl",
      len(sources) > 1,
      f"only {sources} cited; the one-ledger defect is back")

# The stronger form: the scan must not be SCOPED to a subset. Assert it reads the glob
# rather than a list, by checking that a ledger outside experiments.jsonl contributes.
check("a ledger other than experiments.jsonl contributes edges",
      any(not s.endswith("experiments.jsonl") for s in sources),
      f"every edge came from experiments.jsonl; sources={sources}")

# --- 3. the report must state its own population --------------------------------------
# A conclusion whose population is invisible cannot be checked by its reader, and this
# tool's readers delete files. Asserted on the generated listing, not on the source, so
# a print that is removed or never reached fails here.
listing = os.path.join(ROOT, "runs", "reachability.txt")
if os.path.isfile(listing):
    head = open(listing, encoding="utf-8", errors="ignore").read(4000)
    for token in ("TREE:", "POPULATION:", "EDGE KINDS CHECKED:", "LEDGERS READ:"):
        check(f"runs/reachability.txt states {token.rstrip(':')}",
              token in head,
              "regenerate: python3 scripts/reachability.py > runs/reachability.txt")
    check("the population line names the .venv exclusion",
          ".venv" in head,
          "the denominator differs 537 vs 13,665 between a worktree and the integration "
          "tree; the listing must say which it counted")
else:
    check("runs/reachability.txt exists", False, "generate it before committing")

print(f"\n{'ALL OK' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
