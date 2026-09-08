#!/usr/bin/env python3
"""Guards the score-matrix rewrite trace: the writer's half and the check's half.

Read by the pre-commit hook whenever eval/score_matrix.py or scripts/harness.py is
staged. Its output is a PASS/FAIL line per case.

WHAT IS BEING PROTECTED, and why it needs a test rather than a comment. Measured
2026-09-08: of 109 harness checks, three read runs/score_matrix.jsonl and only one was
dedicated to it, asserting a row EXISTS. Nothing read a value and asked whether it should
have moved, so a wrong domain_bpb factor table sat on main for hours -- row present,
well-formed, one line, right ledger, every check that could see it satisfied.

This ledger folds on (ckpt, profile) and its writer REPLACES, so a value change here is
LEGAL, unlike the other eight append-only ledgers. Over the 34 commits that touch the
file: 79 new keys, 44 rewrites, 6 of which changed an existing number. The design follows
from those numbers -- trace the 6, stay silent on the other 38.

Each case below asserts a property with its own negative control where the passing state
could be reached by the machinery not running.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "eval"))

import harness  # noqa: E402
import pod_drift  # noqa: E402
import score_matrix as sm  # noqa: E402

CHECK = harness.check_score_matrix_rewrites_traced
ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}

fails = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        fails.append(name)
        if detail:
            print(f"       {detail}")


def _repo(row):
    """A real git repo holding one committed matrix row. Real history, because the check's
    only source of truth is `git show HEAD:<path>` and a world without it would SKIP --
    and a SKIP that reads as 'not broken' is this check's own failure mode."""
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    p = os.path.join(d, "runs", "score_matrix.jsonl")
    with open(p, "w") as f:
        f.write(json.dumps(row) + "\n")
    for cmd in (["init", "-q", "-b", "main"], ["add", "-A"],
                ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "w",
                 "--no-verify"]):
        subprocess.run(["git", "-C", d, *cmd], env=ENV, capture_output=True)
    return d, p


BASE = {"ckpt": "a.pt", "profile": "full", "metrics": {"x": {"loss": 1.0}, "note": "n"}}

# --- 1. the writer records a numeric change, and ONLY a numeric change ----------------
d = tempfile.mkdtemp()
p = os.path.join(d, "m.jsonl")
sm.write_records(p, [BASE])
sm.write_records(p, [dict(BASE, extra=1)])            # adds a field, moves no number
r = json.loads(open(p).read().strip())
check("a rewrite that only ADDS a field leaves no trace",
      "superseded" not in r,
      "30 of 44 real rewrites are add-only; tracing them makes the guard noise")

sm.write_records(p, [{"ckpt": "a.pt", "profile": "full",
                      "metrics": {"x": {"loss": 2.5}, "note": "n"}, "extra": 1}])
r = json.loads(open(p).read().strip())
ent = (r.get("superseded") or [{}])[-1]
check("a rewrite that MOVES a number records old and new",
      ent.get("changed", {}).get(".metrics.x.loss") == [1.0, 2.5],
      f"got {ent.get('changed')}")

sm.write_records(p, [{"ckpt": "a.pt", "profile": "full",
                      "metrics": {"x": {"loss": 3.5}, "note": "n"}, "extra": 1}])
r = json.loads(open(p).read().strip())
check("a second numeric change APPENDS rather than erasing the first",
      len(r.get("superseded") or []) == 2,
      f"{len(r.get('superseded') or [])} entries; a rewrite must not erase the row's history")

# --- 2. the check fails on an untraced rewrite and passes on a traced one -------------
d, p = _repo(BASE)
st_base, _ = CHECK(d)
check("an unchanged committed matrix passes", st_base == "PASS", st_base)

with open(p, "w") as f:                               # hand-edit: number moves, no trace
    f.write(json.dumps({"ckpt": "a.pt", "profile": "full",
                        "metrics": {"x": {"loss": 2.0}, "note": "n"}}) + "\n")
st_bad, ev_bad = CHECK(d)
check("an untraced numeric rewrite FAILs", st_bad == "FAIL", f"{st_bad}: {ev_bad[:120]}")
check("the refusal names the field that moved",
      ".metrics.x.loss" in ev_bad,
      "a refusal that does not say WHICH number moved sends the reader back to diffing")

subprocess.run(["git", "-C", d, "checkout", "-q", "--", "runs/score_matrix.jsonl"], env=ENV)
sm.write_records(p, [{"ckpt": "a.pt", "profile": "full",
                      "metrics": {"x": {"loss": 2.0}, "note": "n"}}])
st_ok, _ = CHECK(d)
check("the SAME change written through write_records passes",
      st_ok == "PASS",
      "the negative control: if this also failed, the check would be refusing rewrites "
      "rather than untraced ones, and the writer's trace would be doing nothing")

# --- 3. a carried-over trace does not cover a NEW undeclared change -------------------
# Presence of `superseded` is not the property; a NEW entry is. A row that already carries
# one entry from a previous rewrite must not be able to hide a second, hand-made change.
traced = {"ckpt": "b.pt", "profile": "full", "metrics": {"x": {"loss": 1.0}},
          "superseded": [{"at": "2026-01-01T00:00:00Z", "by": "old",
                          "changed": {".metrics.x.loss": [0.5, 1.0]}}]}
d2, p2 = _repo(traced)
with open(p2, "w") as f:
    f.write(json.dumps(dict(traced, metrics={"x": {"loss": 9.0}})) + "\n")
st2, ev2 = CHECK(d2)
check("an old trace does not excuse a new undeclared change",
      st2 == "FAIL",
      f"{st2}: a row carrying one stale entry passed a second, invisible rewrite")

# --- 4. no git means SKIP, never PASS -------------------------------------------------
# THE R12 CASE, and it is the one this check could most easily get wrong about itself:
# /work/aupai is a hand-pushed tree with no .git, so the check's only source of truth is
# absent there. A PASS would be a line green because it never ran -- built, by us, into
# the check written to catch a silent value change.
d3 = tempfile.mkdtemp()
os.makedirs(os.path.join(d3, "runs"), exist_ok=True)
with open(os.path.join(d3, "runs", "score_matrix.jsonl"), "w") as f:
    f.write(json.dumps(BASE) + "\n")
check("the no-git world really is pod-shaped (so the case below is not vacuous)",
      pod_drift.is_pod(d3),
      "if this is False the SKIP below comes from some other branch and proves nothing")
st3, ev3 = CHECK(d3)
# MEASURED, NOT ASSUMED (mutation run, 2026-09-08). This assertion is WEAKER than its name,
# and here is the reason rather than just the warning: TWO DIFFERENT CAUSES PRODUCE THE SAME
# SKIP, AND THE ASSERTION ONLY READS THE RESULT. Remove the pod branch and the check falls
# through to `git show HEAD:<path>`, which also fails in a tree with no git and also returns
# SKIP -- so the mutant that deletes the branch this case exists for leaves this line green.
# The assertion below, which reads the REASON, is the only one with power over it.
# Kept and documented rather than deleted: the next person writes this same assertion, and a
# warning carrying its reason survives being read as a style opinion where a bare one does not.
check("a tree with no .git SKIPs rather than PASSing", st3 == "SKIP", st3)
check("the SKIP names the pod rather than saying a generic 'no git'",
      "pod" in ev3.lower(),
      "a reader who sees SKIP must learn WHERE it cannot run, or the skip is folklore; "
      "this is also the only assertion that dies when the pod branch is removed")

# --- 5. the check and the writer share one comparison ---------------------------------
# R11's own rule (§278): to measure a correction for an instrument, call the instrument.
# If the check grew its own copy of "did a number move", the two would disagree on
# exactly the rows that matter.
src = open(os.path.join(ROOT, "scripts", "harness.py"), encoding="utf-8").read()
check("the check CALLS score_matrix._supersede_entry rather than reimplementing it",
      "score_matrix._supersede_entry" in src,
      "a second copy of the comparison is a second population by construction")

# --- 6. KNOWN ANSWER for the shared comparison itself ---------------------------------
# THE BLIND SPOT SHARING CREATES (4c, 2026-09-08). Case 5 makes both sides call one
# function, which is right -- and it means a leaf class that function MISSES is missed by
# both: the writer does not record it, the check does not count it, and the two agree
# silently. Same shape as a hand-written broken world that shares the check's assumptions,
# except what is shared here is the implementation.
#
# The fix is not a second copy of the logic, which would drift. It is a known answer: a
# hand-built pair whose changed-key set is asserted EXACTLY, not merely as non-empty. A
# "not empty" assertion passes on a function that reports one leaf out of five.
old = {"f": 1.0, "s": "a", "gone": 7, "deep": {"a": {"b": 2}}, "same": 3,
       "flag": False, "lst": [1, 2]}
new = {"f": 2.0, "s": "b", "added": 8, "deep": {"a": {"b": 5}}, "same": 3,
       "flag": True, "lst": [1, 9]}
ent = sm._supersede_entry(old, new, "w", "t")
got = set((ent or {}).get("changed", {}))
# .f moved, .deep.a.b moved two levels down, .lst[1] moved inside a list.
# .s is a string -- not tracked, by design. .same did not move. .gone and .added exist on
# one side only, so there is no old-vs-new pair to record. .flag is a bool, excluded.
want = {".f", ".deep.a.b", ".lst[1]"}
check("_supersede_entry reports EXACTLY the moved numeric leaves",
      got == want,
      f"got {sorted(got)}, want {sorted(want)} -- a missed leaf class is missed by the "
      f"writer and the check together, with no disagreement to notice")
check("_supersede_entry returns None when nothing numeric moved",
      sm._supersede_entry({"a": 1, "s": "x"}, {"a": 1, "s": "y"}, "w", "t") is None,
      "a string-only change must not open a trace, or 30 of 44 real rewrites become noise")
check("_supersede_entry does not treat a bool as a number",
      ".flag" not in got,
      "True == 1 in Python; a status flag flipping is not a measurement moving")

print(f"\n{'ALL OK' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
