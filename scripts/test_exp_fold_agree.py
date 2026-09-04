#!/usr/bin/env python3
"""All three readers of runs/experiments.jsonl fold it identically.

    python3 scripts/test_exp_fold_agree.py [--selftest]

WHY. Three call sites reduce that ledger to one row per run: scripts/exp.py's `fold` (the owner
and only writer), scripts/harness.py's `_exp_fold`, and scripts/launch_gate.py's `_recorded_cmd`.
Each has diverged at least once, and each divergence was found in production rather than by a
test:

  - harness.py once held four re-implementations; three were wrong (two position-based, one keyed
    on name alone, so a re-run of a name silently replaced the earlier run's row).
  - exp.py's own rows() folded by POSITION until e1-18, with a docstring asserting the divergent
    shape was impossible in a repo whose harness.py recorded it happening.
  - launch_gate._recorded_cmd folded by position until 53c62229, and it BLOCKED a launch:
    e1_c11_doccu_rescore closed `ok` at 05:41, a pod pull re-appended its `running` row after the
    close, and the gate then saw two runs claiming running and had no command to check.

All three now delegate to exp.fold, so the live risk has moved to the two inline FALLBACKS -- the
`except` branches that run when exp.py cannot be imported (a selftest world is a partial tree).
A fallback that folds differently from the real one is exactly the divergence the consolidation
ended, and nothing exercised those branches. This asserts both paths.

THE FIXTURE IS THE PRODUCTION SHAPE: a close followed by a re-appended start, which is what a pod
pull produces and what a union merge can order either way. Position-based folding reports the run
as running; terminal-wins reports it closed.

NEGATIVE CONTROLS, each naming the code path whose removal makes it fail (tilerl's rule):
  - case 2 removes exp.py from the import path, forcing every reader onto its `except` branch. A
    fallback reverted to `out[key] = r` fails here and passes case 1.
  - case 3 gives one run two DIFFERENT `started` values. A reader keyed on name alone folds them
    into one row and fails.
  - case 4 orders `ok` after `retracted`. A reader without the retraction rule un-retracts the
    run and fails.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# The production shape: a run that CLOSED, with its `running` event ordered after the close.
CLOSED_LATE_START = [
    {"name": "alpha", "started": "2026-09-04 05:10", "status": "running",
     "cmd": "python train.py --name alpha"},
    {"name": "alpha", "started": "2026-09-04 05:10", "status": "ok", "result": "done",
     "cmd": "python train.py --name alpha"},
    # the pod pull's re-append, AFTER the close
    {"name": "alpha", "started": "2026-09-04 05:10", "status": "running",
     "cmd": "python train.py --name alpha"},
    {"name": "beta", "started": "2026-09-04 06:00", "status": "running",
     "cmd": "python train.py --name beta"},
]
SAME_NAME_TWO_RUNS = [
    {"name": "gamma", "started": "2026-09-04 01:00", "status": "fail", "result": "OOM"},
    {"name": "gamma", "started": "2026-09-04 02:00", "status": "running"},
]
RETRACT_THEN_OK = [
    {"name": "delta", "started": "2026-09-04 03:00", "status": "running"},
    {"name": "delta", "started": "2026-09-04 03:00", "status": "retracted",
     "result": "measured on the wrong checkpoint"},
    {"name": "delta", "started": "2026-09-04 03:00", "status": "ok", "result": "0.42"},
]


def _key(r):
    return (r.get("name"), r.get("started"))


def _as_map(folded):
    return {_key(r): r.get("status") for r in folded}


def _in_subprocess(evs, block_exp):
    """Each reader's fold, in a child so `block_exp` can hide scripts/exp.py from it.

    A child, not importlib surgery: harness and launch_gate insert scripts/ onto sys.path
    themselves inside the function, so hiding exp.py in-process would need the path mutated back
    after every call and the first cached import would defeat it anyway.
    """
    d = tempfile.mkdtemp(prefix="foldagree_")
    try:
        os.makedirs(os.path.join(d, "runs"))
        os.makedirs(os.path.join(d, "scripts"))
        with open(os.path.join(d, "runs", "experiments.jsonl"), "w", encoding="utf-8") as fh:
            for r in evs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        # The real modules, copied into the world. exp.py is copied only when it is not blocked;
        # its ABSENCE is what forces the two fallbacks, and that is case 2's whole content.
        wanted = ["harness.py", "launch_gate.py"] + ([] if block_exp else ["exp.py"])
        for f in wanted:
            src = os.path.join(HERE, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(d, "scripts", f))
        prog = f'''
import ast, json, os, sys
sys.path.insert(0, {os.path.join(d, "scripts")!r})
ROOT = {d!r}
out = {{}}

evs = [json.loads(x) for x in open(os.path.join(ROOT, "runs", "experiments.jsonl"),
                                   encoding="utf-8") if x.strip()]


def take(path, name):
    """The source of one top-level def, bounded by AST line span.

    NOT a regex. `\\ndef NAME.*?\\n(?=def |\\Z)` was the first version and it swallowed the
    NEXT function's decorator -- `@functools.lru_cache` became a bare line and the exec died
    with SyntaxError. lineno/end_lineno are exact.
    """
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            lines = src.splitlines()[n.lineno - 1:n.end_lineno]
            return "\\n".join(lines)
    raise AssertionError("%s does not define %s at top level" % (path, name))


# 1. exp.fold, when it is available at all.
try:
    from exp import fold
    out["exp"] = [(r.get("name"), r.get("started"), r.get("status")) for r in fold(evs)]
except Exception as e:
    out["exp"] = "unavailable: %s" % type(e).__name__

# 2. harness._exp_fold -- taken from source rather than imported, because importing harness
#    pulls its whole module scope (corpus_fingerprint, pod_drift) for one function.
ns = {{"sys": sys, "os": os, "ROOT": ROOT}}
exec(take(os.path.join({HERE!r}, "harness.py"), "_exp_fold"), ns)
out["harness"] = [(r.get("name"), r.get("started"), r.get("status"))
                  for r in ns["_exp_fold"](evs)]

# 3. launch_gate._recorded_cmd -- same technique. It returns (cmd, source) for the ONE running
#    run, so its answer is which run is running, not the whole fold.
#
#    __file__ MUST BE IN THE NAMESPACE. That function's import line is
#    `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`, so an exec namespace
#    without __file__ raises NameError, the except branch catches it, and the fallback runs on
#    EVERY case -- the delegated path would never be measured while all four cases passed. Caught
#    by mutation: reverting the fallback to position-based folding failed case 1 instead of case
#    2, which is only possible if case 1 was already on the fallback. Pointed at the world's copy
#    so the path it inserts is the world's scripts/, which is where exp.py is or is not.
ns2 = {{"sys": sys, "os": os, "json": json,
        "__file__": os.path.join({os.path.join(d, "scripts")!r}, "launch_gate.py")}}
exec(take(os.path.join({HERE!r}, "launch_gate.py"), "_recorded_cmd"), ns2)
out["gate"] = ns2["_recorded_cmd"](ROOT)
out["gate_used_fallback"] = "exp" not in sys.modules
print(json.dumps(out))
'''
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True,
                           env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")})
        if r.returncode != 0:
            raise AssertionError(f"the reader probe failed: {(r.stdout + r.stderr)[-500:]}")
        return json.loads(r.stdout)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _agree(label, evs, block_exp, want):
    got = _in_subprocess(evs, block_exp)
    exp_rows = got["exp"]
    har_rows = {(n, s): st for n, s, st in got["harness"]}
    if isinstance(exp_rows, str):
        assert block_exp, f"{label}: exp.fold was unavailable when it should not be ({exp_rows})"
    else:
        assert {(n, s): st for n, s, st in exp_rows} == want, (
            f"{label}: exp.fold gave {exp_rows}, wanted {sorted(want.items())}")
    # WHICH PATH RAN, asserted rather than assumed. Without this the delegated path can silently
    # never be exercised -- a missing __file__ in the exec namespace put every case on the
    # fallback and all four still passed (see the probe's comment).
    assert got["gate_used_fallback"] == block_exp, (
        f"{label}: the gate took the {'fallback' if got['gate_used_fallback'] else 'delegated'} "
        f"path with exp.py {'hidden' if block_exp else 'present'} -- the case is not measuring "
        f"the path it names")
    assert har_rows == want, (
        f"{label}: harness._exp_fold gave {sorted(har_rows.items())}, wanted "
        f"{sorted(want.items())} -- the two readers disagree, which is the divergence the "
        f"consolidation ended")
    return got


def main():
    # CASE 1: exp.py present. All three read the close as terminal despite the later start.
    want1 = {("alpha", "2026-09-04 05:10"): "ok", ("beta", "2026-09-04 06:00"): "running"}
    got = _agree("close then re-appended start", CLOSED_LATE_START, False, want1)
    cmd, src = got["gate"]
    assert cmd == "python train.py --name beta", (
        f"launch_gate named {cmd!r} as the running command. Position-based folding sees TWO runs "
        f"running and returns None -- the 53c62229 shape, which blocked b0's launch")
    print("  case 1: close beats a later start in all three readers; the gate names beta")

    # CASE 2: THE NEGATIVE CONTROL. exp.py hidden, so both inline fallbacks run. A fallback that
    # reverted to `out[key] = r` reports alpha as running here and passes case 1.
    got = _agree("same shape, exp.py hidden (fallbacks)", CLOSED_LATE_START, True, want1)
    cmd, src = got["gate"]
    assert cmd == "python train.py --name beta", (
        f"the launch_gate FALLBACK named {cmd!r}: it folds differently from exp.fold, which is "
        f"the divergence the consolidation ended, hiding in the except branch")
    print("  case 2: both inline fallbacks agree with exp.fold, so a missing exp.py degrades to "
          "the correct answer")

    # CASE 3: one name, two runs. A reader keyed on name alone folds these into one row.
    want3 = {("gamma", "2026-09-04 01:00"): "fail", ("gamma", "2026-09-04 02:00"): "running"}
    for _blocked in (False, True):
        _agree("same name, two started times", SAME_NAME_TWO_RUNS, _blocked, want3)
    print("  case 3: (name, started) keeps two runs of one name apart, both paths")

    # CASE 4: `ok` ordered after `retracted`. Terminal-by-kind, not by position.
    want4 = {("delta", "2026-09-04 03:00"): "retracted"}
    for _blocked in (False, True):
        _agree("ok ordered after retracted", RETRACT_THEN_OK, _blocked, want4)
    print("  case 4: a retraction survives a later ok, both paths")

    print("exp fold agreement: exp.py, harness._exp_fold and launch_gate._recorded_cmd agree on "
          "4 shapes, with exp.py present and hidden")


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--selftest"]):
        sys.exit(f"usage: {os.path.basename(__file__)} [--selftest]  (got {sys.argv[1:]})")
    main()
