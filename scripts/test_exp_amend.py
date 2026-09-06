#!/usr/bin/env python3
"""exp.py amend adds reading_artifact to a CLOSED row and REFUSES every other row state.

The defect this guards: three rows were closed `ok` with no reading_artifact, so harness.py's
score_matrix_present read them as unscored training runs. `done` refuses (the row is not open)
and `retract` withdraws the RESULT to satisfy a gate about its READING. amend is the field-only
edit; these worlds are what make it not a way to un-retract.

Worlds are built by MUTATING the real ledger (copy, then append rows), never hand-written: a
hand-written world shares the test author's assumptions about the schema.

    python3 scripts/test_exp_amend.py
"""
import json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "scripts", "exp.py")
REAL = os.path.join(ROOT, "runs", "experiments.jsonl")

def world(status):
    """A repo-shaped tree whose ledger is the REAL one plus one zz_amend row in `status`."""
    d = tempfile.mkdtemp(prefix="expamend_")
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    shutil.copy(REAL, os.path.join(d, "runs", "experiments.jsonl"))
    # a real file to point at, so the path check is not what fails
    shutil.copy(os.path.join(ROOT, "runs", "moe_diag.jsonl"), os.path.join(d, "runs", "moe_diag.jsonl"))
    rows = [json.loads(x) for x in open(REAL, encoding="utf-8") if x.strip()]
    tmpl = rows[-1]
    st = "2026-09-02 05:00"
    evs = [dict(tmpl, name="zz_amend", status="running", started=st, result="", ended="",
                reading_artifact=None)]
    if status != "running":
        evs.append(dict(tmpl, name="zz_amend", status="ok", started=st, result="r1",
                        ended="2026-09-02 06:00", reading_artifact=None))
    if status == "retracted":
        evs.append(dict(tmpl, name="zz_amend", status="retracted", started=st, result="r1",
                        retracted_reason="test", retracted_result="r1", reading_artifact=None))
    with open(os.path.join(d, "runs", "experiments.jsonl"), "a", encoding="utf-8") as f:
        for e in evs:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return d, st

def run(d, *args):
    return subprocess.run([sys.executable, EXP, "--root", d, "amend", *args],
                          capture_output=True, text=True)

def folded(d, name):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import importlib, exp
    importlib.reload(exp)
    exp.set_root(d)
    return [r for r in exp.rows() if r["name"] == name]

bad = 0
# 1. CLOSED ok -> the field lands and status is untouched
d, st = world("ok")
r = run(d, "--name", "zz_amend", "--started", st, "--reading_artifact", "runs/moe_diag.jsonl")
rows = folded(d, "zz_amend")
ok = r.returncode == 0 and len(rows) == 1 and rows[0]["reading_artifact"] == "runs/moe_diag.jsonl" \
     and rows[0]["status"] == "ok" and rows[0]["result"] == "r1"
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} a closed ok row gains reading_artifact, status and result unchanged"
      + ("" if ok else f" -- rc={r.returncode} {r.stdout.strip()[:90]}{r.stderr.strip()[:90]} rows={rows}"))
shutil.rmtree(d)

# 2. RETRACTED -> refused, and NOT silently a no-op
d, st = world("retracted")
r = run(d, "--name", "zz_amend", "--started", st, "--reading_artifact", "runs/moe_diag.jsonl")
rows = folded(d, "zz_amend")
ok = r.returncode != 0 and "retracted" in (r.stdout + r.stderr) and rows[0]["status"] == "retracted" \
     and rows[0].get("reading_artifact") in (None, "")
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} a retracted row is REFUSED, so amend is not a way to un-retract"
      + ("" if ok else f" -- rc={r.returncode} {(r.stdout+r.stderr).strip()[:110]} status={rows[0]['status']}"))
shutil.rmtree(d)

# 3. RUNNING -> refused (pick_closed_row's job, asserted here so it stays that way)
d, st = world("running")
r = run(d, "--name", "zz_amend", "--reading_artifact", "runs/moe_diag.jsonl")
ok = r.returncode != 0 and "running" in (r.stdout + r.stderr).lower()
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} a running row is REFUSED -- a reading of an unfinished run is not a reading"
      + ("" if ok else f" -- rc={r.returncode} {(r.stdout+r.stderr).strip()[:110]}"))
shutil.rmtree(d)

# 4. NONEXISTENT PATH -> refused. The negative control for case 1: without this, amend would
#    satisfy the gate with a path nobody wrote, which is worse than the gap it closes.
d, st = world("ok")
r = run(d, "--name", "zz_amend", "--started", st, "--reading_artifact", "runs/no_such_file.jsonl")
rows = folded(d, "zz_amend")
ok = r.returncode != 0 and "does not exist" in (r.stdout + r.stderr) \
     and rows[0].get("reading_artifact") in (None, "")
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} a path that does not exist is REFUSED and nothing is appended"
      + ("" if ok else f" -- rc={r.returncode} {(r.stdout+r.stderr).strip()[:110]}"))
shutil.rmtree(d)

print(f"exp amend selftest: {4 - bad}/4 pass")
sys.exit(1 if bad else 0)
