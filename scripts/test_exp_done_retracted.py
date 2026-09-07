#!/usr/bin/env python3
"""exp.py `done` REFUSES an already-closed row instead of fabricating an orphan.

The defect (measured 2026-09-06, de-46, over all 8 combinations of retracted / --started /
later-minute): pick_open_row's subject is rows whose last event is `running`, so it returns None
for a name nobody started AND for a name already closed. `done` then built a base from `or {...}`
with started=now() and exited 0 printing "logged done". RETRACTION IS NOT THE VARIABLE -- being
closed is; a bare `done` on any closed row fabricated. What it produced depended on the CLOCK,
because now() is minute resolution:

  same minute as the close -> the fabricated `started` collides with the original's and the event
    folds onto that row. On a retracted row fold() then DISCARDS it (`retracted` is terminal by
    kind): success printed, nothing changed.
  a later minute -> TWO folded rows for one run: the original keeping its result, and a new row
    carrying this one with cmd='', hypothesis='' and no commit.

Both clock worlds are here because they are the same command and a fix handling one would leave
the other. Cases 3 and 4 are the negative controls that keep the refusal narrow: a name with NO
row must still close (run_sft.sh and run_pretrain.sh call `done` unconditionally at exit, and
closing a run whose start event was lost is what the fabrication is legitimately for), and the
refusal must name the state it found rather than only its absence.

Worlds MUTATE the real ledger (copy, then append), never hand-written.

    python3 scripts/test_exp_done_retracted.py
"""
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "scripts", "exp.py")
REAL = os.path.join(ROOT, "runs", "experiments.jsonl")


def world(started="2026-09-02 05:00", retract=True, close=True):
    """The REAL ledger plus one zz_done_retr run: start, then optionally ok, then optionally
    retracted. close=False leaves it RUNNING, which is the normal-path control."""
    d = tempfile.mkdtemp(prefix="expdoneretr_")
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    shutil.copy(REAL, os.path.join(d, "runs", "experiments.jsonl"))
    rows = [json.loads(x) for x in open(REAL, encoding="utf-8") if x.strip()]
    # `writer` IS STRIPPED, and that is not cosmetic (de-70, 2026-09-08). tmpl is the real
    # ledger's LAST row, so this fixture inherits whatever fields that row happens to carry today
    # -- and on 2026-09-08 the last row was a monitor close, so every zz_done_retr event silently
    # became writer="monitor". `done` now treats a row closed ONLY by the monitor as re-closable
    # with --reason, so case 1 below started failing with the reclassify refusal instead of the
    # ordinary one: a fixture whose subject changed because someone else's run landed. Every case
    # here is about a HUMAN-closed row, which is a row with no writer.
    tmpl = {k: v for k, v in rows[-1].items() if k != "writer"}
    evs = [
        dict(tmpl, name="zz_done_retr", status="running", started=started, result="", ended="",
             cmd="./run_ddp.sh --name zz_done_retr"),
    ]
    if close:
        evs.append(dict(tmpl, name="zz_done_retr", status="ok", started=started,
                        result="loss 2.10", ended="2026-09-02 06:00",
                        cmd="./run_ddp.sh --name zz_done_retr"))
    if retract:
        evs.append(dict(tmpl, name="zz_done_retr", status="retracted", started=started,
                        result="loss 2.10", retracted_reason="read off the wrong arm",
                        retracted_result="loss 2.10", retracted_at="2026-09-02 07:00",
                        cmd="./run_ddp.sh --name zz_done_retr"))
    with open(os.path.join(d, "runs", "experiments.jsonl"), "a", encoding="utf-8") as f:
        for e in evs:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return d, started


def run(d, *args):
    return subprocess.run([sys.executable, EXP, "--root", d, "done", *args],
                          capture_output=True, text=True)


def folded(d, name):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import exp
    importlib.reload(exp)
    exp.set_root(d)
    return [r for r in exp.rows() if r["name"] == name]


bad = 0
CASES = 6

# 1. RETRACTED, bare `done` -- the shape a human types, and the one that fabricated. Must refuse,
#    name the retraction, and append nothing. len(rows) == 1 is the assertion that catches the
#    orphan: a status check alone passes on the orphan world, because rows[0] is still the original.
d, st = world()
r = run(d, "--name", "zz_done_retr", "--status", "ok", "--result", "corrected")
rows = folded(d, "zz_done_retr")
ok = (r.returncode != 0 and "retracted" in (r.stdout + r.stderr) and len(rows) == 1
      and rows[0]["result"] == "loss 2.10" and all(x.get("cmd") for x in rows))
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} bare `done` on a RETRACTED row is REFUSED, no orphan row"
      + ("" if ok else f" -- rc={r.returncode} {(r.stdout + r.stderr).strip()[:120]} "
                       f"rows={[(x.get('started'), x['status'], x.get('cmd')) for x in rows]}"))
shutil.rmtree(d)

# 2. CLOSED ok, bare `done` -- RETRACTION IS NOT THE VARIABLE. This is the same fabrication on a
#    row that was never retracted, and it is the case a refusal keyed on `status == "retracted"`
#    would miss entirely. Measured: it produced a second row with cmd=''.
d, st = world(retract=False)
r = run(d, "--name", "zz_done_retr", "--status", "ok", "--result", "corrected")
rows = folded(d, "zz_done_retr")
ok = (r.returncode != 0 and "already closed" in (r.stdout + r.stderr) and len(rows) == 1
      and rows[0]["result"] == "loss 2.10" and all(x.get("cmd") for x in rows))
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} bare `done` on a closed-but-NOT-retracted row is also REFUSED"
      + ("" if ok else f" -- rc={r.returncode} {(r.stdout + r.stderr).strip()[:120]} "
                       f"rows={[(x.get('started'), x['status'], x.get('cmd')) for x in rows]}"))
shutil.rmtree(d)

# 3. RETRACTED, --started given. This path exits inside pick_open_row, which used to say only
#    "Open rows: none" -- true, and useless: it reads identically for a name nobody started and
#    for the row the caller is holding, already closed. Both paths must name the state found.
d, st = world()
r = run(d, "--name", "zz_done_retr", "--started", st, "--status", "ok", "--result", "corrected")
rows = folded(d, "zz_done_retr")
out = r.stdout + r.stderr
ok = (r.returncode != 0 and "retracted" in out and "Open rows: none" not in out
      and len(rows) == 1 and rows[0]["status"] == "retracted")
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} `done --started` on a retracted row names the STATE, not "
      f"just the absence"
      + ("" if ok else f" -- rc={r.returncode} {out.strip()[:150]}"))
shutil.rmtree(d)

# 4. NO ROW AT ALL -> must still close. THE NEGATIVE CONTROL. pick_open_row returns None here too,
#    so a refusal keyed on `base is None` alone would break the path that closes a run whose start
#    event was lost -- and run_sft.sh:97 / run_pretrain.sh:45 call `done` unconditionally at exit.
d, _st = world()
r = run(d, "--name", "zz_never_started", "--status", "ok", "--result", "orphan close")
rows = folded(d, "zz_never_started")
ok = r.returncode == 0 and len(rows) == 1 and rows[0]["result"] == "orphan close"
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} a name with NO row still closes -- the refusal is keyed on a "
      f"closed row existing, not on `base is None`"
      + ("" if ok else f" -- rc={r.returncode} {(r.stdout + r.stderr).strip()[:120]} rows={rows}"))
shutil.rmtree(d)

# 5. RUNNING, bare `done` -> the normal path. The other direction of narrowness: a refusal reading
#    "a row for this name exists" would break every close there is.
d, st = world(retract=False, close=False)
r = run(d, "--name", "zz_done_retr", "--status", "ok", "--result", "loss 2.34")
rows = folded(d, "zz_done_retr")
ok = r.returncode == 0 and len(rows) == 1 and rows[0]["result"] == "loss 2.34" \
    and rows[0]["status"] == "ok" and rows[0].get("cmd")
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} an OPEN row still closes normally, keeping its cmd"
      + ("" if ok else f" -- rc={r.returncode} {(r.stdout + r.stderr).strip()[:120]} "
                       f"rows={[(x['status'], x.get('result'), x.get('cmd')) for x in rows]}"))
shutil.rmtree(d)

# 6. RUNNING, --started given -> also closes. pick_open_row's new branch must not intercept a
#    legitimate --started close: the closed-row lookup is keyed on the SAME started, so an open
#    row at that started has to win.
d, st = world(retract=False, close=False)
r = run(d, "--name", "zz_done_retr", "--started", st, "--status", "ok", "--result", "loss 2.34")
rows = folded(d, "zz_done_retr")
ok = r.returncode == 0 and len(rows) == 1 and rows[0]["result"] == "loss 2.34"
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} `done --started` on an OPEN row still closes it"
      + ("" if ok else f" -- rc={r.returncode} {(r.stdout + r.stderr).strip()[:120]} "
                       f"rows={[(x['status'], x.get('result')) for x in rows]}"))
shutil.rmtree(d)

print(f"exp done-closed selftest: {CASES - bad}/{CASES} pass")
sys.exit(1 if bad else 0)
