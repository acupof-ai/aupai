#!/usr/bin/env python3
"""exp.py un-retraction: an explicit `done --started` revives a retracted row; nothing else does.

The contract (fold()'s own comment promised it but the code never did): after a row is
`retracted`, a later EXPLICIT close naming the exact row restores it to `ok`, the retraction
stays in the raw log as history. A plain ok/fail with no `unretracts` marker must STILL be
dropped -- under a union merge it can be the same old terminal re-ordered after the retraction,
and accepting it would un-retract silently.

Worlds mutate the REAL ledger (copy, append a zz_unretr row), never a hand-written schema.

    python3 scripts/test_exp_unretract.py
"""
import json, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "scripts", "exp.py")
REAL = os.path.join(ROOT, "runs", "experiments.jsonl")
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def world():
    """Real ledger + a zz_unretr row closed ok then retracted. Returns (dir, started)."""
    d = tempfile.mkdtemp(prefix="expunretr_")
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    shutil.copy(REAL, os.path.join(d, "runs", "experiments.jsonl"))
    rows = [json.loads(x) for x in open(REAL, encoding="utf-8") if x.strip()]
    tmpl = rows[-1]
    st = "2026-09-25 10:00"
    evs = [
        dict(tmpl, name="zz_unretr", status="running", started=st, result="", ended="",
             retracted_reason=None, retracted_result=None, unretracts=None),
        dict(tmpl, name="zz_unretr", status="ok", started=st, result="bad",
             ended="2026-09-25 11:00", retracted_reason=None, retracted_result=None,
             unretracts=None),
        dict(tmpl, name="zz_unretr", status="retracted", started=st, result="bad",
             retracted_reason="guard mis-closed", retracted_result="bad",
             retracted_at="2026-09-25 11:30", unretracts=None),
    ]
    with open(os.path.join(d, "runs", "experiments.jsonl"), "a", encoding="utf-8") as f:
        for e in evs:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return d, st


def run(d, *args):
    return subprocess.run([sys.executable, EXP, "--root", d, "done", *args],
                          capture_output=True, text=True)


def folded(d, name="zz_unretr"):
    import importlib, exp
    importlib.reload(exp)
    exp.set_root(d)
    return [r for r in exp.rows() if r["name"] == name]


def raw(d, name="zz_unretr"):
    import importlib, exp
    importlib.reload(exp)
    exp.set_root(d)
    return [r for r in exp.rows(raw=True) if r["name"] == name]


bad = 0


def check(cond, msg):
    global bad
    if not cond:
        bad += 1
        print("FAIL:", msg)


# 1. explicit done --started revives the retracted row to ok with the new result.
d, st = world()
r = run(d, "--name", "zz_unretr", "--started", st, "--status", "ok", "--result", "28/164")
rs = folded(d)
check(r.returncode == 0, f"un-retract done rc={r.returncode} {r.stderr[:200]}")
check(len(rs) == 1 and rs[0]["status"] == "ok", f"folded status {[x.get('status') for x in rs]}")
check(rs[0]["result"] == "28/164", f"folded result {rs[0].get('result')!r}")

# 2. the revived folded row carries the unretracts marker and no retraction fields.
check(bool(rs[0].get("unretracts")), "revived row lacks unretracts marker")
check(rs[0]["unretracts"].get("result") == "bad",
      f"unretracts does not name withdrawn result: {rs[0].get('unretracts')}")
for k in ("retracted_reason", "retracted_result", "retracted_at", "superseded_by"):
    check(k not in rs[0], f"revived folded row still carries {k}")

# 3. the raw retraction event is retained as history.
rr = raw(d)
check(any(x.get("status") == "retracted" for x in rr), "raw retraction event was removed")
check(len(rr) == 4, f"expected 4 raw events (run, ok, retract, unretract), got {len(rr)}")

# 4. a BARE done (no --started) is refused, not silently revived.
d2, st2 = world()
rb = run(d2, "--name", "zz_unretr", "--status", "ok", "--result", "x")
check(rb.returncode != 0, f"bare done on retracted row rc={rb.returncode}; must refuse")
check(folded(d2)[0]["status"] == "retracted", "bare done changed the folded row")

# 5. fold() directly: a markerless later ok is STILL dropped (union-merge reorder case).
d3, st3 = world()
with open(os.path.join(d3, "runs", "experiments.jsonl"), "a", encoding="utf-8") as f:
    f.write(json.dumps({"name": "zz_unretr", "started": st3, "status": "ok",
                        "result": "old-reordered", "commit": "x"}) + "\n")
check(folded(d3)[0]["status"] == "retracted",
      "markerless ok after retract un-retracted the row (union-merge hazard)")

# 6. render shows the revived row as ok / 28/164.
d4, st4 = world()
run(d4, "--name", "zz_unretr", "--started", st4, "--status", "ok", "--result", "28/164")
import importlib, exp
importlib.reload(exp)
exp.set_root(d4)
md_path = exp.render()
md = open(md_path, encoding="utf-8").read()
check("| ok | 28/164 |" in md, "rendered table lacks the revived ok / 28/164 row")
for tmp in (d, d2, d3, d4):
    shutil.rmtree(tmp, ignore_errors=True)

print(f"{'ALL %d PASS' % 6 if bad == 0 else '%d FAIL' % bad}")
sys.exit(1 if bad else 0)
