#!/usr/bin/env python3
"""exp.py done must close the row you named, and REFUSE when the name is ambiguous.

The defect this guards: p200m_4b_0902 had three open rows in eight minutes -- two OOMed
launches and the live run -- and `done --name p200m_4b_0902` closes the NEWEST open row, which
was the live one. It would have written the OOM as the running job's result. The newest-row
default is only safe when there is exactly one candidate.

Worlds are built by MUTATING the real ledger (copy, then add rows), never hand-written: a
hand-written world shares the test author's assumptions about the schema.

    python3 scripts/test_exp_done_started.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "scripts", "exp.py")
REAL = os.path.join(ROOT, "runs", "experiments.jsonl")


def _world(n_open, n_closed=0):
    """A repo-shaped tree whose ledger is the REAL one plus open (and optionally closed) rows.

    n_closed rows get BOTH a running event and a later terminal event, which is what the real
    ledger holds -- `done` appends rather than rewriting. Those rows are NOT open, and a
    reader that filters raw events on status=="running" counts them anyway."""
    d = tempfile.mkdtemp(prefix="expdone_")
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    shutil.copy(REAL, os.path.join(d, "runs", "experiments.jsonl"))
    rows = [json.loads(x) for x in open(REAL, encoding="utf-8") if x.strip()]
    template = next(r for r in rows if r.get("status") == "running") if any(
        r.get("status") == "running" for r in rows) else dict(rows[-1], status="running")
    with open(os.path.join(d, "runs", "experiments.jsonl"), "a", encoding="utf-8") as f:
        for i in range(n_closed):
            st = f"2026-09-02 0{i}:00"
            f.write(json.dumps(dict(template, name="zz_probe", status="running",
                                    started=st, result="", ended=""),
                               ensure_ascii=False) + "\n")
            f.write(json.dumps(dict(template, name="zz_probe", status="fail",
                                    started=st, result="oom", ended="2026-09-02 09:00"),
                               ensure_ascii=False) + "\n")
        for i in range(n_open):
            f.write(json.dumps(dict(template, name="zz_probe", status="running",
                                    started=f"2026-09-02 1{i}:00", result="", ended=""),
                               ensure_ascii=False) + "\n")
    return d


def _done(d, *args):
    r = subprocess.run([sys.executable, EXP, "--root", d, "done", "--name", "zz_probe", *args],
                       capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)


def _closed(d):
    """(started, result) of every close event for zz_probe, in file order."""
    p = os.path.join(d, "runs", "experiments.jsonl")
    return [(r.get("started"), r.get("result"))
            for r in (json.loads(x) for x in open(p, encoding="utf-8") if x.strip())
            if r.get("name") == "zz_probe" and r.get("status") != "running"]


def main():
    bad = 0

    # 1. One open row: the default still works, no --started needed.
    d = _world(1)
    rc, out = _done(d, "--result", "single")
    got = _closed(d)
    ok = rc == 0 and got == [("2026-09-02 10:00", "single")]
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} one open row closes without --started (rc={rc}, {got})")

    # 2. Three open rows, no --started: MUST refuse, and MUST NOT write anything. This is the
    #    p200m case -- the newest row was the live run.
    d = _world(3)
    rc, out = _done(d, "--result", "should not land")
    got = _closed(d)
    ok = rc != 0 and not got and "--started" in out
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} three open rows refuse and write nothing "
          f"(rc={rc}, closes={got})")
    if not ok:
        print(f"       output was: {out.strip()[:200]}")

    # 3. --started picks the OLDEST, not the newest: the live run is the newest, so an
    #    older-row close must be reachable at all -- the old code could not reach it.
    d = _world(3)
    rc, out = _done(d, "--started", "2026-09-02 10:00", "--result", "oldest")
    got = _closed(d)
    ok = rc == 0 and got == [("2026-09-02 10:00", "oldest")]
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} --started reaches the OLDEST open row (rc={rc}, {got})")

    # 4. A --started that matches no open row refuses and names what IS open, rather than
    #    silently falling back to the newest.
    d = _world(2)
    rc, out = _done(d, "--started", "1999-01-01 00:00", "--result", "nope")
    got = _closed(d)
    ok = rc != 0 and not got and "10:00" in out
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} an unknown --started refuses and lists the open rows "
          f"(rc={rc}, closes={got})")

    # 5. A CLOSED row is not an open row. This is the case the four above could not see: they
    #    only ever built worlds whose extra rows were open, so a reader that counts raw
    #    running EVENTS rather than folding by (name, started) passed all four. MEASURED on
    #    the real ledger (1e, 2026-09-03): p200m_4b_0902 read 5 open rows while fold() gives
    #    exactly one running. With four closed rows and one open, `done` must take the default
    #    path -- no --started -- and close the open one.
    d = _world(1, n_closed=4)
    rc, out = _done(d, "--result", "one live among four closed")
    got = [g for g in _closed(d) if g[1] != "oom"]
    ok = rc == 0 and got == [("2026-09-02 10:00", "one live among four closed")]
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} four closed rows are not open; the one live row closes "
          f"without --started (rc={rc}, {got})")
    if not ok:
        print(f"       output was: {out.strip()[:200]}")

    # 6. And the refusal still fires on two GENUINELY open rows even with closed ones present,
    #    so 5 did not buy its fix by relaxing the guard.
    d = _world(2, n_closed=3)
    rc, out = _done(d, "--result", "should not land")
    got = [g for g in _closed(d) if g[1] != "oom"]
    ok = rc != 0 and not got and "--started" in out
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} two open + three closed still refuses "
          f"(rc={rc}, closes={got})")

    print(f"test_exp_done_started: {6 - bad}/6 pass")
    return 1 if bad else 0


if __name__ == "__main__":
    # --selftest is accepted and ignored: the hook invokes registered files with that flag, and
    # this test's whole body IS the selftest. Accepting it rather than erroring keeps the file
    # in SELFTEST_FILES, where check_selftests_are_gated can see it.
    sys.exit(main())
