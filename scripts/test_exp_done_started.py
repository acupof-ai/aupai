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


def _note(d, *args):
    r = subprocess.run([sys.executable, EXP, "--root", d, "note", "--name", "zz_probe", *args],
                       capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)


def _events(d):
    """Every zz_probe event, in file order, as (started, status, notes)."""
    p = os.path.join(d, "runs", "experiments.jsonl")
    return [(r.get("started"), r.get("status"), r.get("notes"))
            for r in (json.loads(x) for x in open(p, encoding="utf-8") if x.strip())
            if r.get("name") == "zz_probe"]


def _folded(d):
    """zz_probe rows as the readers see them: exp.fold, which harness reads through."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from exp import fold
    p = os.path.join(d, "runs", "experiments.jsonl")
    evs = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    return [r for r in fold(evs) if r.get("name") == "zz_probe"]


def note_cases():
    """`note` annotates a RUNNING row without closing it.

    Why it exists: run_ddp.sh's chained score_matrix runs after torchrun exits and wrote
    nothing, so nothing distinguished "being scored now" from "never scored" -- b0
    double-scored the params leg on that gap. Two events are the point; one line at the end
    cannot say a score is in flight.
    """
    bad = 0

    # 1. The row stays RUNNING and the text lands. A note that closed the row would make the
    #    monitor's settled() break early and stop watching a live job.
    d = _world(1)
    rc, out = _note(d, "--text", "score_matrix STARTED on card 5")
    ev = _events(d)
    folded = _folded(d)
    ok = (rc == 0 and ev[-1][1] == "running" and "STARTED on card 5" in (ev[-1][2] or "")
          and len(folded) == 1 and folded[0]["status"] == "running")
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} note keeps the row running and lands the text "
          f"(rc={rc}, status={ev[-1][1]}, rows={len(folded)})")

    # 2. TWO notes then done: one folded row, both notes present, terminal status. This is the
    #    real sequence -- STARTED, FINISHED, then the monitor's close -- and the thing that must
    #    not happen is a duplicate row: score_matrix_present joins ok runs to score records by
    #    (name, started), so a second row for one run reads as a second unscored run.
    d = _world(1)
    _note(d, "--text", "score_matrix STARTED on card 5")
    _note(d, "--text", "score_matrix FINISHED on card 5 rc=0")
    rc, _ = _done(d, "--result", "val 1.8")
    folded = _folded(d)
    notes = folded[0].get("notes", "") if len(folded) == 1 else ""
    ok = (rc == 0 and len(folded) == 1 and folded[0]["status"] == "ok"
          and "STARTED" in notes and "FINISHED" in notes)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} STARTED+FINISHED+done fold to ONE ok row keeping both "
          f"notes (rows={len(folded)}, status={folded[0]['status'] if folded else '?'})")

    # 3. Ambiguity refuses and writes NOTHING -- the same p200m_4b_0902 guard as `done`, reached
    #    through the shared pick_open_row rather than a second copy of it.
    d = _world(3)
    before = len(_events(d))
    rc, out = _note(d, "--text", "should not land")
    ok = rc != 0 and len(_events(d)) == before and "--started" in out
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} three open rows refuse and write nothing "
          f"(rc={rc}, events {before}->{len(_events(d))})")

    # 4. --started reaches the OLDEST open row, not the newest.
    d = _world(3)
    rc, out = _note(d, "--started", "2026-09-02 10:00", "--text", "on the oldest")
    ev = _events(d)
    ok = rc == 0 and ev[-1][0] == "2026-09-02 10:00" and "on the oldest" in (ev[-1][2] or "")
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} --started annotates the OLDEST open row "
          f"(rc={rc}, started={ev[-1][0]})")

    # 5. NO OPEN ROW. Default refuses; --quiet-if-absent exits 0 and still writes nothing.
    #    A run launched without `harness launch` has no row, and bookkeeping with nowhere to
    #    write must not turn a score that SUCCEEDED into a nonzero exit -- run_ddp.sh's exit
    #    code is the signal fb's ruling made load-bearing.
    d = _world(0, n_closed=2)
    before = len(_events(d))
    rc_strict, _ = _note(d, "--text", "nowhere to write")
    mid = len(_events(d))
    rc_quiet, _ = _note(d, "--quiet-if-absent", "--text", "nowhere to write")
    ok = rc_strict != 0 and rc_quiet == 0 and mid == before and len(_events(d)) == before
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} no open row: strict refuses, --quiet-if-absent exits 0, "
          f"neither writes (strict={rc_strict}, quiet={rc_quiet}, events {before}->{len(_events(d))})")

    # 6. A note does NOT resurrect a closed row. fold() makes a close terminal regardless of
    #    position, and a note carrying status="running" for a (name, started) that is already
    #    closed would reopen it -- which is exactly what the fold refuses for start events. The
    #    only path to that row is --started, so that is what is tried.
    d = _world(0, n_closed=1)
    closed_at = [e[0] for e in _events(d) if e[1] != "running"][-1]
    rc, out = _note(d, "--started", closed_at, "--text", "reopen me")
    folded = _folded(d)
    ok = rc != 0 and all(r["status"] != "running" for r in folded)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} a closed row cannot be annotated back to running "
          f"(rc={rc}, statuses={[r['status'] for r in folded]})")

    return bad


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
    bad_note = note_cases()
    print(f"test_exp_note: {6 - bad_note}/6 pass")
    return 1 if (bad or bad_note) else 0


if __name__ == "__main__":
    # --selftest is accepted and ignored: the hook invokes registered files with that flag, and
    # this test's whole body IS the selftest. Accepting it rather than erroring keeps the file
    # in SELFTEST_FILES, where check_selftests_are_gated can see it.
    sys.exit(main())
