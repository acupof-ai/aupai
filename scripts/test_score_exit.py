#!/usr/bin/env python3
"""run_ddp.sh must not swallow a scoring failure.

Two layers cancelled each other: score_matrix.py exits nonzero on any per-checkpoint
failure (de, 2026-09-01), and run_ddp.sh's `|| echo WARN` discarded that, then
`exit $rc` returned the TRAINING code. A caller checking the exit status saw success.
Over ~28 milestones on a 66-hour run that is 28 silent misses (fb, 2026-09-02).

Runs the real script with torchrun and score_matrix.py replaced by stubs on PATH, so
what is tested is the script's own control flow, not a copy of it.

    python3 scripts/test_score_exit.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "run_ddp.sh")

STUB_TORCHRUN = "#!/bin/bash\nexit ${FAKE_TRAIN_RC:-0}\n"
STUB_SCORE = "#!/bin/bash\nexit ${FAKE_SCORE_RC:-0}\n"
STUB_HARNESS = "#!/bin/bash\necho \"${FAKE_CARD-7}\"\n"


def _world(script_src):
    """A directory holding the script under test plus stubs it will find first."""
    d = tempfile.mkdtemp()
    bin_ = os.path.join(d, "bin")
    os.makedirs(bin_)
    os.makedirs(os.path.join(d, "scripts"))
    os.makedirs(os.path.join(d, "eval"))
    os.makedirs(os.path.join(d, "runs"))
    os.makedirs(os.path.join(d, ".git"))  # a checkout: skips the pod stamp gate
    with open(os.path.join(d, "run_ddp.sh"), "w") as f:
        f.write(script_src)
    os.chmod(os.path.join(d, "run_ddp.sh"), 0o755)
    for name, body in (("torchrun", STUB_TORCHRUN),):
        p = os.path.join(bin_, name)
        with open(p, "w") as f:
            f.write(body)
        os.chmod(p, 0o755)
    # `python` must dispatch on the script it is handed: score_matrix vs harness vs exp.
    # exp.py note APPENDS to a trace file rather than exiting 0 silently: the note calls are
    # the de-47 annotation, and a stub that swallowed them would let a mistyped flag or a
    # dropped call pass every case below. What a test cannot see, it does not check.
    #
    # `exp.py done` traces to the SAME file for the same reason (de-47 c3). It used to fall
    # through to the `*)` catch-all, which exits 0 and records nothing -- so the chained close
    # could be dropped entirely and every case here would still pass. The `done` arm must come
    # BEFORE `*)`; order is the whole mechanism in a case statement.
    p = os.path.join(bin_, "python")
    with open(p, "w") as f:
        f.write("#!/bin/bash\n"
                "case \"$*\" in\n"
                "  *exp.py*note*) echo \"$*\" >> \"$NOTE_TRACE\"; exit ${FAKE_NOTE_RC:-0};;\n"
                "  *exp.py*done*) echo \"DONE $*\" >> \"$NOTE_TRACE\"; exit ${FAKE_DONE_RC:-0};;\n"
                "  *score_matrix*) exit ${FAKE_SCORE_RC:-0};;\n"
                "  *free-card*) echo \"${FAKE_CARD-7}\"; exit 0;;\n"
                "  *) exit 0;;\n"
                "esac\n")
    os.chmod(p, 0o755)
    open(os.path.join(d, "ckpt_probe.pt"), "w").write("x")
    return d, bin_


def run(script_src, train_rc=0, score_rc=0, card="7", note_rc=0, done_rc=0, log=None):
    d, bin_ = _world(script_src)
    trace = os.path.join(d, "note_trace")
    if log is not None:
        with open(os.path.join(d, "runs", "probe.log"), "w") as f:
            f.write(log)
    try:
        env = dict(os.environ)
        env["PATH"] = bin_ + os.pathsep + env["PATH"]
        env["FAKE_TRAIN_RC"] = str(train_rc)
        env["FAKE_SCORE_RC"] = str(score_rc)
        env["FAKE_CARD"] = card
        env["FAKE_NOTE_RC"] = str(note_rc)
        env["FAKE_DONE_RC"] = str(done_rc)
        env["NOTE_TRACE"] = trace
        r = subprocess.run(["bash", os.path.join(d, "run_ddp.sh"), "--name", "probe"],
                           capture_output=True, text=True, env=env, cwd=d, timeout=120)
        notes = open(trace).read() if os.path.exists(trace) else ""
        return r.returncode, (r.stdout + r.stderr), notes
    finally:
        shutil.rmtree(d, ignore_errors=True)


CASES = [
    # (label, train_rc, score_rc, card, expect_zero_exit)
    ("trained and scored", 0, 0, "7", True),
    ("trained, SCORING FAILED", 0, 1, "7", False),
    ("trained, no free card", 0, 0, "", False),
    ("training failed", 2, 0, "7", False),
]


def main():
    src = open(SCRIPT, encoding="utf-8").read()
    bad = []
    for label, trc, src_rc, card, want_zero in CASES:
        rc, out, _notes = run(src, trc, src_rc, card)
        ok = (rc == 0) if want_zero else (rc != 0)
        print(f"  {'ok  ' if ok else 'FAIL'} {label:28} exit={rc}")
        if not ok:
            bad.append(f"{label}: exit={rc}, wanted {'0' if want_zero else 'nonzero'}\n{out[-300:]}")
    if bad:
        print("\n".join(bad))
        return 1
    # Distinct codes: "training failed" and "trained but unscored" need different
    # responses, so they must not be the same number.
    rc_train, _, _ = run(src, 2, 0, "7")
    rc_score, _, _ = run(src, 0, 1, "7")
    if rc_train == rc_score:
        print(f"  FAIL training-failed and scoring-failed share exit {rc_train}; a caller "
              f"cannot tell 'retrain' from 'rescore'")
        return 1
    print(f"  ok   distinct codes: training={rc_train}, scoring={rc_score}")

    # de-47: the chained score writes STARTED and FINISHED into the run's row. b0
    # double-scored the params leg because this chain wrote nothing, so nothing said a score
    # was in flight. TWO events are the requirement -- a single line at the end cannot tell a
    # reader whether a currently-running score is theirs -- and both must name the card.
    _rc, _out, notes = run(src, 0, 0, "5")
    started = [l for l in notes.splitlines() if "STARTED" in l]
    finished = [l for l in notes.splitlines() if "FINISHED" in l]
    if not (len(started) == 1 and len(finished) == 1):
        print(f"  FAIL the chained score must write exactly one STARTED and one FINISHED note; "
              f"got {len(started)}/{len(finished)}\n{notes}")
        return 1
    if "card 5" not in started[0] or "card 5" not in finished[0]:
        print(f"  FAIL both notes must name the lane card the scorer actually got: {notes}")
        return 1
    print("  ok   STARTED and FINISHED notes both name the card")

    # ORDER, not just presence: STARTED must land BEFORE the scorer is waited on. A pair of
    # notes written back-to-back after the score would satisfy the count above and buy nothing
    # -- the whole point is a reader seeing "in flight" while it is in flight. The scorer stub
    # blocks until a file appears, so STARTED can only be in the trace if it was written first.
    src_probe = src
    rc_b, out_b, notes_b = _run_with_blocking_scorer(src_probe)
    if rc_b != 0 or "STARTED" not in notes_b:
        print(f"  FAIL STARTED must be written before the scorer is waited on "
              f"(rc={rc_b}, notes={notes_b!r})\n{out_b[-300:]}")
        return 1
    print("  ok   STARTED is written before the scorer is waited on")

    # A FAILED note must not change the exit code: the score succeeded, and bookkeeping that
    # cannot write is not a training failure. The inverse of fb's rule, and both directions
    # matter -- a scoring failure must be loud, a note failure must not be.
    #
    # WHAT THIS CATCHES, measured by mutating the real script: a note whose `|| true` becomes
    # `|| SCORING_RC=$?` (the copy-paste from the line above it) turns red here. What it does
    # NOT catch is DELETING `|| true`, which stays green -- run_ddp.sh has no `set -e`, so a
    # bare nonzero note is discarded by the shell anyway. The `|| true` is documentation of
    # intent, not the mechanism, and this assertion cannot claim otherwise.
    rc_n, _out_n, _notes_n = run(src, 0, 0, "7", note_rc=3)
    if rc_n != 0:
        print(f"  FAIL a note that exits nonzero must not fail the run (exit={rc_n})")
        return 1
    print("  ok   a failing note does not change the run's exit code")

    # de-47 c3: THE CHAIN CLOSES THE ROW, on both paths. no_stale_running FAILs on a row
    # still `running` after 24h, and this chain is the last thing that runs -- so before
    # this, every pretrain left its own row open and launch_gate could not tell a finished
    # run from a job still on the cards. That is the NO-GO it was reading.
    _rc_ok, _o, notes_ok = run(src, 0, 0, "6", log="step 10 | loss 2.9\nval 2.135\n")
    done_ok = [l for l in notes_ok.splitlines() if l.startswith("DONE ")]
    if len(done_ok) != 1:
        print(f"  FAIL a successful run must close its row exactly once; got {len(done_ok)}\n{notes_ok}")
        return 1
    if "--status ok" not in done_ok[0]:
        print(f"  FAIL a successful run must close status=ok: {done_ok[0]}")
        return 1
    # The val loss from the log, not a hand-written string: an empty result cell is what
    # `exp.py render` shows when the chain closes a row with nothing in it.
    if "2.135" not in done_ok[0]:
        print(f"  FAIL the close must carry the val loss read from the log: {done_ok[0]}")
        return 1
    # `finding` must NOT read as a human interpretation the chain cannot supply.
    if "pending" not in done_ok[0]:
        print(f"  FAIL the chained close must mark its finding pending, not assert one: {done_ok[0]}")
        return 1
    print("  ok   a successful run closes its row: status ok, val loss, finding pending")

    # THE FAILURE PATH IS THE ONE THAT MATTERS. A run that produced no metrics is the row a
    # human is least likely to return to, and `exit 1` would otherwise leave it open forever.
    # Both scoring-failed and no-free-card reach the exit; both must still close.
    for label, s_rc, card in (("scoring failed", 1, "7"), ("no free card", 0, "")):
        _rc_f, _of, notes_f = run(src, 0, s_rc, card, log="val 3.010\n")
        done_f = [l for l in notes_f.splitlines() if l.startswith("DONE ")]
        if len(done_f) != 1:
            print(f"  FAIL {label}: the row must still close ({len(done_f)} closes)\n{notes_f}")
            return 1
        if "--status error" not in done_f[0]:
            print(f"  FAIL {label}: an unscored run must close status=error, not ok: {done_f[0]}")
            return 1
        print(f"  ok   {label}: row closed status=error")

    # A close that cannot write must not fail the run, for the same reason a note cannot:
    # bookkeeping with nowhere to go is not a training failure. Distinct from the note case
    # because `done` runs AFTER the scoring verdict, where a nonzero could reach the exit.
    rc_d, _od, _nd = run(src, 0, 0, "7", done_rc=4)
    if rc_d != 0:
        print(f"  FAIL a failing close must not fail the run (exit={rc_d})")
        return 1
    print("  ok   a failing close does not change the run's exit code")

    # The close must not fire when there was no checkpoint to score: that branch never
    # opened this chain, and closing a row the chain did not enter would close somebody
    # else's. A training failure exits before the `if` block.
    _rc_t, _ot, notes_t = run(src, 2, 0, "7")
    if [l for l in notes_t.splitlines() if l.startswith("DONE ")]:
        print(f"  FAIL a failed training run must not be closed by this chain: {notes_t}")
        return 1
    print("  ok   a failed training run is not closed by the scoring chain")
    print("score exit OK: a scoring failure reaches the caller")
    return 0


def _run_with_blocking_scorer(script_src):
    """Run the script with a score_matrix stub that blocks until the note trace exists.

    The ORDER assertion needs the scorer to still be running when STARTED is written, and a
    stub that exits immediately cannot distinguish before-the-wait from after it. If the
    script wrote STARTED after `wait`, this deadlocks and the timeout below is the failure.
    """
    d, bin_ = _world(script_src)
    trace = os.path.join(d, "note_trace")
    try:
        p = os.path.join(bin_, "python")
        with open(p, "w") as f:
            f.write("#!/bin/bash\n"
                    "case \"$*\" in\n"
                    "  *exp.py*note*) echo \"$*\" >> \"$NOTE_TRACE\"; exit 0;;\n"
                    "  *score_matrix*) while [ ! -f \"$NOTE_TRACE\" ]; do sleep 0.1; done; exit 0;;\n"
                    "  *free-card*) echo 7; exit 0;;\n"
                    "  *) exit 0;;\n"
                    "esac\n")
        os.chmod(p, 0o755)
        env = dict(os.environ)
        env["PATH"] = bin_ + os.pathsep + env["PATH"]
        env["FAKE_TRAIN_RC"] = "0"
        env["NOTE_TRACE"] = trace
        try:
            r = subprocess.run(["bash", os.path.join(d, "run_ddp.sh"), "--name", "probe"],
                               capture_output=True, text=True, env=env, cwd=d, timeout=30)
        except subprocess.TimeoutExpired:
            return 1, "TIMEOUT: STARTED was not written before the scorer was waited on", ""
        notes = open(trace).read() if os.path.exists(trace) else ""
        return r.returncode, (r.stdout + r.stderr), notes
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
