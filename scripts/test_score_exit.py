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
    p = os.path.join(bin_, "python")
    with open(p, "w") as f:
        f.write("#!/bin/bash\n"
                "case \"$*\" in\n"
                "  *exp.py*note*) echo \"$*\" >> \"$NOTE_TRACE\"; exit ${FAKE_NOTE_RC:-0};;\n"
                "  *score_matrix*) exit ${FAKE_SCORE_RC:-0};;\n"
                "  *free-card*) echo \"${FAKE_CARD-7}\"; exit 0;;\n"
                "  *) exit 0;;\n"
                "esac\n")
    os.chmod(p, 0o755)
    open(os.path.join(d, "ckpt_probe.pt"), "w").write("x")
    return d, bin_


def run(script_src, train_rc=0, score_rc=0, card="7", note_rc=0):
    d, bin_ = _world(script_src)
    trace = os.path.join(d, "note_trace")
    try:
        env = dict(os.environ)
        env["PATH"] = bin_ + os.pathsep + env["PATH"]
        env["FAKE_TRAIN_RC"] = str(train_rc)
        env["FAKE_SCORE_RC"] = str(score_rc)
        env["FAKE_CARD"] = card
        env["FAKE_NOTE_RC"] = str(note_rc)
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
