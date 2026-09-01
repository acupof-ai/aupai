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
    # `python` must dispatch on the script it is handed: score_matrix vs harness.
    p = os.path.join(bin_, "python")
    with open(p, "w") as f:
        f.write("#!/bin/bash\n"
                "case \"$*\" in\n"
                "  *score_matrix*) exit ${FAKE_SCORE_RC:-0};;\n"
                "  *free-card*) echo \"${FAKE_CARD-7}\"; exit 0;;\n"
                "  *) exit 0;;\n"
                "esac\n")
    os.chmod(p, 0o755)
    open(os.path.join(d, "ckpt_probe.pt"), "w").write("x")
    return d, bin_


def run(script_src, train_rc=0, score_rc=0, card="7"):
    d, bin_ = _world(script_src)
    try:
        env = dict(os.environ)
        env["PATH"] = bin_ + os.pathsep + env["PATH"]
        env["FAKE_TRAIN_RC"] = str(train_rc)
        env["FAKE_SCORE_RC"] = str(score_rc)
        env["FAKE_CARD"] = card
        r = subprocess.run(["bash", os.path.join(d, "run_ddp.sh"), "--name", "probe"],
                           capture_output=True, text=True, env=env, cwd=d, timeout=120)
        return r.returncode, (r.stdout + r.stderr)
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
        rc, out = run(src, trc, src_rc, card)
        ok = (rc == 0) if want_zero else (rc != 0)
        print(f"  {'ok  ' if ok else 'FAIL'} {label:28} exit={rc}")
        if not ok:
            bad.append(f"{label}: exit={rc}, wanted {'0' if want_zero else 'nonzero'}\n{out[-300:]}")
    if bad:
        print("\n".join(bad))
        return 1
    # Distinct codes: "training failed" and "trained but unscored" need different
    # responses, so they must not be the same number.
    rc_train, _ = run(src, 2, 0, "7")
    rc_score, _ = run(src, 0, 1, "7")
    if rc_train == rc_score:
        print(f"  FAIL training-failed and scoring-failed share exit {rc_train}; a caller "
              f"cannot tell 'retrain' from 'rescore'")
        return 1
    print(f"  ok   distinct codes: training={rc_train}, scoring={rc_score}")
    print("score exit OK: a scoring failure reaches the caller")
    return 0


if __name__ == "__main__":
    sys.exit(main())
