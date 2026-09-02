#!/usr/bin/env python3
"""Does `harness launch` actually claim and release its cards? (de-30)

    python3 scripts/test_launch_claims.py --selftest

# restartable: spawns short-lived local processes and a temp claim dir. Writes nothing outside
# /tmp. Costs ~25s.

WHY THIS FILE. card_claim.py existed for days and `harness launch` -- the documented way to start
any GPU job -- never called it, so `card_claim.py status` on the pod reported all eight cards
ORPHAN: ownership was inferred from nvidia-smi rather than declared, and on 2026-09-02 two probes
shared cards twice and OOM'd each other. Wiring it is only half the job; the half that fails
silently is the RELEASE, because a claim nobody releases makes a card read held forever, which is
indistinguishable from the state the wiring was meant to fix.

WHICH PID IS CLAIMED, and why it is not the obvious one. cmd_launch's proc.pid is
    bash -c 'set -o pipefail; "$@"; rc=$?; printf %s "$rc" > "$0"; exit "$rc"'
a shell by construction, and card_claim REFUSES a shell (de-34: a claim on one either exits and
leaves the card ORPHAN, or lingers and makes a finished job look live -- both happened on
2026-09-03). So the claim names the job descendant. Measured on harness's own wrapper shape: the
descendant exists by the time Popen returns, both for a python payload and for a shell script
that execs one, as run_ddp.sh does.

The cases run locally with no cards: the wrapper is a shell, a descendant exists at Popen, the
claim records the job pid and a non-empty cmdline, status does not call it ORPHAN-SHELL, release
removes it, cmd_launch's body really calls the helpers, and the monitor carries a release on both
of its exit paths.

TWO DEFECTS IN THIS FILE, both found by trying to make it fail rather than by reading it:

  The wiring assertion grepped the source for `_acquire_cards(` and PASSED with the call site
  stubbed out, because that name also appears in its own `def`. Measured: 9/9 green on a tree
  where the launch claimed nothing. It is now an AST walk over cmd_launch's own body, which goes
  red naming the missing call. Same shape as gate_failure_shapes §61 -- a criterion that
  recomputes what it judges.

  The helper case wrote a claim into the repo's REAL runs/claims/. card_claim.py reads
  AUPAI_CLAIM_DIR at import and the helper shells out to it, so patching this process's
  CC.CLAIM_DIR never reached the subprocess. It now sets AUPAI_CLAIM_DIR and asserts nothing
  landed in the real directory. The pre-commit hook refuses this class for the git config
  (_shared_repo_state); claims had no such guard.

RELEASE CANNOT LIVE IN cmd_launch. That returns while the job is still running, so releasing
there frees a card under a live job. The monitor is the only thing that outlives the job and sees
it end, so the release sits beside the row it writes on death -- and on the settled() path too,
because a run closed by hand (`exp.py done`, `harness kill`) is the majority of runs and would
otherwise leave its claim forever. Release is idempotent, so both firing is harmless.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _case(results, good, text):
    results.append(good)
    print(f"  {'ok  ' if good else 'BUG '} {text}")


def selftest():
    import card_claim as CC

    results = []
    d = tempfile.mkdtemp(prefix="de30_launch_")
    CC.CLAIM_DIR = os.path.join(d, "claims")
    os.makedirs(CC.CLAIM_DIR, exist_ok=True)

    # harness's exact wrapper, with a payload that outlives the check.
    rc_path = os.path.join(d, "x.rc")
    payload = ["python3", "-c", "import time; time.sleep(12)"]
    wrapped = ["bash", "-c", 'set -o pipefail; "$@"; rc=$?; printf %s "$rc" > "$0"; exit "$rc"',
               rc_path, *payload]
    proc = subprocess.Popen(wrapped, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    time.sleep(1.2)

    # 1. The wrapper IS a shell, so claiming it would be refused. This is the premise; if it ever
    #    stops holding, the reason for claiming a descendant is gone and the rest is cargo cult.
    _case(results, CC._argv0_is_shell(CC._cmdline(proc.pid)),
          f"cmd_launch's wrapper pid {proc.pid} is a shell (why the claim cannot name it)")

    # 2. A job descendant exists by now -- what harness claims instead.
    jobs = CC._job_descendants(proc.pid)
    _case(results, bool(jobs), f"a job descendant exists right after Popen ({len(jobs)} found)")

    # 3. Claiming the descendant succeeds, and the claim records that pid, not the wrapper's.
    if jobs:
        ok, msg = CC.acquire("de30_e2e", ["7"], pid=jobs[0][0], note="test_launch_claims")
        _case(results, ok, f"acquire on the job descendant succeeds: {msg[:50]}")
        rec = CC._read(os.path.join(CC.CLAIM_DIR, "de30_e2e.json")) or {}
        _case(results, rec.get("pid") == jobs[0][0] and rec.get("pid") != proc.pid,
              f"the claim records the job pid {rec.get('pid')}, not the wrapper {proc.pid}")
        _case(results, bool(rec.get("cmdline")),
              f"and a non-empty cmdline: {str(rec.get('cmdline'))[:44]!r}")

        # 4. status must NOT call this an ORPHAN-SHELL: the claim names the job.
        _, dup, lines = CC.status()
        _case(results, not any("ORPHAN-SHELL" in x for x in lines),
              "status does not report ORPHAN-SHELL for a correctly-bound claim")

        # 5. RELEASE is the half that fails silently. After the job ends, the claim must be gone
        #    -- and nothing in cmd_launch can do it, because cmd_launch returns while the job
        #    runs. The monitor is the only thing that outlives the job, which is why the release
        #    lives beside the row it writes on death.
        ok, msg = CC.release("de30_e2e")
        _case(results, ok and not os.path.exists(os.path.join(CC.CLAIM_DIR, "de30_e2e.json")),
              "release removes the claim file")

    # 6. The wiring exists at all. NOT by grepping for `_acquire_cards(` -- that name also
    #    appears in its own `def`, so the assertion passed with the call site deleted. Verified:
    #    stubbing the acquire to (False, 'UNWIRED') left this 9/9. Same shape as
    #    gate_failure_shapes §61, a criterion that recomputes what it judges.
    #
    #    Instead, CALL cmd_launch's helpers and check the effect: _acquire_cards must actually
    #    write a claim file for a job pid, and the call site must be reachable from cmd_launch --
    #    checked by AST, on the function's own body rather than the file's text.
    import ast

    src = open(os.path.join(ROOT, "scripts", "harness.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    launch = next((n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "cmd_launch"), None)
    called = set()
    if launch:
        for node in ast.walk(launch):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)
    _case(results, "_acquire_cards" in called,
          f"cmd_launch's body CALLS _acquire_cards (AST, not a text match): {sorted(called & {'_acquire_cards', '_job_pids_for', '_release_cards'})}")
    _case(results, "_job_pids_for" in called,
          "and _job_pids_for, so the pid it claims is the job and not the wrapper shell")

    # And the helper does what its name says, exercised for real -- in a TEMP claim dir.
    # card_claim.py reads AUPAI_CLAIM_DIR at import, and the helper shells out to it, so
    # patching this process's CC.CLAIM_DIR does not reach the subprocess: the first version of
    # this case wrote a claim into the repo's real runs/claims/. A selftest that touches shared
    # state passes while breaking the thing it shares -- the hook refuses exactly this
    # (_shared_repo_state), for the git config rather than for claims.
    import harness

    if jobs:
        os.environ["AUPAI_CLAIM_DIR"] = CC.CLAIM_DIR
        try:
            ok, msg = harness._acquire_cards("de30_helper", "7", jobs[0][0], "test")
            landed = os.path.exists(os.path.join(CC.CLAIM_DIR, "de30_helper.json"))
            _case(results, ok and landed,
                  f"_acquire_cards writes a real claim via card_claim ({msg[:40]}, landed={landed})")
            harness._release_cards("de30_helper")
            _case(results, not os.path.exists(os.path.join(CC.CLAIM_DIR, "de30_helper.json")),
                  "_release_cards removes it")
        finally:
            os.environ.pop("AUPAI_CLAIM_DIR", None)
        _case(results, not os.path.exists(os.path.join(ROOT, "runs", "claims", "de30_helper.json")),
              "and nothing was written to the repo's real runs/claims/")

    mon = re.search(r"monitor_code = f'''(.*?)'''", src, re.S)
    body = mon.group(1) if mon else ""
    calls = [ln for ln in body.splitlines()
             if "card_claim.py" in ln and not ln.strip().startswith("#")]
    _case(results, len(calls) >= 2,
          f"the monitor releases on both exit paths ({len(calls)} release calls in its body)")

    for p, _a in CC._descendants(proc.pid):
        try:
            os.kill(p, 9)
        except OSError:
            pass
    proc.kill()
    proc.wait()
    shutil.rmtree(d, ignore_errors=True)

    bad = results.count(False)
    print(f"\nde-30 launch claims: {len(results) - bad}/{len(results)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(selftest())
