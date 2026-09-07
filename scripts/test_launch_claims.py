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
removes it, cmd_launch really reaches the helpers, and the monitor carries a release on both
of its exit paths.

THREE DEFECTS IN THIS FILE, all found by trying to make it fail rather than by reading it:

  The wiring assertion grepped the source for `_acquire_cards(` and PASSED with the call site
  stubbed out, because that name also appears in its own `def`. Measured: 9/9 green on a tree
  where the launch claimed nothing. It became an AST walk over cmd_launch's own body, which goes
  red naming the missing call. Same shape as gate_failure_shapes §61 -- a criterion that
  recomputes what it judges.

  That AST walk then read one function body, and 16d08b1c reported 4 BUGs on an intact launcher:
  the commit moved cmd_launch's post-row half into _launch_after_row, leaving `return
  _launch_after_row(...)`, so all four helpers sat one call outside the walk. Main was RED and
  blocking every commit that stages scripts/card_claim.py (84 diagnosed it in a clean clone,
  18/22). The walk now follows LOCAL calls to fixpoint, because what the launcher depends on is
  that the call happens when cmd_launch runs -- a property of the call graph, not of one body. A
  behaviour-preserving refactor must not red an assertion about behaviour.

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
        rec = CC._read(os.path.join(CC.CLAIM_DIR, CC.claim_file("de30_e2e", ["7"]))) or {}
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
        _case(results, ok and not os.path.exists(os.path.join(CC.CLAIM_DIR, CC.claim_file("de30_e2e", ["7"]))),
              "release removes the claim file")

    # 6. The wiring exists at all. NOT by grepping for `_acquire_cards(` -- that name also
    #    appears in its own `def`, so the assertion passed with the call site deleted. Verified:
    #    stubbing the acquire to (False, 'UNWIRED') left this 9/9. Same shape as
    #    gate_failure_shapes §61, a criterion that recomputes what it judges.
    #
    #    Instead, CALL cmd_launch's helpers and check the effect: _acquire_cards must actually
    #    write a claim file for a job pid, and the call site must be REACHABLE from cmd_launch --
    #    checked by AST, on the call graph rather than the file's text.
    #
    #    REACHABLE, not "in the body": walking cmd_launch's own body only was the second version
    #    of this assertion, and 16d08b1c broke it without breaking the launcher. That commit moved
    #    the post-row half of cmd_launch into _launch_after_row and left `return
    #    _launch_after_row(...)` behind, so all four helpers moved one call out of the walk and the
    #    test reported 4 BUGs on an intact tree (84 diagnosed it in a clean clone of main; measured
    #    18/22). A refactor that preserves behaviour must not red an assertion about behaviour.
    #    What the launcher depends on is that the call happens when cmd_launch runs, which is a
    #    property of the call graph, so the walk follows local calls to fixpoint.
    #
    #    The closure is 32 of harness.py's 466 module functions, so this is not "anything in the
    #    file": the negative control below deletes the one edge that carries all four helpers and
    #    goes red. Local names only (ast.Name, not Attribute) -- a method on an object is not a
    #    module function and cannot be resolved this way.
    import ast

    src = open(os.path.join(ROOT, "scripts", "harness.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    _fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def _direct(node):
        return {x.func.id for x in ast.walk(node)
                if isinstance(x, ast.Call) and isinstance(x.func, ast.Name)}

    def _reachable_from(fns, root):
        """Every local name called on any path out of `root`, local calls followed to fixpoint."""
        called, reached = set(), {root}
        frontier = [root] if root in fns else []
        while frontier:
            nxt = []
            for name in frontier:
                for c in _direct(fns[name]):
                    called.add(c)
                    if c in fns and c not in reached:
                        reached.add(c)
                        nxt.append(c)
            frontier = nxt
        return called, reached

    called, reached = _reachable_from(_fns, "cmd_launch")
    _case(results, "_acquire_cards" in called,
          f"cmd_launch REACHES _acquire_cards (AST call graph, not a text match): {sorted(called & {'_acquire_cards', '_job_pids_for', '_release_cards'})}")
    _case(results, "_job_pids_for" in called,
          "and _job_pids_for, so the pid it claims is the job and not the wrapper shell")

    # AND IT WAITS FOR A DEVICE (4c item 1, 2026-09-05). "not a shell" is necessary and not
    # sufficient: measured on the pod in this wrapper shape, the first non-shell descendant
    # appears at t=0.11s holding ZERO device fds and the first fd at t=1.33s. b0_mem_m1's claim
    # landed in that 1.22s window, so status read STALE plus "ORPHAN card 1 holds 1179 MiB with
    # no claim" for a live arm. AST again, for the reason in the docstring: a text match for
    # `_device_pid_for` also hits its own def.
    _case(results, "_device_pid_for" in called,
          "cmd_launch waits for a descendant that HOLDS a device before claiming")
    _case(results, "_proc_readable" in called,
          "and asks whether /proc is readable, so macOS falls back instead of blocking 90s")

    # THE CLOSURE IS NOT "ANYTHING IN harness.py". Following local calls to fixpoint makes the
    # walk survive a refactor, and the cost is that it could reach so far that it stops being a
    # statement about cmd_launch. Two controls, because "it passes here" cannot tell those apart:
    #
    #   SCOPE -- the closure is a strict subset. Measured on this tree: 32 of 466 module
    #   functions. Asserted as a fraction rather than 32, because the number moves with any
    #   refactor and the property is that most of the file is out of reach.
    _case(results, len(reached) < len(_fns) // 4,
          f"the call closure is a strict subset of harness.py: {len(reached)} of {len(_fns)} functions")

    #   TEETH -- delete the one edge carrying the four helpers and every one of them must go
    #   missing. All four live in _launch_after_row, so dropping cmd_launch's `return
    #   _launch_after_row(...)` is exactly 16d08b1c's damage without the call left behind. If this
    #   world still resolved them, the assertions above would be about the file and not the path.
    _cut = {k: v for k, v in _fns.items() if k != "cmd_launch"}
    _cut["cmd_launch"] = ast.parse("def cmd_launch(a):\n    return _csv(a)\n").body[0]
    _cut_called, _ = _reachable_from(_cut, "cmd_launch")
    _missing = sorted({"_acquire_cards", "_job_pids_for", "_device_pid_for", "_proc_readable"}
                      - _cut_called)
    _case(results, len(_missing) == 4,
          f"cutting cmd_launch -> _launch_after_row loses all four helpers ({len(_missing)}/4: {_missing})")

    # The predicate's contract, at the level the launcher depends on. A device count and an
    # unreadable pid must not collapse to the same value: None is not 0, and only 0 refuses.
    _case(results, CC.nvidia_fds(999999) is None,
          "nvidia_fds on an unreadable pid is None, which never refuses (macOS has no /proc)")
    _case(results, isinstance(CC.nvidia_fds(os.getpid()), (int, type(None))),
          "and on this process it is an int or None, never a raise")
    # NO CEILING BY DEFAULT. The 90s constant this replaces abandoned a live M1 launch's cards
    # (4c, 2026-09-05): run_ddp.sh's drift gate, mix assertion and 155 GiB cache load all precede
    # the ranks, and that run's own startup gate was 793 s. The poll must end on the JOB, not on a
    # clock -- W2 below asserts the job-ended exit still works, which is what makes an unbounded
    # wait safe rather than a hang.
    _case(results, CC.DEVICE_WAIT_S is None,
          f"the device wait has no default ceiling (got {CC.DEVICE_WAIT_S!r})")

    # THE DEADLINE MUST BE BOUNDED AND MUST END EARLY ON A DEAD WRAPPER. A launch whose job died
    # at once must not block for the full 90s; without this case the poll passes for a version
    # that always burns its deadline.
    _dead = subprocess.Popen(["bash", "-c", "exit 0"])
    _dead.wait()
    _t0 = time.time()
    _got = CC.wait_for_device(_dead.pid, deadline=5.0, interval=0.1)
    _el = time.time() - _t0
    _case(results, _got is None and _el < 2.0,
          f"wait_for_device ends at once when the wrapper is gone ({_el:.2f}s of a 5s deadline)")

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
            landed = os.path.exists(os.path.join(CC.CLAIM_DIR, CC.claim_file("de30_helper", ["7"])))
            _case(results, ok and landed,
                  f"_acquire_cards writes a real claim via card_claim ({msg[:40]}, landed={landed})")
            harness._release_cards("de30_helper")
            _case(results, not os.path.exists(os.path.join(CC.CLAIM_DIR, CC.claim_file("de30_helper", ["7"]))),
                  "_release_cards removes it")
        finally:
            os.environ.pop("AUPAI_CLAIM_DIR", None)
        _case(results, not os.path.exists(os.path.join(ROOT, "runs", "claims", CC.claim_file("de30_helper", ["7"]))),
              "and nothing was written to the repo's real runs/claims/")

    # THE RELEASE, and the scan must be scoped to the function that HOLDS the template. A
    # file-wide `re.search` for `monitor_code = f'''` matched harness.py:13301 instead -- a line
    # inside _selftest_monitor_stop_rules that carries that exact text as a REGEX STRING, and it
    # sits before the real template, so the search captured 5 characters ("(.*?)") and reported
    # "0 release calls in its body" while both releases were present. §187's shape: a scanner that
    # locates its subject by a delimiter finds whatever mentions the delimiter first.
    #
    # AST for the boundary, regex only inside it. The template lives in _arm_monitor, not in
    # cmd_launch -- checked here rather than assumed, since a search that finds nothing must not
    # read as zero releases.
    mon_fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "_arm_monitor"), None)
    seg = ast.get_source_segment(src, mon_fn) if mon_fn else ""
    mon = re.search(r"monitor_code = f'''(.*?)'''", seg, re.S)
    _case(results, mon is not None,
          "the monitor template is found in _arm_monitor (not a stray regex-string match)")
    body = mon.group(1) if mon else ""
    calls = [ln for ln in body.splitlines()
             if "card_claim.py" in ln and not ln.strip().startswith("#")]
    _case(results, len(calls) >= 2,
          f"the monitor releases on both exit paths ({len(calls)} release calls in its body)")

    # THE ROW IDENTITY MUST REACH THE TEMPLATE. settled() is checked against real ledger states by
    # harness._selftest_monitor_suppression; what that world cannot see is the JOIN -- the template
    # is a string, so a refactor that stops interpolating the stamp leaves settled() correct and
    # every monitor back to name-only matching, which is the 2026-09-05 M1 defect verbatim (its
    # relaunch's monitor read the previous run's 19:45 `fail` row, released the claim of a job at
    # step 300 and exited). Assert the interpolation is present, and that _arm_monitor takes the
    # parameter it interpolates.
    _case(results, 'started = "{started}"' in body,
          "the monitor template carries the launching row's started stamp")
    _case(results, mon_fn is not None and any(
        a.arg == "started" for a in (mon_fn.args.args if mon_fn else [])),
          "_arm_monitor takes started, so the stamp comes from the caller rather than a clock read")

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
