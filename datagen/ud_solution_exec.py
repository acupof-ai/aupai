#!/usr/bin/env python3
"""Execute an UltraData-Code L3 solution against its bundled test (3b, 2026-09-10).

One predicate shared by the quality audit and the production L3 keep-filter (fb
ruling 2026-09-10): execute() runs a solution against the dataset's own test in a
fresh sandbox. Ruling 2026-09-11 added nontrivial(), a static AST floor, because
exec-pass alone admits mostly trivial passing exercises (precision 43% on the
3b-21 audit); keep_l3() is the conjunction (exec-pass AND non-trivial), precision
68% at ~unchanged recall. The audit's n=400 yield and the full-set conversion call
these same functions.

Composition: a temp candidate.py of  solution + "\\n\\n" + test + "\\n"  run as a
fresh process. Isolated mode -I (no user site/PYTHONPATH), bytecode off, cwd is
an empty temp dir, environment scrubbed to PATH/LANG so a solution cannot read
repo or cwd state. Bounded by RLIMIT_CPU and RLIMIT_AS and a wall timeout.

    python3 datagen/ud_solution_exec.py --selftest
"""
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile

CPU_SOFT, CPU_HARD = 10, 12           # seconds of CPU time
AS_BYTES = 8 * 1024 ** 3              # 8 GiB: numpy/openblas graph problems need headroom
WALL_TIMEOUT = 15                     # seconds of wall time
ERR_TAIL = 400                       # chars of stderr kept for triage

PASS = "pass"
FAIL = "fail"       # nonzero exit: assertion error, runtime/syntax/name error
TIMEOUT = "timeout"

# Stage-2 non-triviality floor (fb ruling 2026-09-11, calibrated on the 3b-21 n=390
# labelled L3 rows). Exec-pass alone admits mostly trivial passing exercises
# (precision 43%); this AST floor lifts joint precision to 68% at ~unchanged recall.
# A solution is non-trivial if it is a sizeable program OR contains a genuine
# algorithmic core (>=3 branch/loop control points AND a loop).
NODES_MIN = 90
CF_MIN = 3          # if/if-exp/match + for/while control points
LOOPS_MIN = 2


def nontrivial(solution):
    """Stage-2 floor: is the solution above the trivial-exercise bar?

    Static (ast), no execution. True iff the solution parses AND
    (ast node count >= NODES_MIN OR (control-flow points >= CF_MIN with >= LOOPS_MIN
    loops)). The OR branch keeps compact-but-real algorithms that a pure size
    threshold cuts (calibration lost only 3/75 substantive pass-set docs at this
    setting). Unparseable/empty -> False.
    """
    import ast

    try:
        tree = ast.parse(solution or "")
    except SyntaxError:
        return False
    nodes = 0
    cf = 0
    loops = 0
    for n in ast.walk(tree):
        nodes += 1
        if isinstance(n, (ast.For, ast.AsyncFor, ast.While)):
            loops += 1
            cf += 1
        elif isinstance(n, (ast.If, ast.IfExp, ast.Match)):
            cf += 1
    return nodes >= NODES_MIN or (cf >= CF_MIN and loops >= LOOPS_MIN)


def keep_l3(solution, test):
    """The L3 keep rule: solution executes against its test AND clears the
    non-triviality floor. Returns bool; use execute() directly for triage detail."""
    verdict, _ = execute(solution, test)
    return verdict == PASS and nontrivial(solution)


def execute(solution, test, _limits=(CPU_SOFT, CPU_HARD, AS_BYTES), _wall=WALL_TIMEOUT):
    """Run solution+test. Returns (verdict, stderr_tail).

    verdict is PASS only on exit code 0. A missing solution/test is FAIL.
    Each call gets a fresh temp cwd that is removed on exit.
    _limits/_wall are a test seam (selftest tightens them); production uses defaults.
    """
    if not (solution or "").strip() or not (test or "").strip():
        return FAIL, "empty solution or test"
    cpu_s, cpu_h, as_bytes = _limits
    td = tempfile.mkdtemp()
    path = os.path.join(td, "candidate.py")
    with open(path, "w") as f:
        f.write(solution + "\n\n" + test + "\n")

    def _apply():
        # RLIMIT_AS is unsupported on macOS (dev/CI) and enforced on Linux (pod).
        for res, lim in ((resource.RLIMIT_CPU, (cpu_s, cpu_h)),
                         (resource.RLIMIT_AS, (as_bytes, as_bytes))):
            try:
                resource.setrlimit(res, lim)
            except (ValueError, OSError):
                pass

    p = None
    try:
        # start_new_session: candidate leads its own process group so a timed-out
        # candidate's grandchild (the 209-orphan incident, 2026-09-11) can be
        # reaped with one killpg instead of reparenting to init and surviving.
        p = subprocess.Popen(
            [sys.executable, "-I", "-B", path],
            cwd=td, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=_apply, start_new_session=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        )
        try:
            out, err = p.communicate(timeout=_wall)
        except subprocess.TimeoutExpired:
            _kill_pg(p)
            out, err = p.communicate()
            return TIMEOUT, "wall timeout"
        if p.returncode == 0:
            return PASS, ""
        return FAIL, (err or b"")[-ERR_TAIL:].decode("utf-8", "replace")
    finally:
        # Defense in depth: if communicate returned without the explicit timeout
        # path (interpreter shutdown, error after spawn), still reap the group.
        try:
            if p.poll() is None:
                _kill_pg(p)
        except Exception:
            pass
        shutil.rmtree(td, ignore_errors=True)


def _kill_pg(p):
    """SIGKILL the candidate's whole process group and wait for the leader."""
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        p.kill()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _selftest():
    good_sol = "def f(x):\n    return x + 1"
    good_test = "assert f(1) == 2\nassert f(2) == 3"
    assert execute(good_sol, good_test)[0] == PASS

    bad_sol = "def f(x):\n    return x"
    assert execute(bad_sol, good_test)[0] == FAIL  # f(1)==1 != 2, AssertionError

    syntax_sol = "def f(x:\n    return x"
    assert execute(syntax_sol, good_test)[0] == FAIL

    empty = execute("", good_test)
    assert empty == (FAIL, "empty solution or test")

    # wall timeout path: sleeping child burns no CPU, so only the wall limit ends it
    sleep_sol = "import time\nwhile True:\n    time.sleep(60)"
    assert execute(sleep_sol, "assert True", _limits=(CPU_SOFT, CPU_HARD, AS_BYTES), _wall=3)[0] == TIMEOUT

    # orphan-grandchild path: candidate spawns a sleeping child in its process
    # group and then idles. Wall timeout must kill BOTH via killpg (209-orphan
    # incident 2026-09-11: children reparented to init and survived the leader).
    import tempfile as _tf
    gpid_file = os.path.join(_tf.mkdtemp(), "gpid")
    spawn = (
        "import os, subprocess, sys, time\n"
        f"g = subprocess.Popen([sys.executable, '-c', "
        f"'import time; time.sleep(60)'])\n"
        f"open({gpid_file!r}, 'w').write(str(g.pid))\n"
        "time.sleep(60)\n"
    )
    assert execute(spawn, "assert True", _wall=3)[0] == TIMEOUT
    import time as _time
    _time.sleep(1)
    gpid = int(open(gpid_file).read())
    try:
        os.kill(gpid, 0)
        raise AssertionError("grandchild survived the timeout killpg")
    except ProcessLookupError:
        pass
    # CPU bound runaway dies on the CPU rlimit -> nonzero exit -> FAIL
    assert execute("while True:\n    pass", "assert True", _limits=(1, 1, AS_BYTES), _wall=10)[0] == FAIL

    # numpy is importable under the 8 GiB limit (the OpenBLAS 2 GiB false-fail shape)
    try:
        import numpy  # noqa: F401
        np_sol = "import numpy as np\ndef f():\n    return int(np.arange(3).sum())"
        assert execute(np_sol, "assert f() == 3")[0] == PASS
    except ImportError:
        pass

    # nontrivial() floor: print-only / single-builtin exercises are CUT
    assert nontrivial("print(input())") is False
    assert nontrivial("def f(x):\n    return abs(x)") is False
    # syntax-broken solution is not nontrivial even if long
    assert nontrivial("x = (\n" + "\n".join(["a = 1"] * 40)) is False
    # a sizeable program clears the node-count branch
    big = "\n".join(f"v{i} = {i}" for i in range(NODES_MIN + 5))
    assert nontrivial(big) is True
    # a compact real algorithm under the node bar clears the control-flow+loop branch
    compact = (
        "def f(a):\n"
        "    for i in range(len(a)):\n"
        "        if a[i] < 0:\n"
        "            for j in range(i):\n"
        "                if a[j] == -a[i]:\n"
        "                    return i\n"
        "    return -1\n"
    )
    assert nontrivial(compact) is True
    print("ud_solution_exec selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__.strip())
