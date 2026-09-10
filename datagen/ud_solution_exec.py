#!/usr/bin/env python3
"""Execute an UltraData-Code L3 solution against its bundled test (3b, 2026-09-10).

One predicate shared by the quality audit and the production L3 keep-filter (fb
ruling 2026-09-10): an L3 row is kept iff its solution, run together with the
dataset's own test in a fresh sandbox, exits 0. The audit's n=400 yield estimate
and the full-set conversion call this same function.

Composition: a temp candidate.py of  solution + "\\n\\n" + test + "\\n"  run as a
fresh process. Isolated mode -I (no user site/PYTHONPATH), bytecode off, cwd is
an empty temp dir, environment scrubbed to PATH/LANG so a solution cannot read
repo or cwd state. Bounded by RLIMIT_CPU and RLIMIT_AS and a wall timeout.

    python3 datagen/ud_solution_exec.py --selftest
"""
import os
import resource
import shutil
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

    try:
        r = subprocess.run(
            [sys.executable, "-I", "-B", path],
            cwd=td, capture_output=True, timeout=_wall, preexec_fn=_apply,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        )
        if r.returncode == 0:
            return PASS, ""
        return FAIL, (r.stderr or b"")[-ERR_TAIL:].decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return TIMEOUT, "wall timeout"
    finally:
        shutil.rmtree(td, ignore_errors=True)


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
    # CPU bound runaway dies on the CPU rlimit -> nonzero exit -> FAIL
    assert execute("while True:\n    pass", "assert True", _limits=(1, 1, AS_BYTES), _wall=10)[0] == FAIL

    # numpy is importable under the 8 GiB limit (the OpenBLAS 2 GiB false-fail shape)
    try:
        import numpy  # noqa: F401
        np_sol = "import numpy as np\ndef f():\n    return int(np.arange(3).sum())"
        assert execute(np_sol, "assert f() == 3")[0] == PASS
    except ImportError:
        pass
    print("ud_solution_exec selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__.strip())
