#!/usr/bin/env python3
"""Sandboxed Python execution for code eval and RL (fb hard precondition).

Model-generated code is untrusted code executed in a loop. Isolation:
  - new mount namespace, root made rprivate, chroot into a minimal root:
    read-only bind of /usr and /dev, merged-/usr symlinks replicated, tmpfs
    for /tmp. The process cannot see /work/aupai (eval answers, training
    data) or anything outside the minimal root.
  - network namespace (-n): no sockets.
  - pid namespace (-p): the whole tree dies with the runner on timeout.
  - rlimits: CPU 5s, address space 2GB, no core dumps.
  - env scrubbed, python -I (no user site, no PYTHON* vars), wall timeout.

Pod-only: needs root + unshare. Off-pod use is a loud failure, not a silent
fallback — running untrusted code without isolation is not an option.

Usage:
  from sandbox_exec import run_sandboxed
  rc, out, err = run_sandboxed("print(1)")   # (0, "1\\n", "")
"""
import os
import shutil
import subprocess
import tempfile

_SETUP = r"""set -e
ROOT="$1"
mount --make-rprivate /
mkdir -p "$ROOT/usr" "$ROOT/dev" "$ROOT/proc" "$ROOT/tmp" "$ROOT/work"
mount --bind /usr "$ROOT/usr"
mount -o remount,ro,bind "$ROOT/usr"
# merged-/usr: /lib /bin /sbin are symlinks into /usr. Replicate the symlink;
# do NOT bind-mount a symlink source (silent failure leaves an empty dir and
# the dynamic loader chain breaks with a confusing ENOENT). Real dirs (/lib64
# on some systems) get the read-only bind.
for d in lib lib64 bin sbin; do
  if [ -L "/$d" ]; then
    ln -s "$(readlink "/$d")" "$ROOT/$d"
  elif [ -d "/$d" ]; then
    mkdir -p "$ROOT/$d"
    mount --bind "/$d" "$ROOT/$d"
    mount -o remount,ro,bind "$ROOT/$d"
  fi
done
mount --bind /dev "$ROOT/dev"
mount -o remount,ro,bind "$ROOT/dev"
mount -t proc proc "$ROOT/proc"
mount -t tmpfs -o size=64m tmpfs "$ROOT/tmp"
# /work is the per-run mkdtemp with code.py already written by the runner;
# a tmpfs here would shadow it. chroot confines visibility to this tree.
ulimit -t 5 -v 2097152 -c 0
# /usr/bin/python3 is a symlink through /etc/alternatives, which the chroot
# deliberately does not contain; resolve to the real binary on the host.
PY=$(readlink -f /usr/bin/python3)
exec chroot "$ROOT" /usr/bin/env -i PATH=/usr/bin:/bin PYTHONIOENCODING=utf-8 \
  "$PY" -I /work/code.py
"""


def run_sandboxed(code, timeout=10, stdin=None):
    """Run code in the sandbox. Returns (rc, stdout, stderr_tail).
    stdin: optional string fed to the process's stdin (example-based tests)."""
    if os.geteuid() != 0:
        raise RuntimeError("sandbox_exec needs root (chroot + namespaces); run on the pod")
    root = tempfile.mkdtemp(prefix="sandbox.", dir="/tmp")
    try:
        os.makedirs(os.path.join(root, "work"), exist_ok=True)
        with open(os.path.join(root, "work", "code.py"), "w", encoding="utf-8") as f:
            f.write(code)
        p = subprocess.run(
            ["unshare", "-nmp", "--fork", "bash", "-c", _SETUP, "bash", root],
            capture_output=True,
            timeout=timeout,
            input=stdin.encode("utf-8") if stdin is not None else None,
        )
        return (p.returncode,
                p.stdout.decode("utf-8", "replace"),
                p.stderr.decode("utf-8", "replace")[-500:])
    except subprocess.TimeoutExpired as e:
        return -1, (e.stdout or b"").decode("utf-8", "replace"), "TIMEOUT"
    finally:
        # mounts die with the namespace; what remains is empty dirs
        shutil.rmtree(root, ignore_errors=True)


def _self_check():
    """Known-answer: gold runs, cheats and attacks do not."""
    cases = [
        # (code, expect_rc, expect_stdout_contains, label)
        ("print('hello')", 0, "hello", "basic execution"),
        ("print([x for x in range(5)])", 0, "[0, 1, 2, 3, 4]", "list output"),
        ("import math\nprint(math.gcd(12, 18))", 0, "6", "stdlib import"),
        ("raise SystemExit(0)", 0, "", "clean exit"),
        ("while True:\n    pass", 1, "", "cpu limit (SIGXCPU kills; unshare exits 1)"),
        ("import time\ntime.sleep(1000)", -1, "", "wall timeout (sleep burns no CPU)"),
        ("x = [0] * 10**10", 1, "", "memory limit (MemoryError)"),
        ("import socket\nsocket.socket().connect(('1.1.1.1', 80))", 1, "", "network blocked"),
        ("print(open('/work/aupai/data/eval/code_holdout_500.jsonl').read()[:10])",
         1, "", "filesystem isolation (eval answers invisible)"),
        ("import os\nprint(os.listdir('/work'))", 0, "code.py", "only the tmpfs workdir is visible"),
        ("import sys\nprint(sys.stdin.read().strip())", 0, "hello", "stdin passthrough (example-based tests)"),
    ]
    fails = 0
    for code, exp_rc, exp_out, label in cases:
        kw = {"stdin": "hello"} if label.startswith("stdin") else {}
        rc, out, err = run_sandboxed(code, timeout=15, **kw)
        ok = (rc == exp_rc or (exp_rc == -1 and rc < 0)) and exp_out in out
        if not ok:
            fails += 1
        print(f"  {'OK ' if ok else 'FAIL'} rc={rc} exp {exp_rc} | {label} | "
              f"out={out[:40]!r} err={err[:80]!r}")
    print(f"sandbox self-check: {len(cases) - fails}/{len(cases)} pass")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(1 if _self_check() else 0)
