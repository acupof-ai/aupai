#!/usr/bin/env python3
"""Sandboxed Python execution for code eval and RL (fb hard precondition).

Model-generated code is untrusted code executed in a loop. Isolation:
  - new mount namespace, root made rprivate, chroot into a minimal root:
    read-only bind of /usr and /dev, merged-/usr symlinks replicated, tmpfs
    for /tmp. The process cannot see /work/aupai (eval answers, training
    data) or anything outside the minimal root.
  - network namespace (-n): no sockets.
  - pid namespace (-p) + process-group kill on timeout: the whole tree dies
    with the runner (os.killpg, not just the unshare parent).
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
import signal
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
# /dev is a tmpfs with the few devices CREATED BY mknod, not a read-only bind of the
# host's /dev. Three measurements produced this shape, in order (pod, 2026-09-02):
#   1. `mount --bind /dev` + remount ro: pytest's logging plugin opens /dev/null for
#      WRITE and dies before collecting a test --
#      `INTERNALERROR OSError: [Errno 30] Read-only file system: '/dev/null'`.
#   2. tmpfs + a per-device `mount --bind /dev/$f`: null, zero, full and tty did not
#      appear inside the chroot while random and urandom did, so the loop's result
#      depends on the shape of the container's own /dev. pytest then died on
#      `FileNotFoundError: '/dev/null'` -- a different error with the same cause.
#   3. mknod with the fixed Linux major/minor numbers: independent of the host, and
#      the device is writable because nothing remounts it ro.
# Strictly tighter than the whole-/dev bind it replaces: code in the chroot runs as
# root and a full /dev hands it every block device on the box.
mount -t tmpfs -o size=1m,mode=755 tmpfs "$ROOT/dev"
mknod -m 666 "$ROOT/dev/null" c 1 3
mknod -m 666 "$ROOT/dev/zero" c 1 5
mknod -m 666 "$ROOT/dev/full" c 1 7
mknod -m 666 "$ROOT/dev/random" c 1 8
mknod -m 666 "$ROOT/dev/urandom" c 1 9
mknod -m 666 "$ROOT/dev/tty" c 5 0
ln -s /proc/self/fd "$ROOT/dev/fd"
ln -s /proc/self/fd/0 "$ROOT/dev/stdin"
ln -s /proc/self/fd/1 "$ROOT/dev/stdout"
ln -s /proc/self/fd/2 "$ROOT/dev/stderr"
# The assert, because both failures above were a missing or unwritable /dev/null
# surfacing hundreds of lines later as someone else's traceback. Exit 97 names it here.
[ -c "$ROOT/dev/null" ] && : > "$ROOT/dev/null" || {
  echo "sandbox: /dev/null is missing or not writable in the chroot" >&2; exit 97; }
mount -t proc proc "$ROOT/proc"
mount -t tmpfs -o size=64m tmpfs "$ROOT/tmp"
# /work is the per-run mkdtemp with code.py already written by the runner;
# a tmpfs here would shadow it. chroot confines visibility to this tree.
ulimit -t 5 -v 2097152 -c 0
# /usr/bin/python3 is a symlink through /etc/alternatives, which the chroot
# deliberately does not contain; resolve to the real binary on the host.
PY=$(readlink -f /usr/bin/python3)
shift
# The command, from argv rather than baked in: a test runner needs to invoke
# `-m pytest <file>` in a directory holding several files, not one hardcoded
# path. Defaults to code.py when no command is given, so every existing caller
# is unchanged (de-28a, 2026-09-02).
if [ "$#" -eq 0 ]; then
  exec chroot "$ROOT" /usr/bin/env -i PATH=/usr/bin:/bin PYTHONIOENCODING=utf-8 \
    "$PY" -I /work/code.py
fi
# NOT -I and NOT -E for the multi-file form. Both ignore PYTHONPATH, so the test
# could not import the implementation beside it and `-m pytest` could not find
# pytest -- MEASURED on the pod: `No module named pytest` with `-S -E` set, a
# pair of flags that reads as hardening and silently empties sys.path of
# everything this form needs (2026-09-02). Isolation here comes from the
# namespaces and the chroot, not from python's flags; `env -i` already gives a
# clean environment, and PYTHONNOUSERSITE keeps ~/.local out.
exec chroot "$ROOT" /usr/bin/env -i PATH=/usr/bin:/bin PYTHONIOENCODING=utf-8 \
  PYTHONPATH="/work:$SITE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  HOME=/work TMPDIR=/tmp "$PY" "$@"
"""


def run_sandboxed(code, timeout=10, stdin=None, files=None, argv=None, site=False):
    """Run code in the sandbox. Returns (rc, stdout, stderr_tail).

    code:   written to /work/code.py and executed. Pass None with `files`+`argv` to run
            something else instead.
    stdin:  optional string fed to the process's stdin (example-based tests).
    files:  {name: text} written into /work beside code.py. For a test runner that needs
            an implementation module and a test module in one directory.
    argv:   python arguments to run instead of `-I /work/code.py`, e.g.
            ["-m", "pytest", "-q", "/work/test_solution.py"]. Paths are inside the chroot,
            so /work/<name>.
    site:   bind site-packages read-only into the chroot. Off by default: the sandbox is
            deliberately minimal, and pulling the host's whole dependency tree in widens
            what untrusted code can import. Needed only when argv names a third-party
            runner such as pytest.

    Added for de-28a: the single-file form was the only form, so a code-execution reward
    could not run a test file beside an implementation. Every existing caller passes only
    `code` and is unaffected -- the argv default reproduces the old command exactly.
    """
    if os.geteuid() != 0:
        raise RuntimeError("sandbox_exec needs root (chroot + namespaces); run on the pod")
    root = tempfile.mkdtemp(prefix="sandbox.", dir="/tmp")
    try:
        os.makedirs(os.path.join(root, "work"), exist_ok=True)
        if code is not None:
            with open(os.path.join(root, "work", "code.py"), "w", encoding="utf-8") as f:
                f.write(code)
        for name, text in (files or {}).items():
            # basename only: a caller must not be able to write outside /work through a
            # relative path, and the reward's file names come from its own constants.
            with open(os.path.join(root, "work", os.path.basename(name)), "w",
                      encoding="utf-8") as f:
                f.write(text)
        setup = _SETUP.replace('shift\n', 'shift\nSITE=""\n', 1)
        if site:
            # Read-only, and only when asked. Located rather than hardcoded: the path
            # differs between 3.11 and 3.12 and between distro and local installs.
            import sysconfig

            sp = sysconfig.get_paths().get("purelib", "")
            local = "/usr/local/lib/python3.12/dist-packages"
            sp = sp if os.path.isdir(sp) else (local if os.path.isdir(local) else "")
            if sp:
                setup = setup.replace(
                    'mount -t proc proc "$ROOT/proc"',
                    'mount -t proc proc "$ROOT/proc"\n'
                    f'mkdir -p "$ROOT{sp}"\n'
                    f'mount --bind {sp} "$ROOT{sp}"\n'
                    f'mount -o remount,ro,bind "$ROOT{sp}"',
                ).replace('SITE=""', f'SITE="{sp}"')
        p = subprocess.Popen(
            ["unshare", "-nmp", "--fork", "bash", "-c", setup, "bash", root] + list(argv or []),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if stdin is not None else None,
            start_new_session=True,
        )
        try:
            stdout, stderr = p.communicate(
                input=stdin.encode("utf-8") if stdin is not None else None,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # Kill the whole process group, not just unshare — the grandchild
            # python3 holds the stdout pipes open and blocks communicate() otherwise.
            os.killpg(p.pid, signal.SIGKILL)
            stdout, stderr = p.communicate()
            return -1, (stdout or b"").decode("utf-8", "replace"), "TIMEOUT"
        return (p.returncode,
                stdout.decode("utf-8", "replace"),
                stderr.decode("utf-8", "replace")[-500:])
    finally:
        # mounts die with the namespace; what remains is empty dirs
        shutil.rmtree(root, ignore_errors=True)


def _no_sandbox_survivors():
    """True if no sandbox python3 process is running on the host."""
    out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
    return "python3 -I /work/code.py" not in out


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
    # Wall timeout must kill the whole process tree — the old subprocess.run
    # timeout killed only unshare, leaving python3 alive and blocking the pipe.
    import time as _time
    _time.sleep(1)
    if not _no_sandbox_survivors():
        fails += 1
        print("  FAIL: sandbox python3 survived after tests")
    else:
        print("  OK  no sandbox survivors after tests")
    print(f"sandbox self-check: {len(cases) + 1 - fails}/{len(cases) + 1} pass")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(1 if _self_check() else 0)
