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
import sys
import tempfile

_SETUP = r"""set -e
ROOT="$1"
mount --make-rprivate /
# The chroot root must be TRAVERSABLE by the unprivileged uid the test drops to. mkdtemp
# creates it 0700 root-owned, and the failure that causes is completely misdirected:
# every binary inside dies with `error while loading shared libraries: libc.so.6`, which
# reads as a broken /usr bind. MEASURED on the pod (2026-09-02): as root `ls libm.so.6`
# works and shows 644, as 65534 even /bin/sh cannot load libc -- because 65534 cannot
# traverse $ROOT itself, so nothing under it resolves. Confining is the chroot's job, not
# this mode bit's; /work and /tmp below are the only writable paths.
chmod 755 "$ROOT"
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
#
# The workdir and /tmp must be writable BY THE UNPRIVILEGED UID the test runs as, and
# they are owned by root because the runner created them. 65534 is the kernel's own
# overflow uid, present on every Linux, so it needs no /etc/passwd inside the chroot.
chown -R 65534:65534 "$ROOT/work" "$ROOT/tmp"
ulimit -t 5 -v 2097152 -c 0
# -f caps FILE SIZE in 512-byte blocks: 512 MiB. MEASURED MISSING, not reasoned about --
# a test writing a 2 GiB file through this sandbox returned rc=0 and `WROTE 2147483648`
# on 2026-09-03, while the four other axes (network, filesystem, uid, nproc) were all
# provoked and held. The chroot lives on the container's overlay, which was at 92% with
# 164 GiB free that day, and tilerl's `cp -a` had filled the same 2.0T six hours earlier;
# 200 unknown test suites at 2 GiB each is 400 GiB. This is the one axis where the failure
# mode had already happened once the same day rather than being hypothetical.
#
# In blocks because that is ulimit -f's unit, and a limit expressed in the wrong unit is
# 512x wrong in the permissive direction.
ulimit -f 1048576 2>/dev/null || true
# nproc caps the fork bomb, and ONLY WORKS ON A NON-ROOT UID: RLIMIT_NPROC is not
# enforced for uid 0 (fb, survey A.3). It is set here, in the shell that is about to
# setuid, because a limit set after the drop cannot be raised back.
ulimit -u 64 2>/dev/null || true
# /usr/bin/python3 is a symlink through /etc/alternatives, which the chroot
# deliberately does not contain; resolve to the real binary on the host.
PY=$(readlink -f /usr/bin/python3)
shift
# THE TEST NEVER RUNS AS UID 0. chroot alone leaves the process root inside the tree,
# and root in a chroot is a well-known escape: it can mknod a block device for the host
# disk, it ignores every DAC bit on the bind-mounted /usr, RLIMIT_NPROC does not apply
# to it, and the classic chdir-then-chroot trick walks straight out. `setpriv --reuid
# --regid --clear-groups --no-new-privs` drops to 65534 after the namespaces and the
# chroot are in place -- that order matters, because each of those steps needs the
# privilege it is dropping. --no-new-privs makes the drop irreversible through setuid
# binaries (fb ruling, 2026-09-02, survey A.3).
DROP="setpriv --reuid 65534 --regid 65534 --clear-groups --no-new-privs"
# cwd is /work, not the chroot root. Under uid 0 the cwd was `/` and a test writing a
# relative path silently wrote into the chroot root -- which is root-owned, so the same
# test failed with EACCES the moment the uid drop landed. The rollout marker test caught
# it, though its assertion message blamed the wrong cause: the record read as "a peer's
# workdir was visible" when the write had simply been denied. A relative write from a test
# belongs in the workdir; -C puts it there (2026-09-02).
if [ "$#" -eq 0 ]; then
  exec chroot "$ROOT" /usr/bin/env -i -C /work PATH=/usr/bin:/bin PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/work PYTHONDONTWRITEBYTECODE=1 \
    $DROP "$PY" $BOOT -I /work/code.py
fi
# NOT -I and NOT -E for the multi-file form. Both ignore PYTHONPATH, so the test
# could not import the implementation beside it and `-m pytest` could not find
# pytest -- MEASURED on the pod: `No module named pytest` with `-S -E` set, a
# pair of flags that reads as hardening and silently empties sys.path of
# everything this form needs (2026-09-02). Isolation here comes from the
# namespaces and the chroot, not from python's flags; `env -i` already gives a
# clean environment, and PYTHONNOUSERSITE keeps ~/.local out.
exec chroot "$ROOT" /usr/bin/env -i -C /work PATH=/usr/bin:/bin PYTHONIOENCODING=utf-8 \
  PYTHONPATH="/work:$SITE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  HOME=/work TMPDIR=/tmp $DROP "$PY" $BOOT "$@"
"""


def run_sandboxed(code, timeout=10, stdin=None, files=None, argv=None, site=False,
                  seccomp=True):
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
        # seccomp, when this host can install a filter. It goes here rather than in the
        # shell because setpriv cannot load a BPF filter (no --seccomp option, MEASURED),
        # and the filter must land AFTER the uid drop and INSIDE the chroot -- so a tiny
        # bootstrap in /work installs it and execv's the real target. Absent seccomp the
        # sandbox runs exactly as before: the namespaces and the chroot are the guarantee,
        # this is depth. What it adds over the netns is AF_UNIX, socketpair and ptrace,
        # none of which a network namespace blocks.
        seccomp_ok = False
        if seccomp:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "algorithms"))
            try:
                import seccomp as _sec

                ok, _why = _sec.available()
                if ok:
                    shutil.copy(_sec.__file__, os.path.join(root, "work", "seccomp.py"))
                    with open(os.path.join(root, "work", "_boot.py"), "w",
                              encoding="utf-8") as f:
                        f.write(_sec.BOOTSTRAP)
                    seccomp_ok = True
            except ImportError:
                pass
        setup = _SETUP.replace('shift\n', 'shift\nSITE=""\nBOOT=""\n', 1)
        if seccomp_ok:
            setup = setup.replace('BOOT=""', 'BOOT="/work/_boot.py"')
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
    """True if no sandbox python3 process is running on the host.

    Matches on the /work paths the sandbox uses, not on one hardcoded cmdline: the setsid
    double-fork probe runs `/work/code.py` through a different argv shape, and the old
    single-string match would have reported no survivors while one slept for 300s.
    """
    out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
    return not [ln for ln in out.splitlines()
                if "/work/code.py" in ln or "/work/test_solution.py" in ln]


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
        ("import os\nprint(sorted(os.listdir('/work')))", 0, "code.py",
         "the workdir holds code.py plus the seccomp bootstrap when the filter is in force"),
        # seccomp specifically, as opposed to the network namespace: AF_UNIX and socketpair
        # are NOT blocked by a netns, so these two fail only because a filter denied the
        # syscall. Without seccomp they succeed -- asserted from the other side in
        # algorithms/seccomp.py --selftest, which runs them unfiltered first.
        ("import socket\n"
         "try:\n    socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); print('REACHED unix')\n"
         "except PermissionError:\n    print('blocked')\n"
         "except OSError as e:\n    print('oserr', e.errno)\n",
         0, "blocked", "AF_UNIX socket is denied (seccomp, not the netns)"),
        ("import sys\nprint(sys.stdin.read().strip())", 0, "hello", "stdin passthrough (example-based tests)"),
        # The uid drop, as an assertion rather than a claim in a comment. Everything below
        # depends on it: RLIMIT_NPROC is not enforced for uid 0, and root inside a chroot can
        # mknod the host disk and walk out with chdir-then-chroot.
        ("import os\nprint(os.getuid(), os.geteuid())", 0, "65534 65534",
         "the test runs as an unprivileged uid, NOT root (fb ruling, survey A.3)"),
        # Root's chroot escape, tried directly: mknod a block device for the host disk. As
        # 65534 this is EPERM, so the code cannot manufacture a path to the raw disk.
        ("import os\ntry:\n"
         "    os.mknod('/tmp/disk', 0o600 | 0o060000, os.makedev(8, 0))\n"
         "    print('REACHED made a block device')\n"
         "except Exception as e:\n    print('blocked', type(e).__name__)\n",
         0, "blocked", "cannot mknod a host block device (the classic chroot escape)"),
        # fb's probe 1: setsid + double fork to leave the process group the runner kills.
        # The pid namespace is what actually stops this -- every descendant dies with the
        # namespace's init, whatever its pgid.
        ("import os, sys, time\n"
         "if os.fork():\n    print('parent done'); sys.exit(0)\n"
         "os.setsid()\n"
         "if os.fork():\n    os._exit(0)\n"
         "time.sleep(300)\n", 0, "parent done",
         "setsid double fork: the escapee dies with the pid namespace, checked below"),
        # fb's probe 2: read the harness's own environment through /proc. The chroot's /proc
        # is a fresh mount in a new pid namespace, so the harness is not even numbered there.
        ("import glob\n"
         "hits = [p for p in glob.glob('/proc/*/environ')]\n"
         "leaked = []\n"
         "for p in hits:\n"
         "    try:\n"
         "        leaked += [p for k in (b'AWS', b'TOKEN', b'KEY', b'SSH')\n"
         "                   if k in open(p, 'rb').read()]\n"
         "    except Exception:\n        pass\n"
         "print('REACHED ' + str(leaked) if leaked else f'blocked, {len(hits)} procs visible')\n",
         0, "blocked", "cannot read a secret out of another process's environ"),
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
