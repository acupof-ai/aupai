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
import re
import shutil
import signal
import subprocess
import sys
import tempfile


class SandboxLaunchError(RuntimeError):
    """The sandbox could not start the candidate, so there is no verdict to report.

    Distinct from a nonzero rc, which IS a verdict. Raised rather than returned because
    every caller reads rc == 0 as "passed": handing back an rc for a launch that never
    happened turns an infrastructure fault into a benchmark score of zero.
    """


#: stderr signatures of a launch that never reached the candidate: setpriv failing to exec
#: python after the uid drop, and this script's own fail-closed rlimit exits.
_LAUNCH_FAILED = re.compile(r"failed to execute|could not set RLIMIT|cannot execute")

_SETUP = r"""set -e
ROOT="$1"
# Bring loopback up inside THIS network namespace (no `ip` binary in the image;
# SIOCSIFFLAGS on lo from a raw ioctl). The fresh netns has no routes and no
# external interfaces, so lo is the only thing a socket can reach. Done while
# still in the host rootfs so the interpreter is the host's, before chroot.
if [ "$LOOPBACK" = "1" ]; then
  "$(readlink -f /usr/bin/python3)" - <<'PYEOF'
import fcntl, socket, struct
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
fcntl.ioctl(s, 0x8914, struct.pack("16sH22s", b"lo", 0x1, b""))  # IFF_UP, struct ifreq is 40B
PYEOF
fi
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
# Private /dev/shm: multiprocessing spawn needs POSIX semaphores for its Queue, and
# without it the child dies on FileNotFoundError('/dev/shm/sem....'). A fresh tmpfs
# mounted here shadows the host's /dev/shm inside the chroot, so the chapter sees an
# empty 64m shm of its own, never other jobs' semaphores.
mkdir -p "$ROOT/dev/shm"
mount -t tmpfs -o size=64m,mode=1777 tmpfs "$ROOT/dev/shm"
# /work is the per-run mkdtemp with code.py already written by the runner;
# a tmpfs here would shadow it. chroot confines visibility to this tree.
#
# The workdir and /tmp must be writable BY THE UNPRIVILEGED UID the test runs as, and
# they are owned by root because the runner created them. 65534 is the kernel's own
# overflow uid, present on every Linux, so it needs no /etc/passwd inside the chroot.
chown -R 65534:65534 "$ROOT/work" "$ROOT/tmp"
ulimit -t "${CPUSECS:-5}" -v 2097152 -c 0
# -f caps FILE SIZE. MEASURED MISSING, not reasoned about -- a test writing 600 MB in 1 MiB
# chunks through this sandbox returned rc=0 on 2026-09-03, while the four other axes (network,
# filesystem, uid, nproc) were all provoked and held. The chroot lives on the container's
# overlay, which was at 92% with 164 GiB free that day, and tilerl's `cp -a` had filled the
# same 2.0T six hours earlier; 200 unknown test suites at 600 MB each is 120 GB. This is the
# one axis whose failure mode had already happened that day rather than being hypothetical.
#
# THE UNIT IS 1024-BYTE BLOCKS ON THIS SHELL, NOT 512, and that was measured rather than read
# off a man page: `ulimit -f 1048576` then getrlimit returns 1073741824, i.e. 1 GiB. A first
# version wrote 1048576 believing it was 512 MiB, so the cap was 2x more permissive than the
# comment claimed and the 600 MB acceptance case still passed. POSIX says 512; bash's builtin
# uses 1024. Never state a limit's unit from documentation -- set it and read it back.
#
# FAIL CLOSED. This was `2>/dev/null || true`, which swallows a failure to SET the limit
# and runs untrusted code with no file-size cap at all, printing nothing. A limit that
# cannot be set is a boundary that is not there; exit 98 says so here rather than letting
# the caller read the run as clean (fb review, 2026-09-30).
ulimit -f 262144 || {   # 262144 x 1024 = 256 MiB
  echo "sandbox: could not set RLIMIT_FSIZE" >&2; exit 98; }
# nproc caps the fork bomb, and ONLY WORKS ON A NON-ROOT UID: RLIMIT_NPROC is not
# enforced for uid 0 (fb, survey A.3). It is set here, in the shell that is about to
# setuid, because a limit set after the drop cannot be raised back. Same fail-closed rule
# as RLIMIT_FSIZE above, and for the same reason.
#
# THE DEFAULT STAYS 64 AND 64 IS NOT ENOUGH ON THIS POD. This is a known defect with a
# known wrong fix; raising the number is the wrong fix and was reverted before it shipped.
#
# RLIMIT_NPROC counts the tasks of the REAL UID across the whole machine, not this
# sandbox's descendants. Every sandbox here drops to the one shared uid 65534 and creates
# no user namespace, so every sandbox on the box, ours and everyone else's, draws on one
# budget. MEASURED on the pod 2026-09-30, and the two sides agree to the task:
#   - `ps -eL -o ruid=` ON THE HOST counts 121 tasks owned by 65534. Inside the container
#     `ps` counts 0 of them -- a different pid namespace, the same uid accounting -- so a
#     reading taken in the container cannot see the budget at all.
#   - run_sandboxed("print(7)") at nproc=64 returns rc 126 with `setpriv: failed to
#     execute /usr/bin/python3.12: Resource temporarily unavailable`: the execve after the
#     uid drop, before any candidate code runs. 72/80/88/96/104/112/120 fail the same way;
#     128/256/512 return (0, "7\n", ""). The boundary sits in (120, 128], i.e. at the 121.
#   - at nproc=512 a fork loop inside the sandbox got 390 children then EAGAIN. 121 + 390
#     = 511. The cap is the machine-wide count for the uid, exactly.
# So a higher number buys headroom only until those 121 (other containers' nobody tasks,
# which we do not control) grow, and it raises the ceiling for every task sharing the uid
# at the same time -- one runaway then starves the rest. A shared-uid RLIMIT cannot be a
# per-task limit at any value.
# The fix is a per-task cgroup `pids.max` plus a distinct uid (or a user namespace) per
# execution, so the budget is the task's own. Until that lands, callers that need more
# than the shared headroom pass nproc explicitly -- datagen/vet_textbooks.py:37 and
# scripts/sft_verify_code.py:45 both pass 4096 -- and eval/score_code_exec.py --selftest
# stays red on a busy pod.
ulimit -u "${NPROC:-64}" || {
  echo "sandbox: could not set RLIMIT_NPROC=${NPROC:-64}" >&2; exit 98; }
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
                  seccomp=True, profile="hardened", nproc=64, cpu_secs=5, loopback=False):
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
            except ImportError as exc:
                ok, _why, _sec = False, f"algorithms/seccomp.py not importable: {exc}", None
            # A FILTER THAT WAS ASKED FOR AND DID NOT INSTALL MUST NOT BE SILENT. Both arms
            # used to fall through to seccomp_ok=False, so `seccomp=True` ran an UNFILTERED
            # sandbox and said nothing. Measured 2026-10-01: a tree holding datagen/ but not
            # algorithms/ takes the ImportError arm, and AF_UNIX then succeeds under
            # seccomp=True exactly as it does under seccomp=False -- the parameter had no
            # observable effect and no message. The netns still blocks external network, so
            # this was not a wide hole; it was an unannounced one, which is worse to reason
            # about. A caller that genuinely wants no filter passes seccomp=False and says so.
            if not ok:
                raise SandboxLaunchError(
                    f"seccomp=True but the filter could not be installed ({_why}); refusing to "
                    f"run untrusted code in a sandbox weaker than the caller asked for. Pass "
                    f"seccomp=False to accept netns+chroot+rlimits only."
                )
            shutil.copy(_sec.__file__, os.path.join(root, "work", "seccomp.py"))
            with open(os.path.join(root, "work", "_boot.py"), "w", encoding="utf-8") as f:
                f.write(_sec.BOOTSTRAP)
            seccomp_ok = True
        setup = _SETUP.replace('shift\n', 'shift\nSITE=""\nBOOT=""\n', 1)
        setup = setup.replace('set -e\n',
                              f'set -e\nLOOPBACK="{1 if loopback else 0}"\nNPROC="{int(nproc)}"\n'
                              f'CPUSECS="{int(cpu_secs)}"\n', 1)
        setup = setup.replace("PYTHONDONTWRITEBYTECODE=1 \\",
                              f"PYTHONDONTWRITEBYTECODE=1 SANDBOX_SECCOMP_PROFILE={profile} \\")
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
        def _clear_sigmask():
            # util-linux unshare forks and calls sigprocmask(SIG_UNBLOCK) on a fixed set
            # in the child; it returns EINVAL ("sigprocmask unblock failed") when it
            # inherits a polluted mask from a long-running parent (the vet process, after
            # hundreds of chapters). Start unshare from a fully-unblocked mask so the host
            # tool never depends on the caller's signal state. In a forked preexec there
            # is one thread, so pthread_sigmask is equivalent to the process-wide mask.
            signal.pthread_sigmask(signal.SIG_UNBLOCK, signal.valid_signals())

        p = subprocess.Popen(
            ["unshare", "-nmp", "--fork", "bash", "-c", setup, "bash", root] + list(argv or []),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if stdin is not None else None,
            start_new_session=True,
            preexec_fn=_clear_sigmask,
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
        err_tail = stderr.decode("utf-8", "replace")[-500:]
        # A SANDBOX THAT COULD NOT START THE CODE MUST NOT RETURN AN EXIT CODE FOR IT.
        # Every caller reads `rc == 0` as "the candidate passed", so a nonzero rc from a
        # FAILED LAUNCH reads as "the candidate failed" -- print-and-continue, in the one
        # place where the output is a benchmark number. Measured 2026-10-01 on this pod:
        # run_sandboxed("print(1)") at the default nproc=64 returns rc 126 with
        # `setpriv: failed to execute /usr/bin/python3.12: Resource temporarily unavailable`,
        # because RLIMIT_NPROC counts the shared uid 65534's tasks machine-wide (see the
        # NPROC comment above). Every eval scorer in this repo omitted nproc, so on a busy
        # pod each one scored EVERY problem as failed and reported it as a score.
        # exit 98 is this script's own fail-closed rlimit exit; 126/127 are exec failures
        # from setpriv, which happen before the candidate runs at all.
        if p.returncode in (98, 126, 127) and _LAUNCH_FAILED.search(err_tail):
            raise SandboxLaunchError(
                f"sandbox did not run the code (rc={p.returncode}): {err_tail.strip()[-300:]}. "
                f"This is NOT a verdict on the code. nproc={nproc} draws on uid 65534's "
                f"machine-wide task count; callers that need headroom pass nproc explicitly."
            )
        return (p.returncode, stdout.decode("utf-8", "replace"), err_tail)
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
        # "code.py" was the expected substring, which ['code.py'] satisfies with the filter
        # OFF -- the case could not detect the thing its own label names. _boot.py is present
        # only when the bootstrap was installed, so it is the discriminating string.
        ("import os\nprint(sorted(os.listdir('/work')))", 0, "_boot.py",
         "the workdir holds the seccomp bootstrap, which is absent when the filter is off"),
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

    # A FAILED LAUNCH RAISES AND IS NEVER AN rc. Both directions, because a raise that fired
    # on every call would be worse than the bug it replaces. nproc=1 cannot exec CPython under
    # any machine load, so this is deterministic where nproc=64 is not: 64 depends on uid
    # 65534's machine-wide task count and passes on a quiet box (see the NPROC comment above),
    # which is exactly why no eval scorer noticed it was scoring every problem as failed.
    for nproc_arg, want_raise in ((1, True), (4096, False)):
        try:
            rc0, out0, _ = run_sandboxed("print(1)", timeout=15, nproc=nproc_arg)
            raised = False
        except SandboxLaunchError:
            rc0, out0, raised = None, "", True
        ok = raised == want_raise and (raised or (rc0 == 0 and "1" in out0))
        fails += 0 if ok else 1
        print(f"  {'OK ' if ok else 'FAIL'} nproc={nproc_arg} raised={raised} "
              f"exp_raise={want_raise} | launch failure is loud, not a verdict")
    if not _LAUNCH_FAILED.search("setpriv: failed to execute /usr/bin/python3.12"):
        fails += 1
        print("  FAIL: _LAUNCH_FAILED does not match the measured setpriv stderr")
    if _LAUNCH_FAILED.search("AssertionError: expected 3, got 4"):
        fails += 1
        print("  FAIL: _LAUNCH_FAILED matches an ordinary test failure")

    for code, exp_rc, exp_out, label in cases:
        # nproc=4096, not the default 64: these cases assert sandbox SEMANTICS (cpu, memory,
        # network, filesystem, seccomp, pid namespace), and at 64 none of them reaches the
        # candidate at all -- the uid drop cannot exec CPython, so every case would report the
        # launch failure instead of its own property. That is the measured cause of the
        # "eval/score_code_exec.py --selftest stays red on a busy pod" note above: the
        # selftest was as exposed to the shared-uid limit as the scorers were. The nproc cap
        # itself is asserted by the two cases above, which own that variable.
        kw = {"stdin": "hello"} if label.startswith("stdin") else {}
        rc, out, err = run_sandboxed(code, timeout=15, nproc=4096, **kw)
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
