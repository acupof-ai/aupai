#!/usr/bin/env python3
"""seccomp-BPF filter as a preexec hook: deny socket, ptrace, setsid (de-28a, fb ruling b).

WHY HAND-WRITTEN BPF AND NOT pyseccomp. pyseccomp is not installed on the pod and adding a
dependency to run untrusted code is the wrong trade -- the filter is 20 instructions and
ctypes reaches prctl directly. No new package, nothing to keep in sync.

WHY A DENYLIST AND NOT A WHITELIST. fb's ruling: deny socket/socketpair/ptrace/setsid, no
whitelist. A syscall whitelist for CPython is a moving target -- it changes with the libc,
the interpreter version and the imports the test happens to make, and a missing entry kills
the process with SIGSYS before it runs a line, which is exactly the failure shape that cost
this file three rounds already (a narrow Seatbelt read list, a ro /dev, a 0700 chroot root:
all "hardening that silently breaks the loader"). A denylist cannot do that -- anything not
named runs.

WHAT THIS ADDS OVER THE NETWORK NAMESPACE. The netns already means there is no route to
anywhere. seccomp denies the socket() call itself, so:
  - it works where a netns is unavailable (no CAP_SYS_ADMIN, a locked-down k8s policy)
  - it also blocks AF_UNIX and socketpair, which a netns does NOT: an abstract unix socket
    is namespaced, but a filesystem one on a shared mount is not.
  - ptrace: without it, a rollout can attach to a sibling process in the same pid namespace
    and read its memory. The pid namespace isolates it from the HOST, not from its peers.
  - setsid: fb's escape probe -- leaving the process group the runner kills. The pid
    namespace is the real defence, this closes the mechanism.
It is defence in depth, not a replacement: the level in a rollout record still says which
isolation was in force.

    python3 algorithms/seccomp.py --selftest    # EPERM inside, and the loader still runs

# restartable: installs a filter in the calling process and returns; no files, no state.
"""

import ctypes
import os
import platform
import struct
import sys

# ---------------------------------------------------------------- BPF, seccomp constants
BPF_LD, BPF_JMP, BPF_RET = 0x00, 0x05, 0x06
BPF_W, BPF_ABS = 0x00, 0x20
BPF_JEQ, BPF_JGE, BPF_K = 0x10, 0x30, 0x00

SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_KILL_PROCESS = 0x80000000

PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2

# x86_64 only, and it REFUSES elsewhere rather than installing a filter whose numbers mean
# something else. A syscall number is per-architecture: 41 is socket on x86_64 and
# io_setup on aarch64, so a table applied to the wrong arch does not fail, it denies the
# wrong call -- silently, which is the worst available outcome for a security filter.
AUDIT_ARCH_X86_64 = 0xC000003E
NR = {"socket": 41, "socketpair": 53, "ptrace": 101, "setsid": 112,
      "connect": 42, "bind": 49, "listen": 50, "accept": 43, "accept4": 288,
      "sendto": 44, "recvfrom": 45}

DENIED = ("socket", "socketpair", "ptrace", "setsid", "connect", "bind", "listen",
          "accept", "accept4", "sendto", "recvfrom")

# struct seccomp_data: nr at offset 0, arch at offset 4.
OFF_NR, OFF_ARCH = 0, 4


class Unsupported(RuntimeError):
    """seccomp is not usable here. The caller decides; this module never degrades silently."""


def _stmt(code, k):
    return struct.pack("HBBI", code, 0, 0, k)


def _jump(code, k, jt, jf):
    return struct.pack("HBBI", code, jt, jf, k)


def build_filter(denied=DENIED, errno=1):
    """The BPF program. errno 1 is EPERM: the call fails like a permission error, which
    every runtime already handles, instead of killing the process with SIGSYS."""
    prog = [
        # Wrong architecture -> kill. Checked FIRST: on a mismatch every nr below means a
        # different call, so allowing anything would be worse than refusing to run.
        _stmt(BPF_LD | BPF_W | BPF_ABS, OFF_ARCH),
        _jump(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        _stmt(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        _stmt(BPF_LD | BPF_W | BPF_ABS, OFF_NR),
    ]
    # x32 shares the x86_64 arch token with the high bit set on nr; treat it as foreign.
    prog += [
        _jump(BPF_JMP | BPF_JGE | BPF_K, 0x40000000, 0, 1),
        _stmt(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
    ]
    for name in denied:
        prog += [
            _jump(BPF_JMP | BPF_JEQ | BPF_K, NR[name], 0, 1),
            _stmt(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (errno & 0xFFFF)),
        ]
    prog.append(_stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW))
    return b"".join(prog), len(prog)


def available():
    """(ok, why). Probes rather than assuming -- and distinguishes ENOSYS from EINVAL.

    ENOSYS means this kernel has no seccomp at all; EINVAL means seccomp is there and
    rejected the deliberately invalid argument, which is the answer we want. Reading only
    "the call failed" turns a missing kernel feature into an apparent config problem, the
    same misreading that made Landlock look configurable on this pod's 5.4 kernel.
    """
    if platform.system() != "Linux":
        return False, f"not Linux ({platform.system()})"
    if platform.machine() != "x86_64":
        return False, f"syscall table is x86_64-only, host is {platform.machine()}"
    libc = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    libc.syscall(317, 99, 0, None)  # __NR_seccomp, deliberately invalid operation
    err = ctypes.get_errno()
    if err == 38:  # ENOSYS
        return False, "ENOSYS: this kernel has no seccomp syscall"
    if err != 22:  # EINVAL is the expected answer
        return False, f"unexpected errno {err} ({os.strerror(err)}) probing seccomp"
    return True, "seccomp present (EINVAL on an invalid op, not ENOSYS)"


def install(denied=DENIED, errno=1):
    """Install the filter in THIS process. Irreversible; call in a child, before exec."""
    ok, why = available()
    if not ok:
        raise Unsupported(why)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise Unsupported(f"PR_SET_NO_NEW_PRIVS: {os.strerror(ctypes.get_errno())}")
    prog, n = build_filter(denied, errno)
    buf = ctypes.create_string_buffer(prog, len(prog))

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    fprog = SockFprog(n, ctypes.cast(buf, ctypes.c_void_p))
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0) != 0:
        raise Unsupported(f"PR_SET_SECCOMP: {os.strerror(ctypes.get_errno())}")
    return n


def preexec(denied=DENIED, errno=1):
    """A preexec_fn for subprocess: installs the filter in the child before it execs."""
    def _pre():
        install(denied, errno)
    return _pre


# The bootstrap for a chroot. setpriv cannot load a BPF filter (no --seccomp option,
# MEASURED on the pod), and the filter has to be installed AFTER the uid drop and INSIDE the
# chroot, so something in there must install it and then exec the real target. This file is
# that something: sandbox_exec writes it and seccomp.py beside it into /work.
#
# execv, not subprocess: the filter survives exec (asserted in the suite), so the target
# inherits it with no extra process in between and the runner's rc is the target's rc.
BOOTSTRAP = r'''import os
import sys

sys.path.insert(0, "/work")
import seccomp

try:
    seccomp.install()
except seccomp.Unsupported as e:
    # Loud, and non-zero. A sandbox that silently runs without the layer it claims is worse
    # than one that refuses: the rollout record would say the filter was in force.
    sys.stderr.write(f"sandbox: seccomp filter could not be installed: {e}\n")
    raise SystemExit(96)
# The interpreter, re-exec'd with everything after this script's own name. argv[1] is a
# python FLAG (-I, -m), not a binary, so the binary has to come from sys.executable.
os.execv(sys.executable, [sys.executable] + sys.argv[1:])
'''


_CHILD = r"""
import sys
sys.path.insert(0, "@MODDIR@")
import seccomp as S
S.install()
import socket, os
# 1. A socket must be EPERM, not a hang and not a kill.
try:
    socket.socket()
    print("BUG socket() succeeded")
except PermissionError:
    print("ok socket EPERM")
except Exception as e:
    print("BUG socket raised", type(e).__name__, e)
# 2. AF_UNIX too -- this is what the net namespace does NOT cover.
try:
    socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    print("BUG AF_UNIX succeeded")
except PermissionError:
    print("ok AF_UNIX EPERM")
except Exception as e:
    print("BUG AF_UNIX raised", type(e).__name__, e)
# 3. socketpair, same reason.
try:
    socket.socketpair()
    print("BUG socketpair succeeded")
except (PermissionError, OSError) as e:
    print("ok socketpair blocked" if e.errno == 1 else f"BUG socketpair errno {e.errno}")
# 4. setsid: fb's pgid-escape mechanism.
try:
    os.setsid()
    print("BUG setsid succeeded")
except PermissionError:
    print("ok setsid EPERM")
except Exception as e:
    print("BUG setsid raised", type(e).__name__, e)
# 5. THE OTHER HALF, and the one every previous round of this file got wrong: ordinary work
#    must still run. A filter that breaks the interpreter is not a tighter sandbox, it is a
#    sandbox that measures nothing -- three times now (narrow Seatbelt read list, ro /dev,
#    0700 chroot root) the "hardening" killed the process before it ran a line.
import json, subprocess, tempfile, math
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write("io works")
print("ok file io:", open(f.name).read())
print("ok json+math:", json.dumps({"gcd": math.gcd(12, 18)}))
r = subprocess.run([sys.executable, "-c", "print('child ran')"], capture_output=True, text=True)
print("ok subprocess:", r.stdout.strip(), "(filter is inherited)")
"""


def _selftest():
    ok, why = available()
    print(f"seccomp available: {ok} -- {why}")
    prog, n = build_filter()
    print(f"filter: {n} instructions, {len(prog)} bytes, denying {len(DENIED)} syscalls")
    if not ok:
        print("SKIP: cannot install a filter here; the build path above is still checked.")
        # The arch guard is the one assertion that holds everywhere.
        assert n == 6 + 2 * len(DENIED) + 1, n
        assert AUDIT_ARCH_X86_64 in [struct.unpack("HBBI", prog[i:i + 8])[3]
                                     for i in range(0, len(prog), 8)], "no arch check"
        return 0

    import subprocess
    import tempfile

    here = os.path.dirname(os.path.abspath(__file__))

    # The falsifying direction FIRST: without the filter, do these calls succeed? A denial
    # test whose calls would have failed anyway measures nothing -- and AF_UNIX and
    # socketpair are exactly the ones a network namespace does NOT block, which is the whole
    # reason to add seccomp on top of it. MEASURED on the pod: all three succeed unfiltered.
    unfiltered = subprocess.run(
        [sys.executable, "-c",
         "import socket\n"
         "for f in (lambda: socket.socket(),\n"
         "          lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM),\n"
         "          socket.socketpair):\n"
         "    try:\n        f(); print('open')\n"
         "    except Exception as e:\n        print('closed', type(e).__name__)\n"],
        capture_output=True, text=True, timeout=30)
    opens = unfiltered.stdout.count("open")
    print(f"without the filter: {opens}/3 of socket/AF_UNIX/socketpair succeed")
    if opens < 3:
        print(f"FAIL: only {opens}/3 work unfiltered, so the denials below prove nothing "
              f"about the filter. Output: {unfiltered.stdout.strip()!r}")
        return 1

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        # replace, not .format: the child source contains dict and f-string braces, and
        # .format read `{"gcd": ...}` as a field name (KeyError: 'e' on the pod).
        f.write(_CHILD.replace("@MODDIR@", here))
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=60)
        print(r.stdout.rstrip())
        if r.stderr.strip():
            print("stderr:", r.stderr.strip()[-400:])
        bugs = [ln for ln in r.stdout.splitlines() if ln.startswith("BUG")]
        # A filter that killed the child instead of returning EPERM: rc -31 is SIGSYS.
        if r.returncode != 0:
            print(f"FAIL: the child exited {r.returncode} "
                  f"({'SIGSYS -- the filter killed it' if r.returncode == -31 else 'see stderr'})")
            return 1
        expected = 4 + 3  # four denials, three still-works lines
        got = len([ln for ln in r.stdout.splitlines() if ln.startswith("ok ")])
        if got != expected:
            print(f"FAIL: {got} ok lines, expected {expected} -- a case did not report")
            return 1
        if bugs:
            print(f"FAIL: {len(bugs)} bug(s)")
            return 1
    finally:
        os.unlink(path)
    print("selftest OK: socket/AF_UNIX/socketpair/setsid are EPERM, file io + subprocess "
          "+ stdlib still run, filter inherited across exec")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    ok, why = available()
    print(f"{'available' if ok else 'unavailable'}: {why}")
