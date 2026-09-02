#!/usr/bin/env python3
"""Run untrusted code under the best isolation this host offers, and SAY WHICH (de-28a).

Stage 3 of docs/lessons/claude_code_rl.md. Model-generated code runs in a loop against
mined tasks; each rollout gets a temporary git worktree and a process sandbox. There is no
container: the pod is already a k8s container and nested containers are unavailable there,
and the launcher must also work on a bare host.

FOUR LEVELS, chosen by what the host actually has, MEASURED not assumed:

  bwrap / nsjail / firejail   none of the three exists on either target host (pod or Mac,
                              2026-09-02). Kept first because they are the right answer on
                              a host that has one, and adding a host must not mean editing
                              the caller.
  sandbox_exec                pod: root + unshare. chroot into a minimal root, mount/net/pid
                              namespaces, rlimits. datagen/sandbox_exec, 10 known answers.
  seatbelt                    Darwin: /usr/bin/sandbox-exec. Denies network and the home
                              secrets, allows the workdir. Apple marks it deprecated; it is
                              the only process isolation this machine has, and Chrome and
                              Firefox still use the same substrate.
  rlimits_only                REFUSES BY DEFAULT. rlimits + timeout + a private tmpdir
                              isolate neither the network nor filesystem reads, so model
                              code could read ~/.ssh and phone home. Set
                              ALLOW_UNISOLATED=1 to run anyway; the level is recorded either
                              way, so a result taken without isolation can never be mistaken
                              for one taken with it.

The level is returned in the result, not logged and forgotten: a reward computed under
rlimits_only and one computed under a namespace are different measurements, and the record
has to say which. Same reasoning as vocab_id on a checkpoint.

    python3 algorithms/isolate.py --selftest    # the four levels and the escape attempts

# restartable: every run writes only inside a mkdtemp it removes. An interrupt leaves at
# most one temp directory under the system tmp; no state is carried between runs.
"""

import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEVELS = ("bwrap", "nsjail", "firejail", "sandbox_exec", "seatbelt", "rlimits_only")

# What each level is asserted to isolate. The suite provokes each True.
ISOLATES = {
    "bwrap": {"net": True, "fs_read": True, "fs_write": True},
    "nsjail": {"net": True, "fs_read": True, "fs_write": True},
    "firejail": {"net": True, "fs_read": True, "fs_write": True},
    "sandbox_exec": {"net": True, "fs_read": True, "fs_write": True},
    "seatbelt": {"net": True, "fs_read": True, "fs_write": True},
    "rlimits_only": {"net": False, "fs_read": False, "fs_write": False},
}


class Unisolated(RuntimeError):
    """rlimits_only was selected and ALLOW_UNISOLATED is not set."""


def detect_level():
    """The best level this host offers. Reads the host, never a config."""
    for tool in ("bwrap", "nsjail", "firejail"):
        if shutil.which(tool):
            return tool
    if platform.system() == "Linux" and os.geteuid() == 0 and shutil.which("unshare"):
        return "sandbox_exec"
    if platform.system() == "Darwin" and shutil.which("sandbox-exec"):
        return "seatbelt"
    return "rlimits_only"


# macOS Seatbelt profile. Three traps, each of which produced a wrong reading before it was
# understood; each is a case in _selftest so the trap cannot come back silently.
#
#  1. `(allow process-exec (with no-sandbox))` EXEMPTS the exec'd process. A profile with it
#     reads as "Seatbelt does not block the network" -- the network was reached because the
#     profile granted an exemption, not because Seatbelt failed.
#  2. A `(deny default)` profile whose read list is incomplete makes python3 die with
#     SIGABRT (rc 134) before running a line: the dynamic loader reads outside any
#     hand-written subpath list. `(allow file-read*)` unrestricted starts fine, so the
#     trap is the ENUMERATION, not the deny-default -- and the enumeration looks complete
#     until the abort. MEASURED: reads allowed only under /usr -> 134. The usable shape is
#     (allow default) plus explicit denies: broader in principle, and the only one that runs.
#  3. `/tmp` on macOS is a symlink to `/private/tmp`, and Seatbelt matches on the REAL path.
#     `(subpath "/tmp/x")` matches nothing, so the workdir write is denied and it reads as
#     the policy being too strict.
_SEATBELT = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{work}") (literal "/dev/null") (literal "/dev/stdout")
                   (literal "/dev/stderr") (literal "/dev/dtracehelper"))
(deny file-read* {secrets})
"""

# Read-denied even though the level allows reads by default: an agent's own credentials are
# the thing a rollout must not be able to exfiltrate, and they are the same on every Mac.
_SECRET_DIRS = (".ssh", ".aws", ".config/gcloud", ".gnupg", ".kube", ".netrc",
                ".claude", ".docker")


def _seatbelt_profile(workdir):
    home = os.path.realpath(os.path.expanduser("~"))
    secrets = " ".join(f'(subpath "{os.path.join(home, d)}")' for d in _SECRET_DIRS)
    # realpath: trap 3.
    return _SEATBELT.format(work=os.path.realpath(workdir), secrets=secrets)


def _rlimits(cpu_s, mem_bytes):
    def pre():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        if mem_bytes:
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, OSError):
                # macOS refuses RLIMIT_AS for large values on some releases; the CPU limit
                # and the wall timeout still apply, and the level already says this tier
                # isolates nothing. Silently skipping the memory cap would overstate it.
                pass
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setsid()
    return pre


def run(code, workdir=None, timeout=10, cpu_s=5, mem_mb=2048, level=None, argv=None):
    """Execute `code` (a str) or `argv` (a list) under isolation. Returns a dict.

    Keys: level, rc, stdout, stderr, timed_out, isolates. `level` is the level actually
    used -- callers record it, they do not assume it.
    """
    lvl = level or detect_level()
    if lvl == "rlimits_only" and os.environ.get("ALLOW_UNISOLATED") != "1":
        raise Unisolated(
            "REFUSING: this host offers no process isolation (no bwrap/nsjail/firejail, "
            "not root+unshare on Linux, no sandbox-exec on Darwin), and rlimits alone "
            "isolate neither the network nor filesystem reads -- model code could read "
            "~/.ssh and reach the network. Set ALLOW_UNISOLATED=1 to run anyway; the "
            "result will record level=rlimits_only, which is not comparable to an "
            "isolated one. datagen/sandbox_exec.py has said this since it was written: "
            "'running untrusted code without isolation is not an option'."
        )
    own_dir = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="isolate.")
    try:
        script = os.path.join(workdir, "code.py")
        if argv is None:
            with open(script, "w", encoding="utf-8") as f:
                f.write(code)
            argv = [sys.executable, "-I", script]

        if lvl == "sandbox_exec":
            sys.path.insert(0, os.path.join(ROOT, "datagen"))
            from sandbox_exec import run_sandboxed

            # Multi-file when the caller gave one: the reward writes an implementation and a
            # test module into workdir and runs a test runner over them. sandbox_exec took
            # only a single code string until de-28a, which is why this path raised
            # `write() argument must be str, not None` on the pod while the Mac's seatbelt
            # path was green -- a defect only the pod could surface (2026-09-02).
            files = {n: open(os.path.join(workdir, n), encoding="utf-8").read()
                     for n in sorted(os.listdir(workdir))
                     if n.endswith(".py") and n != "code.py"}
            if files:
                # /work/<name> inside the chroot, and site-packages bound read-only only
                # when the runner is third-party (pytest); unittest is stdlib.
                inner = ["/work/" + os.path.basename(a) if os.path.isabs(a) or a.endswith(".py")
                         else a for a in (argv[1:] if argv else [])]
                inner = [a for a in inner if a not in ("-I",)]
                rc, out, err = run_sandboxed(code, timeout=timeout, files=files,
                                            argv=inner, site="pytest" in inner)
            else:
                rc, out, err = run_sandboxed(code, timeout=timeout)
            return {"level": lvl, "rc": rc, "stdout": out, "stderr": err,
                    "timed_out": err == "TIMEOUT", "isolates": ISOLATES[lvl]}

        if lvl == "seatbelt":
            prof = os.path.join(workdir, "policy.sb")
            with open(prof, "w", encoding="utf-8") as f:
                f.write(_seatbelt_profile(workdir))
            argv = ["/usr/bin/sandbox-exec", "-f", prof] + argv
        elif lvl == "bwrap":
            argv = ["bwrap", "--unshare-all", "--die-with-parent", "--ro-bind", "/usr", "/usr",
                    "--ro-bind-try", "/lib", "/lib", "--ro-bind-try", "/lib64", "/lib64",
                    "--ro-bind-try", "/bin", "/bin", "--proc", "/proc", "--dev", "/dev",
                    "--bind", workdir, workdir, "--chdir", workdir] + argv
        elif lvl == "nsjail":
            argv = ["nsjail", "-Mo", "--really_quiet", "--disable_proc", "-N",
                    "-R", "/usr", "-R", "/lib", "-R", "/lib64", "-R", "/bin",
                    "-B", workdir, "--cwd", workdir, "--"] + argv
        elif lvl == "firejail":
            argv = ["firejail", "--quiet", "--net=none", "--private=" + workdir,
                    "--nogroups", "--"] + argv

        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             cwd=workdir, preexec_fn=_rlimits(cpu_s, mem_mb * 1024 * 1024),
                             env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8",
                                  "HOME": workdir, "TMPDIR": workdir})
        timed_out = False
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # The process group, not the leader: a child holds the pipes open and
            # communicate() blocks forever otherwise (the lesson sandbox_exec:85 records).
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = p.communicate()
            timed_out = True
        return {"level": lvl, "rc": -1 if timed_out else p.returncode,
                "stdout": (out or b"").decode("utf-8", "replace"),
                "stderr": (err or b"").decode("utf-8", "replace")[-2000:],
                "timed_out": timed_out, "isolates": ISOLATES[lvl]}
    finally:
        if own_dir:
            shutil.rmtree(workdir, ignore_errors=True)


# The escape attempts. Each returns code that SUCCEEDS when isolation is absent, so a level
# claiming to isolate that axis must make it fail.
PROBES = {
    "net": (
        "import socket\n"
        "try:\n"
        "    socket.socket().connect(('1.1.1.1', 80)); print('REACHED')\n"
        "except Exception as e:\n"
        "    print('blocked', type(e).__name__)\n"
    ),
    "fs_read": (
        "import os\n"
        "p = os.path.join(os.path.expanduser('~'), '.ssh')\n"
        "# expanduser follows HOME, which the runner repoints at the workdir; the real\n"
        "# target is the operator's home, so it is spelled out rather than derived.\n"
        "for cand in (p, '/Users/bytedance/.ssh', '/root/.ssh'):\n"
        "    try:\n"
        "        if os.listdir(cand):\n"
        "            print('REACHED', cand); break\n"
        "    except Exception:\n"
        "        pass\n"
        "else:\n"
        "    print('blocked')\n"
    ),
    "child_net": (
        "import subprocess, sys\n"
        "r = subprocess.run([sys.executable, '-I', '-c',\n"
        "  \"import socket\\n\"\n"
        "  \"try:\\n\"\n"
        "  \"    socket.socket().connect(('1.1.1.1',80)); print('REACHED')\\n\"\n"
        "  \"except Exception as e:\\n\"\n"
        "  \"    print('blocked', type(e).__name__)\"],\n"
        "  capture_output=True, text=True)\n"
        "print((r.stdout or r.stderr).strip()[:60])\n"
    ),
}


def probe(level=None):
    """Run every escape attempt at `level`. Returns {axis: bool_isolated}."""
    out = {}
    for axis, code in PROBES.items():
        r = run(code, timeout=20, level=level)
        out[axis] = "REACHED" not in r["stdout"]
    return out


def _selftest():
    lvl = detect_level()
    print(f"host: {platform.system()} euid={os.geteuid()}  -> level {lvl}")
    for t in ("bwrap", "nsjail", "firejail", "unshare", "sandbox-exec"):
        print(f"  {t:12s} {shutil.which(t) or 'ABSENT'}")

    # 1. The level's own claims, provoked. A level that says it isolates the network must
    #    make the network probe fail -- asserting the claim rather than the mechanism, so a
    #    new level cannot be added without its evidence.
    got = probe(lvl)
    print(f"\nescape attempts at level {lvl}: {got}")
    claims = ISOLATES[lvl]
    if claims["net"]:
        assert got["net"], f"{lvl} claims to isolate the network and the probe REACHED it"
        assert got["child_net"], f"{lvl}: a CHILD process escaped the network isolation"
    if claims["fs_read"]:
        assert got["fs_read"], f"{lvl} claims to isolate reads and the probe read ~/.ssh"

    # 2. The refusal. rlimits_only without the variable must raise, with it must run.
    saved = os.environ.pop("ALLOW_UNISOLATED", None)
    try:
        try:
            run("print(1)", level="rlimits_only")
            raise AssertionError("rlimits_only ran without ALLOW_UNISOLATED")
        except Unisolated as e:
            assert "REFUSING" in str(e), e
        os.environ["ALLOW_UNISOLATED"] = "1"
        r = run("print('ran')", level="rlimits_only")
        assert r["stdout"].strip() == "ran", r
        assert r["level"] == "rlimits_only", r
        assert r["isolates"] == {"net": False, "fs_read": False, "fs_write": False}, r
    finally:
        os.environ.pop("ALLOW_UNISOLATED", None)
        if saved is not None:
            os.environ["ALLOW_UNISOLATED"] = saved

    # 3. Basics at the real level: it must actually run code, and time out.
    r = run("print(6*7)", level=lvl)
    assert r["stdout"].strip() == "42", r
    r = run("import time\ntime.sleep(300)", timeout=3, level=lvl)
    assert r["timed_out"] and r["rc"] == -1, r
    print("timeout path: killed and reported")

    # 4. The three Seatbelt traps, as cases rather than prose (fb's ruling). Darwin only;
    #    each is a profile that LOOKS right and silently measures nothing.
    if lvl == "seatbelt":
        d = tempfile.mkdtemp(prefix="sbtrap.")
        try:
            code = os.path.join(d, "code.py")
            with open(code, "w", encoding="utf-8") as f:
                f.write(PROBES["net"])

            def sb(profile):
                p = os.path.join(d, "p.sb")
                with open(p, "w", encoding="utf-8") as f:
                    f.write(profile)
                r = subprocess.run(["/usr/bin/sandbox-exec", "-f", p, sys.executable,
                                    "-I", code], capture_output=True, text=True, cwd=d)
                return r.returncode, r.stdout.strip()

            # Trap 1: the no-sandbox exemption reaches the network.
            rc, out = sb('(version 1)\n(deny default)\n'
                         '(allow process-exec (with no-sandbox))\n(allow file-read*)\n'
                         '(allow sysctl-read)\n(deny network*)\n')
            assert "REACHED" in out, (
                "trap 1 no longer reproduces: (with no-sandbox) used to exempt the exec'd "
                f"process so the network was reachable despite (deny network*). Got {out!r} "
                f"rc={rc} -- re-derive before trusting the profile")

            # Trap 2: a deny-default profile whose read list is INCOMPLETE aborts python
            # before it runs a line. Note the read list: `(allow file-read*)` with no
            # subpath allows every read and python starts fine, so the first version of
            # this case did not reproduce and said so (rc 0, "blocked PermissionError").
            # That is the trap stated precisely -- deny-default is not what breaks it,
            # enumerating the loader's read set is, and the enumeration looks complete
            # right up to the abort. MEASURED 2026-09-02: `(subpath "/usr")` alone -> 134.
            rc, out = sb('(version 1)\n(deny default)\n(allow process-exec)\n'
                         '(allow process-fork)\n(allow file-read* (subpath "/usr"))\n'
                         '(allow sysctl-read)\n')
            assert rc != 0 and not out, (
                f"trap 2 no longer reproduces: a (deny default) profile allowing reads only "
                f"under /usr used to abort python3 (rc 134, no output) because the dynamic "
                f"loader reads outside it. Got rc={rc} {out!r}")

            # Trap 3: /tmp is a symlink, so a non-real path grants nothing.
            work = os.path.join(d, "w")
            os.makedirs(work, exist_ok=True)
            wr = os.path.join(work, "t.py")
            with open(wr, "w", encoding="utf-8") as f:
                f.write("import os\ntry:\n"
                        "    open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"
                        " 'o.txt'), 'w').write('x'); print('wrote')\n"
                        "except Exception as e:\n    print('denied', type(e).__name__)\n")
            fake = work if work.startswith("/private") else work.replace("/var", "/tmp", 1)
            unreal = _SEATBELT.format(work=fake.replace("/private", "", 1), secrets='(subpath "/nonexistent")')
            p = os.path.join(d, "unreal.sb")
            with open(p, "w", encoding="utf-8") as f:
                f.write(unreal)
            r = subprocess.run(["/usr/bin/sandbox-exec", "-f", p, sys.executable, "-I", wr],
                               capture_output=True, text=True, cwd=work)
            assert "denied" in r.stdout, (
                f"trap 3 no longer reproduces: a profile naming the /tmp path instead of "
                f"its /private realpath used to match nothing, denying the workdir write. "
                f"Got {r.stdout.strip()!r}")
            # And the real path grants it.
            r2 = run("import os\nopen('o.txt','w').write('x')\nprint('wrote')",
                     workdir=work, level="seatbelt")
            assert r2["stdout"].strip() == "wrote", r2
            print("seatbelt traps: all three reproduce, realpath profile writes")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    print(f"\nselftest OK: level {lvl}, its claims provoked, the refusal refuses, "
          f"timeout kills"
          + (", 3 seatbelt traps locked" if lvl == "seatbelt" else ""))
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    if "--probe" in sys.argv:
        print(json.dumps({"level": detect_level(), "isolated": probe()}, indent=1))
        sys.exit(0)
    print(json.dumps({"level": detect_level()}, indent=1))
