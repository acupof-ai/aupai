#!/usr/bin/env python3
"""v42_phi_sft.sh claims the 8 cards by binding them to the LIVE torchrun pid, after it
holds a device -- not before torchrun starts, and not per rank.

Why this is the only shape that works (measured 2026-09-15):
  * The original launcher ran `card_claim acquire --wait-for-device 300` BEFORE torchrun:
    no GPU-holding descendant exists, so it dead-waits and exits without training.
  * sft_math.py cannot self-claim one card per rank. Every torchrun rank inherits the SAME
    CUDA_VISIBLE_DEVICES=0..7, and card_claim.acquire's exec-time CVD check REFUSES a claim
    whose cards differ from the holder's visible devices:
      "visible ['0'..'7'] vs claimed ['0'] ... an orphan behind a healthy claim".
    (Reproduced on the pod: claim_my_cards(cards=['0']) under CVD=0..7 raises SystemExit.)
  * The supported 8-card block claim is the torchrun PARENT, which holds nvidia device fds
    itself and whose CVD is exactly 0..7. `acquire --pid <torchrun> --require-device`
    binds all eight to that one pid; it lapses on torchrun exit and the launcher trap also
    releases it. This is the manual bind fb used to protect the live run.

What this test RUNS (torchrun is a non-shell python stub; the REAL card_claim.py runs
against a temp AUPAI_CLAIM_DIR; no GPU, macOS has no /proc so require-device fails open
exactly as the guard documents):
  1. launcher starts torchrun, then a live 8-card claim appears bound to the torchrun
     stub's OWN pid (not the shell, not an ephemeral pid);
  2. if torchrun dies before the claim binds, the launcher refuses, marks exp fail, and
     leaves NO training running;
  3. after torchrun exits 0 and the launcher unwinds, the claim is RELEASED (trap), so a
     finished run does not read held.
  4. negative control: the OLD launcher never reaches torchrun (its pre-launch acquire
     refuses a shell with no device-holding descendant).

    python3 scripts/test_phisft_launcher.py
"""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER = os.path.join(ROOT, "runs", "v42_phi_sft.sh")

# The PRE-FIX launcher, frozen as a fixture rather than `git show origin/main:...`. Reading
# the live branch rotted the negative control the moment #367 merged: origin/main then held
# the FIXED launcher, so the assertion "old has wait-for-device and no --pid" failed on every
# later merge. This literal is the exact buggy shape -- an acquire --wait-for-device run
# BEFORE torchrun, which refuses (no GPU-holding descendant) and exits without training.
# Keep it self-contained: it must reach the acquire and print "card claim acquire refused".
_OLD_BUGGY_LAUNCHER = """#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
# shellcheck disable=SC1091
source eval/_devs.sh 8
CARDS=$(IFS=,; echo "${_DEVS[*]}")
[ -z "${HYPOTHESIS:-}" ] && { echo "REFUSING no hypothesis"; exit 2; }
[ -f ckpt_v41_r3_0914.pt ] || { echo "REFUSING no ckpt"; exit 2; }
[ -f data/sft/sft_phi_codeexercises_v42_65m_0914.pt ] || { echo "REFUSING no pack"; exit 2; }
CUDA_VISIBLE_DEVICES= python3 sft_math.py --check_pack
python3 scripts/exp.py start --name v42_phi_sft --cmd x --hypothesis "$HYPOTHESIS" >/dev/null
python3 scripts/card_claim.py acquire --name v42_phi_sft --cards "$CARDS" \\
  --note old --wait 0 --wait-for-device 2 || {
  echo "REFUSING to launch: card_claim acquire refused on $CARDS"
  python3 scripts/exp.py done --name v42_phi_sft --status fail --result "card claim refused"
  exit 1
}
torchrun --nproc_per_node=8 sft_math.py
"""


def _build_tree(claim_dir):
    d = tempfile.mkdtemp(prefix="phisft_")
    for sub in ("runs", "scripts", "eval", "bin", "data/sft"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    shutil.copy(LAUNCHER, os.path.join(d, "runs", "v42_phi_sft.sh"))
    os.chmod(os.path.join(d, "runs", "v42_phi_sft.sh"), 0o755)
    shutil.copy(os.path.join(ROOT, "eval", "_devs.sh"), os.path.join(d, "eval", "_devs.sh"))
    # The REAL claim code so the exercise covers the exec-time CVD / shell / require-device
    # guards, not a stub that agrees with everything.
    shutil.copy(os.path.join(ROOT, "scripts", "card_claim.py"),
                os.path.join(d, "scripts", "card_claim.py"))

    import torch
    torch.save({"input_ids": torch.zeros(4, 2, dtype=torch.long),
                "labels": torch.zeros(4, 2, dtype=torch.long),
                "vocab_id": "stub-vocab"},
               os.path.join(d, "data/sft/sft_phi_codeexercises_v42_65m_0914.pt"))
    open(os.path.join(d, "ckpt_v41_r3_0914.pt"), "w").close()

    # check_pack gate (called directly by the launcher) is a no-op; exp records its argv so a
    # test can assert it was closed as fail after an early torchrun death.
    open(os.path.join(d, "sft_math.py"), "w").write("import sys\nsys.exit(0)\n")
    open(os.path.join(d, "scripts", "exp.py"), "w").write(
        "import os, sys\n"
        "p=os.environ.get('EXP_LOG')\n"
        "open(p,'a').write(' '.join(sys.argv[1:])+chr(10)) if p else None\n"
        "sys.exit(0)\n")
    # Post-training humaneval step is absent; launcher exits non-zero there but the claim
    # trap still fires -- that path is not under test, so provide a harmless acceptor.
    os.makedirs(os.path.join(d, "eval"), exist_ok=True)
    open(os.path.join(d, "eval", "humaneval_gen.py"), "w").write("import sys\nsys.exit(0)\n")

    # torchrun = a NON-SHELL python process (a bash stub is refused as a shell holder). It
    # records argv, then either blocks until a die file appears (happy path) or exits
    # immediately non-zero (early-death world), per TORCHRUN_BEHAVIOR.
    #
    # It also HOLDS an fd whose path names nvidia, standing in for the /dev/nvidia* fds a real
    # torchrun holds once it is on a card. The launcher's claim is --require-device, and
    # card_claim.nvidia_fds counts an fd by `"nvidia" in os.readlink(...)`: a stub holding none
    # is refused on Linux, where /proc is readable, while macOS abstains (None, never refuses)
    # and the test passed for the wrong reason. Holding one makes the stub the same OBSERVABLE
    # shape as the process the assertion is about, without weakening the predicate.
    dev_fd_path = os.path.join(d, "nvidia0")
    ready = os.path.join(d, "torchrun_ready")
    die = os.path.join(d, "torchrun_die")
    tr = os.path.join(d, "bin", "torchrun")
    with open(tr, "w") as f:
        f.write(
            "#!" + sys.executable + "\n"
            "import os, sys, time\n"
            f"_devfd = os.open({dev_fd_path!r}, os.O_CREAT | os.O_RDONLY)\n"
            f"open({ready!r}, 'w').close()\n"
            "with open(os.path.join(os.path.dirname(__file__), '..', 'torchrun_argv'), 'w') as a:\n"
            "    a.write(' '.join(sys.argv[1:]))\n"
            "if os.environ.get('TORCHRUN_BEHAVIOR') == 'die':\n"
            "    sys.exit(7)\n"
            f"while not os.path.exists({die!r}):\n"
            "    time.sleep(0.05)\n"
            "sys.exit(int(os.environ.get('TORCHRUN_RC', '0')))\n")
    os.chmod(tr, 0o755)
    return d, ready, die


_LEAKED = []  # pids any world failed to reap; asserted in main() so it cannot mask an error


def _live_claim_files(claim_dir):
    return [f for f in os.listdir(claim_dir) if f.endswith(".json")]


def _reap(popen):
    """Kill `popen` and EVERY descendant it started, children before parents, and wait.

    THE TEST LEAKED ITS OWN STUBS. `proc.kill()` signals the bash launcher only; the torchrun
    stub is a GRANDCHILD (`bash` execs it), so it survived every path that did not let bash
    unwind on its own -- and a surviving stub is a live process holding a claim dir that has
    already been rmtree'd. 3b caught one: ppid=1, cwd pointing at a deleted mkdtemp, idling
    9.5h. That is the test harness leaking, which is a DIFFERENT thing from the launcher under
    test leaking, and the whole point of the assertion this file exists to make is to tell
    those two apart. A harness that orphans its own stub makes "launcher left torchrun
    running" unfalsifiable.

    CHILDREN BEFORE PARENTS. Signalling the parent first reparents the child to init (ppid=1)
    and loses the pid we were about to kill -- which is exactly the shape 3b observed.

    Returns the pids that were still alive when it gave up, so the caller can ASSERT on the
    result instead of trusting this function ran. An empty list is the healthy answer.
    """
    if popen is None:
        return []
    kids = []
    if popen.poll() is None:
        # Snapshot descendants BEFORE signalling anything: the parent's exit reparents them.
        try:
            out = subprocess.run(["pgrep", "-P", str(popen.pid)], capture_output=True,
                                 text=True).stdout
            kids = [int(x) for x in out.split() if x.strip().isdigit()]
        except (OSError, ValueError):
            kids = []
        for pid in kids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            popen.kill()
        except OSError:
            pass
        for pid in kids:
            for _ in range(100):
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
                time.sleep(0.02)
    try:
        popen.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    return [pid for pid in kids if _alive(pid)]


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _env_worlds():
    """The two host shapes this file has to tell apart, driven through the seam card_claim's own
    selftest uses -- so both are exercised on every host, laptop included.

    A GREEN RUN MUST NAME THE SHAPE IT RAN IN. macOS has no /proc: the predicate abstains (None)
    and every device assertion is vacuous there. Linux has /proc: a stub holding nothing reads 0
    and is refused. Only the second outcome is the one the launcher is gated on, and letting it
    go unexercised on the laptop is exactly how the world1 failure reached CI green. So both are
    built here out of real directories and real symlinks.
    """
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import card_claim
    saved = card_claim.PROC_ROOT
    proot = tempfile.mkdtemp(prefix="phisft_env_")
    try:
        card_claim.PROC_ROOT = proot
        # (a) NO /proc entry for this pid: unreadable -> None -> the predicate has no opinion.
        assert card_claim.nvidia_fds(os.getpid()) is None, (
            "a pid absent from /proc must read None (abstain), not 0 (refuse): conflating the "
            "two refuses every claim on a host that simply cannot answer")
        # (b) /proc entry present, no nvidia fd: readable, zero device fds -> 0 -> refusal bites.
        fd = os.path.join(proot, "4242", "fd")
        os.makedirs(fd, exist_ok=True)
        os.symlink("/dev/null", os.path.join(fd, "3"))
        assert card_claim.nvidia_fds(4242) == 0, (
            "a readable pid holding no device must read 0, not None: the refusal is what the "
            "launcher's --require-device depends on")
        print("env worlds: no-/proc -> None (abstain); /proc-without-nvidia -> 0 (refuse)")
    finally:
        card_claim.PROC_ROOT = saved
        shutil.rmtree(proot, ignore_errors=True)


def _run(d, claim_dir, env_extra=None):
    env = dict(os.environ,
               PATH=os.path.join(d, "bin") + os.pathsep + os.environ["PATH"],
               AUPAI_CLAIM_DIR=claim_dir,
               HYPOTHESIS="selftest world",
               CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
               TORCHRUN_RC="0",
               EXP_LOG=os.path.join(d, "exp_calls.log"))
    if env_extra:
        env.update(env_extra)
    return subprocess.Popen(["bash", os.path.join(d, "runs", "v42_phi_sft.sh")],
                            cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main():
    claim_dir = tempfile.mkdtemp(prefix="phisft_claims_")
    d, ready, die = _build_tree(claim_dir)
    proc = _run(d, claim_dir)
    try:
        # 1. torchrun starts and a live claim binds to its pid before training is left
        #    unattended.
        for _ in range(200):
            if os.path.exists(ready):
                break
            time.sleep(0.05)
        assert os.path.exists(ready), "torchrun stub never started"
        argv = open(os.path.join(d, "torchrun_argv")).read()
        assert "--nproc_per_node=8" in argv and "sft_math.py" in argv, argv

        bound = None
        for _ in range(200):
            files = _live_claim_files(claim_dir)
            if files:
                bound = files[0]
                break
            time.sleep(0.05)
        # WHERE THE PREDICATE ABSTAINS, SAY SO. On macOS there is no /proc, so nvidia_fds
        # returns None and --require-device cannot refuse; the claim lands whatever the stub
        # holds. The assert below therefore cannot fail here for the reason it fails on Linux,
        # and a green line would otherwise read as "the acceptance path was exercised". It is
        # not: that path is pinned by card_claim's own w1_dev world on every machine, and by
        # ubuntu CI here. Printed, never silent.
        if not os.path.isdir("/proc"):
            print("SKIP require-device acceptance path linux-only on this host "
                  "(no /proc): covered by card_claim w1_dev + CI ubuntu")
        assert bound, "launcher left torchrun running with NO live card claim"
        import json
        claim = json.load(open(os.path.join(claim_dir, bound)))
        assert sorted(map(str, claim["cards"])) == [str(i) for i in range(8)], claim["cards"]
        # The claimed pid is the live torchrun stub, a python process -- find it by scanning
        # children; assert it is alive and the argv-bearing process, not the shell.
        stub_pid = int(claim["pid"])
        assert stub_pid != proc.pid, "claim bound to the launcher shell, not torchrun"
        os.kill(stub_pid, 0)  # alive

        # 2/3. let torchrun finish 0 -> launcher unwinds -> trap releases the claim.
        open(die, "w").close()
        out, _ = proc.communicate(timeout=60)
        for _ in range(100):
            if not _live_claim_files(claim_dir):
                break
            time.sleep(0.05)
        assert not _live_claim_files(claim_dir), (
            "torchrun exit + trap left a stale claim:\n" + out.decode()[-800:])
    finally:
        _LEAKED.extend(_reap(proc))
        shutil.rmtree(d, ignore_errors=True)

    # EARLY-DEATH world (fb condition 2): torchrun exits non-zero before any claim binds.
    # The launcher must refuse non-zero, close the exp row as fail, and leave NO claim file.
    d2, ready2, die2 = _build_tree(claim_dir2 := tempfile.mkdtemp(prefix="phisft_dieclaims_"))
    proc2 = _run(d2, claim_dir2, {"TORCHRUN_BEHAVIOR": "die"})
    try:
        for _ in range(200):
            if os.path.exists(ready2):
                break
            time.sleep(0.05)
        out2, _ = proc2.communicate(timeout=60)
        assert proc2.returncode != 0, "launcher succeeded though torchrun died pre-claim"
        assert not _live_claim_files(claim_dir2), (
            "early torchrun death left a half-built claim:\n" + out2.decode()[-600:])
        exp_log = open(os.path.join(d2, "exp_calls.log")).read()
        assert "done" in exp_log and "--status" in exp_log and "fail" in exp_log, (
            "early death was not recorded as exp fail:\n" + exp_log)
    finally:
        _LEAKED.extend(_reap(proc2))
        shutil.rmtree(d2, ignore_errors=True)
        shutil.rmtree(claim_dir2, ignore_errors=True)

    # 4. NEGATIVE CONTROL: the frozen pre-launch-acquire launcher must never reach torchrun.
    old = _OLD_BUGGY_LAUNCHER
    assert "wait-for-device" in old and "--pid" not in old, \
        "the frozen old-launcher fixture lost its buggy shape"
    od = tempfile.mkdtemp(prefix="phisft_old_")
    oclaim = tempfile.mkdtemp(prefix="phisft_oldclaims_")
    proc4 = None
    try:
        for sub in ("runs", "scripts", "eval", "bin", "data/sft"):
            os.makedirs(os.path.join(od, sub), exist_ok=True)
        with open(os.path.join(od, "runs", "v42_phi_sft.sh"), "w") as f:
            f.write(old)
        os.chmod(os.path.join(od, "runs", "v42_phi_sft.sh"), 0o755)
        shutil.copy(os.path.join(ROOT, "eval", "_devs.sh"), os.path.join(od, "eval", "_devs.sh"))
        shutil.copy(os.path.join(ROOT, "scripts", "card_claim.py"),
                    os.path.join(od, "scripts", "card_claim.py"))
        os.makedirs(os.path.join(od, "scripts"), exist_ok=True)
        # every gate before the acquire passes; torchrun marker must never appear
        for name, body in (("exp.py", "import sys;sys.exit(0)\n"),
                           ("sft_math.py", "import sys;sys.exit(0)\n")):
            p = os.path.join(od, "scripts", name) if name == "exp.py" else os.path.join(od, name)
            open(p, "w").write(body)
        import torch
        torch.save({"input_ids": torch.zeros(2, 2, dtype=torch.long),
                    "labels": torch.zeros(2, 2, dtype=torch.long), "vocab_id": "v"},
                   os.path.join(od, "data/sft/sft_phi_codeexercises_v42_65m_0914.pt"))
        open(os.path.join(od, "ckpt_v41_r3_0914.pt"), "w").close()
        tr = os.path.join(od, "bin", "torchrun")
        # The stub WRITES A MARKER when reached, so the assertion below can prove torchrun was
        # never entered rather than inferring it from the absence of output.
        open(tr, "w").write("#!" + sys.executable + "\nimport sys, os\n"
                            f"open({os.path.join(od, 'torchrun_reached')!r}, 'w').close()\n")
        os.chmod(tr, 0o755)
        env = dict(os.environ, PATH=os.path.join(od, "bin") + os.pathsep + os.environ["PATH"],
                   AUPAI_CLAIM_DIR=oclaim, HYPOTHESIS="neg",
                   CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
        # THE DEVICE-WAIT SEGMENT IS WHERE THE TWO HOSTS DIVERGE, so name it instead of letting
        # a fast run imply the path was exercised. The fixture's `--wait-for-device 2` resolves
        # the shell to a device-holding descendant; without /proc the resolver cannot follow at
        # all and refuses from its no-opinion branch, and with /proc it polls and refuses from
        # its TIMEOUT branch. Either way the REFUSAL runs and is asserted below -- what is
        # skipped (on macOS only) is the poll.
        if not os.path.isdir("/proc"):
            print("SKIP device-wait segment of the negative control (no /proc): the resolver "
                  "cannot follow a shell here, so it refuses on its no-opinion branch; the "
                  "refusal itself is still asserted (Linux exercises the poll: CI ubuntu)")
        proc4 = subprocess.Popen(["bash", os.path.join(od, "runs", "v42_phi_sft.sh")],
                                 cwd=od, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
        out4, _ = proc4.communicate(timeout=90)
        # ASSERT THE SHAPE, NOT A MESSAGE THAT ACCOMPANIES IT: the old launcher must exit
        # non-zero AND say it refused. A fixture that stopped refusing reds here, which is what
        # keeps this control falsifiable under the SKIP above.
        assert proc4.returncode != 0 and "acquire refused" in out4.decode().lower(), (
            "old launcher did not refuse at the pre-launch claim:\n" + out4.decode()[-800:])
        # NOTHING RAN. The whole point of the frozen shape is that the pre-launch acquire
        # refuses, so torchrun is never reached. A marker written by the stub would prove the
        # opposite, so its absence is asserted rather than assumed.
        assert not os.path.exists(os.path.join(od, "torchrun_reached")), (
            "old launcher REACHED torchrun despite being supposed to refuse pre-launch")
    finally:
        if proc4 is not None:
            _LEAKED.extend(_reap(proc4))
        shutil.rmtree(od, ignore_errors=True)
        shutil.rmtree(oclaim, ignore_errors=True)

    _env_worlds()
    # THE HARNESS MUST NOT LEAK ITS OWN STUBS. Without this the test could pass while orphaning
    # a torchrun of its own -- which is what 3b caught (ppid=1, deleted cwd, 9.5h) -- and then
    # "launcher left torchrun running" would be asserting about a process this file started.
    assert not _LEAKED, (
        f"test harness leaked subprocess(es) {_LEAKED}: a survivor here makes every "
        "'launcher left torchrun running' assertion unfalsifiable")

    print("phi SFT launcher OK: binds an 8-card live claim to the live torchrun pid after "
          "device open, releases on exit; old pre-launch-acquire launcher refuses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
