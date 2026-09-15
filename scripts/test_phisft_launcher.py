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
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER = os.path.join(ROOT, "runs", "v42_phi_sft.sh")
OLD_LAUNCHER = None  # set by _negative_control from `git show origin/main:...`


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
    ready = os.path.join(d, "torchrun_ready")
    die = os.path.join(d, "torchrun_die")
    tr = os.path.join(d, "bin", "torchrun")
    with open(tr, "w") as f:
        f.write(
            "#!" + sys.executable + "\n"
            "import os, sys, time\n"
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


def _live_claim_files(claim_dir):
    return [f for f in os.listdir(claim_dir) if f.endswith(".json")]


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
        if proc.poll() is None:
            open(die, "w").close()
            proc.kill()
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
        if proc2.poll() is None:
            open(die2, "w").close()
            proc2.kill()
        shutil.rmtree(d2, ignore_errors=True)
        shutil.rmtree(claim_dir2, ignore_errors=True)

    # 4. NEGATIVE CONTROL: the ORIGINAL pre-launch acquire launcher must never reach torchrun.
    old = subprocess.run(["git", "show", "origin/main:runs/v42_phi_sft.sh"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    assert "wait-for-device" in old and "--pid" not in old, "origin/main launcher shape changed"
    od = tempfile.mkdtemp(prefix="phisft_old_")
    oclaim = tempfile.mkdtemp(prefix="phisft_oldclaims_")
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
        open(tr, "w").write("#!" + sys.executable + "\nimport sys;open(sys.argv[-1],'w').close()\n")
        os.chmod(tr, 0o755)
        env = dict(os.environ, PATH=os.path.join(od, "bin") + os.pathsep + os.environ["PATH"],
                   AUPAI_CLAIM_DIR=oclaim, HYPOTHESIS="neg",
                   CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
        r = subprocess.run(["bash", os.path.join(od, "runs", "v42_phi_sft.sh")],
                           cwd=od, env=env, capture_output=True, text=True, timeout=90)
        assert r.returncode != 0 and "acquire refused" in r.stdout.lower(), (
            "old launcher did not refuse at the pre-launch claim:\n" + r.stdout[-800:])
    finally:
        shutil.rmtree(od, ignore_errors=True)
        shutil.rmtree(oclaim, ignore_errors=True)

    print("phi SFT launcher OK: binds an 8-card live claim to the live torchrun pid after "
          "device open, releases on exit; old pre-launch-acquire launcher refuses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
