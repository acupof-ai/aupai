#!/usr/bin/env python3
"""run_ddp.sh refuses a direct call and accepts a launched one (de-60).

The defect: `harness launch` writes the experiments row, acquires the card claim and arms the
monitor. Calling run_ddp.sh straight from a shell skips all three and looks identical while it
happens -- runs/friction.jsonl 2026-09-04T09:20Z (b0), a 2-arm A/B with no row, no claim, no
watchdog, cards held invisibly.

Six worlds. The gate is checked by RUNNING the script with torchrun stubbed, not by grepping it:
the question is whether the refusal fires before the launch, and only executing it answers that.

  1  no marker, no escape    -> REFUSE, and torchrun must NOT have run
  2  AUPAI_LAUNCHED_BY set   -> runs
  3  ALLOW_DIRECT_RUN=1      -> runs (the named escape)
  4  marker EMPTY            -> REFUSE. An exported-but-empty var is the shape a wrapper produces
                               when it forwards a variable it never set, and `-z` is what makes
                               that a refusal rather than a pass.
  5  ALLOW_DIRECT_RUN=0      -> REFUSE. The flag is checked against "1", not for being set.
  6  the message names the fix -> the refusal is only useful if it says what to run instead.

The stamp/drift gates above the marker are bypassed with ALLOW_UNSYNCED=1: this test's subject is
the launcher marker, and a world that also had to satisfy the sync stamp would be testing that.

    python3 scripts/test_run_ddp_launcher.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "run_ddp.sh")


def world():
    """A tree holding the REAL run_ddp.sh, with torchrun stubbed so a launch is observable."""
    d = tempfile.mkdtemp(prefix="de60_")
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    os.makedirs(os.path.join(d, "bin"), exist_ok=True)
    shutil.copy(REAL, os.path.join(d, "run_ddp.sh"))
    os.chmod(os.path.join(d, "run_ddp.sh"), 0o755)
    stub = os.path.join(d, "bin", "torchrun")
    with open(stub, "w", encoding="utf-8") as f:
        f.write('#!/bin/bash\ntouch "$LAUNCH_MARKER"\necho "torchrun stub: $*"\nexit 0\n')
    os.chmod(stub, 0o755)
    # train.py must exist for the argv to make sense; it is never executed (torchrun is stubbed).
    open(os.path.join(d, "train.py"), "w").write("# stub\n")
    return d


def run(d, env_extra):
    marker = os.path.join(d, "runs", "LAUNCHED")
    if os.path.exists(marker):
        os.remove(marker)
    env = dict(os.environ)
    env.pop("AUPAI_LAUNCHED_BY", None)
    env.pop("ALLOW_DIRECT_RUN", None)
    env["PATH"] = os.path.join(d, "bin") + os.pathsep + env["PATH"]
    env["LAUNCH_MARKER"] = marker
    env["ALLOW_UNSYNCED"] = "1"       # the sync stamp is a different gate's subject
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.update(env_extra)
    p = subprocess.run(["bash", os.path.join(d, "run_ddp.sh"), "--name", "zz_de60"],
                       capture_output=True, text=True, timeout=300, cwd=d, env=env)
    return p.returncode, (p.stdout + p.stderr), os.path.exists(marker)


bad = 0
CASES = 6

# 1. THE DEFECT: a bare direct call.
d = world()
rc, out, launched = run(d, {})
ok = rc != 0 and not launched and "not invoked by" in out
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} a direct call is REFUSED and torchrun never runs"
      + ("" if ok else f" -- rc={rc} launched={launched} {out.strip()[:120]}"))

# 2. The launcher's marker.
rc, out, launched = run(d, {"AUPAI_LAUNCHED_BY": "harness launch zz_de60"})
ok = rc == 0 and launched
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} AUPAI_LAUNCHED_BY set -> the run proceeds"
      + ("" if ok else f" -- rc={rc} launched={launched} {out.strip()[:120]}"))

# 3. The named escape.
rc, out, launched = run(d, {"ALLOW_DIRECT_RUN": "1"})
ok = rc == 0 and launched
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} ALLOW_DIRECT_RUN=1 -> the run proceeds"
      + ("" if ok else f" -- rc={rc} launched={launched} {out.strip()[:120]}"))

# 4. EMPTY marker -> refuse. `-n` instead of `-z` would pass here, and an exported-but-empty
#    variable is exactly what a wrapper forwarding an unset name produces.
rc, out, launched = run(d, {"AUPAI_LAUNCHED_BY": ""})
ok = rc != 0 and not launched
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} an EMPTY marker is REFUSED, not treated as set"
      + ("" if ok else f" -- rc={rc} launched={launched} {out.strip()[:120]}"))

# 5. ALLOW_DIRECT_RUN=0 -> refuse. Tests the comparison, not merely that the name is present.
rc, out, launched = run(d, {"ALLOW_DIRECT_RUN": "0"})
ok = rc != 0 and not launched
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} ALLOW_DIRECT_RUN=0 is REFUSED -- checked against '1'"
      + ("" if ok else f" -- rc={rc} launched={launched} {out.strip()[:120]}"))

# 6. The refusal has to say what to run instead, or it is a wall.
rc, out, _l = run(d, {})
ok = "harness.py launch" in out and "ALLOW_DIRECT_RUN=1" in out
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} the refusal names both the launcher command and the escape"
      + ("" if ok else f" -- {out.strip()[:200]}"))

shutil.rmtree(d, ignore_errors=True)
print(f"run_ddp launcher gate: {CASES - bad}/{CASES} pass")
sys.exit(1 if bad else 0)
