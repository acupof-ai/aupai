"""The ALLOW_DIRECT_RUN population is CLOSED, and a fifth unnamed caller is refused (4c, de-60).

An escape hatch nobody counts becomes the default. Three scripts take it today and each has a
reason that is not "the gate is inconvenient":

  scripts/lr_probe.sh          does its own busy-card refusal and pins its own world; the launcher
                               would allocate the grant's whole block
  scripts/prove_resume.sh      KILLS its run 1 at step 40 on purpose -- the monitor would read that
                               as a crash and its auto-resume would fight the script's own run 2
  scripts/b0_se_launch_arm2.sh pins two cards from the caller and hands the torchrun pid to
                               card_claim itself; the launcher would override those two with the
                               whole block (§188)

The check is an ALLOWLIST, not a count: a fourth script exporting the variable FAILs here until
somebody writes down why, which is the only way the population stays closed. test_score_exit is
NOT in it -- it takes the marker (AUPAI_LAUNCHED_BY), because its subject is the end-of-run chain
that sits after the gate, and that distinction is the reason the two mechanisms are separate.

Case 4 is the one 4c asked for and the one that makes the rest mean something: an UNNAMED caller,
built here as a fresh script that does not appear in the allowlist, must still be refused. Without
it the allowlist could be satisfied by a gate that no longer refuses anything.

    python3 scripts/test_direct_run_population.py
"""

# restartable: 0.6s and it writes nothing outside one tempdir it removes itself. Case 4's only
# side effect is a stubbed torchrun inside that dir, so an interrupt loses at most the tempdir --
# rerun from the top.

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# path -> the one-line reason. Adding a path here is a decision; adding one WITHOUT a reason is
# rejected by the assertion below, so the allowlist cannot grow silently.
ALLOWED = {
    "scripts/lr_probe.sh": "does its own busy-card refusal and pins its own world; the launcher would allocate the "
    "grant's whole block",
    "scripts/prove_resume.sh": "kills its own run 1 at step 40 on purpose; the monitor would read that as a crash and "
    "auto-resume would fight this script's run 2",
    "scripts/b0_se_launch_arm2.sh": "pins two cards from the caller and hands the torchrun pid to card_claim itself; the "
    "launcher would override those two with the whole block (§188)",
}

bad = 0
CASES = 4

# 1. THE POPULATION IS EXACTLY THE ALLOWLIST. Scanned over tracked files, not a fixed list of
#    directories: a launcher added at the repo root would be missed by a scripts/-only glob.
found = set()
listing = subprocess.run(
    ["git", "-C", ROOT, "ls-files"], capture_output=True, text=True, timeout=120
).stdout.split()
for rel in listing:
    if not rel.endswith((".sh", ".py")):
        continue
    p = os.path.join(ROOT, rel)
    try:
        with open(p, encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except OSError:
        continue
    # An assignment or export, not a mention: run_ddp.sh READS the variable and the tests below
    # name it in prose, neither of which is a caller taking the escape.
    if re.search(r"^\s*(export\s+)?ALLOW_DIRECT_RUN=", txt, re.M) or re.search(
        r"ALLOW_DIRECT_RUN=1\s+\./run_ddp", txt
    ):
        found.add(rel)

# run_ddp.sh itself reads it; this test names it in prose.
found -= {"run_ddp.sh", "scripts/test_direct_run_population.py"}
extra, gone = sorted(found - set(ALLOWED)), sorted(set(ALLOWED) - found)
ok = not extra and not gone
bad += 0 if ok else 1
print(
    f"  {'ok  ' if ok else 'BUG '} the escape population is exactly the {len(ALLOWED)} allowlisted "
    f"caller(s)" + ("" if ok else f" -- unlisted: {extra}; listed but no longer taking it: {gone}")
)

# 2. EVERY ALLOWED PATH CARRIES A REASON, and states it at the call site too. A reason that lives
#    only in this file is invisible to whoever next reads the script.
missing_reason = [p for p, why in ALLOWED.items() if len(why.split()) < 8]
no_comment = []
for rel in ALLOWED:
    with open(os.path.join(ROOT, rel), encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    i = txt.find("ALLOW_DIRECT_RUN=1")
    if i < 0 or "de-60" not in txt[max(0, i - 700) : i]:
        no_comment.append(rel)
ok = not missing_reason and not no_comment
bad += 0 if ok else 1
print(
    f"  {'ok  ' if ok else 'BUG '} each allowed caller carries a reason here AND at its call site"
    + ("" if ok else f" -- thin reason: {missing_reason}; no de-60 note above the export: {no_comment}")
)

# 3. THE ALLOWED CALLERS ARE STILL REAL. A path that no longer exists would let the allowlist
#    vouch for a caller nobody can read.
absent = [p for p in ALLOWED if not os.path.exists(os.path.join(ROOT, p))]
ok = not absent
bad += 0 if ok else 1
print(f"  {'ok  ' if ok else 'BUG '} every allowlisted path exists" + ("" if ok else f" -- absent: {absent}"))

# 4. A FIFTH, UNNAMED CALLER IS STILL REFUSED. The case that makes the three above mean something:
#    the gate must refuse a script that is not in the allowlist, so run the real run_ddp.sh from a
#    world where nothing set the marker.
d = tempfile.mkdtemp(prefix="de60pop_")
try:
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    os.makedirs(os.path.join(d, "bin"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "run_ddp.sh"), os.path.join(d, "run_ddp.sh"))
    marker = os.path.join(d, "runs", "LAUNCHED")
    with open(os.path.join(d, "bin", "torchrun"), "w", encoding="utf-8") as f:
        f.write(f'#!/bin/bash\ntouch "{marker}"\nexit 0\n')
    os.chmod(os.path.join(d, "bin", "torchrun"), 0o755)
    # The unnamed caller: a wrapper that does what the three allowed ones do, minus the escape.
    with open(os.path.join(d, "newprobe.sh"), "w", encoding="utf-8") as f:
        f.write('#!/bin/bash\ncd "$(dirname "$0")"\nNGPU=2 ./run_ddp.sh --name zz_new\n')
    os.chmod(os.path.join(d, "newprobe.sh"), 0o755)
    env = dict(os.environ)
    env.pop("AUPAI_LAUNCHED_BY", None)
    env.pop("ALLOW_DIRECT_RUN", None)
    env["PATH"] = os.path.join(d, "bin") + os.pathsep + env["PATH"]
    env["ALLOW_UNSYNCED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    r = subprocess.run(
        ["bash", os.path.join(d, "newprobe.sh")], capture_output=True, text=True, timeout=300, cwd=d, env=env
    )
    launched = os.path.exists(marker)
    ok = r.returncode != 0 and not launched and "not invoked by" in (r.stdout + r.stderr)
    bad += 0 if ok else 1
    print(
        f"  {'ok  ' if ok else 'BUG '} a FIFTH unnamed caller is still REFUSED and never launches"
        + ("" if ok else f" -- rc={r.returncode} launched={launched} {(r.stdout + r.stderr).strip()[:120]}")
    )
finally:
    shutil.rmtree(d, ignore_errors=True)

print(f"direct-run population: {CASES - bad}/{CASES} pass")
sys.exit(1 if bad else 0)
