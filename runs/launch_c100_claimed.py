#!/usr/bin/env python3
"""Claim all GPUs, supervise the neutral Stage-3 launcher, and record its exit.

This is an operational bridge while the pod's historical drift gate is repaired.
The stable parent PID owns the claim for the full child lifetime. Obsolete SFT
guards then fail closed on the same card-claim lock.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = "/work/aupai"
NAME = "c100_seg1"
CARDS = [str(i) for i in range(8)]

os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import card_claim  # noqa: E402

ok, message = card_claim.acquire(
    NAME,
    CARDS,
    wait=0,
    note="Engram Stage-3: 10B segment to step 66767",
    pid=os.getpid(),
)
print(message, flush=True)
if not ok:
    raise SystemExit(1)

env = os.environ.copy()
env.update(
    {
        "CUDA_VISIBLE_DEVICES": ",".join(CARDS),
        "NGPU": "8",
        "AUPAI_LAUNCHED_BY": f"claimed launcher {NAME}",
        "AUPAI_CLAIMED_BY": NAME,
        "AUPAI_TOKEN_CACHE_DIR": "/data00",
        "ALLOW_UNSYNCED": "1",
        "PYTHONUNBUFFERED": "1",
    }
)

child = subprocess.Popen(
    ["bash", "runs/c100_seg1.sh"],
    env=env,
    cwd=ROOT,
    start_new_session=True,
)
with open("runs/c100_seg1.child.pid", "w", encoding="utf-8") as handle:
    handle.write(f"{child.pid}\n")
rc = child.wait()
with open("runs/c100_seg1.rc", "w", encoding="utf-8") as handle:
    handle.write(f"{rc}\n")
card_claim.release(NAME)
raise SystemExit(rc)
