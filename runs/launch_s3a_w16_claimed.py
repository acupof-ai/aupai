#!/usr/bin/env python3
"""Hold GPU claim for s3a world-16 node 0 and supervise the launcher."""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = "/work/aupai"
NAME = "s3a_seg1"
CARDS = [str(i) for i in range(8)]

if len(sys.argv) != 2:
    raise SystemExit("usage: launch_s3a_w16_claimed.py <checkpoint-path>")
resume = sys.argv[1]
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import card_claim

ok, message = card_claim.acquire(
    NAME,
    CARDS,
    wait=0,
    note="Engram Stage-3 world16: segment to step 66767",
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
        "AUPAI_LAUNCHED_BY": f"claimed launcher {NAME} world16",
        "AUPAI_CLAIMED_BY": NAME,
        "AUPAI_TOKEN_CACHE_DIR": "/data00",
        "ALLOW_UNSYNCED": "1",
        "PYTHONUNBUFFERED": "1",
    }
)
child = subprocess.Popen(
    ["bash", "runs/s3a_w16_node0.sh", resume],
    env=env,
    cwd=ROOT,
    start_new_session=True,
)
with open("runs/s3a_w16_node0.child.pid", "w", encoding="utf-8") as handle:
    handle.write(f"{child.pid}\n")
rc = child.wait()
with open("runs/s3a_w16_node0.rc", "w", encoding="utf-8") as handle:
    handle.write(f"{rc}\n")
card_claim.release(NAME)
raise SystemExit(rc)
