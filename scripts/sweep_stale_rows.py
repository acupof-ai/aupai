#!/usr/bin/env python3
"""Close pre-0901 experiment rows left open by jobs that are gone (fb's sweep).

no_stale_running FAILs a row open past 24h, and 14 rows from yesterday cross that
line over the next few hours -- a rolling wall of refusals against the shipment
window, each needing a manual close at the moment it trips.

Verified dead before closing: `ps -eo args | grep -E '--name <n>'` matches nothing
on the pod for any of them. The row is closed as `fail` with the reason it is being
closed, not as `ok`: nobody observed these finishing, and a swept row that claims
success is worse than one left open.

The 15B milestone row (ms_ckpt_pretrain_15b_s1) is EXCLUDED -- its readout landed
and it is the run whose verdict is under review; fb closes that one with its result.
"""
import json
import os
import subprocess
import sys
import time

ROOT = "/work/aupai"
EXCLUDE = {"ms_ckpt_pretrain_15b_s1"}
REASON = ("swept 2026-09-01: no process on the pod and no terminal row; the job is gone "
          "and nobody observed its outcome")

rows = [json.loads(x) for x in open(f"{ROOT}/runs/experiments.jsonl", encoding="utf-8") if x.strip()]
folded = {}
for r in rows:
    k = (r.get("name"), r.get("started"))
    prev = folded.get(k)
    if prev is None or (prev.get("status") == "running" and r.get("status") != "running"):
        folded[k] = r
    elif prev.get("status") != "running" and r.get("status") != "running":
        folded[k] = r

live = subprocess.run(["ps", "-eo", "args", "--no-headers"], capture_output=True, text=True).stdout

closed, skipped = [], []
for (name, started), r in sorted(folded.items(), key=lambda kv: kv[0][1] or ""):
    if r.get("status") != "running" or name in EXCLUDE:
        continue
    if f"--name {name}" in live or f"name {name} " in live:
        skipped.append((name, "still running"))
        continue
    age_h = (time.time() - time.mktime(time.strptime(started, "%Y-%m-%d %H:%M"))) / 3600
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "exp.py"), "done",
         "--name", name, "--status", "fail",
         "--result", f"swept: open {age_h:.0f}h, process gone",
         "--finding", REASON,
         "--decision", "relaunch if the work is still wanted; the row states no outcome"],
        capture_output=True, cwd=ROOT)
    closed.append((name, started, round(age_h, 1)))

print(f"closed {len(closed)}, skipped {len(skipped)}")
for n, s, a in closed:
    print(f"  {n:28} {s}  {a}h")
for n, why in skipped:
    print(f"  SKIP {n}: {why}")
