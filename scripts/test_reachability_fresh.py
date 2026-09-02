#!/usr/bin/env python3
"""selftest for 44-14 defect 5: runs/reachability.txt went stale for two days
while harness.py grew from 5,940 to 10,176 lines. Regenerating the full graph
takes ~15s, too slow for this hook, so this guards the observed failure mode
cheaply: the harness.py line count the graph records must match the live file.

    python3 scripts/test_reachability_fresh.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
recorded = None
for line in open(os.path.join(ROOT, "runs/reachability.txt"), encoding="utf-8"):
    if line.startswith("scripts/harness.py "):
        recorded = int(line.split()[1])
        break
assert recorded is not None, "reachability.txt no longer lists scripts/harness.py"
live = sum(1 for _ in open(os.path.join(ROOT, "scripts/harness.py"), encoding="utf-8"))
assert recorded == live, f"reachability.txt stale: harness.py {recorded} vs live {live}; rerun python3 scripts/reachability.py > runs/reachability.txt"

print(f"selftest OK: reachability.txt harness.py line count ({recorded}) matches live")
