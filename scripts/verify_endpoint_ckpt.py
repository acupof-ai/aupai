#!/usr/bin/env python3
"""Verify a checkpoint IS the endpoint, by comparing two artifacts to each other.

NO LITERAL STEP COUNT. The obvious version asserts `step == 10172`, and 10172 is itself
derived -- train.py:3348 computes total_steps from the plan length and :3361 adds the resume
point, so it depends on the mix and the resume, not on the token budget. A literal was wrong
once already: 8e9/786432 = 10172.526 floors to 10172 while --max_steps was 10173, so a
`== 10173` assertion would have refused the real endpoint.

So: the checkpoint's recorded step must EQUAL the total printed in the log's own step lines.
Both sides come from artifacts. This also catches the case a literal cannot -- a run that
exits at a step different from its advertised header.
"""
import argparse, re, sys, torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--log", required=True)
a = ap.parse_args()

txt = open(a.log, errors="replace").read()
steps = re.findall(r"^step (\d+)/(\d+)", txt, re.M)
if not steps:
    sys.exit(f"REFUSING: no step lines in {a.log}")
last_step, total = int(steps[-1][0]), int(steps[-1][1])
totals = {int(t) for _, t in steps}
if len(totals) != 1:
    sys.exit(f"REFUSING: the log advertises {len(totals)} different totals {sorted(totals)}")

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
rec = ck.get("step")
print(f"log:  last step {last_step}, advertised total {total}")
print(f"ckpt: recorded step {rec}")
if rec is None:
    sys.exit("REFUSING: the checkpoint records no step")
if rec != total:
    sys.exit(f"REFUSING: ckpt step {rec} != log total {total}. This is not the endpoint.")
if last_step != total:
    print(f"  NOTE: last printed step {last_step} != total {total} (the loop breaks at >= total)")
print(f"ENDPOINT CONFIRMED at step {rec}")
