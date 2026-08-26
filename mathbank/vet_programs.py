#!/usr/bin/env python3
"""Vet every bank module in isolation (subprocess + timeout -> hang-safe).
Prints `<module>: N clean, M FAIL: [...]` per module (or HANG)."""

import glob
import json
import os
import subprocess
import sys

WORKER = r'''
import sys, json, random, importlib
from run_math_short import verify
m = importlib.import_module(sys.argv[1])
rng = random.Random(1234)
clean, fail = [], []
for _lvl, name, fn in m.PROGRAMS:
    bad = False
    for _ in range(30):
        try:
            ins, lines, ans = fn(rng)
        except Exception:
            bad = True; break
        try:
            out, ok = verify(ins, lines, ans)
        except Exception:
            bad = True; break
        if not ok:
            bad = True; break
    (clean if not bad else fail).append(name)
json.dump({"clean": clean, "fail": fail}, sys.stdout, ensure_ascii=False)
'''

def main():
    mods = ["math_programs_l1", "math_programs_l2", "math_programs_l3", "math_programs_l4"] + \
           [os.path.splitext(os.path.basename(f))[0] for f in
            sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "math_programs_l*_ext*.py")))]
    for stem in mods:
        try:
            r = subprocess.run([sys.executable, "-c", WORKER, stem],
                               capture_output=True, text=True, timeout=45)
            d = json.loads(r.stdout)
            print(f"{stem}: {len(d['clean'])} clean, {len(d['fail'])} FAIL: {d['fail']}")
        except subprocess.TimeoutExpired:
            print(f"{stem}: HANG (killed) — programs unverified")
        except Exception as e:
            print(f"{stem}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()