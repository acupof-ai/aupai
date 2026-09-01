#!/usr/bin/env python3
"""score_matrix's failure path must name what failed, keep the diagnosis, and exit
nonzero (de, 2026-09-01).

The real incident: an automatic post-checkpoint run OOMed twice, printed

    ckpt_w7_b32a1.pt: SKIPPED (OutOfMemoryError: CUDA out of memory. Tried to
    allocate 96.00 MiB. GPU 0 has a total capacity of 95.22 GiB o)

exited 0, and wrote no record. Three defects in one line:

1. "SKIPPED" with the checkpoint as subject read as "the checkpoint OOMed while
   saving" -- training fits but writing does not, which would be a large conclusion.
   The save succeeded at 987 MB. Scoring is what failed.
2. [:90] cut the line at "95.22 GiB o", exactly before the allocated/free/reserved
   figures. What survived reads as "a scorer wants 95 GB"; the full line says it
   failed to allocate 96 MiB, which is the opposite diagnosis -- contention, not a
   greedy scorer. The truncation destroyed the only evidence that distinguishes them.
3. exit 0 on a partial failure, so a caller checking the exit code saw success. With
   ~28 planned milestones this fires 28 times and score_matrix_present stays red
   throughout.

    python3 scripts/test_score_matrix_failpath.py
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REAL_OOM = ("CUDA out of memory. Tried to allocate 96.00 MiB. GPU 0 has a total "
            "capacity of 95.22 GiB of which 31.06 MiB is free. Process 1466244 has "
            "14.37 GiB memory in use. Of the allocated memory 12.90 GiB is allocated "
            "by PyTorch, and 1.02 GiB is reserved by PyTorch but unallocated.")

HARNESS = '''
import sys, types
sys.path.insert(0, {root!r})
import eval.score_matrix as sm

class FakeOOM(Exception):
    pass

sm.score = lambda *a, **k: (_ for _ in ()).throw(FakeOOM({msg!r}))
sys.argv = ["score_matrix.py", "--ckpt", "ckpt_fake.pt"]
try:
    sm.main()
    print("EXITCODE 0")
except SystemExit as e:
    print("EXITCODE", e.code)
'''


def run():
    src = HARNESS.format(root=ROOT, msg=REAL_OOM)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        p = f.name
    r = subprocess.run([sys.executable, p], capture_output=True, text=True, cwd=ROOT)
    os.unlink(p)
    return r.stdout + r.stderr


def main():
    out = run()
    bad = []

    def want(cond, name):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            bad.append(name)

    want("EXITCODE 1" in out, "a failed checkpoint exits nonzero")
    want("SCORING FAILED" in out, "the message names SCORING as what failed")
    want("the checkpoint is fine" in out, "it says the checkpoint is not the problem")
    want("96.00 MiB" in out, "the requested-allocation figure survives")
    want("31.06 MiB is free" in out,
         "the free-memory figure survives -- this is what separates contention from a "
         "greedy scorer, and [:90] cut it")
    want("1466244" in out, "the holding process id survives")
    want(not re.search(r"\bSKIPPED\b.*FakeOOM", out),
         "'SKIPPED' is not used for a hard failure")

    if bad:
        print(f"\n{len(bad)} case(s) failed: {bad}")
        print("\n--- captured output ---\n" + out)
        return 1
    print("\n7 cases pass: the failure names its subject, keeps its diagnosis, and "
          "exits nonzero")
    return 0


if __name__ == "__main__":
    sys.exit(main())
