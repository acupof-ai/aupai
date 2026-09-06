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

THE SECOND SHAPE, added 2026-09-06 (e1's finding 1 on 91531b49). Everything above is
score() RAISING, which main()'s whole-checkpoint handler catches. The other shape is
score() returning NORMALLY with every metric inside it an {"error": ...} dict -- what
the cache_guard self-deadlock produced -- and that exited 0 too, writing three rows to
runs/score_matrix.jsonl that looked like scores. The fix was measured by mutation when
it landed and then had NO test: `if False:` in place of its condition left this file,
score_matrix's own --selftest and eval/test_l1_fewshot_2x2.py all green, so the next
edit to main()'s loop could delete it and land clean. Three record shapes below.

    python3 scripts/test_score_matrix_failpath.py
"""
import json
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


#: score() RETURNS this instead of raising. The three shapes, and the exit code each must
#: produce. `partial` and `skipped-only` are the negative controls: without them the
#: condition could be "any errored metric fails the run", which breaks every legitimately
#: partial score, or "an empty metrics dict fails", which fails every base checkpoint.
RETURN_SHAPES = {
    "all-errored": (
        {"metrics": {"domain_loss": {"error": "CoResidentCacheRead: REFUSING: this eval is "
                                              "about to read 166.2 GB of token cache"},
                     "ppl": {"error": "RuntimeError: boom"}},
         "skipped": {}},
        1,
    ),
    "partial": (
        {"metrics": {"domain_loss": {"error": "RuntimeError: boom"},
                     "score_mc": {"acc": 0.31}},
         "skipped": {}},
        0,
    ),
    "skipped-only": (
        {"metrics": {}, "skipped": {"l1_fewshot": "base checkpoint, generative SKIPs"}},
        0,
    ),
}

RETURN_HARNESS = '''
import json, sys
sys.path.insert(0, {root!r})
import eval.score_matrix as sm

REC = json.loads({rec!r})
sm.score = lambda *a, **k: dict(REC)
sm._mix_for = lambda *a, **k: "mix.json"
sm._pick_card = lambda *a, **k: "cpu"
sm.claim_my_cards = lambda *a, **k: []
sm.torch.cuda.is_available = lambda: False
sm.write_records = lambda p, rs: open(p, "w").write(
    "".join(json.dumps(r) + chr(10) for r in rs))
sys.argv = ["score_matrix.py", "--ckpt", "ckpt_fake.pt", "--json", {out!r}]
try:
    sm.main()
    print("EXITCODE 0")
except SystemExit as e:
    print("EXITCODE", e.code if e.code is not None else 0)
'''


def run_return_shape(rec, out_path):
    """Drive the real main() with score() RETURNING rec. Returns (output, rows_written)."""
    src = RETURN_HARNESS.format(root=ROOT, rec=json.dumps(dict(rec, ckpt="ckpt_fake.pt",
                                                               type="base")),
                                out=out_path)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        p = f.name
    r = subprocess.run([sys.executable, p], capture_output=True, text=True, cwd=ROOT)
    os.unlink(p)
    rows = 0
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            rows = sum(1 for ln in fh if ln.strip())
    return r.stdout + r.stderr, rows


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

    # THE SECOND SHAPE: score() returns, every metric inside errored.
    tmp = tempfile.mkdtemp(prefix="score_return_")
    for label, (rec, want_rc) in RETURN_SHAPES.items():
        out_json = os.path.join(tmp, f"{label}.jsonl")
        text, rows = run_return_shape(rec, out_json)
        got = re.search(r"EXITCODE (\d+)", text)
        rc = int(got.group(1)) if got else None
        want(rc == want_rc,
             f"a {label} record exits {want_rc} (got {rc}) -- "
             + ("every metric is an error dict, so the run produced no usable metrics"
                if want_rc else
                "this must NOT fail, or the condition bans partial and base-checkpoint rows"))
        if label == "all-errored":
            # The row is the EVIDENCE: an errored metric carries its traceback, and a
            # failing run that also withholds the record leaves nothing to diagnose from.
            want(rows == 1,
                 f"the all-errored record is still WRITTEN ({rows} row(s)) -- the traceback "
                 f"in it is the only evidence of why the metrics failed")
            want("no usable metrics" in text,
                 "the summary says no USABLE metrics, not 'NO metrics' -- metrics exist, "
                 "they are all errors")

    if bad:
        print(f"\n{len(bad)} case(s) failed: {bad}")
        print("\n--- captured output ---\n" + out)
        return 1
    print("\n12 cases pass: a raised failure names its subject, keeps its diagnosis and exits "
          "nonzero; a record whose every metric errored exits nonzero and is still written; "
          "partial and skipped-only records exit 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
