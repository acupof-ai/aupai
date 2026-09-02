#!/usr/bin/env python3
"""de-13 acceptance, part B: does a SECOND resume's cursor cover the whole run?

Part A is scripts/test_resume_cursor_pod.py, which proved the cursor is read and applied
and that the planned rows move. It hands build_mix a cursor value directly, so it never
touches the two lines part B is about, and it passes today.

This asserts the property those two lines break:

    a checkpoint's row_cursor must count every row the RUN has consumed,
    not only the rows the current segment drew

train.py:1387  rows_done = (step - _origin) * _batch * _accum
train.py:2309  Cfg._plan_step_origin = resume_step
train.py:1906  used[name] = int(row_cursor[name])        # assignment, not accumulation

So a checkpoint written after a resume describes that resume, and every earlier segment is
invisible in it. MEASURED on p500m_20b_0902 (facts/data_scaling.json#ds.second_resume_
rereads_one_segment): .interrupt.step32 sums to 8,192 = 32x256, correct because that
segment began at origin 0; .interrupt.step83 sums to 13,056 = (83-32)x256, while step 83
had truly consumed 21,248 = 83x256. The 8,192 difference is exactly segment one, and every
later resume re-reads it.

THIS TEST IS RED ON PURPOSE until train.py is unfrozen and fixed. It is red on the
ARITHMETIC, not on an import or a missing file -- run it and the failure names the two row
counts and their difference. That is what makes it de-13's acceptance: it turns green when
the defect is fixed and cannot turn green any other way. Verified in both directions before
it was trusted: red on today's train.py, GREEN when either fix is simulated in a clone
(rows_done counting from the run's start, or `used[name] +=`).

DO NOT put it in the hook's SELFTEST_FILES or in CI while it is red. A deliberately-red
test in either place turns every commit and every push red, and a permanent red is the same
as no signal (AGENTS.md). It carries no `--selftest` flag for the same reason, so
selftests_are_gated does not ask for it. Gate it in the same commit that fixes train.py.

No GPU, no corpus, no tokenizer: the quantity in dispute is integer arithmetic over
(step, origin, batch, accum, world), so the test reimplements nothing -- it reads the
expression out of train.py's source and evaluates it. Reading the source rather than
importing train keeps it runnable on a laptop, where train's CUDA imports are absent.

    python3 scripts/test_resume_accumulates.py
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The live 20B run's shape, from the fact's config. Two segments: 0..32 and 32..83.
BATCH, ACCUM, WORLD = 32, 1, 8
SEG = [(0, 32), (32, 83)]


def cursor_rows_at(src, step, origin, batch, accum, world):
    """Rows the checkpoint at `step` would record, per train.py as written.

    Derived from the source rather than restated, so a fix in EITHER place flips the
    answer. The first version keyed the verdict on whether `_plan_step_origin =
    resume_step` was present, which is one particular edit rather than the property --
    both simulated fixes left that line alone and the test stayed red, i.e. it could only
    ever fail. A test that cannot go green is an assertion, not an acceptance test (de,
    2026-09-02, caught by simulating the fix before trusting the red).

    Three independent ways the whole-run count can be correct:
      1. rows_done stops subtracting a per-segment origin
      2. the cursor accumulates: `used[name] += ...` instead of `=`
      3. the write site adds the cursor this plan STARTED from, so a relative per-segment
         count plus that base is the run total. This is the shape de-13 shipped, and the
         test had to learn it -- recognising only shapes 1 and 2 would have kept the test red
         against a correct fix, which is the same defect as a test that cannot go green, one
         level up: a test that can only go green the way its author imagined.
    """
    m = re.search(r"^\s*rows_done\s*=\s*\(step\s*-\s*([\w.]+)\)\s*\*", src, re.M)
    if m is None:
        return None
    sub = m.group(1)
    # Does the subtrahend actually carry the resume step at runtime? `0` or a literal
    # cannot, and a name only does if something assigns resume_step to it.
    per_segment = not sub.isdigit() and bool(
        re.search(r"_plan_step_origin\s*=\s*resume_step", src)
        and re.search(rf"{re.escape(sub)}\s*=.*_plan_step_origin", src)
    )
    seg_rows = (step - (origin if per_segment else 0)) * batch * accum * world
    # An accumulating cursor adds each segment, so the recorded total is whole-run even
    # when each segment is counted relatively.
    if re.search(r"used\[name\]\s*\+=\s*int\(", src):
        return step * batch * accum * world
    # Shape 3: the written cursor adds a per-domain base, and that base is set from the
    # cursor build_mix applied. Both halves are required -- a base that is never populated
    # adds zero, so checking only the write site would accept a broken fix.
    writes_base = re.search(r"row_cursor\"\]\s*=\s*\{[^}]*_base\.get\(", src, re.S)
    fills_base = re.search(r"base\[name\]\s*=\s*used\[name\]", src)
    if writes_base and fills_base:
        return seg_rows + origin * batch * accum * world
    return seg_rows


def main():
    train_py = os.path.join(ROOT, "train.py")
    if not os.path.exists(train_py):
        print("FAIL: no train.py")
        return 1
    src = open(train_py, encoding="utf-8").read()

    bad = []
    for origin, step in SEG:
        got = cursor_rows_at(src, step, origin, BATCH, ACCUM, WORLD)
        if got is None:
            print("FAIL: train.py has no `rows_done = (step - <x>) * ...` line. Either it "
                  "was fixed in a shape this test cannot read, or it moved -- re-read "
                  "train.py before trusting either answer.")
            return 1
        want = step * BATCH * ACCUM * WORLD
        if got != want:
            bad.append(
                f"at step {step} (segment origin {origin}): the cursor records {got:,} rows, "
                f"the run has consumed {want:,} -- {want - got:,} rows are invisible to it, "
                f"and a resume from this checkpoint re-reads them"
            )

    if bad:
        print("FAIL: a checkpoint's row_cursor does not cover the whole run")
        for b in bad:
            print(f"  {b}")
        print("\nThis is de-13's acceptance and it is EXPECTED red while train.py is "
              "frozen for p500m_20b_0902. Either fix turns it green: count rows_done from "
              "the run's start, or accumulate the cursor across segments.")
        return 1

    print(f"OK: the cursor covers the whole run across {len(SEG)} segments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
