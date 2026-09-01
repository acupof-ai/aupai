#!/usr/bin/env python3
"""Read the crash-recovery proof by the user's frozen conditions, and say which are
UNCOVERED rather than letting silence read as green.

Five conditions were handed down; two of them, taken literally, test something that
cannot fail:

  the compensation condition, in its first form ("total_steps shows the resume_step
    compensation") -- train.py:2443 adds resume_step only when _plan_trimmed, which the
    cursor conditions force true, and :2431's min() then clamps it straight back:
    min(20+40, 60) = 60. The compensation fires and is invisible. fb's final version is
    de's EQUATION, total == N + this stage's plan steps, which asserts the result and
    stays silent on the mechanism -- so it reads the same in a trimmed stage 2 and in
    this same-mix restart. (44's catch: it holds here only because run 2 passes the same
    --max_steps, which prove_resume.sh does.)

  cond 5 as written ("optimizer state is non-empty") -- :2374 refuses to start when a
    checkpoint has `step` and no `opt`, and the .stepN path always writes `opt` (:2588
    passes good_opt). Both branches make an empty optimizer unreachable, so the literal
    check is green in every world. What is NOT unreachable is :2380's `if ... "opt" in
    ck` having no else: a key-name change skips the load silently and leaves freshly
    initialised (all-zero) buffers. So this reads the buffers and requires them NONZERO.

    b0's find. Not covered by anything here: :2381's zip(strict=True) catches a changed
    NUMBER of optimizers but not a changed ORDER -- Muon's momentum loaded into AdamW
    raises nothing. 60 steps cannot see it; it is reported as UNCOVERED, not as passed.

    python3 scripts/read_resume_proof.py --selftest
"""
import re
import sys

PASS, FAIL, UNCOVERED = "PASS", "FAIL", "UNCOVERED"

CURSOR_RE = re.compile(r"cursor discarded")
# "mix: <domain> rows ... from row N" -- the adopted-cursor line, whatever its wording,
# always names the domain and the row it starts at.
START_RE = re.compile(r"mix: (\w+).*?(?:from row|start(?:ing)? at row|cursor) (\d+)")
JOIN_RE = re.compile(r"WSD JOIN: resumed at step (\d+)/(\d+)")


def read_log(lines, expect_step, plan_steps):
    """Conditions 1, 3, 4 -- everything the restart log can answer on its own.

    `plan_steps` is what this stage's own plan is worth, so condition 4 is the equation
    total == N + plan_steps rather than a claim about whether the compensation fired.
    """
    expect_total = expect_step + plan_steps
    out = []
    discarded = [ln.strip() for ln in lines if CURSOR_RE.search(ln)]
    out.append((FAIL if discarded else PASS, "1. no cursor discarded",
                f"{len(discarded)} discard line(s): {discarded[:3]}" if discarded
                else "no domain restarted at row 0"))

    join = None
    for ln in lines:
        m = JOIN_RE.search(ln)
        if m:
            join = (int(m.group(1)), int(m.group(2)))
    if join is None:
        out.append((FAIL, "3. step resumes at 40", "no WSD JOIN line -- did run 2 get --resume?"))
        out.append((FAIL, "4. total == N + plan steps", "no WSD JOIN line"))
        return out, None
    step, total = join
    out.append((PASS if step == expect_step else FAIL, f"3. step resumes at {expect_step}",
                f"WSD JOIN says step {step}"))
    out.append((PASS if total == expect_total else FAIL,
                f"4. total == N + plan steps ({expect_step} + {plan_steps} = {expect_total})",
                f"WSD JOIN says total {total}"
                + ("" if total == expect_total else
                   " -- the equation is what is asserted, not whether :2443 fired")))
    return out, join


def read_cursors(lines, ck_cursor):
    """Condition 2: every domain starts where the checkpoint said it stopped."""
    seen = {}
    for ln in lines:
        m = START_RE.search(ln)
        if m:
            seen.setdefault(m.group(1), int(m.group(2)))
    if not seen:
        return [(UNCOVERED, "2. cursors adopted and equal to the checkpoint",
                 "the restart log prints no per-domain start row; read it from the "
                 "next checkpoint's row_cursor instead of calling this green")]
    bad = [f"{d}: log {v} vs ckpt {ck_cursor.get(d)}"
           for d, v in seen.items() if ck_cursor.get(d) != v]
    zero = [d for d, v in seen.items() if v == 0]
    if bad or zero:
        return [(FAIL, "2. cursors adopted and equal to the checkpoint",
                 f"mismatched: {bad}; at row 0: {zero}")]
    return [(PASS, "2. cursors adopted and equal to the checkpoint",
             f"{len(seen)} domains, all nonzero and equal")]


def read_opt(opts):
    """Condition 5, b0's version: the buffers were LOADED, not merely present.

    `opts` is ck["opt"], a LIST in build_optimizers order (train.py:1114) -- [0] Muon,
    [1] AdamW(embed), [2] AdamW(scalar). Index 0 is not cosmetic: fb's condition names
    Muon's momentum, and reading [1] instead would find AdamW's exp_avg, which is also
    nonzero, so the condition would go green having measured the wrong optimizer (b0).

    Empty state and all-zero buffers are both failures, for different reasons: empty
    means Muon never stepped or the load was skipped entirely, all-zero is what a
    key-rename past :2380's else-less `if "opt" in ck` leaves behind.
    """
    if not opts:
        return [(FAIL, "5. Muon momentum loaded and nonzero", 'no "opt" key in the checkpoint'),
                (UNCOVERED, "5b. optimizer ORDER unchanged",
                 "zip(strict=True) catches a changed count, not a swap (b0); a 60-step run "
                 "cannot build a reordered checkpoint to test it -- debt ledger")]
    state = opts[0].get("state") or {}
    if not state:
        return [(FAIL, "5. Muon momentum loaded and nonzero",
                 "opt[0] state is empty -- Muon never stepped, or the load was skipped"),
                (UNCOVERED, "5b. optimizer ORDER unchanged", "see above; debt ledger")]
    bufs = [v["momentum_buffer"] for v in state.values()
            if isinstance(v, dict) and "momentum_buffer" in v]
    if not bufs:
        return [(FAIL, "5. Muon momentum loaded and nonzero",
                 f"opt[0] has {len(state)} param(s) but no momentum_buffer -- is [0] really "
                 f"Muon? AdamW would carry exp_avg instead"),
                (UNCOVERED, "5b. optimizer ORDER unchanged", "see above; debt ledger")]
    peak = max(float(b.norm()) for b in bufs)
    return [(PASS if peak > 0 else FAIL, "5. Muon momentum loaded and nonzero",
             f"{len(bufs)} momentum_buffer(s), max norm {peak:.4g}"
             + ("" if peak > 0 else " -- ALL ZERO: present in the file, never loaded")),
            (UNCOVERED, "5b. optimizer ORDER unchanged",
             "zip(strict=True) catches a changed count, not a swap (b0); a 60-step run "
             "cannot build a reordered checkpoint to test it -- debt ledger")]


def report(rows):
    for status, name, detail in rows:
        print(f"  [{status:9}] {name}\n              {detail}")
    fails = [r for r in rows if r[0] == FAIL]
    unc = [r for r in rows if r[0] == UNCOVERED]
    print()
    if fails:
        print(f"NOT PROVEN: {len(fails)} condition(s) failed. Per the user's order, "
              f"this blocks the launch.")
        return 1
    print("PROVEN: every condition that this test can answer, passed."
          + (f" {len(unc)} reported UNCOVERED above -- they were not tested, which is "
             f"not the same as passing." if unc else ""))
    return 0


def _selftest():
    good = [
        "mix: cot rows 1234 from row 900",
        "mix: zh_web rows 9999 from row 4400",
        "WSD JOIN: resumed at step 40/60 under mix mix_500m | lr_mult 1.0000",
    ]
    ck = {"cot": 900, "zh_web": 4400}
    rows, _ = read_log(good, 40, 20)
    rows += read_cursors(good, ck)
    assert [r[0] for r in rows] == [PASS] * 4, rows

    # A discarded cursor is the failure the negative condition exists for.
    bad = good + ["mix: zh_web cursor discarded -- written at sample_seed 42, this run uses 7"]
    assert read_log(bad, 40, 20)[0][0][0] == FAIL

    # step 0 after a resume: ck["step"] was never read.
    zero = ["WSD JOIN: resumed at step 0/60 under mix mix_500m"]
    assert [r[0] for r in read_log(zero, 40, 20)[0]] == [PASS, FAIL, PASS], read_log(zero, 40, 20)[0]

    # total 100 means the min() clamp did not apply -- run 2 was launched without --max_steps.
    infl = ["WSD JOIN: resumed at step 40/100 under mix mix_500m"]
    assert read_log(infl, 40, 20)[0][2][0] == FAIL

    # A trimmed stage 2 -- the equation reads the same there, which is why it replaced
    # "total must be 60": N=4768, plan 4769 steps, total 9537 and the compensation DID fire.
    st2 = ["WSD JOIN: resumed at step 4768/9537 under mix mix_500m"]
    assert [r[0] for r in read_log(st2, 4768, 4769)[0]][1:] == [PASS, PASS]

    # A cursor at row 0 fails even when the log prints it without the word "discarded".
    z = ["mix: cot rows 1 from row 0", "WSD JOIN: resumed at step 40/60"]
    assert read_cursors(z, {"cot": 0})[0][0] == FAIL

    class T:
        def __init__(self, v):
            self.v = v

        def abs(self):
            return self

        def max(self):
            return self.v

        def norm(self):
            return self.v

    assert read_opt([])[0][0] == FAIL                                   # no opt key
    assert read_opt([{"state": {}}])[0][0] == FAIL                      # Muon never stepped
    assert read_opt([{"state": {0: {"exp_avg": T(0.3)}}}])[0][0] == FAIL  # [0] is not Muon
    assert read_opt([{"state": {0: {"momentum_buffer": T(0.0)}}}])[0][0] == FAIL  # all zero
    assert read_opt([{"state": {0: {"momentum_buffer": T(0.3)}}}])[0][0] == PASS
    assert read_opt([{"state": {0: {"momentum_buffer": T(0.3)}}}])[1][0] == UNCOVERED
    print("read_resume_proof selftest OK: 11 cases "
          "(clean pass, discard, step 0, inflated total, stage-2 equation, row 0, "
          "no-opt/never-stepped/wrong-index/all-zero/loaded)")
    return 0


def main(argv):
    if "--selftest" in argv:
        return _selftest()
    if len(argv) < 3:
        print(__doc__)
        return 0
    log, ckpt = argv[1], argv[2]
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    with open(log, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    rows, _ = read_log(lines, 40, 20)
    rows += read_cursors(lines, ck.get("row_cursor") or {})
    # The optimizer buffers come from the checkpoint run 2 WROTE (step 60), not the one it
    # read: the question is whether the load happened, and step 60's buffers answer it.
    import glob
    later = sorted(glob.glob(ckpt.rsplit(".step", 1)[0] + ".step*"),
                   key=lambda p: int(p.rsplit(".step", 1)[1]))
    tail = later[-1] if later and later[-1] != ckpt else None
    if tail is None:
        rows += [(UNCOVERED, "5. optimizer buffers loaded and nonzero",
                  "run 2 wrote no later checkpoint to read the buffers from")]
    else:
        d = torch.load(tail, map_location="cpu", weights_only=False, mmap=True)
        rows += read_opt(d.get("opt") or [])
    print(f"reading {log}")
    print(f"  resume source: {ckpt}")
    print(f"  buffers from:  {tail or '(none)'}")
    print()
    return report(rows)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
