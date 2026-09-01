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


def read_cursors(lines, before, after):
    """Condition 3, criterion B (fb): the SUM strictly increased and no domain fell.

    Criterion A -- every domain strictly greater -- needs an exemption table. At the test
    shape the two smallest domains gain only ~4.6 and ~4.7 rows between step 40 and 60, so
    the shuffle leaves one of them empty about 1.84% of the time: one false red in every
    54 runs of a gate whose red blocks a launch. A table fixes that and then rots -- change
    a mix weight and the stale exemption waves through a real defect.

    B needs no table. The sum gains 640 rows, which cannot come out zero, so there is no
    false red; and "no domain fell" is exactly the re-read signature, because a domain that
    restarts at row 0 comes back BELOW where it stopped rather than merely level. Immune to
    weight changes by construction.

    Read from the checkpoints, not the log: build_mix prints on both discard branches
    (:1886 seed, :1895 fingerprint) and prints nothing when the cursor IS adopted, so a
    silent adopt and a silent skip look identical in the restart output.
    """
    if not before:
        return [(FAIL, "3. cursor sum grew, no domain fell",
                 "the resume-source checkpoint has no row_cursor -- nothing to advance from")]
    if not after:
        return [(UNCOVERED, "3. cursor sum grew, no domain fell",
                 "run 2 wrote no checkpoint carrying row_cursor")]
    missing = sorted(set(before) - set(after))
    fell = sorted(d for d in before if d in after and after[d] < before[d])
    grew = sum(after.get(d, 0) for d in before) - sum(before.values())
    if missing or fell or grew <= 0:
        detail = []
        if missing:
            detail.append(f"absent from the later cursor: {missing}")
        if fell:
            detail.append("went BACKWARD (the re-read signature): "
                          + ", ".join(f"{d} {before[d]}->{after[d]}" for d in fell))
        if grew <= 0:
            detail.append(f"sum did not grow: {grew:+d} rows")
        return [(FAIL, "3. cursor sum grew, no domain fell", "; ".join(detail))]
    still = sorted(d for d in before if after[d] == before[d])
    return [(PASS, "3. cursor sum grew, no domain fell",
             f"sum +{grew} rows over {len(before)} domains"
             + (f"; {len(still)} level ({', '.join(still)}) -- allowed, the shuffle can miss "
                f"a small domain in 640 rows" if still else ""))]


def read_sum(cursor, step, batch, accum, world):
    """Condition 6 (de): the nine cursors must sum to the GLOBAL rows consumed.

    Every other condition compares domains one at a time, so a scale error in the write
    path -- :1404 multiplies this rank's bincount by world -- passes all of them while
    the resume position is off by a factor of world. The sum is the only reading that
    carries a dimension, and it costs nothing: the cursor is already loaded.

    Bound, not equality: each domain's count is an int() truncation, so the sum can fall
    up to (domains x world) below step*batch*accum*world and still be right.
    """
    if not cursor:
        return [(FAIL, "6. cursors sum to the global rows consumed", "no row_cursor to sum")]
    got = sum(cursor.values())
    top = step * batch * accum * world
    floor = top - len(cursor) * world
    if floor <= got <= top:
        return [(PASS, f"6. cursors sum to the global rows consumed [{floor}, {top}]",
                 f"sum {got} over {len(cursor)} domains")]
    factor = f" -- that is {got / top:.3g}x the expected total" if top else ""
    return [(FAIL, f"6. cursors sum to the global rows consumed [{floor}, {top}]",
             f"sum {got}{factor}; a world-factor error reads exactly like this while every "
             f"per-domain comparison stays green")]


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
    print("NOT COVERED by this test, and not implied by its green: the startup refusal "
          "logic itself (the six conditions all take the happy path and never reach it); "
          "an optimizer ORDER swap (zip(strict=True) sees a changed count, not a swap); "
          "the rows_done > plan refusal at :1387 (only a full 20B run reaches it).")
    print()
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
    b4 = {"cot": 900, "zh_web": 4400}
    aft = {"cot": 1350, "zh_web": 6600}
    rows, _ = read_log(good, 40, 20)
    rows += read_cursors(good, b4, aft)
    assert [r[0] for r in rows] == [PASS] * 4, rows

    # Re-read from row 0: the domain ends up nonzero but BELOW where it stopped. This is
    # the case "nonzero" would have passed and strictly-greater catches.
    assert read_cursors(good, b4, {"cot": 450, "zh_web": 6600})[0][0] == FAIL
    # One domain level is ALLOWED under B: the shuffle can miss a small domain in 640 rows.
    assert read_cursors(good, b4, {"cot": 900, "zh_web": 6600})[0][0] == PASS
    assert read_cursors(good, b4, {"cot": 1350})[0][0] == FAIL                  # domain vanished

    # Condition 6: 1280 rows/rank x 7 = 8960, int() truncation can cost 9x7=63.
    nine = {f"d{i}": 995 for i in range(9)}          # sums to 8955, inside [8897, 8960]
    assert read_sum(nine, 40, 32, 1, 7)[0][0] == PASS
    assert read_sum({f"d{i}": 142 for i in range(9)}, 40, 32, 1, 7)[0][0] == FAIL  # /7: per-rank, world dropped
    assert read_sum({}, 40, 32, 1, 7)[0][0] == FAIL

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
    print("read_resume_proof selftest OK: 16 cases "
          "(clean pass, discard, step 0, inflated total, stage-2 equation, cursor re-read/level-ok/vanish, "
          "sum ok/world-dropped/empty, no-opt/never-stepped/wrong-index/all-zero/loaded)")
    return 0


def main(argv):
    if "--selftest" in argv:
        return _selftest()
    if len(argv) < 3:
        print(__doc__)
        return 0
    log, ckpt = argv[1], argv[2]
    # world is the rank count the run used; the cursor is stored global (:1404 multiplies
    # this rank's count by it), so condition 6 cannot be checked without it.
    world = int(argv[3]) if len(argv) > 3 else 8
    import glob

    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    with open(log, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Conditions 3 and 5 both read the checkpoint run 2 WROTE, not the one it read: the
    # cursor question is whether it advanced past the resume point, and the optimizer
    # question is whether the load happened at all. Only the later file answers either.
    later = sorted(glob.glob(ckpt.rsplit(".step", 1)[0] + ".step*"),
                   key=lambda q: int(q.rsplit(".step", 1)[1]))
    tail = later[-1] if later and later[-1] != ckpt else None
    d = torch.load(tail, map_location="cpu", weights_only=False, mmap=True) if tail else {}

    rows, _ = read_log(lines, 40, 20)
    rows += read_cursors(lines, ck.get("row_cursor") or {}, d.get("row_cursor") or {})
    cfg = ck.get("cfg") or {}
    rows += read_sum(ck.get("row_cursor") or {}, ck.get("step") or 40,
                     cfg.get("batch") or 32, cfg.get("accum") or 1, world)
    if tail is None:
        rows += [(UNCOVERED, "5. Muon momentum loaded and nonzero",
                  "run 2 wrote no later checkpoint to read the buffers from"),
                 (UNCOVERED, "5b. optimizer ORDER unchanged", "debt ledger")]
    else:
        rows += read_opt(d.get("opt") or [])
    print(f"reading {log}")
    print(f"  resume source: {ckpt}")
    print(f"  run 2 wrote:   {tail or '(none)'}")
    print()
    return report(rows)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
