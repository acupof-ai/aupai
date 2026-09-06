#!/usr/bin/env python3
"""The MoE diagnostics write is rank 0's, and every rank still enters the collective.

WHY THIS EXISTS. b0_moe_e1 (2026-09-05) logged `moe_diag write FAILED: UnboundLocalError: cannot
access local variable 'tps'` at steps 30 and 100, and runs/moe_diag.jsonl held 12 rows for 7 due
steps. Both came from one missing `if is_main` around the write: the memory diagnostics block has
it, the MoE block did not.

  - `tps` is bound in the `if is_main and step % 10 == 0` block, so on every other rank it is
    unbound. `tok_s_gpu=(tps if step in (30, 100) else None)` evaluates it ONLY at 30 and 100, so
    rank 1 raised there and the broad `except` logged a write failure. Readout 5's number survived
    only because rank 0 had already written it.
  - At every OTHER due step the conditional short-circuits to None before touching `tps`, so both
    ranks wrote and the ledger gained a DUPLICATE row. That is the harmful half and it was silent:
    a reader counting rows or averaging a field sees two identical samples and a confidence that is
    not there.

THE LEDGER'S DISTRIBUTION IS THE INVERSE OF WHAT THE LOG SUGGESTED, which is the fact worth
carrying: per step {10: 2, 20: 2, 30: 1, 100: 1, 200: 2, 300: 2, 400: 2}. The two steps that
printed "write FAILED" are the only two with a SINGLE row -- rank 1 raised there before writing --
while every step that logged nothing has two. So the failures marked the clean steps, and the
silence marked the corrupted ones.

WHY A TEST AND NOT A RE-READ. Nothing covered this write -- grep for moe_diag across scripts/
finds the writer, the audit and the harness, and no test. A single-rank test cannot see it either:
the defect is that a NON-main rank reaches code only rank 0 should run, so the population has to
include a rank that is not rank 0. That is the whole reason it shipped green.

WHAT THIS DOES NOT COVER, stated because a reader would otherwise assume it does (e1's review
point): the block is EXTRACTED and executed against a fake torch.distributed and a fake model, so
these checks prove the block's rank logic and NOT that the guard fires from train.py at world 2.
If `_diag_due` or `is_main` were computed differently at the real call site, this fixture would
still be green -- the same shape as a parameter whose caller never passes it. The in-situ
assertion is the ledger after a real world-2 run: each due step must hold exactly one row.

WHAT IS CHECKED, by executing the real block extracted from train.py rather than by reading it:
  1. rank 0 at step 30 writes exactly one row carrying tok_s_gpu.
  2. rank 1 at step 30 writes NOTHING and raises nothing -- the pre-fix code raised here.
  3. rank 1 at step 10 writes NOTHING -- the pre-fix code wrote a duplicate here, silently.
  4. every due step in the real ledger's step set yields exactly one row, and the pre-fix
     distribution is NOT reproduced (asserting the count alone would pass for code that writes
     nothing on either rank).
  5. the all_reduce stays OUTSIDE the guard: every rank must enter it or the next reduction hangs.

    python3 scripts/test_moe_diag_rank.py
"""
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_fails = []


def _check(name, got, want):
    if got != want:
        _fails.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {name}: {got!r}")


def _extract(src):
    """The MoE diag block, from its `_diag_due` test to the end of its except clause.

    Sliced by INDENTATION rather than to the next known statement: an earlier version of a
    different extractor in this repo sliced to the next `if fp8:` and swallowed unrelated lines
    when the code moved, so the check died on a NameError instead of failing its assertion
    (test_e2e check 12, 2026-09-05). The block ends at the first non-blank line indented no deeper
    than the `if` itself.
    """
    lines = src.split("\n")
    starts = [i for i, l in enumerate(lines)
              if '_diag_due and getattr(raw_model, "moe_layers"' in l]
    if len(starts) != 1:
        raise SystemExit(f"expected exactly one MoE diag block in train.py, found {len(starts)}")
    i = starts[0]
    base = len(lines[i]) - len(lines[i].lstrip())
    block = [lines[i]]
    for l in lines[i + 1:]:
        if l.strip() and (len(l) - len(l.lstrip())) <= base:
            break
        block.append(l)
    return textwrap.dedent("\n".join(block))


class _FakeMoE:
    def __init__(self):
        self.top_k = 3
        self.tokens_per_expert = "counter"
        self.reduced = 0
        self.resets = 0

    def diagnostics(self, reset=True):
        self.resets += 1
        return {"usage_frac": 1.0, "entropy_norm": 0.98, "load_gini": 0.1,
                "tokens": 7864320, "window_steps": 10, "n_routed": 24}


class _FakeBlock:
    def __init__(self, ffn):
        self.ffn = ffn


class _FakeModel:
    def __init__(self, moe):
        self.moe_layers = [0]
        self.blocks = [_FakeBlock(moe)]


def _run(block, *, step, is_main, bind_tps):
    """Execute the extracted block as one rank. Returns (rows, reduced, error).

    `bind_tps` models the real binding rule: `tps` exists only where
    `if is_main and step % 10 == 0` ran, i.e. on rank 0. A non-main rank has no such name, which
    is what NameError below stands in for -- exec's globals cannot hold an "unbound local", so the
    absence of the key is the faithful model of it.
    """
    rows = []
    moe = _FakeMoE()

    class _Diag:
        @staticmethod
        def log_diag(**kw):
            rows.append(kw)

    reduced = []

    class _ReduceOp:
        SUM = "sum"
        MAX = "max"

    class _Dist:
        # ReduceOp hangs off torch.distributed, not beside it -- the block calls
        # `torch.distributed.ReduceOp.SUM`. My first fake put it one level out and every rank
        # died on AttributeError BEFORE reaching the write, which made three checks pass for the
        # wrong reason: "rank 1 wrote nothing" was true because nothing ran at all.
        ReduceOp = _ReduceOp

        @staticmethod
        def all_reduce(t, op=None):
            reduced.append((t, op))

    ns = {
        "_diag_due": step % 100 == 0 or step in (10, 20, 30),
        "raw_model": _FakeModel(moe),
        "ddp": True,
        "is_main": is_main,
        "step": step,
        "total_steps": 3815,
        "torch": type("T", (), {"distributed": _Dist, "int32": "i32"}),
        "runlog": lambda m: rows.append({"_runlog": m}),
        "Cfg": type("C", (), {"moe_arm": "e1"}),
        "moe_diag": _Diag,
    }
    if bind_tps:
        ns["tps"] = 61329.0
    # The import inside the block resolves from sys.modules, so the fake has to be there.
    sys.modules["moe_diag"] = _Diag
    try:
        exec(block, ns)
        err = None
    except Exception as e:  # noqa: BLE001 -- the point is which error, if any, escapes
        err = f"{type(e).__name__}: {e}"
    return rows, len(reduced), err


def main():
    with open(os.path.join(ROOT, "train.py"), encoding="utf-8") as f:
        src = f.read()
    block = _extract(src)
    if "moe_diag.log_diag(" not in block:
        raise SystemExit("extracted block does not contain the write -- the slice is wrong")

    # 1. rank 0 at step 30: one row, and it carries the throughput.
    rows, reduced, err = _run(block, step=30, is_main=True, bind_tps=True)
    _check("rank 0 step 30 error", err, None)
    _check("rank 0 step 30 rows", len(rows), 1)
    _check("rank 0 step 30 carries tok_s_gpu", rows and rows[0].get("tok_s_gpu"), 61329.0)
    _check("rank 0 step 30 entered the collective", reduced, 1)

    # 2. rank 1 at step 30 -- THE LOUD HALF. Pre-fix this raised UnboundLocalError and the
    #    except turned it into a "write FAILED" log line.
    rows, reduced, err = _run(block, step=30, is_main=False, bind_tps=False)
    _check("rank 1 step 30 error", err, None)
    _check("rank 1 step 30 rows", [r for r in rows if "_runlog" not in r], [])
    _check("rank 1 step 30 logged no failure", [r for r in rows if "_runlog" in r], [])
    _check("rank 1 step 30 entered the collective", reduced, 1)

    # 3. rank 1 at step 10 -- THE SILENT HALF. Pre-fix the conditional short-circuited to None
    #    before touching tps, so this rank wrote a duplicate row and nothing said so.
    rows, reduced, err = _run(block, step=10, is_main=False, bind_tps=False)
    _check("rank 1 step 10 error", err, None)
    _check("rank 1 step 10 rows (duplicate source)", [r for r in rows if "_runlog" not in r], [])
    _check("rank 1 step 10 entered the collective", reduced, 1)

    # 4. THE LEDGER COUNT, anchored to the REAL ledger rather than to a number this test computes.
    #    b0_moe_e1's runs/moe_diag.jsonl (read on the pod 2026-09-05, 12 rows for name="e1") holds
    #    per step: {10: 2, 20: 2, 30: 1, 100: 1, 200: 2, 300: 2, 400: 2}. That distribution is the
    #    mechanism's signature and it is the INVERSE of what the log suggested: the two steps that
    #    printed "write FAILED" are the only two with a single row, because rank 1 raised there
    #    before writing, while every step that logged nothing has two rows because both ranks wrote.
    #    So the failing steps were the clean ones. Post-fix every due step must have exactly one.
    #    e1 asked for this anchor specifically: a count measured against my own fake proves the
    #    fake, and a fixture fact has to come from outside the thing under test.
    _pre_fix_ledger = {10: 2, 20: 2, 30: 1, 100: 1, 200: 2, 300: 2, 400: 2}
    _due = sorted(_pre_fix_ledger)
    per_step = {}
    for step in _due:
        n = 0
        for is_main in (True, False):
            rows, _, _ = _run(block, step=step, is_main=is_main, bind_tps=is_main)
            n += len([r for r in rows if "_runlog" not in r])
        per_step[step] = n
    _check("one row per due step, both ranks run", per_step, {s: 1 for s in _due})
    _check("total rows over the real ledger's due steps", sum(per_step.values()), len(_due))
    # AND THE PRE-FIX DISTRIBUTION IS NOT REPRODUCED. Asserting the post-fix count alone would
    # pass for code that writes one row because it writes none on rank 0 either.
    _check("pre-fix distribution no longer produced", per_step == _pre_fix_ledger, False)

    # 5. THE COLLECTIVE IS NOT INSIDE THE GUARD. Checked structurally as well as by the counts
    #    above, because this is the half that hangs rather than mismeasures if it regresses: an
    #    all_reduce only rank 0 enters leaves rank 1 waiting at the next reduction forever.
    guard = block.find("if is_main:")
    allred = block.find("all_reduce")
    _check("all_reduce precedes the is_main guard", guard > allred > 0, True)
    _check("diagnostics(reset=...) precedes the guard", 0 < block.find("diagnostics(reset") < guard,
           True)

    if _fails:
        print(f"\ntest_moe_diag_rank: {len(_fails)} failure(s)")
        return 1
    print("\ntest_moe_diag_rank OK: the write is rank 0's, the collective is every rank's, and "
          "each due step in the real ledger's step set yields exactly one row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
