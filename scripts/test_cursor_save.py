#!/usr/bin/env python3
"""save_checkpoint must write the row cursor, for every shape `cfg` arrives in (de-8 D1).

THIS TEST FAILS ON train.py AS OF 2026-09-01 12:20, and that is the point of landing it
first. `train.py:1289` rebinds `cfg` to a dict on every path:

    :1289   cfg = cfg if isinstance(cfg, dict) else vars(cfg)
    :1315   cur = getattr(cfg, "_row_cursor", None) if not isinstance(cfg, dict) else None

so by :1315 the isinstance test is always true, `cur` is always None, and the whole
`if cur:` block -- row_cursor, row_cursor_as_of_step, row_cursor_srcfp, row_cursor_seed --
never executes. No checkpoint any run has ever written carries a cursor. Verified on the
live pod: ckpt_pretrain_30b_s2.pt.step21500 has keys [cfg, vocab_id, corpus_fp, env_fp,
step] and no row_cursor; the stage-1 checkpoint has one only because replay_cursor.py
injected it, which its row_cursor_reconstructed marker records.

Why this test rather than the de-7 rehearsal that already passed: the rehearsal exercised
the RECONSTRUCTION path -- replay_cursor writes the dict, a resume reads it -- which
touches neither :1289 nor :1315. The production writer was never called by any test. This
one calls save_checkpoint itself, which is the only thing that would have caught it.

Three cfg shapes because train.py passes the Cfg CLASS while other callers pass an
instance or a namespace, and `vars()` behaves differently on each (mappingproxy vs dict).
A fix that works for one and not the others is the same defect wearing a smaller hat.

    python3 scripts/test_cursor_save.py

Exit 0 = the cursor survives every shape. Exit 1 = D1 is still open.
"""

import os
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def shapes(mid_run=False):
    """The three ways `cfg` reaches save_checkpoint, each carrying a cursor.

    mid_run selects the branch. Without a step the plan-complete branch runs and
    `cfg.batch` is never read -- so the plan-complete case alone would pass with the
    SECOND defect still in place. See main() for why that matters.
    """

    class Cfg:
        mix = None  # no mix file: skips the corpus_fp walk, which is not under test
        seq = 4096
        batch = 16
        accum = 2
        vocab = 32784
        _row_cursor = {"code_rp1t": 1_074_090, "zh_web": 401_178}
        _row_cursor_srcfp = {"code_rp1t": "d8b9b18b", "zh_web": "a0d44fc4"}
        _plan_domains = None  # no step given below, so the plan-complete branch is taken
        _plan_names = None
        seed = 42
        sample_seed = None

    if mid_run:
        # 8000 plan rows, first half domain 0 and second half domain 1. At step 100 with
        # batch 16 accum 2 the run has read 3200 rows, all inside domain 0's half -- a
        # count the plan-complete branch could not produce, so the assertion distinguishes
        # the two branches rather than merely observing that some cursor was written.
        import torch

        idx = torch.zeros(8000, dtype=torch.int8)
        idx[4000:] = 1
        Cfg._plan_domains = idx
        Cfg._plan_names = ["code_rp1t", "zh_web"]

    inst = Cfg()
    ns = types.SimpleNamespace(**{k: v for k, v in vars(Cfg).items() if not k.startswith("__")})
    return [("Cfg class", Cfg), ("instance", inst), ("SimpleNamespace", ns)]


def main():
    import torch

    from train import save_checkpoint

    bad = []
    for label, cfg in shapes():
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ck.pt")
            save_checkpoint(p, {"w": torch.zeros(1)}, cfg, "deadbeefdeadbeef")
            ck = torch.load(p, map_location="cpu", weights_only=False)
            got = ck.get("row_cursor")
            if not got:
                bad.append(f"{label}: no row_cursor in the written checkpoint "
                           f"(keys: {sorted(k for k in ck if k != 'model')})")
                continue
            if got != {"code_rp1t": 1_074_090, "zh_web": 401_178}:
                bad.append(f"{label}: row_cursor is {got}, not what cfg carried")
            if not ck.get("row_cursor_srcfp"):
                bad.append(f"{label}: row_cursor written without its srcfp -- a count "
                           f"against an unknown corpus is not interpretable")

    # The MID-RUN branch, which the plan-complete cases above never enter. It is the
    # only path that reads cfg.batch and cfg.accum, and those were read off the same
    # rebound mapping -- so the two defects cancelled: the write block was unreachable,
    # and had it been reachable it would have raised AttributeError at the first
    # checkpoint of every run. Fixing one without the other trades a silent wrong
    # answer for a crash, so the test has to cover both. --auto-resume makes a mid-plan
    # checkpoint the expected case, not the rare one.
    for label, cfg in shapes(mid_run=True):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ck.pt")
            try:
                save_checkpoint(p, {"w": torch.zeros(1)}, cfg, "deadbeefdeadbeef", step=100)
            except Exception as e:
                bad.append(f"{label} (mid-run): {type(e).__name__}: {e}")
                continue
            ck = torch.load(p, map_location="cpu", weights_only=False)
            got = ck.get("row_cursor")
            if not got:
                bad.append(f"{label} (mid-run): no row_cursor")
                continue
            if ck.get("row_cursor_as_of_step") != 100:
                bad.append(f"{label} (mid-run): as_of_step {ck.get('row_cursor_as_of_step')}, "
                           f"expected 100 -- a plan-complete count would seed stage 2 past "
                           f"everything between the checkpoint and the plan's end")
            if got.get("code_rp1t") != 3200:
                bad.append(f"{label} (mid-run): code_rp1t {got.get('code_rp1t')}, expected "
                           f"3200 = 100 steps * batch 16 * accum 2, all within domain 0")
            if ck.get("row_cursor_seed") != 42:
                bad.append(f"{label} (mid-run): seed {ck.get('row_cursor_seed')}, expected "
                           f"cfg's 42 -- a cursor replayed under a different shuffle seed "
                           f"names different rows")

    # RESUMED: absolute step against a relative plan. This is the case that survived the
    # first fix, because stage 1 starts at step 0 where absolute and relative are equal --
    # so every test written against stage 1 passes with the defect intact. At step 24000
    # against a 523,158-row plan the index runs past the array, Python's slice CLAMPS
    # instead of raising, and the cursor written is the plan-complete count wearing an
    # as-of-step label (tilerl, 2026-09-01). Two sub-cases: the origin known, and the
    # origin missing so the count is impossible.
    for label, cfg in shapes(mid_run=True):
        cfg._plan_step_origin = 16000  # resumed here; the plan below is stage 2's
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ck.pt")
            save_checkpoint(p, {"w": torch.zeros(1)}, cfg, "deadbeefdeadbeef", step=16100)
            ck = torch.load(p, map_location="cpu", weights_only=False)
            got = ck.get("row_cursor")
            if not got:
                bad.append(f"{label} (resumed): no row_cursor -- refused "
                           f"({ck.get('row_cursor_refused', 'no reason given')})")
                continue
            # 100 RELATIVE steps * 16 * 2 = 3200 rows, all in domain 0. The absolute
            # reading would be 16100*32 = 515,200 rows, clamped to the 8000-row plan,
            # giving code_rp1t 4000 and zh_web 4000 -- plan-complete, not as-of-step.
            if got.get("code_rp1t") != 3200:
                bad.append(f"{label} (resumed): code_rp1t {got.get('code_rp1t')}, expected "
                           f"3200 from 100 RELATIVE steps; the absolute step would clamp to "
                           f"the plan end and report {got.get('code_rp1t')} as if measured")

    # The origin is wrong or absent: the count is impossible and the writer must refuse,
    # not clamp. A missing cursor costs a resume that repeats rows; a wrong one is
    # indistinguishable from a right one to every later reader.
    for label, cfg in shapes(mid_run=True):
        cfg._plan_step_origin = 0  # forgotten after a resume
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ck.pt")
            save_checkpoint(p, {"w": torch.zeros(1)}, cfg, "deadbeefdeadbeef", step=16100)
            ck = torch.load(p, map_location="cpu", weights_only=False)
            if ck.get("row_cursor"):
                bad.append(f"{label} (bad origin): wrote a cursor {ck['row_cursor']} for "
                           f"515,200 rows against an 8000-row plan -- a clamp, reported as "
                           f"a measurement")
            elif "row_cursor_refused" not in ck:
                bad.append(f"{label} (bad origin): refused silently; the checkpoint must "
                           f"say why no cursor was written")
            # the srcfp/seed writes must still happen on the refusal path
            elif not ck.get("row_cursor_srcfp"):
                bad.append(f"{label} (bad origin): the refusal path skipped row_cursor_srcfp")

    if bad:
        print("FAIL: save_checkpoint dropped the cursor")
        for b in bad:
            print(f"  {b}")
        print("\nD1 is open: train.py:1289 rebinds cfg to a dict, so the guard at :1315 "
              "always takes its else-None branch.")
        return 1
    print(f"OK: row_cursor and row_cursor_srcfp survive all {len(shapes())} cfg shapes, "
          f"plan-complete and mid-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
