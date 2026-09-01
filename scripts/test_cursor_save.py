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


def shapes():
    """The three ways `cfg` reaches save_checkpoint, each carrying a cursor."""

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

    if bad:
        print("FAIL: save_checkpoint dropped the cursor")
        for b in bad:
            print(f"  {b}")
        print("\nD1 is open: train.py:1289 rebinds cfg to a dict, so the guard at :1315 "
              "always takes its else-None branch.")
        return 1
    print(f"OK: row_cursor and row_cursor_srcfp survive all {len(shapes())} cfg shapes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
