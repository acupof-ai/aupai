#!/usr/bin/env python3
"""An eval takes its mix from the checkpoint, never from a hardcoded default (de-26).

The incident (fb, 2026-09-02): score_matrix defaulted `--mix` to the ladder mix
`data/mix_scale_3.24b.json` while scoring a p500m step-2500 checkpoint. The eval went
looking for domains that run never touched, and cache_guard refused on web_hq's empty
.vocab stamp. The guard was right; the default was wrong. A default that names a real,
existing file cannot be caught by "does this path exist" -- it does exist, it is just not
this checkpoint's.

Two known answers, and the second is the one that matters more:
  1. a checkpoint carrying cfg.mix yields THAT mix, and never the ladder default
  2. a checkpoint carrying no mix REFUSES, rather than falling back

Plus the shape that made the first attempt at this fix wrong: `cfg` is a DICT on every
checkpoint this repo writes, so a getattr-only read returns None and the guard refuses
every real checkpoint -- a correct-looking check that converts a wrong default into a
false refusal, which is worse than the bug. Both shapes are read and both are asserted.

    python3 scripts/test_ckpt_mix_source.py --selftest

# restartable: writes only tiny checkpoints under a mkdtemp it removes.
"""

import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))


class _ObjCfg:
    """cfg as an object, the older checkpoint shape. Module level, not nested: pickle cannot
    serialise a class defined inside a function."""

    mix = "data/mix_500m.json"


def _selftest():
    try:
        import torch
    except ImportError:
        print("SKIP: torch not importable here")
        return 0
    from domain_loss import mix_from_ckpt

    d = tempfile.mkdtemp(prefix="mixsrc.")
    fails = 0
    try:
        # 1. cfg as a DICT -- the shape every checkpoint this repo writes actually uses.
        p = os.path.join(d, "dict.pt")
        torch.save({"cfg": {"mix": "data/mix_500m.json", "seq": 4096}}, p)
        got = mix_from_ckpt(p)
        ok = got.endswith("mix_500m.json") and "3.24b" not in got
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} dict cfg -> {os.path.basename(got)}")

        # 2. cfg as an OBJECT -- older checkpoints, and what the first fix only handled.
        p = os.path.join(d, "obj.pt")
        torch.save({"cfg": _ObjCfg()}, p)
        got = mix_from_ckpt(p)
        ok = got.endswith("mix_500m.json")
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} object cfg -> {os.path.basename(got)}")

        # 3. No mix at all: REFUSE. Falling back is the bug being fixed, so a silent
        #    default here would reintroduce it while the test above still passed.
        for label, obj in (("cfg=None", {"cfg": None}),
                           ("cfg dict without mix", {"cfg": {"seq": 4096}})):
            p = os.path.join(d, "no.pt")
            torch.save(obj, p)
            try:
                got = mix_from_ckpt(p)
                fails += 1
                print(f"  BUG  {label} did not refuse, returned {got}")
            except SystemExit as e:
                ok = "refusing" in str(e)
                fails += 0 if ok else 1
                print(f"  {'ok  ' if ok else 'BUG '} {label} refuses")

        # 4. An explicit --mix wins and the checkpoint is not consulted -- otherwise the
        #    escape hatch the refusal message points at would not work.
        p = os.path.join(d, "no2.pt")
        torch.save({"cfg": None}, p)
        got = mix_from_ckpt(p, "data/mix_scale_3.24b.json")
        ok = got == "data/mix_scale_3.24b.json"
        fails += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} explicit --mix overrides ({got})")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print(f"ckpt mix source: {5 - fails}/5 pass")
    if fails:
        print("FAIL: an eval could score a checkpoint against the wrong mix, or refuse a "
              "checkpoint that carries a valid one.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else _selftest())
