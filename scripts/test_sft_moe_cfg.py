#!/usr/bin/env python3
"""Known-answer tests for sft_math.assert_moe_matches_ckpt.

    python3 scripts/test_sft_moe_cfg.py

WHY THIS EXISTS. sft_math.py:151 copies every key of ck["cfg"] onto the Cfg CLASS, then
:263 builds HybridLM(Cfg). That is how "build from ck['cfg']" is implemented here and it
works for the keys the checkpoint carries. It leaves two holes, and MoE is where they bite:

  1. A key the checkpoint does NOT carry keeps whatever the live Cfg class holds. A dense
     checkpoint saved before moe_experts existed, loaded by a Cfg whose default has since
     become non-zero, builds an MoE model from a dense checkpoint.
  2. Anything that mutates Cfg between the copy and the build wins. No MoE flag has a CLI
     override in sft_math.py today, which makes this a guard against the next one.

moe_top_k and moe_layers are the reason the guard cannot be left to load_state_dict: they
change ROUTING and leave every tensor shape intact, so a mismatch loads cleanly and trains
a different model than the one that was pretrained.

CPU-only, no checkpoint, no card: the worlds are real Cfg values mutated one field at a
time, and the model is a stub carrying the one tensor the assertion reads.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# liger_kernel is pod-only and sft_math.py imports it at module scope, so on a laptop this
# file could only SKIP -- and a check that never runs where people commit is not a check
# (scripts/test_sft_prefix.py:55 skips three cases for exactly this reason). The stub
# satisfies ONE unrelated import; the subject is the real sft_math.assert_moe_matches_ckpt,
# imported from the real module. If the loss class is ever used by the code under test this
# stub stops being harmless, so it raises rather than pretending to be a loss.
if "liger_kernel" not in sys.modules:
    try:
        import liger_kernel  # noqa: F401
    except ModuleNotFoundError:
        import types

        _lk = types.ModuleType("liger_kernel")
        _tr = types.ModuleType("liger_kernel.transformers")

        class _NotHere:
            def __init__(self, *a, **k):
                raise RuntimeError(
                    "LigerFusedLinearCrossEntropyLoss is stubbed in this test: it is pod-only "
                    "and nothing here should construct it. If you reached this, the code under "
                    "test now needs liger and this file must run on the pod instead."
                )

        _tr.LigerFusedLinearCrossEntropyLoss = _NotHere
        _lk.transformers = _tr
        sys.modules["liger_kernel"] = _lk
        sys.modules["liger_kernel.transformers"] = _tr

import sft_math  # noqa: E402,F401
from sft_math import MOE_KEYS, assert_moe_matches_ckpt  # noqa: E402
from train import Cfg  # noqa: E402

FAILS = []


class _FFN(nn.Module):
    """Carries w13 with the leading dim the assertion reads, or nothing if dense."""

    def __init__(self, n_routed):
        super().__init__()
        if n_routed:
            self.w13 = nn.Parameter(torch.zeros(n_routed, 2, 2))


class _Block(nn.Module):
    def __init__(self, n_routed):
        super().__init__()
        self.ffn = _FFN(n_routed)


class _Model(nn.Module):
    def __init__(self, n_routed, n_moe=2, n_dense=1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_Block(n_routed) for _ in range(n_moe)] + [_Block(0) for _ in range(n_dense)]
        )


def _live_moe():
    """The live Cfg's MoE fields -- the REAL defaults, read, not transcribed."""
    return {k: getattr(Cfg, k) for k in MOE_KEYS if hasattr(Cfg, k)}


def _expect_ok(model, ck_cfg, what):
    try:
        assert_moe_matches_ckpt(model, ck_cfg)
    except SystemExit as e:
        FAILS.append(f"{what}: refused a matching pair -- {e}")


def _expect_refuse(model, ck_cfg, what, must_name=()):
    try:
        assert_moe_matches_ckpt(model, ck_cfg)
    except SystemExit as e:
        for token in must_name:
            if token not in str(e):
                FAILS.append(f"{what}: refused but the message never names {token!r}: {e}")
        return
    FAILS.append(f"{what}: ACCEPTED a mismatch that must refuse")


def main():
    live = _live_moe()
    if "moe_experts" not in live:
        FAILS.append("Cfg has no moe_experts: this test's premise is gone, it tests nothing")
        print(f"sft moe cfg test: {len(FAILS)} BUG(S)", file=sys.stderr)
        return 1

    saved = {k: getattr(Cfg, k) for k in live}
    try:
        # 1. THE MATCHING CASES PASS. Both directions: dense against a dense checkpoint and
        #    MoE against an MoE one. Without these the whole file is satisfied by a function
        #    that raises unconditionally.
        for k, v in saved.items():
            setattr(Cfg, k, v)
        _expect_ok(_Model(0, n_moe=0, n_dense=3), {}, "dense model, checkpoint with no MoE keys")
        _expect_ok(_Model(0, n_moe=0, n_dense=3), {"moe_experts": 0}, "dense model, moe_experts=0")

        moe_ck = dict(saved, moe_experts=8)
        for k, v in moe_ck.items():
            setattr(Cfg, k, v)
        _expect_ok(_Model(8), moe_ck, "8-expert model, 8-expert checkpoint")

        # 2. THE COUNT MISMATCH REFUSES, both ways round. This is the case load_state_dict
        #    would also catch -- included because the message must name the numbers, and
        #    because a guard that only fires when the state dict already fails is redundant.
        _expect_refuse(_Model(4), moe_ck, "4-expert model vs 8-expert checkpoint", ("8", "4"))
        for k, v in saved.items():
            setattr(Cfg, k, v)
        _expect_refuse(
            _Model(0, n_moe=0, n_dense=3), moe_ck, "dense model vs 8-expert checkpoint", ("8", "0")
        )
        for k, v in moe_ck.items():
            setattr(Cfg, k, v)
        _expect_refuse(_Model(8), {"moe_experts": 0}, "8-expert model vs dense checkpoint", ("0", "8"))

        # 3. THE SHAPE-PRESERVING MISMATCHES REFUSE. moe_top_k and moe_layers change routing
        #    and leave every tensor identical, so load_state_dict accepts them. This is the
        #    half of the guard that is not redundant with the load, and each field is tested
        #    on its own so a check that reads only the first one is caught.
        for field, other in (
            ("moe_top_k", 5),
            ("moe_shared", 2),
            ("moe_expert_ffn", 999),
            ("moe_layers", "0-5"),
        ):
            if field not in saved:
                continue
            ck = dict(moe_ck)
            ck[field] = other if ck[field] != other else saved[field]
            for k, v in moe_ck.items():
                setattr(Cfg, k, v)
            _expect_refuse(_Model(8), ck, f"{field} differs (shape-preserving)", (field,))

        # 4. A CHECKPOINT THAT OMITS A KEY IS NOT A MISMATCH. The copy at :151 cannot set what
        #    is absent, and an older checkpoint legitimately lacks fields added later. The
        #    check must compare only keys the checkpoint actually carries -- a version that
        #    compared all of MOE_KEYS would refuse every pre-MoE checkpoint.
        for k, v in saved.items():
            setattr(Cfg, k, v)
        _expect_ok(
            _Model(0, n_moe=0, n_dense=3),
            {"vocab_id": "x", "step": 100},
            "checkpoint carrying no MoE key at all",
        )

        # 5. THE COUNT IS READ FROM THE TENSOR, NOT FROM THE CONFIG. A model whose w13 says 4
        #    while Cfg says 8 must refuse against an 8-expert checkpoint -- if the assertion
        #    read n_routed or Cfg.moe_experts it would compare the config to itself and pass.
        #    This is the world that separates the real check from the tautological one.
        for k, v in moe_ck.items():
            setattr(Cfg, k, v)
        stub = _Model(4)
        stub.blocks[0].ffn.n_routed = 8  # the attribute LIES; the tensor does not
        _expect_refuse(stub, moe_ck, "w13 says 4 while n_routed attribute says 8", ("8", "4"))

        # 6. A DENSE-ONLY MODEL WITH NO w13 ANYWHERE reads as 0, not as a crash. The blocks
        #    list is scanned with hasattr, so a model built before MoE existed must be a clean
        #    pass rather than an AttributeError that looks like a different bug.
        for k, v in saved.items():
            setattr(Cfg, k, v)
        bare = nn.Module()
        bare.blocks = nn.ModuleList([nn.Module() for _ in range(3)])
        _expect_ok(bare, {}, "model whose blocks have no ffn at all")
    finally:
        for k, v in saved.items():
            setattr(Cfg, k, v)

    for f in FAILS:
        print(f"BUG {f}", file=sys.stderr)
    n = 6
    print(f"sft moe cfg test: {f'PASS ({n} worlds)' if not FAILS else f'{len(FAILS)} BUG(S)'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
