#!/usr/bin/env python3
"""Parameter accounting for candidate p1 configurations, computed rather than derived.

p1 is the V2 architecture (user order 2026-09-09): every layer attention, CSA and HCA
interleaved, partial RoPE for position, AttnRes on the residual, sparse MoE for the FFN.
Two people are currently blocked on numbers this file produces -- d1 needs `d` to finish the
vocabulary accounting, and the schedule needs an active-parameter count to size the run.

The counts come from building each candidate on the meta device and calling train.py's own
`_n_active_params`, not from arithmetic in a comment. Hand arithmetic is how a config gets
proposed at one size and trained at another: this repo's MoE active/total split alone has
three interacting knobs (experts, top_k, expert_ffn) plus a shared expert whose width must
equal ffn_hidden exactly, and the padded vocabulary is a fourth term nobody remembers.

Run: python3 scripts/p1_size.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402
from train import Cfg, _n_active_params  # noqa: E402


def cfg(**over):
    c = type("C", (Cfg,), {})()
    for k, v in over.items():
        setattr(c, k, v)
    return c


# The architecture is fixed across every candidate; only the size knobs move. Written once so a
# candidate cannot silently differ in architecture as well as in size, which would make the
# comparison below answer a different question than the one it is labelled with.
V2 = dict(attn_every=1, attn_hybrid=True, csa=True, rope_dims=64, attn_res=True,
          moe_top_k=3, moe_shared=1, mem_values=0, head_mixed=0, value_embed=False)

CANDIDATES = [
    # phi-1-small is 350M dense at 45% HumanEval and is the acceptance gate in
    # docs/standards/p1_data_recipe.md. This is the V2 model at that active budget: the cheap
    # run that says whether the corpus is right before the full one spends five days.
    # Chosen by sweeping depth and width against that budget rather than by picking round
    # numbers: d1024/L24 lands at 0.92x of 350M active, where d1280/L20 is 1.18x and d1280/L24
    # is 1.41x. Depth was the knob that moved it least violently -- widening d moves active
    # parameters faster than deepening because the shared expert and the attention projections
    # both scale with d^2.
    ("p1-small", dict(d=1024, layers=24, heads=8, ffn_hidden=2816,
                      moe_experts=32, moe_expert_ffn=704, **V2)),
    # phi-1 is 1.3B dense at 50.6%. Matching its ACTIVE budget is what the reference score
    # is a reference for -- total parameters are what the MoE adds on top, not what does the
    # computing at any one token.
    ("p1", dict(d=2048, layers=24, heads=16, ffn_hidden=5632,
                moe_experts=48, moe_expert_ffn=1408, **V2)),
]

VOCABS = [20000, 32784]


def main():
    print(f"{'name':10s} {'V':>6s} {'d':>5s} {'L':>3s} {'E':>3s} "
          f"{'total':>13s} {'active':>13s} {'act/tot':>8s}")
    for name, over in CANDIDATES:
        for V in VOCABS:
            c = cfg(vocab=V, **over)
            with torch.device("meta"):
                m = M.HybridLM(c)
            total = sum(p.numel() for p in m.parameters())
            active = _n_active_params(m, c)
            print(f"{name:10s} {V:6d} {c.d:5d} {c.layers:3d} {c.moe_experts:3d} "
                  f"{total:13,d} {active:13,d} {active / total:7.1%}")
    print()
    print("Reference, from docs/standards/p1_data_recipe.md: phi-1-small is 350M dense at "
          "HumanEval 45%, phi-1 is 1.3B dense at 50.6%. Those are DENSE parameter counts, so the "
          "column to read against them is `active` -- an MoE's total is what it stores, its "
          "active is what computes a token. A candidate whose active count sits far below the "
          "reference is not the reference's architecture at a discount; it is a smaller model.")


if __name__ == "__main__":
    sys.exit(main())
