#!/usr/bin/env python3
"""Does the wired balancer actually move expert_bias, once per optimizer step?

Three worlds, run on GPU because chunk_kda/l2norm are Triton with no CPU fallback
(memory: model-cannot-forward-on-cpu). This tests the MODULE contract that the
train.py call site depends on, not the call site itself -- the call site is proved
by the 300-step probe, whose checkpoint must come back with a nonzero bias.

World A: gamma 0.0 -> bias must stay exactly zero (negative control: if it moves,
         something other than gamma*sign(err) is writing it).
World B: gamma 0.001, ONE update on a deliberately skewed load -> every entry must
         move by exactly gamma, sign opposite to the load error.
World C: the accum trap. Two forwards then ONE update, vs two forwards each with
         its own update. The first must move the bias half as far as the second.
         This is the failure the docstring names and the reason the call sits at
         the accum boundary rather than inside the micro-batch loop.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

D = 64
N_EXPERTS = 8
TOP_K = 2


def _cfg(**kw):
    """A config whose parity holds: (top_k + shared) * expert_ffn == ffn_hidden.

    Field names come from scripts/test_moe_module.py's fixture, which is the working one --
    the module reads cfg.d, not cfg.dim, and a fixture that guesses raises inside __init__.
    """
    base = dict(d=D, ffn_hidden=96, layers=4, vocab=100, seq=16, attn_every=4,
                moe_experts=N_EXPERTS, moe_top_k=TOP_K, moe_expert_ffn=32, moe_shared=1,
                moe_bias_gamma=0.001, moe_balance_alpha=1e-4)
    base.update(kw)
    return type("MoECfg", (), base)


def build(gamma):
    from model import MoEFFN
    m = MoEFFN(_cfg(moe_bias_gamma=gamma)).cuda().to(torch.bfloat16)
    m.train()
    return m


def main():
    assert torch.cuda.is_available(), "needs a GPU: chunk_kda/l2norm are Triton, no CPU fallback"
    torch.manual_seed(0)
    fails = []

    # ---- World A: gamma 0 is inert
    m = build(0.0)
    counts = torch.tensor([100, 0, 50, 50, 50, 50, 50, 50], device="cuda")
    m.update_bias(counts)
    mv = float(m.expert_bias.abs().max())
    print(f"A gamma=0.0            absmax after update = {mv:.6f}   (want exactly 0)")
    if mv != 0.0:
        fails.append(f"A: gamma 0 moved the bias by {mv}")

    # ---- World B: one update moves every entry by exactly gamma, against the load
    g = 0.001
    m = build(g)
    counts = torch.tensor([100, 0, 50, 50, 50, 50, 50, 50], device="cuda")
    mean = counts.float().mean()
    m.update_bias(counts)
    b = m.expert_bias.float().cpu()
    print(f"B gamma={g} mean_count={float(mean):.1f}")
    print(f"   counts {counts.tolist()}")
    print(f"   bias   {[round(float(x), 6) for x in b]}")
    for i, (cnt, bias) in enumerate(zip(counts.float().cpu().tolist(), b.tolist(), strict=True)):
        want = -g if cnt > mean else (g if cnt < mean else 0.0)
        # bf16 cannot represent 0.001 exactly; compare at bf16 resolution
        if abs(bias - want) > 4e-6:
            fails.append(f"B: expert {i} count {cnt} mean {float(mean)} -> bias {bias}, want {want}")
    if not [f for f in fails if f.startswith("B")]:
        print("   OK: every entry moved by gamma, sign opposite to the load error")

    # ---- World C: the accum trap. one update per STEP vs one per MICRO-BATCH.
    torch.manual_seed(1)
    x = torch.randn(2, 16, D, device="cuda", dtype=torch.bfloat16)

    m1 = build(g)                      # correct: 2 forwards, 1 update on the summed counts
    torch.manual_seed(2)
    m1(x)
    m1(x)
    c1 = m1.step_tokens_per_expert.clone()
    m1.update_bias(c1)
    once = m1.expert_bias.float().abs().max().item()

    m2 = build(g)                      # wrong: 2 forwards, 1 update EACH
    torch.manual_seed(2)
    m2(x)
    m2.update_bias(m2.step_tokens_per_expert.clone())
    m2.step_tokens_per_expert.zero_()
    m2(x)
    m2.update_bias(m2.step_tokens_per_expert.clone())
    twice = m2.expert_bias.float().abs().max().item()

    print(f"C accum 2: one update per step absmax {once:.6f} | one per micro-batch {twice:.6f}")
    print(f"   ratio {twice / once if once else float('nan'):.2f}x  (want 2.00x)")
    if once <= 0:
        fails.append("C: the per-step update did not move the bias at all")
    elif abs(twice / once - 2.0) > 0.01:
        fails.append(f"C: per-micro-batch is {twice / once:.3f}x the per-step move, want 2.0")
    else:
        print("   OK: calling per micro-batch would double the effective gamma, as documented")

    # ---- the buffer the call site reads must exist and count the step's load
    m3 = build(g)
    assert hasattr(m3, "step_tokens_per_expert"), "call site reads step_tokens_per_expert"
    torch.manual_seed(3)
    m3(x)
    a = int(m3.step_tokens_per_expert.sum())
    m3(x)
    bsum = int(m3.step_tokens_per_expert.sum())
    want = 2 * 16 * TOP_K
    print(f"D step counter: after 1 fwd {a}, after 2 fwd {bsum}, want {want} then {2 * want}")
    if a != want or bsum != 2 * want:
        fails.append(f"D: counter {a}/{bsum}, want {want}/{2 * want}")
    else:
        print("   OK: accumulates across micro-batches, tokens x top_k per forward")

    print()
    if fails:
        print("FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS: all four worlds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
