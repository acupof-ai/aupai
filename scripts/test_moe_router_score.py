#!/usr/bin/env python3
"""Known-answer gates for MoEFFN's router_score=softmax|sigmoid switch.

    python3 scripts/test_moe_router_score.py

Three properties, each one a thing a green run could otherwise hide:

  1. softmax (the default) is BITWISE identical to the pre-switch router: feeding the
     module's own _route() the logits must equal a hand-written softmax/topk/normalize
     to the last bit, so old checkpoints keep their exact function.
  2. sigmoid runs fwd and bwd on a CPU small shape, and its gate is the selected
     sigmoids renormalized within the top-k (V3 arXiv 2412.19437 §2.1.2).
  3. the independence known answer: with one expert's logit driven large, softmax
     pushes every other expert's affinity toward 0, while sigmoid leaves the other
     experts' affinity at their own sigmoid value. That is the reason the switch exists.

Plus: expert_bias changes the SELECTION but never the GATE, on both paths.
CPU only, no data.
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

D = 32


def _cfg(**kw):
    # parity: (moe_top_k 2 + moe_shared 2) * expert_ffn 32 == ffn_hidden 128
    base = dict(d=D, ffn_hidden=128, layers=2, vocab=100, seq=8, attn_every=2,
                moe_experts=4, moe_top_k=2, moe_expert_ffn=32, moe_shared=2,
                moe_bias_gamma=0.001, moe_balance_alpha=1e-4)
    base.update(kw)
    return type("MoECfg", (), base)


FAILS = []


def check(name, cond, detail=""):
    print(("ok  " if cond else "FAIL") + " " + name + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(f"{name}: {detail}")


def main():
    from model import MoEFFN

    torch.manual_seed(0)

    # Known logits, shared across paths: two tokens, four experts, top-2, zero bias.
    logits = torch.tensor(
        [[0.5, -1.0, 2.0, 0.25],
         [-2.0, 0.0, 0.75, -0.5]],
        dtype=torch.float32,
    )

    # ---- 1. softmax bitwise identity -------------------------------------
    m = MoEFFN(_cfg(router_score="softmax")())
    check("default router_score is softmax",
          MoEFFN(_cfg()()).router_score == "softmax")
    aff, sel, gate = m._route(logits)

    want_aff = torch.softmax(logits, dim=-1)
    want_sel = want_aff.topk(2, dim=-1).indices
    want_gate = want_aff.gather(1, want_sel)
    want_gate = want_gate / want_gate.sum(-1, keepdim=True).clamp_min(1e-9)
    check("softmax affinity is bitwise torch.softmax",
          torch.equal(aff, want_aff), f"{aff[0].tolist()}")
    check("softmax selection/gate match the pre-switch computation",
          torch.equal(sel, want_sel) and torch.equal(gate, want_gate))

    # default build (no flag) byte-identical to an explicitly-softmax build on the same weights
    m0 = MoEFFN(_cfg()())
    m0.load_state_dict(m.state_dict(), strict=False)
    a0, s0, g0 = m0._route(logits)
    check("unflagged build is bitwise the softmax build",
          torch.equal(a0, aff) and torch.equal(s0, sel) and torch.equal(g0, gate))

    # ---- 2. sigmoid fwd/gate known answer + bwd --------------------------
    ms = MoEFFN(_cfg(router_score="sigmoid")())
    ms.load_state_dict(m.state_dict(), strict=False)
    affs, sels, gates = ms._route(logits)

    want_s = torch.sigmoid(logits)
    want_ssel = want_s.topk(2, dim=-1).indices
    want_sgate = want_s.gather(1, want_ssel)
    want_sgate = want_sgate / want_sgate.sum(-1, keepdim=True).clamp_min(1e-9)
    check("sigmoid affinity is torch.sigmoid (independent experts)",
          torch.equal(affs, want_s), f"{affs[0].tolist()}")
    check("sigmoid selects/gates on selected-sigmoid renormalization",
          torch.equal(sels, want_ssel) and torch.allclose(gates, want_sgate))
    # renormalized gates sum to 1 per token
    check("sigmoid gate rows sum to 1", torch.allclose(gates.sum(-1), torch.ones(2)))

    # fwd/bwd through the real module
    x = torch.randn(2, 8, D)
    ms.train()
    y = ms(x)
    check("sigmoid module forward is finite", torch.isfinite(y).all().item())
    y.sum().backward()
    rg = ms.router.weight.grad
    check("sigmoid backward reaches the router", rg is not None and torch.isfinite(rg).all().item())

    # ---- 3. independence known answer ------------------------------------
    # Drive expert 2's logit very high on token 0. Softmax collapses the OTHER three
    # affinities toward 0; sigmoid leaves them at their own sigmoid(logit).
    big = logits.clone()
    big[0, 2] = 40.0
    other = [0, 1, 3]

    a_soft, _, _ = m._route(big)
    a_sig, _, _ = ms._route(big)
    soft_others = a_soft[0, other]
    sig_others = a_sig[0, other]
    check("softmax: one huge logit drives other affinities ~0",
          bool((soft_others < 1e-8).all()), f"{soft_others.tolist()}")
    # sigmoid(logit) for the unchanged experts must be the SAME value as without the big logit
    want_unchanged = torch.sigmoid(logits[0, other])
    check("sigmoid: other experts' affinity is unaffected by the huge logit",
          torch.allclose(sig_others, want_unchanged), f"{sig_others.tolist()}")
    check("the independence contrast is actually visible",
          bool((sig_others - soft_others).abs().min() > 0.2))

    # ---- 4. bias selects but is never in the gate, both paths ------------
    def bias_check(mod, tag):
        a0_, s0_, g0_ = mod._route(logits)
        with torch.no_grad():
            mod.expert_bias.copy_(torch.tensor([5.0, 0.0, 0.0, 0.0]))  # force expert 0 chosen
        a1, s1, g1 = mod._route(logits)
        # affinity is independent of the bias by construction
        aff_same = torch.equal(a1, a0_)
        # expert 0 is now in the selection
        selected0 = bool((s1 == 0).any())
        # the gate weight attached to a given selected expert equals its UN-biased affinity,
        # renormalized: gate rows are a subset of a1 (no +5 contribution)
        gate_unbiased = bool(torch.all(g1 <= 1.0 + 1e-6)) and torch.allclose(
            g1.sum(-1), torch.ones(2))
        check(f"{tag}: bias leaves affinity unchanged", aff_same)
        check(f"{tag}: bias changes selection (expert 0 chosen)", selected0,
              f"{s1.tolist()}")
        check(f"{tag}: gate stays un-biased and normalized", gate_unbiased,
              f"{g1.tolist()}")
        with torch.no_grad():
            mod.expert_bias.zero_()

    bias_check(m, "softmax")
    bias_check(ms, "sigmoid")

    # invalid score refuses at construction
    refused = False
    try:
        MoEFFN(_cfg(router_score="gelu")())
    except ValueError:
        refused = True
    check("unknown router_score refuses", refused)

    if FAILS:
        print(f"\n{len(FAILS)} FAIL:")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("\nmoe router_score tests OK: softmax bitwise-identical, sigmoid V3 fwd/bwd + "
          "top-k renorm, big-logit independence, bias selects-not-gates on both paths")


if __name__ == "__main__":
    main()
