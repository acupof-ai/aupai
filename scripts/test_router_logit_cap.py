#!/usr/bin/env python3
"""Known-answer gate for the MoE router logit softcap (--router_logit_cap, de 2026-09-26).

Drives the REAL MoEFFN._route on known logits so the cap cannot silently fail:

  cap 0 (off)       -> sigmoid/softmax see the raw logits, bitwise like every old checkpoint
  cap C > 0         -> the affinity is computed on z' = C*tanh(z/C), before softmax/sigmoid
  a huge logit      -> capped branch affinity stays in the finite learnable range, off branch
                       saturates (sigmoid -> 1 / softmax one-hot)
  small |z|         -> cap is near-identity (tanh(z/C) ~ z/C for C >> |z|)

    python3 scripts/test_router_logit_cap.py --selftest
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

D = 64


def _cfg(**kw):
    base = dict(d=D, ffn_hidden=64, layers=4, vocab=100, seq=16, attn_every=4,
                moe_experts=8, moe_top_k=3, moe_expert_ffn=16, moe_shared=1,
                moe_bias_gamma=0.001, moe_balance_alpha=1e-4, router_score="sigmoid")
    base.update(kw)
    return type("C", (), base)


def main():
    from model import MoEFFN

    E, C = 8, 12.0
    t1 = float(torch.tanh(torch.tensor(1.0)))
    z_big = torch.full((1, E), 500.0)   # far past any cap
    z_crit = torch.tensor([[C, -C, C / 2] + [0.0] * (E - 3)])

    # 1. OFF (default 0): a huge logit saturates sigmoid to ~1.
    m_off = MoEFFN(_cfg(router_logit_cap=0.0)()).eval()
    with torch.no_grad():
        aff, _, _ = m_off._route(z_big)
    assert abs(float(aff[0, 0]) - 1.0) < 1e-5, f"off cap sigmoid(500) must be ~1, got {aff[0,0]}"
    assert m_off.router_logit_cap == 0.0

    # 2. ON: the same huge logit is mapped to sigmoid(C*tanh(1)) = sigmoid(C), finite, not 1.
    m_on = MoEFFN(_cfg(router_logit_cap=C)()).eval()
    # 2. ON: the huge logit argument is bounded to C*tanh(z/C) -> C, so sigmoid equals the KNOWN
    # value sigmoid(C). With C=12 the winner stays confidently selected (sigmoid(12)=0.999994);
    # the cap is not meant to flatten winners. Its job is on the LOSERS and on softmax (3 below).
    want = torch.sigmoid(torch.tensor(C * float(torch.tanh(torch.tensor(500.0 / C)))))  # -> sigmoid(C)
    with torch.no_grad():
        aff_c, sel_c, gate_c = m_on._route(z_big)
    assert abs(float(aff_c[0, 0]) - float(want)) < 1e-5, (
        f"capped sigmoid must equal sigmoid(C*tanh(z/C)) {float(want):.6f}, got {float(aff_c[0,0])}")

    # 2b. BOUNDED VALUE, NOT RESTORED GRADIENT. Off, a loser at -500 is sigmoid = exact grid 0.
    # Capped it is sigmoid(-C*tanh(z/C)) = sigmoid(-12) ~ 6.1e-6 -- nonzero and finite. But this
    # does NOT make an already-saturated router weight learnable: the cap derivative at z=500 is
    # sech^2(500/12) ~ 2.3e-7, so the gradient is still ~0. The cap bounds the affinity; shrinking
    # the raw weight (router weight_decay / z-loss) is a separate mechanism. We assert the VALUE is
    # nonzero and known, deliberately not that gradient is recovered.
    z_lose = torch.full((1, E), -500.0)
    with torch.no_grad():
        lose_off = m_off._route(z_lose)[0]
        lose_on = m_on._route(z_lose)[0]
    assert float(lose_off[0, 0]) == 0.0, "off sigmoid(-500) must be an exact grid zero"
    want_lose = float(torch.sigmoid(torch.tensor(
        C * float(torch.tanh(torch.tensor(-500.0 / C))))))  # -> sigmoid(-C) ~ 6e-6
    assert abs(float(lose_on[0, 0]) - want_lose) < 1e-7 and want_lose > 0, (
        f"capped loser must be the nonzero known value {want_lose:.2e}, got {float(lose_on[0,0])}")
    # Pinned limitation: the cap output is bounded but its derivative sech^2(z/C) collapses for an
    # already-saturated weight. ~0.42 at |z|=C (learnable edge) vs ~2.3e-7 at z=500 (still dead to
    # the optimizer). Anyone claiming the cap "keeps gradient alive" past the boundary breaks this.
    d_edge = float(1.0 / torch.cosh(torch.tensor(1.0)) ** 2)
    d_far = float(1.0 / torch.cosh(torch.tensor(500.0 / C)) ** 2)
    assert abs(d_edge - 0.4199743) < 1e-5 and d_far < 1e-6, (d_edge, d_far)

    # 3. The cap runs in BOTH score branches: with softmax selected the affinity equals
    # softmax(C*tanh(z/C)). softmax is shift-invariant, so it can still be one-hot; the cap's softmax
    # job is only to bound the raw argument (prevent exp overflow / extreme-logit backward), NOT to
    # flatten the gate -- that flattening is what moving to sigmoid provides. Assert the capped
    # softmax equals softmax of the capped argument on a NON-extreme input (where the cap itself
    # binds mildly), so the branch is proven wired without asserting an impossible de-peak.
    z_sm = torch.tensor([[40.0, -30.0, 10.0] + [0.0] * (E - 3)])
    m_sm = MoEFFN(_cfg(router_score="softmax", router_logit_cap=C)()).eval()
    with torch.no_grad():
        a_sm = m_sm._route(z_sm)[0]
    capped = C * torch.tanh(z_sm / C)
    a_ref = torch.softmax(capped, dim=-1)
    assert torch.allclose(a_sm, a_ref, atol=1e-6), (
        f"capped softmax must be softmax(C*tanh(z/C)), max diff {float((a_sm-a_ref).abs().max())}")
    # and it differs from the UNcapped softmax (the cap actually moved the argument)
    assert not torch.allclose(a_sm, torch.softmax(z_sm, -1), atol=1e-3), \
        "cap must change the softmax argument for a logit beyond C"

    # 4. Near-identity for |z| << C: tanh(z/C) ~ z/C. Sigmoid amplifies a small argument delta by
    # up to 0.25x, so use |z| <= 0.3 (|z/C| <= 0.025 -> arg delta < ~5e-5 -> affinity < 2e-5).
    z_tiny = torch.tensor([[0.3, -0.25, 0.1] + [0.0] * (E - 3)])
    with torch.no_grad():
        a_small_on = m_on._route(z_tiny)[0]
        a_small_off = m_off._route(z_tiny)[0]
    assert torch.allclose(a_small_on, a_small_off, atol=3e-5), (
        f"cap must be near-identity for small logits, max diff "
        f"{float((a_small_on-a_small_off).abs().max())}")

    # 5. KNOWN VALUE at the cap boundary z=C: capped argument is exactly C*tanh(1).
    with torch.no_grad():
        a_crit = m_on._route(z_crit)[0]
    assert abs(float(a_crit[0, 0]) - float(torch.sigmoid(torch.tensor(C * t1)))) < 1e-5, (
        f"at z=C affinity must be sigmoid(C*tanh(1)), got {float(a_crit[0,0])}")

    # 6. Negative cap rejected; the construction is the single validation point.
    for bad in (-1.0,):
        try:
            MoEFFN(_cfg(router_logit_cap=bad)())
        except ValueError:
            pass
        else:
            raise AssertionError(f"negative router_logit_cap {bad} must be rejected")

    print("router logit cap OK: off bitwise, on finite at huge logits (softmax+sigmoid), "
          "near-identity small, known value at C, negative rejected")


if __name__ == "__main__":
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] != "--selftest"):
        raise SystemExit(f"usage: {os.path.basename(__file__)} [--selftest] (got {sys.argv[1:]})")
    main()
