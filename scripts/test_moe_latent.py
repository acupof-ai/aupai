#!/usr/bin/env python3
"""The latent MoE runs its routed experts at d_latent, and parity is still exact.

WHY THIS EXISTS (prereg runs/prereg.jsonl#moe_0905 amendment 13). E1' asks whether a latent
projection buys back the MoE's time penalty: the 24-expert arm costs 16.2% throughput at equal
precision (facts/efficiency.json#eff.moe_structure_cost_e1_bf16), and shrinking the routed path's
width shrinks both the dispatch payload and every expert matmul. The registered cell is
moe_latent 384, moe_expert_ffn 2048, moe_shared_ffn 512, which is EXACT equal-active parity
(2*1024*384 + 3*3*384*2048 + 3*1024*512 = 9,437,184 = 3*d*ffn_hidden) at 1.000x the 24-expert
arm's routed parameters (3*24*384*2048 = 56,623,104).

THE NUMBER 1308 IS WHY THE PARITY CHECK IS TESTED AND NOT TRUSTED. The controller's first ruling
was d_latent 512 with expert_ffn 1308, which came from my own non-integer solution 1308.444; at
1308 the active multiplies are 9,435,136 against the dense 9,437,184, short by 2,048. The
constructor asserts EQUALITY, so that config does not construct at all. A test that only exercised
the registered cell would never have said so, which is why check 3 below drives the refusal.

WHAT IS CHECKED, by constructing real modules and running real forwards on CPU where the dtypes
allow it:
  1. SHAPES: w13 is (E, 2*expert_ffn, d_latent) and w2 is (E, d_latent, expert_ffn) -- the experts
     contract over the LATENT width -- while the shared expert stays at the FULL width d with its
     own moe_shared_ffn, and down/up are (d_latent, d) and (d, d_latent).
  2. PARAMETER COUNT equals the arithmetic, and the routed count matches the 24-expert arm's
     56,623,104 exactly. Asserted against a literal, not against a recomputation of the same
     expression, so a wrong shape cannot agree with a wrong expectation.
  3. PARITY IS REFUSED when it misses, including at the exact off-by-2048 config that was ruled
     before the arithmetic was checked, and the refusal names the three terms.
  4. THE INITIALISATION USES d_latent AS w13's fan_in, not d. This is the silent one: at the
     registered cell the correct bound is 1/sqrt(384) = 0.05103 and the `d` bound would be
     1/sqrt(1024) = 0.03125, a 1.63x scale error that no shape check can see and that would make
     the arm's first steps measure its initialisation.
  5. THE OFF PATH IS BYTE-UNCHANGED: with moe_latent 0 the module has no down/up at all and every
     shape matches the 24-expert arm, so the running arm's behaviour cannot have moved.
  6. FORWARD runs and returns the input's shape and dtype, and the LINEARITY IDENTITY the
     implementation relies on holds: up(sum_k g_k y_k) == sum_k g_k up(y_k), which is what makes
     accumulating at d_latent before the projection exact rather than an approximation.

CPU ONLY WHERE IT IS HONEST. torch._grouped_mm has no CPU kernel for this shape, so check 6 runs
the forward with the grouped op monkey-patched to an equivalent loop -- and the patch is asserted
to be equivalent on a shape the real op DOES handle when a GPU is present. Everything else here is
construction and arithmetic, which need no device. The in-situ assertion is the smoke run.

    python3 scripts/test_moe_latent.py
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import model as M  # noqa: E402

_fails = []


def _check(name, got, want):
    if got != want:
        _fails.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {name}: {got!r}")


def _close(name, got, want, tol):
    if abs(got - want) > tol:
        _fails.append(f"{name}: got {got!r}, want {want!r} +/-{tol}")
        print(f"  FAIL {name}: got {got!r}, want {want!r} +/-{tol}")
    else:
        print(f"  ok   {name}: {got:.6g}")


class _Cfg:
    """Only the fields MoEFFN reads."""

    d = 1024
    ffn_hidden = 3072
    moe_experts = 24
    moe_top_k = 3
    moe_shared = 1
    moe_expert_ffn = 2048
    moe_latent = 384
    moe_shared_ffn = 512
    moe_bias_step = 0.001
    moe_balance_alpha = 0.0

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# The registered cell's arithmetic, written out as literals so a wrong implementation cannot be
# confirmed by a wrong expectation.
D, DL, W, WSH, E, K = 1024, 384, 2048, 512, 24, 3
ROUTED_PARAMS = 56623104          # 3 * 24 * 384 * 2048, and the 24-expert arm's routed count
DENSE_MULTIPLIES = 9437184        # 3 * 1024 * 3072


def main():
    # 1. SHAPES.
    m = M.MoEFFN(_Cfg())
    _check("w13 shape (experts, 2*expert_ffn, d_latent)", tuple(m.w13.shape), (E, 2 * W, DL))
    _check("w2 shape (experts, d_latent, expert_ffn)", tuple(m.w2.shape), (E, DL, W))
    _check("down is (d_latent, d)", tuple(m.down.weight.shape), (DL, D))
    _check("up is (d, d_latent)", tuple(m.up.weight.shape), (D, DL))
    _check("shared sh13 stays at FULL width", tuple(m.sh13.weight.shape), (2 * WSH, D))
    _check("shared sh2 stays at FULL width", tuple(m.sh2.weight.shape), (D, WSH))
    _check("router reads the full-width token", tuple(m.router.weight.shape), (E, D))

    # 2. PARAMETER COUNT, against literals.
    routed = m.w13.numel() + m.w2.numel()
    _check("routed params match the 24-expert arm exactly", routed, ROUTED_PARAMS)
    proj = m.down.weight.numel() + m.up.weight.numel()
    _check("projection params", proj, 2 * D * DL)
    shared = m.sh13.weight.numel() + m.sh2.weight.numel()
    _check("shared params at its own width", shared, 3 * D * WSH)

    # 3. PARITY IS REFUSED WHEN IT MISSES -- including the config that was ruled before the
    #    arithmetic was checked. 1308 at d_latent 512 is 2,048 multiplies short of dense.
    for name, kw, expect in (
        ("the pre-arithmetic ruling (d_latent 512, expert_ffn 1308)",
         dict(moe_latent=512, moe_expert_ffn=1308, moe_shared_ffn=768), True),
        ("expert_ffn one step off the registered cell",
         dict(moe_expert_ffn=W + 1), True),
        ("shared width off the registered cell",
         dict(moe_shared_ffn=WSH + 64), True),
        ("the registered cell itself", {}, False),
    ):
        try:
            M.MoEFFN(_Cfg(**kw))
            refused, msg = False, ""
        except ValueError as e:
            refused, msg = True, str(e)
        _check(f"refused: {name}", refused, expect)
        if refused and expect:
            _check(f"  ...and the message names the three terms ({name[:28]})",
                   all(t in msg for t in ("2*d*d_latent", "3*top_k*d_latent*expert_ffn",
                                          "3*d*shared_ffn")), True)

    # The off-by-2048 is the finding, so state it as a number rather than only as a refusal.
    _check("1308 at d_latent 512 is short by exactly 2048 multiplies",
           DENSE_MULTIPLIES - (2 * D * 512 + 3 * K * 512 * 1308 + 3 * D * 768), 2048)
    _check("the registered cell hits dense multiplies exactly",
           2 * D * DL + 3 * K * DL * W + 3 * D * WSH, DENSE_MULTIPLIES)

    # 4. INITIALISATION FAN-IN. w13 contracts over d_latent, so its bound is 1/sqrt(384).
    #    Checked as the observed max |value| against the two candidate bounds: with 24*2048*384
    #    uniform draws the sample max is within a hair of the true bound, and the two bounds differ
    #    by 1.63x, so this distinguishes them decisively.
    obs = m.w13.abs().max().item()
    _close("w13 init bound is 1/sqrt(d_latent)", obs, 1.0 / math.sqrt(DL), 0.002)
    _check("w13 init bound is NOT 1/sqrt(d)", obs > 1.2 * (1.0 / math.sqrt(D)), True)
    _close("w2 init bound is 1/sqrt(expert_ffn)", m.w2.abs().max().item(),
           1.0 / math.sqrt(W), 0.002)

    # 5. THE OFF PATH IS UNCHANGED -- the arm training right now must not have moved.
    off = M.MoEFFN(_Cfg(moe_latent=0, moe_expert_ffn=768, moe_shared_ffn=0))
    _check("moe_latent 0: no down projection exists", hasattr(off, "down"), False)
    _check("moe_latent 0: no up projection exists", hasattr(off, "up"), False)
    _check("moe_latent 0: w13 is (E, 2w, d)", tuple(off.w13.shape), (E, 2 * 768, D))
    _check("moe_latent 0: w2 is (E, d, w)", tuple(off.w2.shape), (E, D, 768))
    _check("moe_latent 0: shared follows expert_ffn", tuple(off.sh13.weight.shape), (2 * 768, D))
    _check("moe_latent 0: d_latent reads 0", off.d_latent, 0)

    # 6. FORWARD, with _grouped_mm replaced by an equivalent loop because it has no CPU kernel.
    #    THE PATCH IS ASSERTED EQUIVALENT rather than assumed: on CUDA the same inputs go through
    #    both and must agree; without CUDA that check is SKIPPED and said to be skipped.
    def _grouped_loop(a, b_t, offs=None):
        out = []
        lo = 0
        for i, hi in enumerate(offs.tolist()):
            if hi > lo:
                out.append(a[lo:hi] @ b_t[i])
            lo = hi
        return torch.cat(out, 0) if out else a.new_zeros((0, b_t.shape[-1]))

    if torch.cuda.is_available():
        aa = torch.randn(32, DL, device="cuda", dtype=torch.bfloat16)
        bb = torch.randn(E, 2 * W, DL, device="cuda", dtype=torch.bfloat16)
        oo = torch.tensor([8, 16, 24, 32] + [32] * (E - 4), device="cuda", dtype=torch.int32)
        _check("the loop stand-in matches torch._grouped_mm",
               torch.allclose(torch._grouped_mm(aa, bb.transpose(-2, -1), offs=oo).float(),
                              _grouped_loop(aa, bb.transpose(-2, -1), offs=oo).float(),
                              atol=2e-2), True)
    else:
        print("  SKIP the loop-vs-op equivalence check: no CUDA, so it cannot be run. The "
              "forward checks below therefore exercise the loop, NOT the op.")

    def _trace(mod, names):
        """Record the shape of every input each named submodule is called with."""
        seen = {n: [] for n in names}
        for n in names:
            lin = getattr(mod, n)
            orig = lin.forward

            def _w(t, _o=orig, _n=n):
                seen[_n].append(tuple(t.shape))
                return _o(t)

            lin.forward = _w
        return seen

    _real = torch._grouped_mm
    torch._grouped_mm = _grouped_loop
    try:
        # A SMALL CELL SATISFYING PARITY EXACTLY, solved before the module sees it rather than
        # patched onto the object afterwards: my first version assigned moe_shared_ffn after _Cfg
        # was built, at a width the constraint does not admit (the remainder was 128, not 0), so the
        # constructor refused and the forward never ran at all.
        # d 64 / ffn 192 / d_latent 48 / expert_ffn 128 / shared 64:
        # 2*64*48 + 3*1*48*128 + 3*64*64 = 6144 + 18432 + 12288 = 36864 = 3*64*192.
        small = _Cfg(d=64, ffn_hidden=192, moe_experts=4, moe_top_k=1, moe_shared=1,
                     moe_expert_ffn=128, moe_latent=48, moe_shared_ffn=64)
        _check("small-cell parity is exact before construction",
               2 * 64 * 48 + 3 * 1 * 48 * 128 + 3 * 64 * 64, 3 * 64 * 192)
        sm = M.MoEFFN(small)
        x = torch.randn(2, 5, 64)
        y = sm(x)
        _check("forward returns the input shape", tuple(y.shape), (2, 5, 64))
        _check("forward returns the input dtype", y.dtype, x.dtype)
        _check("forward is finite", bool(torch.isfinite(y).all()), True)

        # THE LINEARITY IDENTITY the implementation rests on: accumulating the gated sum at
        # d_latent and projecting ONCE equals projecting each contribution and summing, so the
        # cheap order is exact rather than an approximation.
        g = torch.rand(7, 1)
        v = torch.randn(7, small.moe_latent)
        _check("up(sum g*y) == sum g*up(y)",
               torch.allclose(sm.up((g * v).sum(0, keepdim=True)),
                              (g * sm.up(v)).sum(0, keepdim=True), atol=1e-5), True)

        # WHICH TENSORS THE PROJECTIONS TOUCH, at top_k=2. Two mutants survived the shape-only
        # assertions and both are PLACEMENT rather than shape: feeding the shared expert from
        # up(xr) instead of the full-width token, and down-projecting each DISPATCHED ROW instead
        # of each token -- identical shapes, top_k times the projection cost, which breaks the
        # parity the constructor asserts. top_k MUST be > 1 here or "per token" and "per row" are
        # the same number and neither mutant is distinguishable.
        # d 64 / ffn 192 / d_latent 24 / expert_ffn 64 / top_k 2 / shared 128:
        # 3072 + 9216 + 24576 = 36864 = 3*64*192. SOLVED BY ENUMERATION, not by hand -- my first
        # two attempts at a small cell were arithmetic slips (28672 and 37120 against 36864), and
        # both only surfaced when the constructor refused, which is the constructor doing its job
        # and me wasting two runs of the test.
        c2 = _Cfg(d=64, ffn_hidden=192, moe_experts=4, moe_top_k=2, moe_shared=1,
                  moe_expert_ffn=64, moe_latent=24, moe_shared_ffn=128)
        _check("top_k=2 cell parity is exact",
               2 * 64 * 24 + 3 * 2 * 24 * 64 + 3 * 64 * 128, 3 * 64 * 192)
        m2 = M.MoEFFN(c2)
        seen = _trace(m2, ("down", "up", "sh13"))
        ntok = 3 * 4
        m2(torch.randn(3, 4, 64))
        _check("down projects each TOKEN once, not each dispatched row", seen["down"][0][0], ntok)
        _check("  a per-row projection would have read this many rows", ntok * 2, 24)
        _check("down is called once per forward", len(seen["down"]), 1)
        _check("up is called once per forward", len(seen["up"]), 1)
        _check("up reads the d_latent-wide accumulator", seen["up"][0][-1], 24)
        _check("shared expert is fed the FULL-width token", seen["sh13"][0][-1], 64)
        _check("shared expert is called once per forward", len(seen["sh13"]), 1)
    finally:
        torch._grouped_mm = _real

    if _fails:
        print(f"\ntest_moe_latent: {len(_fails)} failure(s)")
        return 1
    print("\ntest_moe_latent OK: routed experts run at d_latent with the 24-expert arm's exact "
          "routed parameter count, parity is refused when it misses (including the off-by-2048 "
          "config), w13's fan_in is d_latent, the projections are once-per-token with the shared "
          "expert at full width, and moe_latent 0 is unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
