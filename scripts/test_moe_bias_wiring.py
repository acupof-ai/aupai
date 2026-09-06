#!/usr/bin/env python3
"""Does the wired balancer actually move expert_bias, once per optimizer step, without windup?

Seven worlds, run on GPU because chunk_kda/l2norm are Triton with no CPU fallback
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
World D: the counter the call site reads accumulates tokens x top_k per forward.

E, F and G are the post-mortem of b0_moe48_8b, stopped at step ~1080 with load gini
0.5959 (0.0935 at step 500). Cause: update_bias was an unbounded integrator whose
runaway term could not affect routing at all, and whose windup consumed the bf16
resolution that the routing-relevant spread needed.

World E: 1000 updates under a persistent 5-of-48 asymmetry -> |mean| < 1e-6 and the
         spread must GROW. Unprojected, the mean reaches +0.79 here and did reach
         +0.4997 on the real run.
World F: fp32 SURVIVING train.py's cast (.to(bfloat16) then .cuda(), both of which
         route through _apply). A dtype assertion on a fresh module is vacuous:
         torch.zeros is already fp32.
World G: the cast must not ROUND THE VALUES, which F cannot see. Sub-bf16
         differences (0.5, 0.501, 0.502) must survive .to(bfloat16) exactly; an
         override that re-floats AFTER the cast passes F and fails G, because the
         differences are already gone. train.py loads before it casts, so that
         rounding would apply once per resume.
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

    # SEED BEFORE build, not after: build() draws the router weights from the RNG, so seeding
    # only the forwards left m1 and m2 with DIFFERENT routers. Measured on card 7: m1 counted
    # [5,9,9,6,10,6,8,11] and m2 [7,7,7,12,4,7,9,11], and the projected absmax is
    # g*max|sign(err) - mean(sign(err))| -- 1.125g for m1's 4+/3-/1-zero pattern, 1.25g for
    # m2's 3+/5-. The ratio read 2*1.25/1.125 = 2.222x and the world went red on a fixture
    # defect, in all three trees including the pre-fix ancestor. The doubling claim is only
    # about calling update_bias twice; it needs the two models to route identically.
    torch.manual_seed(2)
    m1 = build(g)                      # correct: 2 forwards, 1 update on the summed counts
    m1(x)
    m1(x)
    c1 = m1.step_tokens_per_expert.clone()
    m1.update_bias(c1)
    once = m1.expert_bias.float().abs().max().item()

    torch.manual_seed(2)
    m2 = build(g)                      # wrong: 2 forwards, 1 update EACH
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

    # ---- E: the integrator does not wind up. THE WORLD THAT b0_moe48_8b DIED IN.
    # A PERSISTENT ASYMMETRY, not a random one: 5 of 48 experts below mean every step is what
    # the failed run measured (common mode drifted 0.786 gamma/step, which is (48-2*5)/48 =
    # 0.79). Without the zero-mean projection the mean bias integrates linearly and reached
    # +0.4997 by step 1000, at which point bf16's 0.00391 spacing had rounded the SPREAD -- the
    # only part that routes -- down to 2-3 distinct values across 48 experts.
    m4 = build(g)
    nr = m4.expert_bias.numel()   # model.py:801 -- n_routed == moe_experts
    n_below = 5
    counts_e = torch.full((nr,), 100, dtype=torch.long, device="cuda")
    counts_e[:n_below] = 10  # these read below mean every single step
    for _ in range(1000):
        m4.update_bias(counts_e)
    mean_abs = m4.expert_bias.float().mean().abs().item()
    spread_e = (m4.expert_bias.float().max() - m4.expert_bias.float().min()).item()
    drift_per_step = (1.0 - 2.0 * n_below / nr)
    print(f"E 1000 updates, {n_below}/{nr} below mean every step:")
    print(f"   |mean bias| {mean_abs:.3e}  (want < 1e-6; unprojected would reach "
          f"{1000 * g * drift_per_step:.4f})")
    print(f"   spread {spread_e:.6f}  (want > 0: the differential signal must GROW)")
    if mean_abs >= 1e-6:
        fails.append(f"E: |mean bias| {mean_abs:.3e} after 1000 updates -- the integrator is "
                     f"winding up; unprojected drift would be {1000 * g * drift_per_step:.4f}")
    elif spread_e <= 0:
        fails.append(f"E: spread {spread_e:.6f} -- zero-mean must not flatten the signal it "
                     f"exists to preserve")
    else:
        print("   OK: mean pinned at zero, spread grows -- windup cannot consume the resolution")

    # ---- F: fp32 SURVIVING train.py's CAST, which is the only version of this check that means
    # anything. A dtype assertion on a freshly-constructed module passes trivially: torch.zeros
    # defaults to fp32, so register_buffer's dtype= argument changes nothing on its own. The
    # buffer became bf16 because train.py's --fp8 branch and its --bf16 branch both call
    # `raw_model.to(torch.bfloat16)`, which walks every floating buffer -- which is why
    # ckpt_b0_moe48_8b.pt.step1000 holds bf16 despite the declaration. So this world casts the
    # way the arms cast, and asserts AFTER. On 87ef5985 (dtype= only, no _apply override) it is
    # RED; with the override it is green. 4c caught that the first version of this check would
    # have passed on a module that never went through the cast.
    #
    # BOTH CLAUSES: the dtype after the cast, AND the behaviour at the magnitude that broke --
    # a gamma step at 0.5 must move the value by ~gamma, which bf16's 0.00391 spacing cannot do.
    # The dtype alone would miss a later re-cast; the behaviour alone would not say why.
    m5 = build(g)                      # build() already ends in .to(torch.bfloat16)
    m5 = m5.to(torch.bfloat16)         # and again, explicitly, exactly as train.py does
    m5 = m5.cuda()                     # .cuda() routes through _apply too, so it must also hold
    print(f"F dtype after .to(bfloat16) and .cuda(): {m5.expert_bias.dtype}  (want torch.float32)")
    if m5.expert_bias.dtype != torch.float32:
        fails.append(f"F: expert_bias is {m5.expert_bias.dtype}, want float32 -- bf16 spacing at "
                     f"0.5 is 0.00391, four times the gamma {g} step")
    else:
        with torch.no_grad():
            m5.expert_bias.fill_(0.5)
        before = m5.expert_bias[0].item()
        counts_f = torch.full((m5.expert_bias.numel(),), 100, dtype=torch.long, device="cuda")
        counts_f[0] = 1000  # expert 0 alone is overloaded -> -gamma, others +gamma/...
        m5.update_bias(counts_f)
        moved = abs(m5.expert_bias[0].item() - before)
        # the projection subtracts the mean, so expert 0 moves gamma plus the mean shift
        print(f"   0.5 -> {m5.expert_bias[0].item():.6f}, |delta| {moved:.6f} "
              f"(bf16 could not represent a {g} step here)")
        if moved < g * 0.5:
            fails.append(f"F: a {g} step at magnitude 0.5 moved the value by only {moved:.6f} "
                         f"-- the resolution loss that destroyed the spread is still present")
        else:
            print("   OK: a gamma step is representable at the magnitude the run reached")

    # ---- G: the CAST MUST NOT ROUND THE VALUES, not merely leave the dtype fp32. World F is
    # blind to this and that blindness is the defect it missed (4c, reviewing 2f95a797): an
    # override that does `fn(eb).float()` restores the dtype AFTER fn has already rounded to
    # bf16's grid, so every dtype assertion passes on numbers that have lost their differences.
    #
    # SUB-bf16 DIFFERENCES ARE THE SUBJECT. At magnitude 0.5 the grid step is 0.00391, so
    # 0.5/0.501/0.502 -- three biases a gamma-0.001 loop produces routinely -- collapse to
    # [0.5, 0.5, 0.50390625]: three distinct values become two. That is the differential collapse
    # in miniature, and it is what would be applied ONCE PER RESUME, because train.py loads the
    # checkpoint at :3043 and casts at :3134/:3162 -- load before cast.
    # RED on 2f95a797, green with the keep-the-original-tensor form.
    m6 = build(g)
    probe = torch.tensor([0.5, 0.5 + g, 0.5 + 2 * g], dtype=torch.float32,
                         device=m6.expert_bias.device)
    with torch.no_grad():
        m6.expert_bias[:3].copy_(probe)
    kept = m6.expert_bias[:3].clone()
    m6 = m6.to(torch.bfloat16)          # exactly what train.py's --fp8 and --bf16 branches do
    got = m6.expert_bias[:3]
    same = torch.equal(kept, got)
    print(f"G sub-bf16 differences through .to(bfloat16): {[round(v, 6) for v in got.tolist()]}")
    print(f"   want {[round(v, 6) for v in kept.tolist()]}  (bf16 grid at 0.5 is 0.00391, "
          f"so a {g} difference cannot survive a round-trip)")
    if not same:
        fails.append(f"G: the cast ROUNDED the values -- {kept.tolist()} became {got.tolist()}. "
                     f"The dtype is restored but the differences are gone, which is the collapse "
                     f"this override exists to prevent, applied once per resume")
    else:
        print("   OK: values identical, so a resume does not re-round what the last run learned")

    print()
    if fails:
        print("FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS: all seven worlds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
