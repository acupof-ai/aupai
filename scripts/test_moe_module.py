#!/usr/bin/env python3
"""MoEFFN's correctness contract: it computes the dense FFN's function, at the dense FFN's cost.

WHAT THIS FILE IS FOR. The MoE arm's whole claim is "same active compute, more reachable
parameters". Two things can silently break that claim, and neither shows up as a crash:
  - the experts compute a DIFFERENT FUNCTION than the SwiGLU they replace, so readout 1's loss
    delta measures the activation change and not the sparsity;
  - the config's active width misses parity, so the arm gets more FLOPs than the control and the
    delta measures compute.
Both already happened once in this program's own bench (b0's review of moe_dispatch_bench.py,
2026-09-05): the dense side ran K3 SiTU-GLU while both MoE paths ran plain a*sigmoid(b), AND an
import-time equal-compute assertion certified a configuration the timed code never ran. That is
why the checks below run the CODE and compare against the REAL module, rather than asserting a
formula.

_grouped_mm RUNS ON CPU, verified 2026-09-05 on torch 2.12, so these tests exercise the real
dispatch path rather than a stand-in. One caveat is recorded in check 6 and it matters: CPU
ACCEPTS a contiguous mat2 while CUDA refuses it, so the layout constraint cannot be tested by
letting the op reject it here.

    python3 scripts/test_moe_module.py
"""
import os
import sys

import torch

# TOLERANCES ARE PER-DEVICE, and this block exists because a single number is wrong on one of them.
# Measured by tilerl on card 7, 2026-09-05: on CUDA at bf16 the GROUPED GEMM's own reduction order
# differs from an identical plain matmul by 0.001953 for the SAME function -- while fp32-on-CUDA and
# plain-matmul-at-bf16 are both exactly 0.0, so the noise belongs to the grouped kernel, not to
# bf16. Measured here on CPU: exactly 0.0 at both fp32 and bf16.
#
# So a 1e-3 bound REFUSES A CORRECT MODULE on the card, which is the bug tilerl hit in their own
# witness and reported so I would not repeat it. CPU keeps the tight bound because there the floor
# really is 0.0 and loosening it would stop discriminating; CUDA takes 4e-3, twice the measured
# floor and 4.3x below the SMALLER of the two defect signals this witness exists to catch
# (0.0172 for swapped betas, 0.023 for plain a*sigmoid(b) instead of SiTU-GLU).
#
# THE DEFECT FLOOR IS THE SAME ON BOTH DEVICES: it is a function difference, not a reduction-order
# difference, so it does not shrink with precision.
_CUDA = torch.cuda.is_available()
# 1e-2 on CUDA, ruled by 4c 2026-09-05: ~5x the measured floor, still below the smaller defect
# signal (0.0172). The margin is deliberate -- the floor was measured at ONE shape on ONE card, and
# a bound sitting 2x above a single measurement is a bound that fails on the next shape.
AGREE_TOL = 1e-2 if _CUDA else 1e-6      # "the same function", after kernel noise
GATE_TOL = 1e-2 if _CUDA else 1e-5       # same, for the gate-source comparison
DEFECT_MIN = 1e-3                        # "a different function", floor of the signals above

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

D = 64


def _cfg(**kw):
    """A config whose parity holds: (top_k + shared) * expert_ffn == ffn_hidden."""
    base = dict(d=D, ffn_hidden=64, layers=4, vocab=100, seq=16, attn_every=4,
                moe_experts=8, moe_top_k=1, moe_expert_ffn=32, moe_shared=1,
                moe_bias_gamma=0.001, moe_balance_alpha=1e-4)
    base.update(kw)
    return type("MoECfg", (), base)


def main():
    from model import MoEFFN, SwiGLU
    bad, n = 0, 9

    # 1. THE TIED-WEIGHTS WITNESS, and it is a precondition rather than a diagnostic. With one
    # routed expert (so the gate is a softmax over one score = 1.0) and one shared expert, both
    # holding a dense module's own weights, the module must reproduce dense(x) + dense(x). If it
    # does not, whatever it computes is not the FFN it replaces and no loss delta means anything.
    torch.manual_seed(1)
    m = MoEFFN(_cfg(moe_experts=1, moe_top_k=1, moe_expert_ffn=32)())
    m.eval()
    ref = SwiGLU(_cfg(ffn_hidden=32)())
    ref.eval()
    with torch.no_grad():
        m.w13.copy_(ref.w13.weight.unsqueeze(0))
        m.w2.copy_(ref.w2.weight.unsqueeze(0))
        m.sh13.weight.copy_(ref.w13.weight)
        m.sh2.weight.copy_(ref.w2.weight)
        x = torch.randn(2, 5, D)
        got = m(x)
        want = ref(x) + ref(x)
    d_ok = float((got - want).abs().max())

    # THE SAME WITNESS AGAINST THE DEFECT IT EXISTS FOR. "The module matches dense" alone cannot
    # tell a correct module from one where the comparison is insensitive; this drops the two tanh
    # bounds and requires the witness to move. Threshold from the measured signal: the defect
    # shows 2.3e-02 on outputs of scale ~0.7, so 1e-3 sits ~23x below it.
    _orig = MoEFFN._situ
    MoEFFN._situ = lambda self, a, b, f: f(a * torch.sigmoid(b))
    try:
        with torch.no_grad():
            d_bad = float((m(x) - want).abs().max())
    finally:
        MoEFFN._situ = _orig
    ok = d_ok < AGREE_TOL and d_bad > DEFECT_MIN
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} one expert tied to the dense module reproduces it "
          f"(max|d| {d_ok:.2e}); plain a*sigmoid(b) instead of SiTU-GLU moves it to {d_bad:.2e}, "
          f"so the witness can see a dropped activation")

    # 2. EQUAL-ACTIVE PARITY IS REFUSED AT CONSTRUCTION, at the two shapes that miss it in the
    # directions a launch line actually misses them: top-4 + shared (the 1.25x config 4c's ruling
    # rejected) and a wrong expert width.
    e1 = dict(d=1024, ffn_hidden=3072, layers=12, moe_experts=24, moe_top_k=3,
              moe_expert_ffn=768, moe_shared=1)
    refused = []
    for label, kw in (("k=4 (1.25x)", dict(moe_top_k=4)), ("w=512", dict(moe_expert_ffn=512))):
        c = dict(e1)
        c.update(kw)
        try:
            MoEFFN(_cfg(**c)())
            refused.append(f"{label} ACCEPTED")
        except ValueError:
            pass
    ok = not refused
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} a config missing (top_k+shared)*w == ffn_hidden is "
          f"refused at construction{'' if ok else ': ' + ', '.join(refused)}")

    # 3. THE BIAS AFFECTS SELECTION ONLY: the gate multiplied into the expert output comes from
    # the UN-BIASED affinity (arXiv:2412.19437 2.1.2, "the bias term is only used for routing").
    #
    # THIS CHECK IS BUILT AS A KNOWN-ANSWER COMPARISON AGAINST THE MODULE'S OUTPUT, and the first
    # version was not. That version recomputed both candidate gates itself and asserted they
    # DIFFER -- which is a fact about softmax arithmetic, not about the module, and it never asked
    # the module which one it used. Measured 2026-09-05: a mutant gating on the biased score
    # passed all 8 checks. So instead, tie every expert to the SAME weights: then the module's
    # output is exactly (sum of selected gates) * expert(x), the gate sum is 1.0 under the correct
    # implementation and != 1.0 under the biased one, and the difference is visible in the OUTPUT.
    torch.manual_seed(2)
    m3 = MoEFFN(_cfg(moe_experts=8, moe_top_k=3, moe_expert_ffn=16)())
    m3.eval()
    ref3 = SwiGLU(_cfg(ffn_hidden=16)())
    ref3.eval()
    with torch.no_grad():
        # Every routed expert identical, shared expert zeroed, so the routed sum is isolated.
        for e in range(8):
            m3.w13[e].copy_(ref3.w13.weight)
            m3.w2[e].copy_(ref3.w2.weight)
        m3.sh13.weight.zero_()
        m3.sh2.weight.zero_()
        xb = torch.randn(1, 4, D)
        flat = xb.reshape(-1, D)
        aff = torch.softmax(m3.router(flat).float(), dim=-1)
        # A bias big enough to change the ranking AND to make the biased gate sum far from 1.
        m3.expert_bias.zero_()
        m3.expert_bias[int(aff[0].argmin())] = 10.0
        got3 = m3(xb)
        # Under the correct rule the renormalised UN-BIASED gates of the selected set sum to 1,
        # so the output is exactly one expert application. Shared is zero, but SwiGLU's second
        # tanh is applied to w2(gate) and a zeroed shared expert contributes tanh(0) = 0.
        want3 = ref3(xb)
        d_unbiased = float((got3 - want3).abs().max())
        # Under the biased rule the gate sum is (sum of biased scores)/(their sum) = 1 as well
        # AFTER renormalisation -- so the sum is not the discriminator. The WEIGHTS differ: the
        # biased rule puts almost all mass on the +10 expert. With identical experts that still
        # sums to one expert application, so identical experts cannot separate the two rules
        # either. Use DISTINCT experts and compare against the explicitly un-biased prediction.
    m3d = MoEFFN(_cfg(moe_experts=8, moe_top_k=3, moe_expert_ffn=16)())
    m3d.eval()
    with torch.no_grad():
        m3d.sh13.weight.zero_()
        m3d.sh2.weight.zero_()
        aff_d = torch.softmax(m3d.router(flat).float(), dim=-1)
        m3d.expert_bias.zero_()
        m3d.expert_bias[int(aff_d[0].argmin())] = 10.0
        got_d = m3d(xb)

        def predict(gate_source):
            """The module's arithmetic, with the gate taken from a named source."""
            sel = (aff_d + m3d.expert_bias.float()).topk(3, dim=-1).indices
            g = gate_source.gather(1, sel)
            g = g / g.sum(-1, keepdim=True)
            acc = torch.zeros_like(flat)
            for j in range(3):
                for i in range(flat.shape[0]):
                    e = int(sel[i, j])
                    h = flat[i:i + 1] @ m3d.w13[e].t()
                    a, b = h.chunk(2, dim=-1)
                    gt = m3d.beta1 * torch.tanh(a / m3d.beta1) * torch.sigmoid(b)
                    y = m3d.beta2 * torch.tanh((gt @ m3d.w2[e].t()) / m3d.beta2)
                    acc[i] += y[0] * g[i, j].to(y.dtype)
            return acc.view(1, 4, D)

        pred_unbiased = predict(aff_d)
        pred_biased = predict(aff_d + m3d.expert_bias.float())
        d_to_unbiased = float((got_d - pred_unbiased).abs().max())
        d_to_biased = float((got_d - pred_biased).abs().max())
    # The module must match the UN-BIASED prediction and NOT the biased one. The second clause is
    # what makes the first discriminating: if the two predictions coincided, matching one would say
    # nothing.
    ok = (d_unbiased < GATE_TOL and d_to_unbiased < GATE_TOL and d_to_biased > DEFECT_MIN)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the gate is the UN-BIASED affinity: module matches the "
          f"un-biased prediction (max|d| {d_to_unbiased:.2e}) and NOT the biased one "
          f"({d_to_biased:.2e}); identical-experts sanity {d_unbiased:.2e}")

    # 4. THE BIAS UPDATE MOVES AGAINST LOAD, by a fixed gamma on the SIGN of the error (the
    # paper's rule, arXiv:2412.19437 2.1.2). An overloaded expert's bias must DECREASE by exactly
    # gamma, an underloaded one's INCREASE by exactly gamma -- not by an amount proportional to
    # the error, which would be a different balancer than the one whose value we borrowed.
    m4 = MoEFFN(_cfg(moe_experts=4, moe_top_k=1, moe_expert_ffn=32)())
    counts = torch.tensor([100, 1, 1, 1])
    before = m4.expert_bias.clone()
    m4.update_bias(counts)
    delta = (m4.expert_bias - before)
    g = m4.gamma
    ok = (abs(float(delta[0]) + g) < 1e-9
          and all(abs(float(delta[i]) - g) < 1e-9 for i in (1, 2, 3)))
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the bias step is +/- gamma on the sign of the load error "
          f"(overloaded {float(delta[0]):+.4f}, underloaded {float(delta[1]):+.4f}, "
          f"gamma {g})")

    # 5. THE BIAS IS PERSISTENT AND A MISSING KEY LOADS ZEROS (4c's ruling: the cards can be
    # recalled mid-run, so a resume is expected and a cold balancer would make readout 4
    # unattributable). Both halves are asserted: it IS in the state dict, and an old checkpoint
    # without it still loads.
    sd = m4.state_dict()
    in_sd = "expert_bias" in sd
    m5 = MoEFFN(_cfg(moe_experts=4, moe_top_k=1, moe_expert_ffn=32)())
    old = {k: v for k, v in sd.items() if k != "expert_bias"}
    missing, unexpected = m5.load_state_dict(old, strict=False)
    loads_zero = ("expert_bias" in missing and not unexpected
                  and bool((m5.expert_bias == 0).all()))
    ok = in_sd and loads_zero
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} the bias is persistent ({in_sd}) and a checkpoint "
          f"without it loads zeros ({loads_zero})")

    # 6. THE WEIGHT LAYOUT IS (E, N, K), asserted DIRECTLY rather than by letting the op refuse it.
    # _grouped_mm needs a transposed mat2 and CUDA refuses a contiguous (E, K, N) operand -- but
    # measured here 2026-09-05, CPU ACCEPTS it, so a CPU test that relied on the refusal would
    # pass on a layout that dies on the pod. The parameter's own shape is the checkable invariant.
    m6 = MoEFFN(_cfg(moe_experts=8, moe_top_k=1, moe_expert_ffn=32)())
    w13_ok = tuple(m6.w13.shape) == (8, 2 * 32, D)
    w2_ok = tuple(m6.w2.shape) == (8, D, 32)
    t_ok = tuple(m6.w13.transpose(-2, -1).shape) == (8, D, 2 * 32)
    ok = w13_ok and w2_ok and t_ok
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} expert weights are stored (E, N, K) so the transpose "
          f"_grouped_mm needs is a view (w13 {tuple(m6.w13.shape)}, w2 {tuple(m6.w2.shape)})")

    # 7. READOUT 4's DIAGNOSTICS, including the field 44's review made a condition: window_steps.
    # And the counters must NOT advance in eval, or a validation pass would pollute the window
    # whose fraction the stop rule reads.
    m7 = MoEFFN(_cfg(moe_experts=8, moe_top_k=3, moe_expert_ffn=16)())
    m7.train()
    xx = torch.randn(2, 8, D)
    m7(xx)
    m7(xx)
    dd = m7.diagnostics(reset=False)
    fields_ok = {"usage_frac", "entropy_norm", "load_gini", "window_steps"} <= set(dd)
    win_ok = dd["window_steps"] == 2
    m7.eval()
    with torch.no_grad():
        m7(xx)
    still = m7.diagnostics(reset=True)["window_steps"]
    eval_clean = still == 2
    reset_ok = m7.diagnostics(reset=False)["window_steps"] == 0
    ok = fields_ok and win_ok and eval_clean and reset_ok
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} diagnostics carry window_steps and the three metrics, "
          f"count only in training (2 train + 1 eval -> {still}), and reset ({reset_ok})")

    # 8. THE E1 PARAMETER COUNT, as the integer the launch gate will enforce. THIS IS NOT THE
    # CHARTER'S 800,670,792: that figure is experts + shared + non-FFN and predates the router,
    # which the charter itself said it could not count yet ("depends on the router and bias vector
    # counts, which do not exist yet. The probe writes it"). The router is 1024*24 = 24,576 per
    # layer, 294,912 over 12 layers, so E1 is 800,965,704. Recorded here so the row is amended
    # from a measurement rather than from arithmetic nobody ran.
    m8 = MoEFFN(_cfg(**e1)())
    per_layer = sum(p.numel() for p in m8.parameters())
    non_ffn = 92_881_992
    total = non_ffn + 12 * per_layer
    active = (3 + 1) * 3 * 1024 * 768
    ok = (per_layer == 59_006_976 and total == 800_965_704 and active == 9_437_184
          and m8.expert_bias.numel() == 24)
    bad += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'BUG '} E1 is {per_layer:,}/layer -> {total:,} total "
          f"(charter's pre-router figure was 800,670,792; the {12 * 1024 * 24:,} router weights "
          f"are the difference), active FFN/layer {active:,}")

    # 9. WIRING AND OPTIMIZER ROUTING, on a real HybridLM rather than the module alone, because
    # both properties are decided by train.build_optimizers over named_parameters() of the whole
    # net and by HybridLM's construction -- neither is visible from MoEFFN.
    #
    # CONSTRUCTION ONLY, no forward: chunk_kda and l2norm are Triton with no CPU fallback, so this
    # model cannot forward here. Routing and refusals are settled at construction, so that costs
    # nothing.
    try:
        import train
        wiring = _wiring_shape(train)
        ok = not wiring
        bad += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'BUG '} ruling (f) holds end to end: experts in Muon, router "
              f"in AdamW at the dense lr, and grad_ckpt + out-of-range moe_layers both refused"
              f"{'' if ok else ' -- ' + '; '.join(wiring)}")
    except Exception as e:  # noqa: BLE001 -- an import failure here is a finding, not a skip
        bad += 1
        print(f"  BUG  could not read the wiring: {type(e).__name__}: {e}")

    print(f"test_moe_module: {n - bad}/{n} pass")
    return 1 if bad else 0


def _wiring_shape(train):
    """Ruling (f)'s two routing rules and the module's two refusals. Returns a list of violations.

    WHY ALL FOUR IN ONE CHECK: they are the four ways a launch line can produce a running arm that
    is not the registered one. Verified by mutation 2026-09-05 in a tree with its own gitdir --
    disabling either routing branch or either refusal is caught, each by its own clause. The first
    version of this harness tested only the routing and BOTH refusal mutants survived it.
    """
    def mk(**kw):
        d = dict(d=64, heads=8, layers=4, ffn_hidden=64, vocab=100, seq=16, attn_every=4,
                 mem_values=0, head_mixed=0, grad_ckpt=False, attn_res=False,
                 moe_experts=8, moe_top_k=1, moe_expert_ffn=32, moe_shared=1, moe_layers="0-3")
        d.update(kw)
        return type("CfgMoeWiring", (train.Cfg,), d)

    out = []
    m = train.HybridLM(mk()())
    opts = train.build_optimizers(m, mk()())
    names = {id(p): n for n, p in m.named_parameters()}
    router = experts = None
    router_lr = None
    for o in opts:
        got = {names.get(id(p), "?") for g in o.param_groups for p in g["params"]}
        if any(n.endswith("router.weight") for n in got):
            router = type(o).__name__
            router_lr = {g["lr"] for g in o.param_groups}
        if any(n.endswith("ffn.w13") for n in got):
            experts = type(o).__name__
    # THE ROUTER IN AdamW: it is 2D, so the p.ndim == 2 branch claims it for Muon without an
    # explicit branch above that one. Measured -- that is where it landed first.
    if router != "AdamW":
        out.append(f"router is in {router}, not AdamW (ruling (f))")
    # THE EXPERTS IN MUON, exactly as the dense FFN they replace. They are 3D (E, N, K), so the
    # p.ndim == 3 branch sends them to the AttnRes pseudo-query group at attn_res_lr without an
    # explicit branch. Measured -- 56M parameters of FFN landed in a group meant for 51,200.
    if experts != "Muon":
        out.append(f"experts are in {experts}, not Muon (ruling (f))")
    # THE ROUTER'S LR IS THE DENSE lr, not the expert/table lr. cfg.lr does NOT exist (Cfg carries
    # muon_lr / embed_lr / scalar_lr), so the resolution reads muon_lr -- the rate the FFN the
    # router routes trains at.
    if router_lr != {float(train.Cfg.muon_lr)}:
        out.append(f"router lr {router_lr} is not the dense lr {float(train.Cfg.muon_lr)}")
    # grad_ckpt: the counter is written under no_grad inside forward, and recomputation runs the
    # forward twice, so readout 4 would divide by a doubled denominator and report a healthier
    # spread than the arm has -- the direction that makes a stop rule fail to fire.
    try:
        train.HybridLM(mk(grad_ckpt=True)())
        out.append("grad_ckpt with moe_experts was NOT refused")
    except ValueError:
        pass
    # An out-of-range layer index would convert fewer layers than the launch line says.
    try:
        train.HybridLM(mk(moe_layers="0,9")())
        out.append("an out-of-range moe_layers index was NOT refused")
    except ValueError:
        pass
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1:] != ["--selftest"]:
        sys.exit(f"usage: {os.path.basename(__file__)} [--selftest]  (got {sys.argv[1:]})")
    sys.exit(main())
