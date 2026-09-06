#!/usr/bin/env python3
"""Where does MoE-48's extra ~0.5 s/step go? Per-component, against the dense FFN.

Stage 2 of prereg#moe_0905. THE PREMISE IS NARROWER THAN IT LOOKS: active params per
token are 206.7M against the dense arm's 206.1M (+0.29%), and the per-layer active FFN
is bit-for-bit the dense FFN's 9,437,184 because parity holds exactly
((top_k+shared)*expert_ffn = 4*768 = 3072 = ffn_hidden). So the +27% s/step is OVERHEAD,
not work -- there is no extra compute to attribute it to.

WHAT IS TIMED, and why each boundary is where it is. The regions follow MoEFFN.forward's
own order, so a share here names a line in that method rather than a stage in a rewrite:

  router      self.router(flat) -> softmax -> topk with the bias -> gate normalise
  seqloss     the sequence-wise balance loss (training only; f/P scatter_add + mean)
  dispatch    argsort of the expert ids, bincount, cumsum to offsets, and the gather
              rows = xr[tok[order]] -- the permutation, not the matmul
  gemm        both torch._grouped_mm calls plus the bounded activation between them
  combine     index_add_ scatter-back with the gate, into the fp32 accumulator
  shared      the shared expert (sh13 -> _situ -> sh2), which the dense arm also pays
              in a different shape

There is NO all-to-all in this implementation and that is a finding, not an omission:
the experts are DDP-REPLICATED (w13/w2 are plain 3-D Parameters on every rank, model.py),
so a token never leaves its rank and no expert-parallel exchange exists. Stage 2's brief
listed all-to-all as a candidate; it cannot be one here.

CUDA EVENTS, NOT time.time(): the kernels are async, so a wall-clock read returns launch
cost. Events are recorded on the same stream and read after one synchronize at the end of
each iteration, which is also why the sum of regions is compared against a separately
timed whole -- an unattributed residual is reported rather than hidden.

FORWARD ONLY, STATED PLAINLY. These regions are the forward pass. Backward is timed as a
whole (loss.backward on the module's output) because autograd does not decompose along
these boundaries from outside, and a per-region backward number would be invented. The
forward split is what locates the overhead; the backward total is what says how much of
the step it can explain.

    python3 scripts/profile_moe_step.py --experts 48            # the arm
    python3 scripts/profile_moe_step.py --dense                 # the control
    python3 scripts/profile_moe_step.py --selftest              # no card needed
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Arm D's shape, from prereg#moe_0905 amendment 14's launch line. Kept as data so --selftest
# can assert it matches without a card: a profiler configured differently from the run
# measures another program.
ARM_D = dict(d=1024, layers=12, heads=8, ffn_hidden=3072, seq=4096, batch=16, accum=2,
             moe_experts=48, moe_top_k=3, moe_expert_ffn=768, moe_shared=1,
             moe_bias_gamma=0.001, moe_balance_alpha=1e-4)


def _cfg(**kw):
    base = dict(ARM_D, vocab=32768, attn_every=4)
    base.update(kw)
    return type("Cfg", (), base)


def _regions(m, x, ev, torch):
    """One forward, with events around each region. Mirrors MoEFFN.forward's order.

    Re-implements the SHAPE of that forward to place event boundaries inside it, which is
    the one thing an outside timer cannot do. Every line here is copied from the method, so
    a divergence is a bug in this file -- checked by --equiv, which runs both on the same
    input and compares outputs. The d_latent branch is NOT reproduced: arm D sets no
    d_latent, and _equiv refuses a config that has one rather than timing a path this
    function does not implement.
    """
    B, T, d = x.shape
    n = B * T
    flat = x.reshape(n, d)

    ev("router")
    logits = m.router(flat).float()
    affinity = torch.softmax(logits, dim=-1)
    sel = (affinity + m.expert_bias.float()).topk(m.top_k, dim=-1).indices
    gate = affinity.gather(1, sel)
    gate = gate / gate.sum(-1, keepdim=True).clamp_min(1e-9)

    ev("seqloss")
    f = torch.zeros(B, m.n_routed, device=x.device, dtype=torch.float32)
    sel_b = sel.view(B, T * m.top_k)
    f.scatter_add_(1, sel_b, torch.ones_like(sel_b, dtype=torch.float32))
    f = f * (m.n_routed / (T * m.top_k))
    P = affinity.view(B, T, m.n_routed).mean(1)
    aux = m.balance_alpha * (f * P).sum(-1).mean()

    ev("dispatch")
    tok = torch.arange(n, device=x.device).repeat_interleave(m.top_k)
    e_of_row = sel.reshape(-1)
    order = torch.argsort(e_of_row)
    counts = torch.bincount(e_of_row, minlength=m.n_routed)
    offs = torch.cumsum(counts, 0).to(torch.int32)
    xw = flat.to(m.w13.dtype)
    rows = xw[tok[order]]

    ev("gemm")
    h = torch._grouped_mm(rows, m.w13.transpose(-2, -1), offs=offs)
    a, b = h.chunk(2, dim=-1)
    y = m._situ(a, b, lambda g: torch._grouped_mm(
        g.contiguous(), m.w2.transpose(-2, -1), offs=offs))

    ev("combine")
    gflat = gate.reshape(-1)[order]
    out = torch.zeros(n, d, device=x.device, dtype=torch.float32)
    out.index_add_(0, tok[order], y.float() * gflat[:, None])

    ev("shared")
    hs = m.sh13(xw)
    sa, sb = hs.chunk(2, dim=-1)
    out = out + m._situ(sa, sb, m.sh2).float()

    ev(None)
    return out.reshape(B, T, d).to(x.dtype), aux


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dense", action="store_true", help="time the dense SwiGLU control")
    ap.add_argument("--experts", type=int, default=ARM_D["moe_experts"])
    ap.add_argument("--top_k", type=int, default=ARM_D["moe_top_k"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5, help="a first pass carries compile/alloc")
    ap.add_argument("--batch", type=int, default=1,
                    help="SEQUENCES PER FORWARD into one layer, not the run's --batch: this "
                         "times ONE MoEFFN, and the run's batch 16 x accum 2 spreads over 12 "
                         "layers and two ranks")
    ap.add_argument("--json", default=None)
    ap.add_argument("--equiv", action="store_true",
                    help="GPU: assert _regions reproduces MoEFFN.forward on the same input, "
                         "then exit. Run this before trusting any share below -- a timed "
                         "reimplementation that computes something else measures nothing.")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    import torch
    assert torch.cuda.is_available(), "needs a GPU: _grouped_mm and the Triton kernels have no CPU path"
    from model import MoEFFN, SwiGLU

    if a.equiv:
        return equiv(torch, MoEFFN, a)

    cfg = _cfg(moe_experts=a.experts, moe_top_k=a.top_k)
    dev = "cuda"
    mod = (SwiGLU(cfg) if a.dense else MoEFFN(cfg)).to(dev).to(torch.bfloat16)
    mod.train()
    x = torch.randn(a.batch, ARM_D["seq"], ARM_D["d"], device=dev, dtype=torch.bfloat16,
                    requires_grad=True)

    names, starts, ends = [], {}, {}
    cur = [None]

    def ev(name):
        if cur[0] is not None:
            ends[cur[0]].record()
        cur[0] = name
        if name is not None:
            if name not in starts:
                names.append(name)
                starts[name] = torch.cuda.Event(enable_timing=True)
                ends[name] = torch.cuda.Event(enable_timing=True)
            starts[name].record()

    whole_s = torch.cuda.Event(enable_timing=True)
    whole_e = torch.cuda.Event(enable_timing=True)
    bwd_s = torch.cuda.Event(enable_timing=True)
    bwd_e = torch.cuda.Event(enable_timing=True)
    acc, whole_ms, bwd_ms, n_timed = {}, 0.0, 0.0, 0

    for i in range(a.warmup + a.iters):
        timed = i >= a.warmup
        whole_s.record()
        if a.dense:
            out = mod(x)
            aux = None
        else:
            out, aux = _regions(mod, x, ev, torch)
        whole_e.record()
        loss = out.float().pow(2).mean()
        if aux is not None:
            loss = loss + aux
        bwd_s.record()
        loss.backward()
        bwd_e.record()
        torch.cuda.synchronize()
        if timed:
            n_timed += 1
            whole_ms += whole_s.elapsed_time(whole_e)
            bwd_ms += bwd_s.elapsed_time(bwd_e)
            if not a.dense:
                for nm in names:
                    acc[nm] = acc.get(nm, 0.0) + starts[nm].elapsed_time(ends[nm])
        mod.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None

    label = "dense SwiGLU" if a.dense else f"MoE-{a.experts} top_k {a.top_k}"
    print(f"=== {label} | one layer | {a.batch} seq x {ARM_D['seq']} tok | "
          f"{n_timed} timed iters after {a.warmup} warmup")
    print(f"forward whole   {whole_ms / n_timed:8.3f} ms")
    print(f"backward whole  {bwd_ms / n_timed:8.3f} ms")
    if not a.dense:
        print()
        print("region        ms/fwd    share of fwd")
        s = sum(acc.values())
        for nm in names:
            print(f"{nm:12s} {acc[nm] / n_timed:8.3f}   {100 * acc[nm] / s:5.1f}%")
        print(f"{'sum':12s} {s / n_timed:8.3f}")
        resid = whole_ms - s
        print(f"{'residual':12s} {resid / n_timed:8.3f}   "
              f"{100 * resid / whole_ms:5.1f}% of the whole, not attributed")
    if a.json:
        rec = {"label": label, "experts": None if a.dense else a.experts,
               "top_k": None if a.dense else a.top_k, "iters": n_timed,
               "fwd_ms": whole_ms / n_timed, "bwd_ms": bwd_ms / n_timed,
               "regions": {k: v / n_timed for k, v in acc.items()}}
        with open(a.json, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"\nappended one record to {a.json}")
    return 0


def equiv(torch, MoEFFN, a):
    """Does _regions compute what MoEFFN.forward computes? On a card, same input, same weights.

    This is the load-bearing check for every share the profiler prints. _regions is a copy of
    the forward with event boundaries inserted; if the copy drifts, the timings are real
    measurements OF THE WRONG PROGRAM, which is worse than no measurement because they look
    like an answer. A negative control is included: perturbing one weight must break equality,
    otherwise the comparison is passing on something other than the computation.
    """
    cfg = _cfg(moe_experts=a.experts, moe_top_k=a.top_k)
    assert not getattr(cfg, "d_latent", None), "_regions does not implement the d_latent branch"
    torch.manual_seed(0)
    m = MoEFFN(cfg).cuda().to(torch.bfloat16)
    m.train()
    # expert_bias off zero: with an all-zero bias the `affinity + expert_bias` term is inert
    # and a copy that dropped it entirely would still match.
    with torch.no_grad():
        m.expert_bias.copy_(torch.randn_like(m.expert_bias.float()).to(m.expert_bias.dtype) * 0.05)
    x = torch.randn(1, ARM_D["seq"], ARM_D["d"], device="cuda", dtype=torch.bfloat16)

    ref = m(x)
    ref_aux = m.aux_loss
    got, got_aux = _regions(m, x, lambda _n: None, torch)

    dmax = float((ref.detach().float() - got.detach().float()).abs().max())
    amax = abs(float(ref_aux.detach()) - float(got_aux.detach()))
    print(f"output max|diff| {dmax:.3e}   aux |diff| {amax:.3e}")
    bad = 0
    # THE BAR IS SET FROM A MEASURED NOISE FLOOR, not chosen. bf16 round-trip on this shape:
    # adding 1.0 to w13 and subtracting it again leaves max|diff| 0.0547 (measured, card 3),
    # because +-1.0 rounds away part of a bf16 mantissa. So a bar below that would fail on
    # arithmetic noise. 0.25 is ~4.6x that floor and ~19x below the smallest real perturbation
    # the control produces (4.85), so it separates the two.
    FLOOR = 0.25
    if dmax > FLOOR:
        print(f"BUG: _regions diverges from MoEFFN.forward by {dmax:.3e}")
        bad += 1
    if amax > 1e-6:
        print(f"BUG: sequence-loss diverges by {amax:.3e}")
        bad += 1

    # NEGATIVE CONTROL, AND CHOOSING WHICH EXPERT TO PERTURB IS THE WHOLE DIFFICULTY. Two
    # earlier versions of this check passed vacuously, both measured on card 3:
    #   w13[0,0,0] (one element)      -> max|diff| 0.000e+00
    #   w13[0]     (expert 0 entire)  -> max|diff| 0.000e+00 once expert_bias is non-zero
    # The second is the instructive one: with a random bias on random weights, EXPERT 0
    # RECEIVES 0 OF 12288 ROUTED SLOTS and 15 of 48 experts get nothing at all. Perturbing an
    # expert no token reaches cannot change the output, so the control was exercising a dead
    # path and its 0.000e+00 was not evidence of anything. The counter says which experts are
    # live, so the perturbation goes to the BUSIEST one.
    m.step_tokens_per_expert.zero_()
    m(x)
    counts = m.step_tokens_per_expert
    victim = int(counts.argmax())
    n_dead = int((counts == 0).sum())
    print(f"routing: expert {victim} is busiest with {int(counts[victim])} of "
          f"{int(counts.sum())} slots; {n_dead}/{m.n_routed} experts got none")
    assert int(counts[victim]) > 0, "no expert received a token -- the control cannot fire"
    with torch.no_grad():
        m.w13[victim] += 1.0
    pert = m(x)
    pmax = float((pert.detach().float() - got.detach().float()).abs().max())
    print(f"negative control: perturbing expert {victim}'s w13, max|diff| {pmax:.3e} "
          f"(must exceed the {FLOOR} bar)")
    if pmax <= FLOOR:
        print("BUG: perturbing a LIVE expert did not change the output -- comparison is vacuous")
        bad += 1

    print("FAIL" if bad else "PASS: _regions reproduces MoEFFN.forward, and the check can fail")
    return 1 if bad else 0


def selftest():
    """No card. Asserts the shape config matches arm D's launch line and that the region
    list is the one the docstring documents -- the two things that make a share nameable."""
    bad = 0
    want = dict(d=1024, layers=12, heads=8, ffn_hidden=3072, batch=16, accum=2,
                moe_experts=48, moe_top_k=3, moe_expert_ffn=768, moe_shared=1)
    for k, v in want.items():
        if ARM_D.get(k) != v:
            print(f"BUG: ARM_D[{k}] = {ARM_D.get(k)}, arm D's launch line says {v}")
            bad += 1
    # parity, the reason active FLOPs are unchanged; if this breaks the whole premise breaks
    parity = (ARM_D["moe_top_k"] + ARM_D["moe_shared"]) * ARM_D["moe_expert_ffn"]
    if parity != ARM_D["ffn_hidden"]:
        print(f"BUG: parity {parity} != ffn_hidden {ARM_D['ffn_hidden']}")
        bad += 1
    else:
        print(f"OK: parity holds, (top_k+shared)*expert_ffn = {parity} = ffn_hidden")
    with open(os.path.abspath(__file__), encoding="utf-8") as fh:
        src = fh.read()
    for nm in ("router", "seqloss", "dispatch", "gemm", "combine", "shared"):
        if f'ev("{nm}")' not in src:
            print(f"BUG: region {nm} is documented but never recorded")
            bad += 1
    if 'all-to-all' not in src.lower().replace("all_to_all", "all-to-all"):
        print("BUG: the absence of an all-to-all is a finding and must be stated")
        bad += 1
    print("FAIL" if bad else "PASS: config matches arm D, all six regions recorded")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
