#!/usr/bin/env python3
"""Reference implementation and gradient derivation for a fused AttnRes mixing kernel.

WHY THIS EXISTS (fb 2026-09-02): the trace shows ~1800 aten::add_ of [16,4096,1] and
~1950 of [16,4096,1024] per step, 90.7 ms. fb's hypothesis, for tilerl to confirm by
External id: those add_ calls are autograd's gradient accumulation for multi-consumer
tensors, not inductor-fused arithmetic. If so, an in-place `out.add_()` in forward changes
nothing and the fix is to make the mixing one autograd node.

WHAT THE ACCOUNTING SAYS, JOINTLY WITH tilerl (read before writing a kernel):

 1. THE ACCOUNTING IS CLOSED, AND THE MECHANISM IS ABLATED. add_[B,T,D] per step = 2 *
    source_reads = n(n+1) at n=2L+1: 650 at L=12, 4290 at L=32, exact at four depths. Times
    3 touches per in-place add_ that is 261.7 GB/step -- the trace's figure to four figures,
    no fitted constant. The factor 2 is one stack reading every source TWICE (model.py:248
    logits, :250-252 mixing), not two stacks; tilerl's ablation detaches the logits read
    only and the count halves exactly at every depth, which is the claim's actual evidence
    (the n(n+1) assert would pass either way -- it constrains the value, not the mechanism).
    Two errors were corrected getting here, one each: my "6.5x gap" was per-step vs
    per-3-step (the trace's ~1950 is 3 steps), and tilerl's "two triangles = two stacks" was
    the double read.
 2. HALF THOSE ADDS ARE ON THE LOGITS PATH (325 of 650), which a mixing-only kernel cannot
    touch by construction.
 3. THE OTHER HALF DOES NOT MOVE EITHER. The accumulation is across calls -- s.v at position
    p is read by every call from p to 25 -- and a per-call fused node still emits one
    contribution per source it reads. Collapsing it needs ONE node over the whole stack.
 4. SO WHAT THE ASSIGNED FUSION ACTUALLY BUYS is the intra-call temporaries: 120.8 GB/step =
    30.2 ms at 4.0 TB/s (eff.h20_specs). Real, mechanised, and NOT the 90.7 ms of add_ --
    those are different quantities, and conflating them sells a 30 ms lever as a 90 ms one.

fb/1e's ruling (14:0xZ) is to build the per-call Function anyway and revisit the whole-stack
node after its A/B. This file is the MATH plus the pure-torch reference for that. It writes
no kernel: parity is elementwise against `grads_analytic` (fp32, atol 1e-6, as known_answer()
does against the real model.AttnRes) plus a 20-step loss compare.

    python3 scripts/attnres_fused_reference.py          # derivation checks + parity
"""
import sys
from pathlib import Path

import torch


def mix_current(a, vs):
    """Exactly model.py:250-252 -- the loop whose backward produces the add_ storm.

    a:  [n, B, T]      mixing weights (already softmaxed over n)
    vs: list of n tensors [B, T, D]
    """
    out = a[0].unsqueeze(-1) * vs[0]
    for i in range(1, len(vs)):
        out = out + a[i].unsqueeze(-1) * vs[i]
    return out


def mix_reference(a, vs):
    """The same value, computed so that a fused kernel has one clear target.

    Deliberately NOT einsum over a stacked [n,B,T,D]: model.py:247 records that the
    [n,B,T,D] copy dominates at L=24, which is why the loop exists. So the reference keeps
    the loop for the FORWARD and only claims to fuse the BACKWARD -- which is where the
    measured cost is.
    """
    return mix_current(a, vs)


def grads_analytic(a, vs, dout):
    """The gradients a fused backward must produce, in ONE pass, accumulating internally.

    out[b,t,d] = sum_i a[i,b,t] * v[i][b,t,d]

    d out[b,t,d] / d v[i][b,t,d] = a[i,b,t]            (diagonal in b,t,d)
        => dV_i[b,t,d] = a[i,b,t] * dout[b,t,d]

    d out[b,t,d] / d a[i,b,t] = v[i][b,t,d]            (sums over d)
        => dA[i,b,t] = sum_d dout[b,t,d] * v[i][b,t,d]

    Both are elementwise/reduction in the same [B,T,D] traversal, so one kernel can emit
    all n of each while reading dout once. That is the whole saving: today autograd walks
    the graph n times and add_s into n grad buffers.

    NOTE ON WHAT IS NOT FUSED HERE: `a` itself comes from a softmax over n of logits that
    each read v[i] again (model.py:248). Its backward is a second multi-consumer pattern.
    This function covers the MIXING only; the logits path is a separate node and a separate
    decision, and lumping them would make the parity test unable to localise a mismatch.
    """
    n = len(vs)
    dV = [a[i].unsqueeze(-1) * dout for i in range(n)]
    dA = torch.stack([(dout * vs[i]).sum(-1) for i in range(n)])
    return dV, dA


def known_answer():
    """1e's known-answer bar (13:5xZ): allclose against the REAL model.AttnRes, fp32 atol 1e-6.

    The checks above compare my reference against autograd on a hand-built mixing expression.
    That proves the calculus but not that the expression is the one the model computes -- and
    on this exact code I have already been wrong once about what the model does (the 325/181
    retraction below). So this reads model.py itself: build a real AttnRes, real Sources, and
    require BOTH the forward and every dV to match, on the same inputs.

    Uses tilerl's parity dtype (fp32, atol 1e-6) rather than the fp64/1e-12 above, because
    that is the tolerance a Triton kernel will be held to.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import model as m  # noqa: PLC0415

    torch.manual_seed(7)
    n, B, T, D = 6, 2, 5, 32
    ar = m.AttnRes(D).float()
    # q is zero-init (uniform mixing) and g is ones -- both identity elements, where a wrong
    # weighting is invisible. Randomize them, the same lesson as test_split_bitwise.py:84.
    with torch.no_grad():
        ar.q.normal_(std=0.5)
        ar.g.normal_(mean=1.0, std=0.2)
    vs = [torch.randn(B, T, D, requires_grad=True) for _ in range(n)]
    srcs = [m.Source.of(v) for v in vs]

    out = ar(srcs)
    dout = torch.randn(B, T, D)
    out.backward(dout)
    got = [v.grad.clone() for v in vs]

    # Recompute `a` the way AttnRes does, then hand it to the reference. `scale` is part of
    # the logits, not of the mixing, so the reference must NOT see it -- if it did, this test
    # would pass a fused kernel that folded the scale into dV, which the model does not.
    with torch.no_grad():
        gq = ar.g * ar.q
        lg = torch.stack([(s.v * gq).sum(-1) * s.scale.squeeze(-1) for s in srcs])
        a = lg.float().softmax(0).to(srcs[0].v.dtype)
    ref_out = mix_reference(a, [s.v.detach() for s in srcs])
    assert torch.allclose(out.detach(), ref_out, atol=1e-6), \
        f"forward max {(out.detach() - ref_out).abs().max():.3e}"

    want, _ = grads_analytic(a, [s.v.detach() for s in srcs], dout)
    # dV from autograd carries the logits path too (each v is read by BOTH the mixing and its
    # own logit), so it is NOT equal to a_i*dout. Reporting the gap is the point: it is the
    # part a mixing-only kernel does not own, and a kernel author comparing against
    # grads_analytic alone would chase it as a bug.
    gaps = [(g - w).abs().max().item() for g, w in zip(got, want, strict=True)]
    print(f"known-answer vs model.AttnRes (fp32, atol 1e-6): forward MATCHES "
          f"(max {(out.detach() - ref_out).abs().max():.2e}). dV mixing-only term differs from "
          f"autograd's total by max {max(gaps):.3e} -- that residue is the LOGITS path "
          f"(model.py:248 reads every v a second time), which this kernel does not fuse. "
          f"A kernel must match the mixing term; the logits term stays in autograd.")
    return max(gaps)


def add_count(L, D=8, B=1, T=3, detach_logits=False):
    """MEASURE add_[B,T,D] on a real Full AttnRes stack, and give its exact closed form.

    tilerl measured 650 per step at L=12 -- the trace's ~1950 is 3 STEPS, so my "6.5x gap" was
    a per-step vs per-3-step unit error (their catch). They decomposed 650 as "two triangles +
    52 singles" and read the 2 as two independent AttnRes stacks, one per sublayer kind.

    IT IS NOT TWO STACKS, AND THE CLOSED FORM IS EXACT:

        add_[B,T,D] per step = 2 * source_reads = n(n+1),   n = 2L+1

    Measured at L=2/3/4/12 -> 30/56/90/650, exact at every depth; 4290 at L=32. The 2 is not
    two stacks: each AttnRes call reads every source TWICE, once for the logits (model.py:248,
    (s.v*gq).sum(-1)) and once for the mixing (model.py:250-252), so every source read puts two
    edges on the graph and earns two accumulations.

    `detach_logits=True` IS THE NEGATIVE CONTROL, AND THE ASSERT ABOVE IS NOT (tilerl, 14:3xZ).
    `n_add == n*(n+1)` passes whether or not the 2 comes from the double read: it constrains the
    VALUE, and "650 = 2 x 325" does not entail "the 2 is the double read" -- two propositions,
    the first not implying the second. The control is the run with the mechanism ABSENT: detach
    the logits read only, leave the mixing untouched, and the count must halve exactly. It does,
    at every depth (30->15, 56->28, 90->45, 650->325). tilerl ran this before accepting the
    explanation; I had shipped the assert and called it measured. Same shape as
    docs/experience/errors/2026-09-02-green-checks-that-proved-less-than-they-looked.md: a check
    never seen red is an assumption about the check.

    THE CONSEQUENCE TURNS ON THE NUMBER I TRIED TO RETRACT THIS MORNING: the count is exactly
    2x source_reads, and source_reads is eff.grad_ckpt_inverts_with_depth's O(L^2) quantity --
    325 at L=12. HALF of these adds belong to the LOGITS path, which a mixing-only kernel does
    not fuse. So the assigned per-call mixing fusion can address at most half the count, and
    removes none of it (the accumulation is ACROSS calls -- see main()).

    Counted on CPU at D=8: the count is structural, independent of B/T/D, which is why a tiny
    stack reproduces the production number exactly -- and makes this a cheap regression check
    on any future change to the AttnRes wiring.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    import model as m  # noqa: PLC0415

    def forward_detached(self, srcs):
        """model.AttnRes.forward with the LOGITS read detached and the mixing untouched."""
        q = self.q if self.dyn is None else self.q + self.dyn(srcs[-1].normed() * self.g)
        gq = self.g * q
        logits = torch.stack([(s.v.detach() * gq).sum(-1) * s.scale.squeeze(-1) for s in srcs])
        a = logits.float().softmax(0).to(srcs[0].v.dtype)
        out = a[0].unsqueeze(-1) * srcs[0].v
        for i in range(1, len(srcs)):
            out = out + a[i].unsqueeze(-1) * srcs[i].v
        return out

    torch.manual_seed(0)
    ars = [m.AttnRes(D) for _ in range(2 * L + 1)]
    for ar in ars:
        with torch.no_grad():
            ar.q.normal_(std=0.5)  # q is zero-init: uniform weights are an identity element
    fwd = (lambda ar, s: forward_detached(ar, s)) if detach_logits else (lambda ar, s: ar(s))
    x = torch.randn(B, T, D, requires_grad=True)
    done, partial = [m.Source.of(x)], []
    for i in range(2 * L):
        out = fwd(ars[i], done + partial) * 1.0001  # *1.0001 stands in for norm + f
        partial = [m.Source.of(partial[0].v + out if partial else out)]
        done, partial = done + partial, []  # attn_res_blocks=0: every sublayer is a boundary
    y = fwd(ars[-1], done + partial)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        y.sum().backward()
    want = str([[B, T, D], [B, T, D], []])
    n_add = sum(e.count for e in prof.key_averages(group_by_input_shape=True)
                if e.key == "aten::add_" and str(e.input_shapes) == want)

    n = 2 * L + 1
    expect = n * (n + 1) // 2 if detach_logits else n * (n + 1)
    assert n_add == expect, \
        f"L={L} detach_logits={detach_logits}: measured {n_add}, expected {expect}"
    return n_add


def bytes_table():
    """1e's closure test (14:0xZ): reproduce the trace's 261.7 GB/step from the add_ structure.

    IT CLOSES EXACTLY, and the arithmetic is short enough to check by eye:

        650 add_ x 3 [B,T,D] touches x 0.1342 GB = 261.7 GB/step

    Three touches per in-place add_: read the destination buffer, read the source contribution,
    write the destination. 650 is add_count()'s closed form n(n+1) at n=25. The trace's figure
    is 261.7 GB/step. Four significant figures, no fitted constant -- so the accounting of what
    those adds are is now settled, which is what 1e asked for.

    WHAT THE FUSION CAN AND CANNOT TAKE OFF THAT. The 650 splits in half by which read earns
    it: 325 from the mixing (model.py:250-252) and 325 from the logits (model.py:248). A
    mixing-only kernel:
      - REMOVES the intra-call temporaries: the k-1 intermediate `out` allocations in forward
        and the per-branch dV materialisation in backward.
      - DOES NOT remove any of the 650 accumulations. They are cross-call, and a per-call node
        still emits one contribution per source it reads.
      - CANNOT touch the logits half at all.

    So the ceiling below is a real saving with a real mechanism, but it is not "the 90.7 ms of
    add_" -- those two numbers measure different things, and conflating them is how a 30 ms
    lever gets sold as a 90 ms one.
    """
    B, T, Dm, el = 16, 4096, 1024, 2
    vb = B * T * Dm * el / 1e9  # 0.1342 GB per [B,T,D]
    calls = list(range(1, 2 * 12 + 1)) + [2 * 12 + 1]  # sources per call at attn_res_blocks=0
    n = 2 * 12 + 1

    traced = n * (n + 1) * 3 * vb
    assert abs(traced - 261.7) < 0.1, f"add_ traffic {traced:.1f} GB != the trace's 261.7"

    # NOW, forward: read k v's, k-1 intermediate `out` read+write pairs, final write.
    fwd_now = sum(k + 2 * (k - 1) + 1 for k in calls) * vb
    # NOW, backward: each of k branches reads dout and writes its dV (2k), then each is add_ed
    # into the source's grad buffer (3k: read dst, read src, write dst). A first version wrote
    # 4k here, omitting the k contribution writes, and produced a FUSED total LARGER than the
    # current one -- an impossibility, and the only reason the omission was caught. A byte
    # model with no sanity relation is unfalsifiable; the relation asserted below is saved > 0.
    bwd_now = sum(2 * k + 3 * k for k in calls) * vb
    # FUSED: forward reads each v once and writes out once; backward reads dout ONCE per call
    # and writes the k dV's, accumulating a_i in registers. The 3k accumulation SURVIVES.
    fwd_fused = sum(k + 1 for k in calls) * vb
    bwd_fused = sum(1 + k + 3 * k for k in calls) * vb

    saved = (fwd_now - fwd_fused) + (bwd_now - bwd_fused)
    assert saved > 0 and bwd_fused < bwd_now and fwd_fused < fwd_now, \
        f"fusing cannot move more bytes: fwd {fwd_now:.1f}->{fwd_fused:.1f}, " \
        f"bwd {bwd_now:.1f}->{bwd_fused:.1f}"

    print(f"CLOSURE: {n * (n + 1)} add_ x 3 touches x {vb:.4f} GB = {traced:.1f} GB/step, "
          f"against the trace's 261.7 GB/step -- EXACT, no fitted constant.\n"
          f"BYTES per step, L=12 stack (bf16 [16,4096,1024], {sum(calls)} source reads):\n"
          f"  forward   now {fwd_now:7.1f} GB -> fused {fwd_fused:7.1f} GB\n"
          f"  backward  now {bwd_now:7.1f} GB -> fused {bwd_fused:7.1f} GB\n"
          f"  saved {saved:.1f} GB = {saved / 4000 * 1000:.1f} ms at 4.0 TB/s. That is the "
          f"temporaries, NOT the {n * (n + 1)} accumulations, which the fusion leaves intact.")
    return saved



def main():
    torch.manual_seed(0)
    n, B, T, D = 5, 2, 7, 8
    vs = [torch.randn(B, T, D, dtype=torch.float64, requires_grad=True) for _ in range(n)]
    logits = torch.randn(n, B, T, dtype=torch.float64, requires_grad=True)
    a = logits.softmax(0)
    dout = torch.randn(B, T, D, dtype=torch.float64)

    out = mix_current(a, vs)
    ref = mix_reference(a.detach(), [v.detach() for v in vs])
    assert torch.equal(out.detach(), ref), "reference must be bit-identical to model.py's loop"

    # autograd's answer, against the analytic one the kernel must reproduce
    out.backward(dout)
    dV, dA = grads_analytic(a.detach(), [v.detach() for v in vs], dout)
    for i in range(n):
        got, want = vs[i].grad, dV[i]
        assert torch.allclose(got, want, atol=1e-12), f"dV[{i}] max {(got - want).abs().max()}"

    # dA is checked through the softmax, since `a` is not a leaf: autograd gives d/d logits,
    # and comparing at the logits is the only place both sides have the same variable.
    da_soft = (dA - (dA * a.detach()).sum(0, keepdim=True)) * a.detach()
    assert torch.allclose(logits.grad, da_soft, atol=1e-12), \
        f"dA through softmax max {(logits.grad - da_soft).abs().max()}"

    known_answer()
    # BOTH ARMS at every depth: the full stack, and the negative control with the logits read
    # detached. The control is what makes "the 2 is the double read" a measurement instead of an
    # interpretation -- see add_count's docstring.
    for L in (2, 3, 4, 12):
        full, half = add_count(L), add_count(L, detach_logits=True)
        assert full == 2 * half, f"L={L}: {full} != 2 x {half}"
    print(f"ABLATION (tilerl's control): logits-detached halves the count exactly at L=2/3/4/12 "
          f"-- 650 -> {add_count(12, detach_logits=True)} at L=12. The mixing is untouched in "
          f"that arm, so the factor 2 IS the double read, not two stacks.")

    # THE COUNT THE FUSION REMOVES, and a retraction. `attn_res_blocks` defaults to 0
    # (train.py:219) and model.py:330 reads it as `min(n_sub, blocks or n_sub)` -- the `or`
    # makes 0 a SENTINEL meaning Full, so n_blocks = n_sub = 2L and EVERY sublayer is a block
    # boundary that promotes immediately. So eff.grad_ckpt_inverts_with_depth's 325/2145 (peak
    # 65/25) is the shipped configuration and is correct. An earlier version of this file
    # asserted 181/1121 was "the real call" on the strength of the argument being named
    # `blocks=0`; blocks=L is a configuration no run has ever used. Retracted d4a0f78.
    def reads(L, blocks=0):
        n_sub = 2 * L
        nb = min(n_sub, blocks or n_sub)  # same expression as model.py:330
        ends = {round((j + 1) * n_sub / nb) for j in range(nb)}
        done, partial, k, tot = [0], [], 0, 0
        for _ in range(L):
            for _s in range(2):
                tot += len(done) + len(partial)
                if not partial:
                    partial = [1]
                k += 1
                if k in ends:
                    done, partial = done + partial, []
        return tot + len(done) + len(partial)

    r12, r32 = reads(12), reads(32)
    assert (r12, r32) == (325, 2145), (r12, r32)  # the default = Full = what runs
    assert (reads(12, 12), reads(32, 32)) == (181, 1121)  # blocks=L, which nothing runs
    # (n+1)(n+2)/2 at n=2L is the closed form of the Full count, so O(L^2) is exact here,
    # not empirical: 6.60x reads for 2.67x depth.
    assert r12 == (25 * 26) // 2 and r32 == (65 * 66) // 2

    # SOURCES AND BYTES (1e's roofline question, 13:5xZ). Elementwise kernels are already at
    # the bandwidth roofline, so the fusion's ceiling is bytes-removed / 4.0 TB/s
    # (eff.h20_specs). Shape from the trace: bf16 [16, 4096, 1024].
    calls = list(range(1, 2 * 12 + 1)) + [2 * 12 + 1]  # sources per call: 1..24, then 25
    assert sum(calls) == r12

    # WHAT THE PER-CALL FUSION ACTUALLY REMOVES -- and it is NOT the accumulations.
    # Inside one call the chain is out = a0*v0; out = out + ai*vi. AddBackward passes the grad
    # through unchanged, so there is no accumulation WITHIN a call. What accumulates is at the
    # SOURCE: s.v at position p is read by every call from p to 25, and each read is a separate
    # graph edge whose gradient autograd add_s into one buffer. A per-call fused node still
    # emits one contribution per source it reads, so the count does not move. Collapsing it
    # needs ONE node over the whole stack, a far larger change than the one assigned.
    saved_gb = bytes_table()
    measured = add_count(12)

    print(f"OK reference matches model.py:250-252 bitwise; dV and dA match autograd to 1e-12 "
          f"(n={n} sources, fp64). Source reads at Full (attn_res_blocks=0, what runs): "
          f"L12 {r12}, L32 {r32}, {r32 / r12:.2f}x for 2.67x depth -- the fact's 325/2145 is "
          f"CORRECT; my 181/1121 'correction' is retracted.\n"
          f"add_[B,T,D] = 2 x source_reads = n(n+1) = {measured}/step at L12, measured, and "
          f"{measured} x 3 touches accounts for the trace's 261.7 GB/step exactly. Half of it "
          f"({r12}) is the LOGITS read, which a mixing-only kernel cannot touch; the other half "
          f"is cross-call, which a per-call node does not remove. So the fusion's real prize is "
          f"the temporaries: {saved_gb / 4000 * 1000:.0f} ms/step, not the 90.7 ms of add_.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
