#!/usr/bin/env python3
"""Reference implementation and gradient derivation for a fused AttnRes mixing kernel.

WHY THIS EXISTS (fb 2026-09-02): the trace shows ~1800 aten::add_ of [16,4096,1] and
~1950 of [16,4096,1024] per step, 90.7 ms. fb's hypothesis, for tilerl to confirm by
External id: those add_ calls are autograd's gradient accumulation for multi-consumer
tensors, not inductor-fused arithmetic. If so, an in-place `out.add_()` in forward changes
nothing and the fix is to make the mixing one autograd node.

WHAT THIS FILE FOUND, WHICH ARGUES AGAINST THE ASSIGNED SHAPE (read before writing a
kernel):

 1. The accumulation is at the SOURCE, across calls -- not inside one call. Within a call
    the chain is out = out + a_i*v_i, and AddBackward passes grad through unchanged, so no
    accumulation happens there. Source.v at position p is read by every AttnRes call from p
    to 25 (at L=12, attn_res_blocks=0), so it collects 26-p separate dV tensors that
    autograd add_s into one buffer. A PER-CALL fused node still emits one dV per source it
    reads, so it leaves that count exactly where it is: 300 -> 300 per micro-batch.
    Collapsing it needs ONE node over the whole stack, a much bigger change.
 2. The roofline ceiling does not reach the measured cost. Elementwise kernels are already
    bandwidth-bound (1e, 13:5xZ), so the ceiling is bytes-removed / 4.0 TB/s
    (eff.h20_specs). The per-call fusion removes intra-call temporaries only: 120.8 GB/step
    = 30.2 ms against 90.7 ms. Attribution does not close.
 3. The counted add_ (300) is 6.5x short of the trace's ~1950, so the mechanism behind
    those calls is still unidentified. That gap should be explained first.

This file is the MATH plus a pure-torch reference. It writes no kernel. If a kernel is
still wanted, parity is elementwise against `grads_analytic` plus a 20-step loss compare.

    python3 scripts/attnres_fused_reference.py          # derivation checks + parity
"""
import sys

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
    B, T, Dm, el = 16, 4096, 1024, 2
    vb = B * T * Dm * el / 1e9  # GB per [B,T,D] tensor = 0.134
    calls = list(range(1, 2 * 12 + 1)) + [2 * 12 + 1]  # sources per call: 1..24, then 25
    assert sum(calls) == r12

    # WHAT THE PER-CALL FUSION ACTUALLY REMOVES -- and it is NOT the cross-call accumulation.
    # Inside one call the chain is out = a0*v0; out = out + ai*vi. AddBackward passes the grad
    # through unchanged, so there is no accumulation WITHIN a call; every mul node just emits
    # its own dV_i. What accumulates is at the SOURCE: s.v at position p is read by every call
    # from p to 25, so it collects (26-p) separate dV tensors that autograd add_s into one
    # buffer. A per-call fused node still emits one dV contribution per source it reads, so it
    # leaves that count EXACTLY where it was. Removing it needs one node over the whole stack,
    # which is a far larger change than the one assigned.
    accum_now = r12 - len(calls)  # 300 add_ of [B,T,D] per micro-batch
    accum_fused = accum_now       # unchanged -- this is the finding
    # So the saving is intra-call temporaries only: forward's k-1 intermediate out
    # read/write pairs, and backward's per-branch materialisation.
    fwd_saved = sum(2 * (k - 1) for k in calls) * vb   # out reads + out writes removed
    bwd_saved = sum((k - 1) for k in calls) * vb       # dV temporaries kept in registers
    saved_gb = fwd_saved + bwd_saved
    ms_ceiling = saved_gb / 4000 * 1000

    print(f"OK reference matches model.py:250-252 bitwise; dV and dA match autograd to 1e-12 "
          f"(n={n} sources, fp64). Source reads at Full (attn_res_blocks=0, what runs): "
          f"L12 {r12}, L32 {r32}, {r32 / r12:.2f}x for 2.67x depth -- the fact's 325/2145 is "
          f"CORRECT; my 181/1121 'correction' is retracted.\n"
          f"ROOFLINE: the per-call fusion removes {saved_gb:.1f} GB/step = {ms_ceiling:.1f} ms "
          f"at 4.0 TB/s, against the trace's 90.7 ms of add_. ATTRIBUTION DOES NOT CLOSE.\n"
          f"AND THE add_ COUNT DOES NOT MOVE: {accum_now} -> {accum_fused} per micro-batch, "
          f"because a per-call node still emits one dV per source it reads; the accumulation "
          f"is at the source, across calls. The trace's ~1950 is {1950 / accum_now:.1f}x my "
          f"{accum_now}, so the mechanism is not yet identified -- that gap must be explained "
          f"before a kernel is worth writing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
