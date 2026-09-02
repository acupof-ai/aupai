#!/usr/bin/env python3
"""Reference implementation and gradient derivation for a fused AttnRes mixing kernel.

WHY THIS EXISTS (fb 2026-09-02): the trace shows ~1800 aten::add_ of [16,4096,1] and
~1950 of [16,4096,1024] per step, 90.7 ms. fb's hypothesis, for tilerl to confirm by
External id: those add_ calls are NOT inductor-fused arithmetic but autograd's gradient
accumulation for a multi-consumer tensor -- each Source.v is read by up to 13 later
AttnRes calls, so backward produces 13 separate dV contributions that autograd adds into
one buffer with add_. If that holds, an in-place `out.add_()` in forward changes nothing
(it is not where the adds come from) and the fix is to make the whole mixing one autograd
node, so autograd sees one dV per source instead of thirteen.

This file is the MATH plus a pure-torch reference. It writes no kernel. tilerl writes the
Triton kernel; parity is checked elementwise against `mix_reference` here plus a 20-step
loss comparison.

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

    # THE COUNT THE FUSION REMOVES. Each v[i] is read once here; in the real model source i
    # is read by every later AttnRes call, and each read is a separate graph edge whose
    # gradient autograd add_s into one buffer.
    def reads(L):
        n_sub = 2 * L
        ends = {round((j + 1) * n_sub / L) for j in range(L)}
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
    assert (r12, r32) == (181, 1121), (r12, r32)
    # eff.grad_ckpt_inverts_with_depth records 325 and 2145, and "65 live [B,T,D] tensors at
    # peak against 25". ROOT CAUSE, found not guessed: probes/t71_depth_lr_rule.py:130 is
    # `source_reads(L, blocks=0)`, and blocks=0 sets nb = n_sub = 2L, making EVERY sublayer a
    # block boundary that promotes its own output to `done` immediately. The recorded figures
    # are that default-argument call; the real model passes blocks=L, where model.py:399 keeps
    # `partial` out of the source list until a block boundary. Verified by calling t71's own
    # function both ways: blocks=0 -> 325/2145 exactly, blocks=L -> 181/1121, matching `reads`
    # here. The O(L^2) conclusion survives (6.19x for 2.67x depth, against its 6.60x); the
    # constant and the peak count (33 vs 13, not 65 vs 25) are wrong.
    assert (25 * 26) // 2 == 325 and (65 * 66) // 2 == 2145  # = the blocks=0 closed form

    print(f"OK reference matches model.py:250-252 bitwise; dV and dA match autograd to 1e-12 "
          f"(n={n} sources, fp64). Source reads: L12 {r12}, L32 {r32} -- O(L^2) with ratio "
          f"{r32 / r12:.2f}x for 2.67x depth. The fact's 325/2145 is t71's source_reads called "
          f"with its DEFAULT blocks=0, which makes every sublayer a block boundary; the real "
          f"call is blocks=L.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
