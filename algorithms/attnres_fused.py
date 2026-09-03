"""Fused AttnRes mixing: one autograd node over the whole call.

WHY. AttnRes plus the gradient accumulation it causes is 318 ms/step, 19.7% of wall
on p200m (measured, runs/trace_p200m_3step.json), and its source reads are n(n+1)/2
with n=2L+1 -- O(L^2) against an O(L) rest, so the share grows to 37.8% at L=32. The
eager form reads every v TWICE, once for the logits (model.py:269) and once for the
mixing (:271-273), and each read puts an edge on the graph: b0 measured add_ per step
= 2 x source_reads exactly, and an ablation detaching the logits read halved it at
L=2/3/4/12. One node reads v once and emits one gradient per source.

Design and gates: docs/standards/attnres_logits_kernel.md, scripts/attnres_logits_reference.py.

WHAT IS AND IS NOT HERE. This is the autograd node and its torch implementation --
correct, one traversal, and already removing the double read. The Triton kernel that
takes it to the bandwidth roofline is the next commit; this file's `_forward_impl` and
`_backward_impl` are where it plugs in, and the gates do not change when it does.
# ponytail: torch ops, so it does not yet hit the 10.9 ms byte floor -- the kernel is
# the upgrade path, and this node is the interface it needs either way.
"""

from __future__ import annotations

import torch


def _forward_impl(v, gq, scale):
    """out [B,T,D] and a [n,B,T], one pass over v, online softmax over the source axis.

    a_i needs every logit, so a streaming form cannot know it while reading v_i. Running
    max and denominator with the accumulator rescaled by exp(m_old - m_new) -- the
    exponent is <= 0 by construction, so the factor is in (0,1] and cannot overflow.
    The accumulator is fp32: bf16 here reads 3.8e-03 against a 1e-5 bar (measured).
    """
    n = len(v)
    B, T, D = v[0].shape
    dev = v[0].device
    # fp32 unless the input is already wider: fp32 is the floor the measurement sets
    # (bf16 accumulation reads 3.8e-03 against a 1e-5 bar), not a ceiling, and
    # downcasting an fp64 input would make the self-check measure rounding.
    acc_dt = v[0].dtype if v[0].dtype == torch.float64 else torch.float32
    m = torch.full((B, T), -float("inf"), dtype=acc_dt, device=dev)
    ell = torch.zeros(B, T, dtype=acc_dt, device=dev)
    acc = torch.zeros(B, T, D, dtype=acc_dt, device=dev)
    logit = torch.empty(n, B, T, dtype=acc_dt, device=dev)
    for i in range(n):
        logit[i] = (v[i].to(acc_dt) * gq).sum(-1) * scale[i].squeeze(-1)
        mi = torch.maximum(m, logit[i])
        rescale = torch.where(torch.isinf(m), torch.zeros_like(m), (m - mi).exp())
        p = (logit[i] - mi).exp()
        ell = ell * rescale + p
        acc = acc * rescale.unsqueeze(-1) + p.unsqueeze(-1) * v[i].to(acc_dt)
        m = mi
    return acc / ell.unsqueeze(-1), (logit - m).exp() / ell


def _backward_impl(v, gq, scale, a, dout):
    """dV_i, dgq and dscale_i in one pass, given `a` from forward.

    Storing `a` rather than recomputing it: a[n,B,T] fp32 is 0.0066 GB per call against
    3.36 GB to re-read v and rebuild the logits -- 512x, not a close call.

    Each v is read twice by the block, so its gradient carries both terms:
        dV_i = a_i.dout + (dlogit_i . scale_i) . gq
    Dropping the second reads as relative error 1.00, not as noise (gate 1).
    """
    n = len(v)
    acc_dt = a.dtype
    dA = torch.stack([(dout * v[i].to(acc_dt)).sum(-1) for i in range(n)])
    s = (a * dA).sum(0)
    dlogit = a * (dA - s.unsqueeze(0))
    dv, dscale = [], []
    dgq = torch.zeros_like(gq, dtype=acc_dt)
    for i in range(n):
        w = (dlogit[i] * scale[i].squeeze(-1)).unsqueeze(-1)  # [B,T,1]
        dv.append((a[i].unsqueeze(-1) * dout + w * gq).to(v[i].dtype))
        # dgq and dscale ride the same traversal -- no extra pass over v.
        dgq += (w * v[i].to(acc_dt)).sum((0, 1))
        dscale.append(((dlogit[i] * (v[i].to(acc_dt) * gq).sum(-1)).unsqueeze(-1)).to(scale[i].dtype))
    return dv, dgq.to(gq.dtype), dscale


class FusedAttnRes(torch.autograd.Function):
    """One node for logits + softmax + mixing. Inputs are flattened because
    autograd.Function does not accept a list of tensors as a differentiable argument."""

    @staticmethod
    def forward(ctx, gq, n, *tensors):
        v, scale = list(tensors[:n]), list(tensors[n:])
        out, a = _forward_impl(v, gq, scale)
        ctx.save_for_backward(gq, a, *v, *scale)
        ctx.n = n
        return out.to(v[0].dtype)

    @staticmethod
    def backward(ctx, dout):
        gq, a, *rest = ctx.saved_tensors
        n = ctx.n
        v, scale = rest[:n], rest[n:]
        dv, dgq, dscale = _backward_impl(v, gq, scale, a, dout.to(a.dtype))
        return (dgq, None, *dv, *dscale)


def fused_attn_res(v, gq, scale):
    """The call `AttnRes.forward` makes when the flag is on.

    PASS `scale` LIVE, never detached. Source.scale is rms_scale(v) -- a FUNCTION of the
    same v (model.py:244), so v reaches the output by two routes and the node only owns
    one of them. It returns dscale precisely so autograd can chain the other back through
    rms_scale; detaching scale silently drops that route and dV lands 7.6% low against
    the module while the FORWARD still matches to 1.5e-07. A forward check cannot see
    this, which is why the gate is on dV.
    """
    return FusedAttnRes.apply(gq, len(v), *v, *scale)


if __name__ == "__main__":  # pragma: no cover - self-check
    # Every gradient against autograd on the same expression, including the two the
    # design's contract did not name (dgq, dscale). fp64 so the check is about the
    # algebra and not about rounding.
    torch.manual_seed(0)
    n, B, T, D = 6, 2, 5, 32
    dt = torch.float64
    v = [torch.randn(B, T, D, dtype=dt, requires_grad=True) for _ in range(n)]
    sc = [(torch.rand(B, T, 1, dtype=dt) + 0.5).requires_grad_(True) for _ in range(n)]
    gq = torch.randn(D, dtype=dt, requires_grad=True)
    dout = torch.randn(B, T, D, dtype=dt)

    lg = torch.stack([(v[i] * gq).sum(-1) * sc[i].squeeze(-1) for i in range(n)])
    ref = sum(lg.softmax(0)[i].unsqueeze(-1) * v[i] for i in range(n))
    ref.backward(dout)
    want = ([x.grad.clone() for x in v], gq.grad.clone(), [s.grad.clone() for s in sc])
    for x in (*v, *sc, gq):
        x.grad = None

    out = fused_attn_res(v, gq, sc)
    assert torch.allclose(out, ref.detach(), atol=1e-12), (out - ref).abs().max()
    out.backward(dout)
    for i in range(n):
        assert torch.allclose(v[i].grad, want[0][i], atol=1e-12), f"dV[{i}]"
        assert torch.allclose(sc[i].grad, want[2][i], atol=1e-12), f"dscale[{i}]"
    assert torch.allclose(gq.grad, want[1], atol=1e-12), "dgq"

    # The point of the node: one backward pass, not one per source read. A wrong
    # implementation that drops the logits term still matches the FORWARD, so the
    # forward assertion above proves nothing about it -- this is what does.
    bad = [(lg.softmax(0).detach()[i].unsqueeze(-1) * dout) for i in range(n)]
    gap = max((b - w).abs().max().item() for b, w in zip(bad, want[0], strict=True))
    assert gap > 1e-3, "the mixing-only gradient must differ from the total, or gate 1 is blind"
    # Against the REAL module, with scale live: dV must be exact, not merely close.
    # This is the check that caught the detached-scale trap -- the forward agreed to
    # 1.5e-07 in both worlds while dV was 7.6% off in one of them.
    import os
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import model as _m

    torch.manual_seed(11)
    n2, B2, T2, D2 = 8, 2, 16, 64
    ar = _m.AttnRes(D2).float()
    with torch.no_grad():
        ar.q.normal_(std=0.5)
        ar.g.normal_(mean=1.0, std=0.2)

    def _fresh():
        return [torch.randn(B2, T2, D2, generator=torch.Generator().manual_seed(7 + i),
                            requires_grad=True) for i in range(n2)]

    d2 = torch.randn(B2, T2, D2, generator=torch.Generator().manual_seed(99))
    a1 = _fresh()
    ar([_m.Source.of(x) for x in a1]).backward(d2)
    ref_g = [x.grad.clone() for x in a1]
    a2 = _fresh()
    fused_attn_res(a2, ar.g * ar.q, [_m.rms_scale(x) for x in a2]).backward(d2)
    worst = max((x.grad - r).abs().max().item() for x, r in zip(a2, ref_g, strict=True))
    assert worst == 0.0, f"dV vs model.AttnRes must be exact with scale live, got {worst:.2e}"

    print(f"fused AttnRes: forward and all three gradients match autograd to 1e-12; "
          f"a mixing-only dV differs by {gap:.2e}, so the check has something to catch. "
          f"Against model.AttnRes with scale live, dV is exact (0.0).")
