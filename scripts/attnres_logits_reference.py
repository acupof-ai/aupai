#!/usr/bin/env python3
"""Reference implementation and gates for the fused AttnRes logits+mixing kernel.

Design: docs/standards/attnres_logits_kernel.md. This file is the executable half --
the pure-torch reference a Triton kernel must match, and the gates that decide whether
it matches. No kernel here.

WHY A SECOND FILE. attnres_fused_reference.py covers the MIXING-only fusion, whose
contract is dV_i = a_i*dout with the logits term deliberately left in autograd. This
kernel fuses both paths, so it owns both terms and the tolerance that file uses does
not apply. Keeping them apart means neither file's assertions have to carry an "unless
the other design" clause.

    python3 scripts/attnres_logits_reference.py        # gates + negative controls
"""
import sys

import torch


def fused_forward(v, gq, scale):
    """out[b,t,d] and a[i,b,t], computed the way the kernel will: ONE pass over v.

    a_i needs every logit, so the accumulator is rescaled as the running max moves --
    flash-attention's online softmax over the SOURCE axis. `a` is returned because
    backward needs it and recomputing it costs 512x what storing it does.

    v:     list of n tensors [B,T,D]
    gq:    [D]        g*q, one per call
    scale: list of n [B,T,1]   the RMS factor; part of the LOGITS, never the mixing
    """
    n, (B, T, D) = len(v), v[0].shape
    dt = torch.float32
    m = torch.full((B, T), -float("inf"), dtype=dt)
    ell = torch.zeros(B, T, dtype=dt)
    acc = torch.zeros(B, T, D, dtype=dt)  # fp32: bf16 here fails parity by 4 orders
    logit = torch.empty(n, B, T, dtype=dt)
    for i in range(n):
        lg = (v[i].float() * gq).sum(-1) * scale[i].squeeze(-1)
        logit[i] = lg
        mi = torch.maximum(m, lg)
        # exp(m_old - m_new) with m_new >= m_old: the exponent is <= 0 by construction,
        # so the factor is in (0,1] and cannot overflow. First source: m is -inf.
        rescale = torch.where(torch.isinf(m), torch.zeros_like(m), (m - mi).exp())
        p = (lg - mi).exp()
        ell = ell * rescale + p
        acc = acc * rescale.unsqueeze(-1) + p.unsqueeze(-1) * v[i].float()
        m = mi
    return acc / ell.unsqueeze(-1), (logit - m).exp() / ell


def fused_backward(v, gq, scale, a, dout):
    """dV_i and dlogit_i in one pass over v, given `a` from forward.

    dlogit needs ALL dA before any of it is final and dV needs a; both dependencies are
    per-row scalars, which is what lets a streaming kernel close them in one traversal.
    """
    n = len(v)
    dA = torch.stack([(dout * v[i].float()).sum(-1) for i in range(n)])
    s = (a * dA).sum(0)
    dlogit = a * (dA - s.unsqueeze(0))
    # Each v is read TWICE by the block -- once by the mixing, once by its own logit --
    # so its gradient carries both terms. Dropping the second is the failure this
    # file's gate 1 exists to catch; it reads as relative error ~1.0, not as noise.
    dv = [a[i].unsqueeze(-1) * dout
          + (dlogit[i] * scale[i].squeeze(-1)).unsqueeze(-1) * gq
          for i in range(n)]
    return dv, dlogit


def _case(n=25, B=2, T=64, D=1024, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    v = [torch.randn(B, T, D, generator=g, dtype=dtype) for _ in range(n)]
    gq = torch.randn(D, generator=g)
    scale = [torch.rand(B, T, 1, generator=g) + 0.5 for _ in range(n)]
    dout = torch.randn(B, T, D, generator=g)
    return v, gq, scale, dout


def autograd_truth(v, gq, scale, dout):
    """The fp32 mathematical definition -- NOT model.AttnRes.

    model.py:270 casts the softmax back to v.dtype unconditionally, so the module
    carries ~6.4e-3 of its own bf16 rounding in training and no kernel can be held to
    1e-12 against it. The math is the judge; the module is what the math describes.
    """
    vs = [x.clone().requires_grad_(True) for x in v]
    logit = torch.stack([(vs[i] * gq).sum(-1) * scale[i].squeeze(-1) for i in range(len(vs))])
    a = logit.softmax(0)
    out = sum(a[i].unsqueeze(-1) * vs[i] for i in range(len(vs)))
    out.backward(dout)
    return out.detach(), a.detach(), [x.grad.clone() for x in vs]


def rel(got, want):
    num = max((g - w).abs().max().item() for g, w in zip(got, want, strict=True))
    den = max(w.abs().max().item() for w in want)
    return num / den


def gates(verbose=True):
    """Gates 1, 2 and 4's numeric half. Returns the measured relative errors."""
    v, gq, scale, dout = _case()
    out_ref, a_ref, dv_ref = autograd_truth(v, gq, scale, dout)

    out, a = fused_forward(v, gq, scale)
    fwd = (out - out_ref).abs().max().item() / out_ref.abs().max().item()
    a_err = (a - a_ref).abs().max().item()

    dv, dlogit = fused_backward(v, gq, scale, a, dout)
    # GATE 1: dV against autograd's TOTAL, relative. Absolute 1e-6 was calibrated at
    # D=32 and rejects a correct implementation at D=1024 (b0-review-7899ea1).
    g1 = rel(dv, dv_ref)
    assert g1 <= 1e-5, f"gate 1: dV relative {g1:.2e} > 1e-5"
    # GATE 2: the softmax backward, checked at the logits where both sides have the
    # same variable -- `a` is not a leaf, so there is nowhere else to compare it.
    lg = torch.stack([(v[i] * gq).sum(-1) * scale[i].squeeze(-1) for i in range(len(v))])
    lg.requires_grad_(True)
    aa = lg.softmax(0)
    (sum(aa[i].unsqueeze(-1) * v[i] for i in range(len(v)))).backward(dout)
    g2 = (dlogit - lg.grad).abs().max().item() / lg.grad.abs().max().item()
    assert g2 <= 1e-5, f"gate 2: dlogit relative {g2:.2e} > 1e-5"

    if verbose:
        print(f"gate 1  dV vs autograd total   rel {g1:.2e}  (bar 1e-5)")
        print(f"gate 2  dlogit through softmax rel {g2:.2e}  (bar 1e-5)")
        print(f"        forward                rel {fwd:.2e}   a max abs {a_err:.2e}")
    return g1, g2


def negative_controls(verbose=True):
    """GATE 5. Each control breaks one thing; every gate above must go red.

    A gate never seen red is a hypothesis about a gate. These are the runs where the
    mechanism is absent -- the only evidence that a green run means anything.
    """
    v, gq, scale, dout = _case()
    _, _, dv_ref = autograd_truth(v, gq, scale, dout)
    _, a = fused_forward(v, gq, scale)
    n = len(v)
    out = []

    # (a) drop the logits term from dV -- the mixing-only contract, wrong for this kernel
    dv_mix = [a[i].unsqueeze(-1) * dout for i in range(n)]
    r = rel(dv_mix, dv_ref)
    assert r > 1e-5, "control (a) did not fail: gate 1 cannot see a dropped logits term"
    out.append(("drops the logits term from dV", r))

    # (b) fold `scale` into the mixing, where it does not belong
    dv_scaled = [(a[i] * scale[i].squeeze(-1)).unsqueeze(-1) * dout for i in range(n)]
    r = rel(dv_scaled, dv_ref)
    assert r > 1e-5, "control (b) did not fail: gate 1 cannot see scale in the mixing"
    out.append(("folds scale into the mixing", r))

    # (c) bf16 accumulator -- the design says fp32 is a hard constraint, so show it
    vb = [x.bfloat16() for x in v]
    m = torch.full(v[0].shape[:2], -float("inf"))
    ell = torch.zeros(v[0].shape[:2])
    acc = torch.zeros(*v[0].shape, dtype=torch.bfloat16)
    for i in range(n):
        lg = (v[i] * gq).sum(-1) * scale[i].squeeze(-1)
        mi = torch.maximum(m, lg)
        rs = torch.where(torch.isinf(m), torch.zeros_like(m), (m - mi).exp())
        p = (lg - mi).exp()
        ell = ell * rs + p
        acc = acc * rs.unsqueeze(-1).bfloat16() + p.unsqueeze(-1).bfloat16() * vb[i]
        m = mi
    out_bf = acc.float() / ell.unsqueeze(-1)
    out_ref, _, _ = autograd_truth(v, gq, scale, dout)
    r = (out_bf - out_ref).abs().max().item() / out_ref.abs().max().item()
    assert r > 1e-5, "control (c) did not fail: a bf16 accumulator passes, so R could grow"
    out.append(("uses a bf16 accumulator", r))

    if verbose:
        for what, r in out:
            print(f"control  a kernel that {what:32s} -> rel {r:.2e}  REJECTED")
    return out


if __name__ == "__main__":  # pragma: no cover - self-check
    g1, g2 = gates()
    print()
    negative_controls()
    print()
    print(f"gates pass (dV {g1:.1e}, dlogit {g2:.1e}) and all three controls are rejected. "
          "add_count is NOT in this file: its post-fusion form n(n+1)/2 = 325 equals the "
          "detach_logits count, so it is blind to control (a) and is a wiring check only.")
    sys.exit(0)
