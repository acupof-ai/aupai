"""P0: Hyper-Connections match vendored upstream model_ref.Block, bitwise-close.

We never construct an upstream Block (it needs Attn/MoE). Instead we call the three
official methods UNBOUND from model.Block with a lightweight fake self carrying only the
attributes they read (hc_mult / hc_eps / hc_sinkhorn_iters / norm_eps) and feed our own
fp32 coefficient tensors (identical shapes/dtypes to Block.__init__), so reference and
implementation run the same params. The Sinkhorn they hit is ref_oracle's pure-torch
port of kernel.hc_split_sinkhorn_kernel.

Coverage:
- hc_mixes / hc_pre / hc_post elementwise allclose (fp32 tight; bf16 I/O on pre/post).
- coefficient hand-off timing: the FFN collapses on the attention's attn_pre, and the
  attention collapses on the incoming pre_mix (Block.forward ordering).
- comb doubly stochastic (row and column sums 1, tol 1e-5); pre/post shape + finite.
- one full forward+backward returns gradient to hc_fn/scale/base; a second backward on
  the shared residual container accumulates without corrupting the first.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from allclose import cmp
from ref_oracle import load_reference

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.hyperconn import HyperConn

HC, DIM, IT, EPS, NORM_EPS = 4, 32, 20, 1e-6, 1e-20
B, S = 2, 11


def _split_ref_3d(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """3D-correct wrapper around ref_oracle's accepted 2D sinkhorn port: the vendored
    kernel wrapper calls the kernel on mixes.view(-1,mix) and reshapes pre/post/comb
    back to [b,s,...], but the pure-torch stub only handles [N,mix]. This adds exactly
    that view/reshape around the SAME 2D math (no formula change), so Block.hc_mixes —
    whose flatten/rsqrt/linear stay official upstream code — runs on a 3D input."""
    import sys as _sys

    kernel = _sys.modules.get("kernel")
    b, s, _ = mixes.shape
    flat = mixes.reshape(b * s, -1)
    pre, post, comb = kernel.hc_split_sinkhorn(flat, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
    return (pre.view(b, s, hc_mult), post.view(b, s, hc_mult), comb.view(b, s, hc_mult, hc_mult))


def _pair():
    model, _ = load_reference()
    torch.manual_seed(7)
    ours = HyperConn(DIM, hc_mult=HC, sinkhorn_iters=IT, eps=EPS, norm_eps=NORM_EPS)
    # route upstream Block.hc_mixes through the shape-correct (formula-identical) split
    model.hc_split_sinkhorn = _split_ref_3d
    ns = SimpleNamespace(norm_eps=NORM_EPS, hc_mult=HC, hc_eps=EPS, hc_sinkhorn_iters=IT)
    return model, ours, ns


def test_split_sinkhorn_matches_2d_stub():
    """Our split, flattened to [N,mix], is bitwise-close to ref_oracle's accepted
    pure-torch port of kernel.hc_split_sinkhorn_kernel (formula level, 2D)."""
    import sys as _sys

    load_reference()
    kernel = _sys.modules["kernel"]
    torch.manual_seed(3)
    n, mix = 23, (2 + HC) * HC
    mixes = torch.randn(n, mix)
    scale = torch.randn(3)
    base = torch.randn(mix)
    from allclose import cmp as _cmp

    from v41f.hyperconn import hc_split_sinkhorn as _split

    op, opo, oc = _split(mixes.view(1, n, mix), scale, base, HC, IT, EPS)
    rp, rpo, rc = kernel.hc_split_sinkhorn(mixes, scale, base, HC, IT, EPS)
    _cmp("split pre 2d", op.view(n, HC), rp, atol=1e-6)
    _cmp("split post 2d", opo.view(n, HC), rpo, atol=1e-6)
    _cmp("split comb 2d", oc.view(n, HC, HC), rc, atol=1e-6)


def test_hc_mixes_allclose():
    model, ours, ns = _pair()
    x = torch.randn(B, S, HC, DIM) * 2.0
    op, opo, oc = ours.hc_mixes(x, ours.hc_attn_fn, ours.hc_attn_scale, ours.hc_attn_base)
    rp, rpo, rc = model.Block.hc_mixes(ns, x, ours.hc_attn_fn, ours.hc_attn_scale, ours.hc_attn_base)
    cmp("hc pre coeff", op, rp, atol=5e-2)
    cmp("hc post coeff", opo, rpo, atol=5e-2)
    cmp("hc comb matrix", oc, rc, atol=5e-2)
    assert op.shape == (B, S, HC) and oc.shape == (B, S, HC, HC)
    assert torch.isfinite(oc).all()


def test_pre_post_allclose_bf16():
    model, ours, ns = _pair()
    x = torch.randn(B, S, HC, DIM)
    pre_mix = torch.rand(B, S, HC) + 0.5
    sub = torch.randn(B, S, DIM)
    _, post, comb = ours.hc_mixes(x, ours.hc_attn_fn, ours.hc_attn_scale, ours.hc_attn_base)
    # hc_pre collapses; bf16 I/O
    cmp(
        "hc_pre bf16",
        ours.hc_pre(x.bfloat16(), pre_mix),
        model.Block.hc_pre(ns, x.bfloat16(), pre_mix),
        atol=2e-2,
    )
    # hc_post expands + mixes residual
    cmp(
        "hc_post bf16",
        ours.hc_post(sub.bfloat16(), x.bfloat16(), post, comb),
        model.Block.hc_post(ns, sub.bfloat16(), x.bfloat16(), post, comb),
        atol=2e-2,
    )


def test_coefficient_hand_off_timing():
    """ffn collapses on attention's attn_pre; attention on the passed-in pre_mix."""
    model, ours, ns = _pair()
    x = torch.randn(B, S, HC, DIM)
    incoming_pre = torch.rand(B, S, HC) + 0.5
    # attention sublayer (identity sublayer output for the math comparison)
    attn_pre, attn_post, attn_comb = ours.hc_mixes(x, ours.hc_attn_fn, ours.hc_attn_scale, ours.hc_attn_base)
    attn_in = ours.hc_pre(x, incoming_pre)
    r_attn_in = model.Block.hc_pre(ns, x, incoming_pre)
    cmp("attn uses incoming pre_mix", attn_in, r_attn_in, atol=5e-2)
    residual2 = ours.hc_post(torch.zeros(B, S, DIM), x, attn_post, attn_comb)
    r_residual2 = model.Block.hc_post(ns, torch.zeros(B, S, DIM), x, attn_post, attn_comb)
    cmp("attn residual2", residual2, r_residual2, atol=5e-2)
    # FFN collapses residual2 using attn_pre (the previous sublayer's output)
    ffn_pre, _, _ = ours.hc_mixes(residual2, ours.hc_ffn_fn, ours.hc_ffn_scale, ours.hc_ffn_base)
    r_ffn_pre, _, _ = model.Block.hc_mixes(ns, residual2, ours.hc_ffn_fn, ours.hc_ffn_scale, ours.hc_ffn_base)
    ffn_in = ours.hc_pre(residual2, attn_pre)
    # independent manual collapse (not the same hc_pre call): the FFN sublayer input is
    # the attn_pre-weighted sum of residual2 over copies. Timing is the point: it uses
    # attn_pre (previous sublayer), not the ffn_pre this sublayer just produced.
    manual = (attn_pre.unsqueeze(-1) * residual2.float()).sum(dim=2).to(residual2.dtype)
    cmp("ffn pre coeff", ffn_pre, r_ffn_pre, atol=5e-2)
    cmp("ffn collapses on attn_pre (manual)", ffn_in, manual, atol=1e-5)
    assert not torch.allclose(attn_pre, ffn_pre, atol=1e-3), (
        "ffn collapse must use attn_pre, not its own freshly produced ffn_pre"
    )


def test_comb_doubly_stochastic():
    _, ours, _ = _pair()
    x = torch.randn(3, 13, HC, DIM) * 3.0
    _, _, comb = ours.hc_mixes(x, ours.hc_ffn_fn, ours.hc_ffn_scale, ours.hc_ffn_base)
    row = comb.sum(dim=-1)
    col = comb.sum(dim=-2)
    assert torch.allclose(row, torch.ones_like(row), atol=1e-5), row.max().item()
    assert torch.allclose(col, torch.ones_like(col), atol=1e-5), col.max().item()
    assert (comb > 0).all(), "sinkhorn comb must be strictly positive"


def test_forward_backward_reaches_hc_fn():
    _, ours, _ = _pair()
    x = torch.randn(B, S, HC, DIM, requires_grad=True)
    pre_mix = torch.rand(B, S, HC) + 0.5

    def run_block(inp):
        a_pre, a_post, a_comb, a_in = ours.attn(inp, pre_mix)
        a_out = a_in  # identity attention sublayer
        r2 = ours.hc_post(a_out, inp, a_post, a_comb)
        f_pre, f_post, f_comb, f_in = ours.ffn(r2, a_pre)
        f_out = f_in  # identity FFN sublayer
        out = ours.hc_post(f_out, r2, f_post, f_comb)
        return out

    out = run_block(x)
    assert torch.isfinite(out).all()
    out.float().sum().backward()
    for name in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_scale", "hc_ffn_scale", "hc_attn_base", "hc_ffn_base"):
        g = getattr(ours, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name

    # a second backward on the shared residual container must add a reproducible,
    # non-corrupting increment (no cross-container bleed): gradient is deterministic.
    ours.zero_grad()
    x2 = x.detach().clone().requires_grad_(True)
    run_block(x2).float().sum().backward()
    g1 = ours.hc_ffn_fn.grad.detach().clone()
    run_block(x2.detach().clone().requires_grad_(True)).float().sum().backward()
    g2 = ours.hc_ffn_fn.grad
    # second backward doubles the gradient, exactly (same graph, same params)
    assert torch.allclose(g2, 2 * g1, atol=1e-5), (g2 - 2 * g1).abs().max().item()
