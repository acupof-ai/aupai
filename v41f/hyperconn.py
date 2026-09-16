"""Hyper-Connections (DeepSeek-V4.1 Block), faithful to upstream model_ref.Block.

The residual stream is `hc_mult` parallel copies `x` [b,s,hc,d]. Each sublayer sits
between `hc_pre` (collapse the copies into one sublayer input weighted by `pre`) and
`hc_post` (expand the output back out, mixing the residual through the doubly-stochastic
`comb`). One normalized projection of the flattened stream yields pre/post/comb, with
`comb` projected to a doubly-stochastic matrix by Sinkhorn.

The coefficients a sublayer computes are consumed by the NEXT one: attention collapses on
the pre_mix handed in from the previous layer, the FFN collapses on the attention's own
attn_pre. This module holds both sublayers' coefficient parameters (as Block does) and
exposes the three math methods with Block's exact signatures so the allclose test can
call them unbound against the reference.
"""

import torch
import torch.nn.functional as F
from torch import nn


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """Pure-torch port of kernel.hc_split_sinkhorn_kernel.

    mixes: [b,s,(2+hc)*hc]; per token layout pre[hc] | post[hc] | comb[hc*hc].
    Returns pre[b,s,hc], post[b,s,hc], comb[b,s,hc,hc]. `comb` is softmaxed over rows,
    then alternating row/column normalization (one column round before the serial loop,
    then iters-1 full rounds) drives it doubly stochastic.
    """
    b, s, _ = mixes.shape
    n = b * s
    flat = mixes.reshape(n, -1)
    pre = torch.sigmoid(flat[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult].view(1, -1)) + eps
    post = 2 * torch.sigmoid(
        flat[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult].view(1, -1)
    )
    comb = flat[:, 2 * hc_mult :].reshape(n, hc_mult, hc_mult) * hc_scale[2] + hc_base[2 * hc_mult :].view(
        1, hc_mult, hc_mult
    )
    comb = torch.softmax(comb, dim=-1) + eps
    # one column normalization outside the serial loop, then iters-1 row+column rounds
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return (pre.view(b, s, hc_mult), post.view(b, s, hc_mult), comb.view(b, s, hc_mult, hc_mult))


class HyperConn(nn.Module):
    """The two sublayers' hyper-connection coefficient tables for one block."""

    def __init__(self, dim, hc_mult=4, sinkhorn_iters=20, eps=1e-6, norm_eps=1e-20):
        super().__init__()
        self.dim = dim
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = sinkhorn_iters
        self.hc_eps = eps
        self.norm_eps = norm_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim
        # Block builds these in fp32; keep them fp32 regardless of the stream dtype.
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.empty(3))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3))
        self.reset_parameters()

    def reset_parameters(self):
        for p in (self.hc_attn_fn, self.hc_ffn_fn):
            nn.init.normal_(p, std=0.02)
        for p in (self.hc_attn_base, self.hc_ffn_base):
            nn.init.zeros_(p)
        nn.init.ones_(self.hc_attn_scale)
        nn.init.ones_(self.hc_ffn_scale)

    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        """x [b,s,hc,d] -> (pre,post,comb). One RMS-normalized linear over the flat
        hc*d stream (one statistic per token), then the Sinkhorn split."""
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x, pre_mix):
        """[b,s,hc,d] x [b,s,hc] -> [b,s,d]: weighted collapse onto one sublayer input."""
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype)

    def hc_post(self, x, residual, post, comb):
        """x [b,s,d], residual [b,s,hc,d], post [b,s,hc], comb [b,s,hc,hc] -> [b,s,hc,d]:
        expand the sublayer output weighted by post and add the comb-mixed residual."""
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)

    def attn(self, x, pre_mix):
        """Coefficients for the attention sublayer from residual x; returns the collapsed
        attention input and (post,comb) for its expansion."""
        pre, post, comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        return pre, post, comb, self.hc_pre(x, pre_mix)

    def ffn(self, x, attn_pre):
        """FFN collapses on the attention's pre (previous-sublayer coefficient hand-off)."""
        pre, post, comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        return pre, post, comb, self.hc_pre(x, attn_pre)
