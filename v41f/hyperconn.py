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

try:  # liger_kernel 0.8.3: Triton mHC kernels, CUDA only
    from liger_kernel.transformers.functional import liger_mhc_coeffs, liger_mhc_post_res, liger_mhc_pre

    HAS_LIGER_MHC = True
except ImportError:
    HAS_LIGER_MHC = False


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
        self.impl = "torch"  # "liger" set by V41FModel from cfg.hc_impl
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim
        # ref Block builds these under set_dtype(float32). Construct them explicitly fp32 so
        # the dtype does not depend on the global default at construction time: Block is
        # instantiated inside a set_default_dtype(bfloat16) context, under which a plain
        # torch.empty would wrongly make the coefficient tables bf16 (ref keeps them fp32).
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.reset_parameters()

    # The paper's gating factor alpha (arXiv 2512.24880 Eq. 7, Table 5 "Gating Factor Init 0.01").
    # The reference is inference-only and carries no init (model_ref :941-946), so this is ours.
    # With fn std 0.02 over hc*d = 4096 inputs the raw mixes have std 1.28; alpha 1.0 made pre/
    # post/comb fully token-dependent from step 0 (post std 0.49 around 1.0), and the v42 arch_b
    # trial's block outputs grew 2x per 100 steps through post. At 0.01 the start is the static
    # residual: pre = 0.5, post = 1.0, comb = 1/hc uniform, and the dynamic term ramps as alpha
    # learns (scratchpad hc_gain.py, 2026-09-30).
    HC_SCALE_INIT = 0.01

    def reset_parameters(self):
        for p in (self.hc_attn_fn, self.hc_ffn_fn):
            nn.init.normal_(p, std=0.02)
        for p in (self.hc_attn_base, self.hc_ffn_base):
            nn.init.zeros_(p)
        nn.init.constant_(self.hc_attn_scale, self.HC_SCALE_INIT)
        nn.init.constant_(self.hc_ffn_scale, self.HC_SCALE_INIT)

    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        """x [b,s,hc,d] -> (pre,post,comb). One RMS-normalized linear over the flat
        hc*d stream (one statistic per token), then the Sinkhorn split."""
        if self._liger(x):
            # phi is [hc*d, m] (ours is [m, hc*d]); alphas are our hc_scale[0:3]; eps roles map 1:1
            return liger_mhc_coeffs(
                x, hc_fn.t().contiguous(), hc_base, hc_scale[0], hc_scale[1], hc_scale[2],
                allow_fp32=True, tmax=self.hc_sinkhorn_iters, rms_eps=self.norm_eps,
                pre_eps=self.hc_eps, sinkhorn_eps=self.hc_eps, post_mult=2.0)
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x, pre_mix):
        """[b,s,hc,d] x [b,s,hc] -> [b,s,d]: weighted collapse onto one sublayer input."""
        if self._liger(x):
            return liger_mhc_pre(x, pre_mix)
        # bmm over hc: no [b,s,hc,d] fp32 product materialized (the elementwise form kept one
        # for backward per call, 256 MiB at B4 T4096 d1024 hc4)
        y = torch.matmul(pre_mix.unsqueeze(-2), x.float()).squeeze(-2)
        return y.to(x.dtype)

    def hc_post(self, x, residual, post, comb):
        """x [b,s,d], residual [b,s,hc,d], post [b,s,hc], comb [b,s,hc,hc] -> [b,s,hc,d]:
        expand the sublayer output weighted by post and add the comb-mixed residual."""
        if self._liger(residual):
            # liger: out[o] = sum_i h_res[o,i] x[i]; ours sums comb[i,j] over i -> hand it comb^T
            return liger_mhc_post_res(residual, x.to(residual.dtype), post, comb.transpose(-1, -2).contiguous())
        # y[j] = post[j] x + sum_i comb[i,j] residual[i]: comb^T @ residual as one bmm instead of
        # the [b,s,hc,hc,d] outer product (512 MiB per call at B4 T4096, OOM on the full trunk)
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), residual.float())
        return y.type_as(x)

    def _liger(self, x):
        return self.impl == "liger" and HAS_LIGER_MHC and x.is_cuda

    def attn(self, x, pre_mix):
        """Coefficients for the attention sublayer from residual x; returns the collapsed
        attention input and (post,comb) for its expansion."""
        pre, post, comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        return pre, post, comb, self.hc_pre(x, pre_mix)

    def ffn(self, x, attn_pre):
        """FFN collapses on the attention's pre (previous-sublayer coefficient hand-off)."""
        pre, post, comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        return pre, post, comb, self.hc_pre(x, attn_pre)
