"""One Metal kernel for the 4x4 HyperConnection Sinkhorn.

The reference loop is ``sinkhorn_loop`` in ``v41f.mlx.model``. hc_mult is 4
on the v42 gate, so one thread owns one 4x4 matrix. Other sizes use the loop.
"""
from __future__ import annotations

import mlx.core as mx

from ..model import hc_mix_presinkhorn, sinkhorn_loop

_KERNEL = mx.fast.metal_kernel(
    name="v42_sinkhorn4",
    input_names=["comb", "nmat", "iters", "eps"],
    output_names=["out"],
    source="""
        uint idx = thread_position_in_grid.x;
        if (idx >= nmat[0]) return;
        int n_iter = iters[0];
        float e = eps[0];
        const int base = int(idx) * 16;
        float m[16];
        for (int i = 0; i < 16; ++i) m[i] = comb[base + i];

        for (int c = 0; c < 4; ++c) {
            float s = m[c] + m[4 + c] + m[8 + c] + m[12 + c] + e;
            m[c] /= s; m[4 + c] /= s; m[8 + c] /= s; m[12 + c] /= s;
        }
        for (int t = 1; t < n_iter; ++t) {
            for (int r = 0; r < 4; ++r) {
                int o = r * 4;
                float s = m[o] + m[o + 1] + m[o + 2] + m[o + 3] + e;
                m[o] /= s; m[o + 1] /= s; m[o + 2] /= s; m[o + 3] /= s;
            }
            for (int c = 0; c < 4; ++c) {
                float s = m[c] + m[4 + c] + m[8 + c] + m[12 + c] + e;
                m[c] /= s; m[4 + c] /= s; m[8 + c] /= s; m[12 + c] /= s;
            }
        }
        for (int i = 0; i < 16; ++i) out[base + i] = m[i];
    """,
)


def sinkhorn_fused(comb: mx.array, sinkhorn_iters: int, eps: float) -> mx.array:
    """Match ``sinkhorn_loop`` for a [..., 4, 4] matrix. Other ranks use the loop."""
    if comb.shape[-1] != 4 or comb.shape[-2] != 4:
        return sinkhorn_loop(comb, sinkhorn_iters, eps)
    flat_shape = (comb.size // 16, 16)
    nmat = comb.size // 16
    out = _KERNEL(
        inputs=[
            comb.reshape(flat_shape),
            mx.array([nmat], dtype=mx.int32),
            mx.array([sinkhorn_iters], dtype=mx.int32),
            mx.array([eps], dtype=mx.float32),
        ],
        output_shapes=[flat_shape],
        output_dtypes=[mx.float32],
        grid=(nmat, 1, 1),
        threadgroup=(min(64, max(1, nmat)), 1, 1),
    )[0]
    return out.reshape(comb.shape)


def hc_mixes_fast(x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps, norm_eps):
    """Same split as ``hc_mixes``, with the 4x4 Sinkhorn in one kernel."""
    pre, post, comb = hc_mix_presinkhorn(
        x, hc_fn, hc_scale, hc_base, hc_mult, eps, norm_eps)
    if hc_mult == 4:
        comb = sinkhorn_fused(comb, sinkhorn_iters, eps)
    else:
        comb = sinkhorn_loop(comb, sinkhorn_iters, eps)
    return pre, post, comb


def warmup() -> None:
    comb = mx.softmax(mx.ones((1, 1, 4, 4), dtype=mx.float32), axis=-1) + 1e-6
    mx.eval(sinkhorn_fused(comb, 20, 1e-6))
