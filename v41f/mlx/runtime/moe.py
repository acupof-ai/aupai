"""v42 routed MoE for the runtime: QuantizedSwitchLinear / SwitchGLU.

Design §5 / §3:
  * Routing = sqrtsoftplus gate, bias, top-k, normalize, route_scale=1.5.
  * Routed experts evaluated with fused ``mx.gather_qmm`` over Q8/Q4 affine
    weights (no Python expert loop on the hot path).
  * SwitchGLU = w1 * sigmoid(w1) * w3, clamped like the dense path
    (swiglu_limit). Shared expert stays BF16.
  * v42 sqrtsoftplus route is the gold reference; any change must re-run §10.
"""
from __future__ import annotations

import mlx.core as mx

from ..model import _deq, _topk  # validated pure math (immutable semantics)


def route(xf: mx.array, gate_w: mx.array, gate_b: mx.array, cfg):
    """sqrtsoftplus routing. xf: [T,d] fp32. Returns (topk_scores[T,k], idx[T,k])."""
    gw = _deq(gate_w, mx.float32)
    scores = (xf @ gw.T) / cfg.gate_temp
    if cfg.score_func == "softmax":
        scores = mx.softmax(scores, axis=-1)
    elif cfg.score_func == "sigmoid":
        scores = mx.sigmoid(scores)
    else:  # sqrtsoftplus
        scores = mx.sqrt(mx.logaddexp(scores, 0.0))
    topk_scores, topk_idx = _topk(
        scores + gate_b.astype(mx.float32), k=cfg.n_activated_experts, axis=-1)
    if cfg.norm_topk_prob and cfg.n_activated_experts > 1:
        topk_scores = topk_scores / (mx.sum(topk_scores, axis=-1, keepdims=True) + 1e-20)
    return topk_scores * cfg.route_scale, topk_idx.astype(mx.int32)


class SwitchGLU:
    """Fused quantized SwitchGLU over routed experts via gather_qmm."""

    def __init__(self, qstore, group_size: int = 64, bits: int = 8):
        self.qstore = qstore
        self.group_size = group_size
        self.bits = bits

    def __call__(self, x, gate_w, gate_b, shared_w1, shared_w3, shared_w2, layer_id, cfg):
        b, s, d = x.shape
        t = b * s
        k = cfg.n_activated_experts
        xf = x.reshape(t, d).astype(mx.float32)
        topk_scores, topk_idx = route(xf, gate_w, gate_b, cfg)

        qweights = self.qstore.get_layer(layer_id)
        rhs = topk_idx.reshape(-1)
        lhs = mx.repeat(mx.arange(t, dtype=mx.int32), k)
        x3 = xf.astype(mx.bfloat16)[:, None, :]

        def qmm(inp, pack, lhs_idx):
            q, scale, bias = pack
            return mx.gather_qmm(
                inp, q, scale, bias,
                lhs_indices=lhs_idx, rhs_indices=rhs,
                transpose=True, group_size=self.group_size, bits=self.bits,
                mode="affine")

        h1 = qmm(x3, qweights["w1"], lhs).reshape(t, k, cfg.moe_inter_dim)
        h3 = qmm(x3, qweights["w3"], lhs).reshape(t, k, cfg.moe_inter_dim)
        if cfg.swiglu_limit > 0:
            h3 = mx.clip(h3, -cfg.swiglu_limit, cfg.swiglu_limit)
            h1 = mx.clip(h1, None, cfg.swiglu_limit)
        gate = h1 * mx.sigmoid(h1) * h3  # SwitchGLU

        q2, s2, b2 = qweights["w2"]
        slot_rows = mx.arange(t * k, dtype=mx.int32)
        expert_out = mx.gather_qmm(
            gate.reshape(t * k, 1, cfg.moe_inter_dim).astype(mx.bfloat16),
            q2, s2, b2, lhs_indices=slot_rows, rhs_indices=rhs,
            transpose=True, group_size=self.group_size, bits=self.bits,
            mode="affine").reshape(t, k, d)
        routed = mx.sum(expert_out.astype(mx.float32) * topk_scores[..., None], axis=1)

        # Shared expert BF16.
        xb = xf.astype(mx.bfloat16)
        sw1 = _deq(shared_w1, mx.bfloat16)
        sw3 = _deq(shared_w3, mx.bfloat16)
        sw2 = _deq(shared_w2, mx.bfloat16)
        sh1 = xb @ sw1.T
        sh3 = xb @ sw3.T
        if cfg.swiglu_limit > 0:
            sh3 = mx.clip(sh3, -cfg.swiglu_limit, cfg.swiglu_limit)
            sh1 = mx.clip(sh1, None, cfg.swiglu_limit)
        shared = (sh1 * mx.sigmoid(sh1) * sh3).astype(mx.bfloat16) @ sw2.T
        return (routed + shared.astype(mx.float32)).reshape(b, s, d)
