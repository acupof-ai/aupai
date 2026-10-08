"""Fused quantized MoE for v42 MLX inference.

The routed expert tensors are quantized per layer with MLX affine Q8 and kept
resident. ``mx.gather_qmm`` evaluates only the selected top-k experts, without a
Python loop over experts. Shared experts remain BF16.
"""
from __future__ import annotations

import json
import os
import time

import mlx.core as mx
import numpy as np

from .model import _deq, _topk


class QExpertStore:
    """Lazy, persistent Q8 store for all routed experts."""

    def __init__(self, weight_dir="ckpt_local/sft_mlx_bf16", group_size=64,
                 bits=8):
        self.weight_dir = weight_dir
        self.group_size = group_size
        self.bits = bits
        with open(os.path.join(weight_dir, "manifest.json")) as f:
            manifest = json.load(f)
        self.entries = {e["src_key"]: e for e in manifest["entries"]}
        self.layers = {}
        self.quantize_s = 0.0

    def _load_bf16(self, key):
        e = self.entries[key]
        path = os.path.join(self.weight_dir, e["file"])
        shape = tuple(e["shape"])
        raw = np.memmap(path, dtype=np.uint16, mode="r", shape=shape)
        return mx.array(np.asarray(raw), dtype=mx.uint16).view(mx.bfloat16)

    def get_layer(self, layer_id):
        cached = self.layers.get(layer_id)
        if cached is not None:
            return cached
        t0 = time.perf_counter()
        p = f"layers.{layer_id}.ffn."
        out = {}
        for name in ("w1", "w3", "w2"):
            w = self._load_bf16(p + name)
            q, scale, bias = mx.quantize(
                w, group_size=self.group_size, bits=self.bits, mode="affine")
            mx.eval(q, scale, bias)
            out[name] = (q, scale, bias)
            del w
        self.quantize_s += time.perf_counter() - t0
        self.layers[layer_id] = out
        return out

    def prewarm(self, n_layers):
        for layer_id in range(n_layers):
            self.get_layer(layer_id)
        mx.clear_cache()


_QSTORE = None


def get_qstore():
    global _QSTORE
    if _QSTORE is None:
        _QSTORE = QExpertStore()
    return _QSTORE


def _route(xf, gate_w, gate_b, cfg):
    gw = _deq(gate_w, mx.float32)
    scores = (xf @ gw.T) / cfg.gate_temp
    if cfg.score_func == "softmax":
        scores = mx.softmax(scores, axis=-1)
    elif cfg.score_func == "sigmoid":
        scores = mx.sigmoid(scores)
    else:
        scores = mx.sqrt(mx.logaddexp(scores, 0.0))
    topk_scores, topk_idx = _topk(
        scores + gate_b.astype(mx.float32),
        k=cfg.n_activated_experts,
        axis=-1,
    )
    if cfg.norm_topk_prob and cfg.n_activated_experts > 1:
        topk_scores = topk_scores / (
            mx.sum(topk_scores, axis=-1, keepdims=True) + 1e-20)
    return topk_scores * cfg.route_scale, topk_idx.astype(mx.int32)


def q8_moe_forward(x, gate_w, gate_b, shared_w1, shared_w3, shared_w2,
                   layer_id, cfg):
    """Compute routed top-k experts with three fused gather_qmm operations."""
    b, s, d = x.shape
    t = b * s
    k = cfg.n_activated_experts
    xf = x.reshape(t, d).astype(mx.float32)
    topk_scores, topk_idx = _route(xf, gate_w, gate_b, cfg)

    qweights = get_qstore().get_layer(layer_id)
    rhs = topk_idx.reshape(-1)
    lhs = mx.repeat(mx.arange(t, dtype=mx.int32), k)
    x3 = xf.astype(mx.bfloat16)[:, None, :]

    def qmm(inp, pack, lhs_idx):
        q, scale, bias = pack
        return mx.gather_qmm(
            inp, q, scale, bias,
            lhs_indices=lhs_idx,
            rhs_indices=rhs,
            transpose=True,
            group_size=64,
            bits=8,
            mode="affine",
        )

    h1 = qmm(x3, qweights["w1"], lhs).reshape(t, k, cfg.moe_inter_dim)
    h3 = qmm(x3, qweights["w3"], lhs).reshape(t, k, cfg.moe_inter_dim)
    if cfg.swiglu_limit > 0:
        h3 = mx.clip(h3, -cfg.swiglu_limit, cfg.swiglu_limit)
        h1 = mx.clip(h1, None, cfg.swiglu_limit)
    gate = h1 * mx.sigmoid(h1) * h3

    q2, s2, b2 = qweights["w2"]
    slot_rows = mx.arange(t * k, dtype=mx.int32)
    expert_out = mx.gather_qmm(
        gate.reshape(t * k, 1, cfg.moe_inter_dim).astype(mx.bfloat16),
        q2, s2, b2,
        lhs_indices=slot_rows,
        rhs_indices=rhs,
        transpose=True,
        group_size=64,
        bits=8,
        mode="affine",
    ).reshape(t, k, d)
    routed = mx.sum(
        expert_out.astype(mx.float32) * topk_scores[..., None], axis=1)

    # Shared expert remains BF16 for accuracy.
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
