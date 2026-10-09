"""Sparse active-expert MoE forward with a resident LRU expert cache.

Two entry points:
  * sparse_moe_forward            -- per-expert loop (legacy, used by accurate fallback).
  * sparse_moe_forward_batched   -- batched active-expert GEMM for decode (T small).

Routing math is an exact copy of model.py:moe_forward
(sqrtsoftplus gate -> bias -> top-k -> norm -> route_scale 1.5). Expert matmuls are
bf16 with swiglu clamp 10, matching the corrected bf16-active-sparse path.

The expert LRU holds resident BF16 mx.arrays keyed by (layer, expert, matrix) with a
hard byte budget. Hot experts stay resident; cold experts are mmap-read from the
per-expert .bin files on a miss. We deliberately do NOT call mx.clear_cache() after
every expert -- that evicts resident weights and forces re-reads. The MLX temp cache
is dropped only when RSS approaches the memguard ceiling (maybe_collect).
"""
from __future__ import annotations

import os
import resource
from collections import OrderedDict

import mlx.core as mx
import numpy as np

from .model import _topk, _deq


# ---------------------------------------------------------------------------
# Resident expert LRU
# ---------------------------------------------------------------------------
class ExpertLRUCache:
    """key=(layer_id, expert_id, matrix_name) -> resident bf16 mx.array."""

    def __init__(self, budget_gb: float = 2.0):
        self.budget_bytes = int(budget_gb * 1e9)
        self.cache: "OrderedDict[tuple, tuple[mx.array, int]]" = OrderedDict()
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.evicted_bytes = 0

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key][0]
        self.misses += 1
        return None

    def put(self, key, arr):
        nbytes = getattr(arr, "nbytes", 0) or 0
        while self.current_bytes + nbytes > self.budget_bytes and self.cache:
            old_key, (old_arr, old_bytes) = self.cache.popitem(last=False)
            self.current_bytes -= old_bytes
            self.evictions += 1
            self.evicted_bytes += old_bytes
        self.cache[key] = (arr, nbytes)
        self.current_bytes += nbytes

    def stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "evicted_bytes": self.evicted_bytes,
            "current_bytes": self.current_bytes,
            "budget_bytes": self.budget_bytes,
            "size": len(self.cache),
        }


_EXPERT_CACHE: ExpertLRUCache | None = None


def get_expert_cache(budget_gb: float = 2.0) -> ExpertLRUCache:
    global _EXPERT_CACHE
    if _EXPERT_CACHE is None:
        _EXPERT_CACHE = ExpertLRUCache(budget_gb=budget_gb)
    return _EXPERT_CACHE


def _rss_bytes() -> int:
    # macOS ru_maxrss is in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


_RSS_COLLECT_THRESHOLD = int(5.2e9)


def maybe_collect():
    """Drop MLX's temporary buffer cache only when RSS is near the guard.
    Never called per-expert."""
    if _rss_bytes() > _RSS_COLLECT_THRESHOLD:
        mx.clear_cache()


def _load_expert_matrix(path, shape):
    return mx.array(np.fromfile(path, dtype=np.uint16).reshape(shape),
                    dtype=mx.uint16).view(mx.bfloat16)


def _route(xf, gate_w, gate_b, cfg):
    """Exact copy of model.py:moe_forward routing."""
    gw = _deq(gate_w, mx.float32)
    scores = (xf @ gw.T) / cfg.gate_temp
    if cfg.score_func == "softmax":
        scores = mx.softmax(scores, axis=-1)
    elif cfg.score_func == "sigmoid":
        scores = mx.sigmoid(scores)
    else:  # sqrtsoftplus
        scores = mx.sqrt(mx.logaddexp(scores, 0.0))
    bias = gate_b.astype(mx.float32)
    topk_scores, topk_idx = _topk(scores + bias, k=cfg.n_activated_experts, axis=-1)
    if cfg.norm_topk_prob and cfg.n_activated_experts > 1:
        topk_scores = topk_scores / (mx.sum(topk_scores, axis=-1, keepdims=True) + 1e-20)
    topk_scores = topk_scores * cfg.route_scale
    return topk_scores, topk_idx


def _shared_out(xb, sw1, sw3, sw2, cfg):
    sw1 = _deq(sw1, mx.bfloat16)
    sw3 = _deq(sw3, mx.bfloat16)
    sw2 = _deq(sw2, mx.bfloat16)
    sh1 = (xb @ sw1.T).astype(mx.float32)
    sh3 = (xb @ sw3.T).astype(mx.float32)
    if cfg.swiglu_limit > 0:
        sh3 = mx.clip(sh3, -cfg.swiglu_limit, cfg.swiglu_limit)
        sh1 = mx.clip(sh1, None, cfg.swiglu_limit)
    return (sh1 * mx.sigmoid(sh1) * sh3).astype(mx.bfloat16) @ sw2.T


# ---------------------------------------------------------------------------
# Batched active-expert forward (decode path)
# ---------------------------------------------------------------------------
def sparse_moe_forward_batched(x, gate_w, gate_b, shared_w1, shared_w3, shared_w2,
                               expert_dir, layer_id, cfg):
    """x: [b,s,d]. Returns [b,s,d]. T = b*s is small (decode / prefill replay)."""
    b, s, d = x.shape
    T = b * s
    xf = x.reshape(T, d).astype(mx.float32)
    xb = xf.astype(mx.bfloat16)

    topk_scores, topk_idx = _route(xf, gate_w, gate_b, cfg)  # [T,k]

    # One CPU sync to learn active expert ids (needed for per-expert file paths).
    idx_np = np.array(topk_idx).reshape(-1)
    active = sorted(int(e) for e in np.unique(idx_np))
    nE = len(active)

    moe_inter = cfg.moe_inter_dim
    dim = cfg.dim
    cache = get_expert_cache()

    W1, W3, W2 = [], [], []
    for e in active:
        w1 = cache.get((layer_id, e, "w1"))
        if w1 is None:
            w1 = _load_expert_matrix(os.path.join(expert_dir, f"L{layer_id}_exp{e}_w1.bin"),
                                     (moe_inter, dim))
            cache.put((layer_id, e, "w1"), w1)
        w3 = cache.get((layer_id, e, "w3"))
        if w3 is None:
            w3 = _load_expert_matrix(os.path.join(expert_dir, f"L{layer_id}_exp{e}_w3.bin"),
                                     (moe_inter, dim))
            cache.put((layer_id, e, "w3"), w3)
        w2 = cache.get((layer_id, e, "w2"))
        if w2 is None:
            w2 = _load_expert_matrix(os.path.join(expert_dir, f"L{layer_id}_exp{e}_w2.bin"),
                                     (dim, moe_inter))
            cache.put((layer_id, e, "w2"), w2)
        W1.append(w1); W3.append(w3); W2.append(w2)

    out = mx.zeros((T, d), dtype=mx.float32)
    if nE > 0:
        B1 = mx.stack(W1, axis=0)   # [nE, inter, dim]
        B3 = mx.stack(W3, axis=0)   # [nE, inter, dim]
        B2 = mx.stack(W2, axis=0)    # [nE, dim, inter]
        h1 = mx.einsum("td,eid->tei", xb, B1)   # [T,nE,inter]
        h3 = mx.einsum("td,eid->tei", xb, B3)
        if cfg.swiglu_limit > 0:
            h3 = mx.clip(h3, -cfg.swiglu_limit, cfg.swiglu_limit)
            h1 = mx.clip(h1, None, cfg.swiglu_limit)
        gate = h1 * mx.sigmoid(h1) * h3                            # [T,nE,inter]
        oe = mx.einsum("tei,edi->ted", gate.astype(mx.bfloat16), B2)  # [T,nE,dim]

        # per-active-expert gate weight: sum over k-slots selecting e
        topk_idx_np = np.array(topk_idx)
        topk_sc_np = np.array(topk_scores)
        e_gate = np.zeros((T, nE), dtype=np.float32)
        for j, e in enumerate(active):
            e_gate[:, j] = np.sum((topk_idx_np == e) * topk_sc_np, axis=-1)
        e_gate_mx = mx.array(e_gate, dtype=mx.float32)
        out = mx.sum(oe.astype(mx.float32) * e_gate_mx[..., None], axis=1)  # [T,d]

    out = out + _shared_out(xb, shared_w1, shared_w3, shared_w2, cfg).astype(mx.float32)
    maybe_collect()
    return out.reshape(b, s, d).astype(x.dtype)


# ---------------------------------------------------------------------------
# Legacy per-expert loop (accurate fallback parity)
# ---------------------------------------------------------------------------
def sparse_moe_forward(x, gate_w, gate_b, shared_w1, shared_w3, shared_w2,
                       expert_dir, layer_id, cfg):
    """Sparse MoE: only load top-8 active experts. Matches model.py:moe_forward."""
    b, s, d = x.shape
    T = b * s
    xf = x.reshape(T, d).astype(mx.float32)
    xb = xf.astype(mx.bfloat16)

    topk_scores, topk_idx = _route(xf, gate_w, gate_b, cfg)

    out = mx.zeros((T, d), dtype=mx.float32)
    idx_np = np.array(topk_idx).reshape(-1)
    topk_idx_np = np.array(topk_idx)
    topk_sc_np = np.array(topk_scores)
    cache = get_expert_cache()
    moe_inter = cfg.moe_inter_dim
    dim = cfg.dim

    for e in (int(x) for x in np.unique(idx_np)):
        w1 = cache.get((layer_id, e, "w1"))
        if w1 is None:
            w1 = _load_expert_matrix(os.path.join(expert_dir, f"L{layer_id}_exp{e}_w1.bin"),
                                     (moe_inter, dim)); cache.put((layer_id, e, "w1"), w1)
        w3 = cache.get((layer_id, e, "w3"))
        if w3 is None:
            w3 = _load_expert_matrix(os.path.join(expert_dir, f"L{layer_id}_exp{e}_w3.bin"),
                                     (moe_inter, dim)); cache.put((layer_id, e, "w3"), w3)
        w2 = cache.get((layer_id, e, "w2"))
        if w2 is None:
            w2 = _load_expert_matrix(os.path.join(expert_dir, f"L{layer_id}_exp{e}_w2.bin"),
                                     (dim, moe_inter)); cache.put((layer_id, e, "w2"), w2)

        e_gate = np.sum((topk_idx_np == e) * topk_sc_np, axis=-1)  # [T]
        if not np.any(e_gate > 0):
            continue
        h1 = xb @ w1.T
        h3 = xb @ w3.T
        if cfg.swiglu_limit > 0:
            h3 = mx.clip(h3, -cfg.swiglu_limit, cfg.swiglu_limit)
            h1 = mx.clip(h1, None, cfg.swiglu_limit)
        g = h1 * mx.sigmoid(h1) * h3
        expert_out = g.astype(mx.bfloat16) @ w2.T
        out = out + expert_out.astype(mx.float32) * mx.array(e_gate[:, None], dtype=mx.float32)

    out = out + _shared_out(xb, shared_w1, shared_w3, shared_w2, cfg).astype(mx.float32)
    return out.reshape(b, s, d).astype(mx.float32)
