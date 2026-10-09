"""Native MLX (Metal) forward for the v42 gate stack.

Ported line-by-line from v41f/ (attention.py, block.py, model.py, projections.py,
compressor.py, indexer.py, moe.py, hyperconn.py, engram.py, rope.py,
sparse_attn.py, window.py, norm_gate.py). No PyTorch, no MPS wrapper.

Prefill path first (whole prompt at once, one document per row). Decode with KV
cache is in generate.py; this file owns the per-token / per-chunk math.

Design notes:
  * fp8 weights are uint8 MLX arrays; dequantise with mx.from_fp8 at matmul.
  * Routed experts stay uint8; only the active top-k experts are dequantised.
  * The residual stream is hc_mult=4 parallel copies [b,s,hc,d].
  * Cross-layer attention state (compressed KV, index keys, topk) is threaded
    explicitly, matching SharedAttnState.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .config import MLXV42Config


# ---------------------------------------------------------------------------
# Small tensor helpers
# ---------------------------------------------------------------------------

def _rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """x: [..., d], weight: [d]. fp32 accumulate, match v41f.norm_gate.RMSNorm."""
    x_f = x.astype(mx.float32)
    var = mx.mean(x_f * x_f, axis=-1, keepdims=True)
    y = x_f * mx.rsqrt(var + eps)
    return (weight.astype(mx.float32) * y).astype(x.dtype)


def _deq(w: mx.array, dtype=mx.bfloat16) -> mx.array:
    """Dequantise fp8 uint8 weights to dtype; pass through if already float."""
    if w.dtype == mx.uint8:
        return mx.from_fp8(w, dtype)
    return w


def _topk(x: mx.array, k: int, axis: int = -1):
    """Returns (values, indices) for top-k along axis. MLX topk returns values only."""
    n = x.shape[axis]
    idx = mx.argpartition(x, kth=n - k, axis=axis)
    top_idx = idx[..., n - k:]
    top_val = mx.take_along_axis(x, top_idx, axis=axis)
    return top_val, top_idx

def _linear(x: mx.array, w: mx.array) -> mx.array:
    """x: [..., in], w: [out, in]. w may be uint8 fp8 or already float. Returns [..., out]."""
    return x @ _deq(w).T


# ---------------------------------------------------------------------------
# RoPE (real cos/sin path, matching rope_impl="real")
# ---------------------------------------------------------------------------

def precompute_freqs(head_dim_rope: int, max_seq: int, theta: float) -> mx.array:
    """Return cos/sin tables [max_seq, rope_dim/2]."""
    freqs = 1.0 / (theta ** (mx.arange(0, head_dim_rope, 2).astype(mx.float32) / head_dim_rope))
    t = mx.arange(max_seq, dtype=mx.float32)
    angles = mx.outer(t, freqs)  # [seq, dim/2]
    return mx.cos(angles), mx.sin(angles)


def apply_rope_real(x: mx.array, cos: mx.array, sin: mx.array, inverse: bool = False) -> mx.array:
    """Rotate the last rope_head_dim dims of x.

    x: [..., n_heads, head_dim] or [..., head_dim] (MQA). cos/sin: [seq, rope_dim/2].
    The rotation is on adjacent pairs of the tail, matching apply_rotary_real.
    """
    rd = cos.shape[-1] * 2  # rope_head_dim
    # reshape cos/sin to broadcast against x: x is [b, s, ..., d], seq at axis=1
    # x.ndim=4 (b,s,h,d): cos -> [1,s,1,rd/2]; x.ndim=3 (b,s,d): [1,s,rd/2]
    mid = [1] * (x.ndim - 3)  # extra head dims between seq and pairs
    cos_b = cos.reshape(1, x.shape[1], *mid, cos.shape[-1])
    sin_b = sin.reshape(1, x.shape[1], *mid, sin.shape[-1])
    nope = x[..., :-rd]
    rot = x[..., -rd:]
    rot = rot.reshape(*rot.shape[:-1], -1, 2)
    a, b = rot[..., 0], rot[..., 1]
    s = -sin_b if inverse else sin_b
    out_a = a * cos_b - b * s
    out_b = a * s + b * cos_b
    out = mx.stack([out_a, out_b], axis=-1).reshape(*x.shape[:-1], rd)
    return mx.concatenate([nope, out], axis=-1)


# ---------------------------------------------------------------------------
# HyperConnections (mHC)
# ---------------------------------------------------------------------------

def hc_mix_presinkhorn(x: mx.array, hc_fn: mx.array, hc_scale: mx.array, hc_base: mx.array,
                       hc_mult: int, eps: float, norm_eps: float):
    """Linear mix and softmax. Returns (pre, post, comb) before Sinkhorn."""
    b, s, hc, d = x.shape
    flat = x.reshape(b, s, hc * d).astype(mx.float32)
    rstd = mx.rsqrt(mx.mean(flat * flat, axis=-1, keepdims=True) + norm_eps)
    mixes = (flat @ hc_fn.astype(mx.float32).T) * rstd  # [b,s,mix_hc]

    pre = mx.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * mx.sigmoid(mixes[..., hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])
    comb = mixes[..., 2 * hc_mult:].reshape(b, s, hc_mult, hc_mult) * hc_scale[2] + \
           hc_base[2 * hc_mult:].reshape(1, 1, hc_mult, hc_mult)
    comb = mx.softmax(comb, axis=-1) + eps
    return pre, post, comb


def sinkhorn_loop(comb: mx.array, sinkhorn_iters: int, eps: float) -> mx.array:
    """One column norm, then iters-1 row/col rounds. comb: [..., n, n]."""
    comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    return comb


def hc_mixes(x: mx.array, hc_fn: mx.array, hc_scale: mx.array, hc_base: mx.array,
             hc_mult: int, sinkhorn_iters: int, eps: float, norm_eps: float):
    """x: [b,s,hc,d]. hc_fn: [mix_hc, hc*d]. -> (pre, post, comb).

    One RMS-normalised linear over the flattened hc*d stream, then Sinkhorn split.
    Matches hyperconn.hc_split_sinkhorn.
    """
    pre, post, comb = hc_mix_presinkhorn(x, hc_fn, hc_scale, hc_base, hc_mult, eps, norm_eps)
    return pre, post, sinkhorn_loop(comb, sinkhorn_iters, eps)


def hc_pre(x: mx.array, pre_mix: mx.array) -> mx.array:
    """x: [b,s,hc,d], pre_mix: [b,s,hc] -> [b,s,d]."""
    return mx.sum(pre_mix[..., None] * x.astype(mx.float32), axis=2).astype(x.dtype)


def hc_post(out: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> mx.array:
    """out: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc] -> [b,s,hc,d]."""
    # y[j] = post[j]*out + sum_i comb[i,j] * residual[i]
    res = residual.astype(mx.float32)
    combined = mx.matmul(comb.transpose(0, 1, 3, 2), res)  # [b,s,hc,d]
    return (post[..., None] * out.astype(mx.float32)[:, :, None, :] + combined).astype(out.dtype)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

def moe_forward(x: mx.array, W: dict, cfg: MLXV42Config) -> mx.array:
    """x: [b,s,d]. W has w1,w3,w2 (uint8 [E,inter,dim]), gate (weight fp8, bias bf16),
    shared (w1,w3,w2 bf16 linears). Returns [b,s,d].
    """
    b, s, d = x.shape
    xf = x.reshape(b * s, d).astype(mx.float32)

    # gate
    gw = _deq(W["gate"]["weight"], mx.float32)  # [E, d]
    scores = (xf @ gw.T) / cfg.gate_temp
    if cfg.score_func == "softmax":
        scores = mx.softmax(scores, axis=-1)
    elif cfg.score_func == "sigmoid":
        scores = mx.sigmoid(scores)
    else:  # sqrtsoftplus
        scores = mx.sqrt(mx.logaddexp(scores, 0.0))

    bias = W["gate"]["bias"].astype(mx.float32)
    topk_scores, topk_idx = _topk(scores + bias, k=cfg.n_activated_experts, axis=-1)  # [T, k]

    if cfg.norm_topk_prob and cfg.n_activated_experts > 1:
        topk_scores = topk_scores / (mx.sum(topk_scores, axis=-1, keepdims=True) + 1e-20)
    topk_scores = topk_scores * cfg.route_scale

    # routed experts: batched einsum over all E experts, weighted by routing matrix.
    # Pure Metal, no CPU sync. Non-selected experts get zero routing weight.
    T = xf.shape[0]
    E = cfg.n_routed_experts

    # routing weights [T, E]: scatter topk scores into (t, expert) positions
    routing = mx.zeros((T, E), dtype=mx.float32)
    rows = mx.arange(T)
    for slot in range(cfg.n_activated_experts):
        cols = topk_idx[:, slot]
        vals = topk_scores[:, slot]
        routing = routing.at[(rows, cols)].add(vals)

    w1 = _deq(W["w1"], mx.bfloat16)  # [E, inter, d]
    w3 = _deq(W["w3"], mx.bfloat16)
    w2 = _deq(W["w2"], mx.bfloat16)  # [E, d, inter]

    xb = xf.astype(mx.bfloat16)
    h1 = mx.einsum("td,eid->tei", xb, w1)  # [T, E, inter]
    h3 = mx.einsum("td,eid->tei", xb, w3)
    if cfg.swiglu_limit > 0:
        h3 = mx.clip(h3, -cfg.swiglu_limit, cfg.swiglu_limit)
        h1 = mx.clip(h1, None, cfg.swiglu_limit)
    gate = h1 * mx.sigmoid(h1) * h3
    oe = mx.einsum("tei,edi->ted", gate.astype(mx.bfloat16), w2)  # [T, E, d]
    out = mx.sum(oe.astype(mx.float32) * routing[:, :, None], axis=1)  # [T, d]

    # shared expert (unconditional)
    sw1 = _deq(W["shared"]["w1"]["weight"], mx.bfloat16)
    sw3 = _deq(W["shared"]["w3"]["weight"], mx.bfloat16)
    sw2 = _deq(W["shared"]["w2"]["weight"], mx.bfloat16)
    sh1 = (xf.astype(mx.bfloat16) @ sw1.T).astype(mx.float32)
    sh3 = (xf.astype(mx.bfloat16) @ sw3.T).astype(mx.float32)
    if cfg.swiglu_limit > 0:
        sh3 = mx.clip(sh3, -cfg.swiglu_limit, cfg.swiglu_limit)
        sh1 = mx.clip(sh1, None, cfg.swiglu_limit)
    shared_out = (sh1 * mx.sigmoid(sh1) * sh3).astype(mx.bfloat16) @ sw2.T
    out = out + shared_out.astype(mx.float32)

    return out.reshape(b, s, d).astype(x.dtype)


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------

def compressor_forward(x: mx.array, W: dict, ratio: int, cfg: MLXV42Config) -> mx.array:
    """x: [b,s,d]. Returns compressed latent [b, n_groups, head_dim] pre-RoPE.

    ratio==1: norm(wkv(x)) per token.
    ratio>1: softmax-gated pool over consecutive groups of `ratio` tokens (fp32).
    """
    wkv = _deq(W["wkv"]["weight"], mx.bfloat16)
    if ratio == 1:
        lat = x.astype(mx.bfloat16) @ wkv.T  # [b,s,head_dim]
        return _rms_norm(lat, W["norm"]["weight"], cfg.norm_eps)

    xf = x.astype(mx.float32)
    kv = xf @ wkv.astype(mx.float32).T  # [b,s,head_dim]
    wgate = _deq(W["wgate"]["weight"], mx.float32)
    score = xf @ wgate.T  # [b,s,head_dim]

    b, s, hd = kv.shape
    n_groups = s // ratio
    kv = kv[:, :n_groups * ratio].reshape(b, n_groups, ratio, hd)
    score = score[:, :n_groups * ratio].reshape(b, n_groups, ratio, hd)
    weight = mx.softmax(score, axis=2)
    pooled = mx.sum(kv * weight, axis=2)  # [b,n_groups,hd]
    return _rms_norm(pooled, W["norm"]["weight"], cfg.norm_eps).astype(x.dtype)


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

@dataclass
class IndexerWeights:
    wq_b: mx.array  # uint8 [n_heads*index_head_dim, q_lora]
    weights_proj: mx.array  # uint8 [n_heads, dim]


def indexer_forward(x, qr, index_k, cos, sin, W: IndexerWeights, cfg,
                    start_pos=0, key_ratio=None):
    """Returns topk entry indices [b,s,k] (int32), -1 for unreachable."""
    b, s, d = x.shape
    n = index_k.shape[1]
    wq_b = _deq(W.wq_b, mx.bfloat16)  # [n_heads*ihd, q_lora]
    q = (qr.astype(mx.bfloat16) @ wq_b.T).reshape(b, s, cfg.index_n_heads, cfg.index_head_dim)

    # rotate last rope_head_dim of index q
    qrot = apply_rope_real(q, cos, sin)

    weights = _deq(W.weights_proj, mx.float32)  # [n_heads, d]
    w_scale = (cfg.index_head_dim ** -0.5) * (cfg.index_n_heads ** -0.5)
    head_weights = (x.astype(mx.float32) @ weights.T) * w_scale  # [b,s,n_heads]

    # scores: [b,s,n_heads,n]
    scores = mx.einsum("bshd,bnd->bshn", qrot.astype(mx.float32), index_k.astype(mx.float32))
    scores = mx.maximum(scores, 0.0)  # relu
    scores = mx.sum(scores * head_weights[..., None], axis=2)  # [b,s,n]

    # causal visibility: query at global position t can see entries whose group
    # ends no later than t. The compressed key ratio is fixed by the source
    # layer configuration; use the global start_pos for incremental decoding.
    if key_ratio is None:
        key_ratio = max(1, (start_pos + s) // n) if n else 1
    key_ratio = max(1, int(key_ratio))
    positions = start_pos + mx.arange(s)
    compress_lens = (positions + 1) // key_ratio  # [s]
    col = mx.arange(n)
    mask = col[None, :] >= compress_lens[:, None]  # [s,n]
    scores = mx.where(mask[None], -float("inf"), scores)

    k = min(cfg.index_topk, n)
    _, idx = _topk(scores, k=k, axis=-1)
    idx = mx.sort(idx, axis=-1)
    return idx.astype(mx.int32)


# ---------------------------------------------------------------------------
# Sparse attention (single softmax: window KV + compressed KV + sink)
# ---------------------------------------------------------------------------

def _batch_valid(idx: mx.array, batch: int) -> mx.array:
    """idx [b,s,w] or [1,s,w]. Returns a [batch,s,w] mask."""
    valid = idx >= 0
    if valid.shape[0] == 1 and batch > 1:
        valid = mx.broadcast_to(valid, (batch, valid.shape[1], valid.shape[2]))
    return valid


def _take_slots(kv: mx.array, idx: mx.array) -> mx.array:
    """kv [b,n,d], idx [b,s,w] or [1,s,w] with -1 invalid. Returns [b,s,w,d]."""
    b, n, d = kv.shape
    s, w = idx.shape[1], idx.shape[2]
    if idx.shape[0] == 1 and b > 1:
        idx = mx.broadcast_to(idx, (b, s, w))
    flat = kv.reshape(b * n, d)
    base = (mx.arange(b) * n).astype(idx.dtype).reshape(b, 1, 1)
    safe = mx.where(idx >= 0, idx + base, mx.array(0, dtype=idx.dtype))
    return mx.take(flat, safe.reshape(-1), axis=0).reshape(b, s, w, d)


def sparse_attention(q: mx.array, kv: mx.array, attn_sink: mx.array,
                     window_idx: mx.array, comp_idx: mx.array,
                     scale: float, softcap: float = 0.0) -> mx.array:
    """q: [b,s,h,d], kv: [b,n,d] (MQA, shared), attn_sink: [h].
    window_idx: [b,s,w] positions into kv for window slots.
    comp_idx: [b,s,k] positions into kv for selected compressed slots (offset by window_len),
               or None.
    Returns o: [b,s,h,d].
    """
    b, s, h, d = q.shape
    # gather window KV per batch row. -1 slots are masked after the gather.
    win_kv = _take_slots(kv, window_idx)  # [b,s,w,d]
    win_scores = mx.einsum("bshd,bswd->bshw", q, win_kv.astype(q.dtype)) * scale
    win_valid = _batch_valid(window_idx, b)
    win_scores = mx.where(win_valid[:, :, None, :], win_scores, -float("inf"))

    parts = [win_scores]
    gathered_values = [win_kv]

    if comp_idx is not None and comp_idx.shape[-1] > 0:
        comp_kv = _take_slots(kv, comp_idx)  # [b,s,k,d]
        comp_scores = mx.einsum("bshd,bskd->bshk", q, comp_kv.astype(q.dtype)) * scale
        if softcap > 0:
            comp_scores = softcap * mx.tanh(comp_scores / softcap)
        comp_valid = _batch_valid(comp_idx, b)
        comp_scores = mx.where(comp_valid[:, :, None, :], comp_scores, -float("inf"))
        parts.append(comp_scores)
        gathered_values.append(comp_kv)

    all_scores = mx.concatenate(parts, axis=-1)  # [b,s,h,w+k]
    # sink: add exp(sink - rowmax) to denominator, no value
    row_max = mx.max(all_scores, axis=-1, keepdims=True)
    exp_scores = mx.exp(all_scores - row_max)
    sink_term = mx.exp(attn_sink.astype(mx.float32) - row_max[..., 0])  # [b,s,h]
    denom = mx.sum(exp_scores, axis=-1) + sink_term

    # weighted sum
    probs = exp_scores / denom[..., None]
    out = mx.zeros_like(q)
    offset = 0
    for i, gv in enumerate(gathered_values):
        w = probs[..., offset:offset + gv.shape[2]]  # [b,s,h,w_i]
        out = out + mx.einsum("bshw,bswd->bshd", w.astype(gv.dtype), gv)
        offset += gv.shape[2]
    return out


# ---------------------------------------------------------------------------
# Engram (n-gram injection)
# ---------------------------------------------------------------------------

def engram_forward_with_embed(h, hash_ids, embed_lookup, W, cfg):
    """h: [b,s,hc,d]. hash_ids: [b,s,n_hash_cols]. embed_lookup: [b,s,n_hash_cols,ed] bf16."""
    n_hash_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
    ed = cfg.engram_head_dim
    flat = embed_lookup.reshape(*embed_lookup.shape[:2], n_hash_cols * ed)
    wkv = _deq(W["wkv"]["weight"], mx.bfloat16)
    kv = flat @ wkv.T
    hc = cfg.hc_mult
    key = kv[..., :hc * cfg.dim].reshape(*kv.shape[:2], hc, cfg.dim)
    value = kv[..., hc * cfg.dim:]
    qw = W["q_weight"].astype(mx.float32)
    kw = W["k_weight"].astype(mx.float32)
    weight = qw * kw
    hf = h.astype(mx.float32)
    rstd = mx.rsqrt(mx.mean(hf * hf, axis=-1) + cfg.norm_eps) * \
           mx.rsqrt(mx.mean(key * key, axis=-1) + cfg.norm_eps)
    dot = mx.sum(hf * weight[None, None] * key, axis=-1) * rstd * (cfg.dim ** -0.5)
    dot_sign = mx.where(dot >= 0, 1.0, -1.0)
    gate = mx.sigmoid(mx.sqrt(mx.abs(dot) + 1e-6) * dot_sign)
    return (hf + gate[..., None] * value[:, :, None, :]).astype(h.dtype)


def engram_forward(h: mx.array, hash_ids: mx.array, W: dict, cfg: MLXV42Config) -> mx.array:
    """h: [b,s,hc,d]. hash_ids: [b,s,n_hash_cols] int.
    W: embed (uint8 [num_emb, head_dim]), wkv, q_weight, k_weight.
    """
    embed = _deq(W["embed"], mx.bfloat16)
    lookup = mx.take(embed, hash_ids.astype(mx.int32), axis=0)
    return engram_forward_with_embed(h, hash_ids, lookup, W, cfg)


# ---------------------------------------------------------------------------
# The assembled model
# ---------------------------------------------------------------------------

@dataclass
class LayerState:
    """Threaded cross-layer attention state (SharedAttnState equivalent)."""
    compress_kv: mx.array | None = None   # [b,n_comp,head_dim] RoPE'd
    index_k: mx.array | None = None      # [b,n_comp,index_head_dim] RoPE'd
    topk_idxs: mx.array | None = None    # [b,s,k]


class MLXV42Model:
    def __init__(self, cfg: MLXV42Config, weights: dict):
        self.cfg = cfg
        self.w = weights
        self.attn_cos, self.attn_sin = precompute_freqs(cfg.rope_head_dim, 8192, cfg.rope_theta)
        self.comp_cos, self.comp_sin = precompute_freqs(cfg.rope_head_dim, 8192, cfg.compress_rope_theta)
        self.engram_hash = None  # MLXNgramHash, set externally
        self.ssd_lookups = {}    # {layer_id: EngramSSDLookup}

    def _qproj(self, x: mx.array, W: dict):
        """x: [b,s,d]. -> q [b,s,h,d], qr [b,s,q_lora]."""
        wq_a = _deq(W["qproj"]["wq_a"]["weight"], mx.bfloat16)
        qr = _rms_norm(x @ wq_a.T, W["qproj"]["q_norm"]["weight"], self.cfg.norm_eps)
        wq_b = _deq(W["qproj"]["wq_b"]["weight"], mx.bfloat16)
        q = (qr @ wq_b.T).reshape(*x.shape[:2], self.cfg.n_heads, self.cfg.head_dim)
        return q, qr

    def _kvproj(self, x: mx.array, W: dict):
        wkv = _deq(W["kvproj"]["wkv"]["weight"], mx.bfloat16)
        return _rms_norm(x @ wkv.T, W["kvproj"]["kv_norm"]["weight"], self.cfg.norm_eps)

    def _oproj(self, o: mx.array, W: dict):
        """o: [b,s,h,d] -> [b,s,d]. Grouped block-diagonal wo_a + wo_b."""
        b, s, h, d = o.shape
        g = self.cfg.o_groups
        pg = h // g
        wo_a = _deq(W["oproj"]["wo_a"], mx.bfloat16)  # [g, o_lora, pg*d]
        og = o.reshape(b, s, g, pg * d)  # [b,s,g,pg*d]
        lat = mx.einsum("bsgd,grd->bsgr", og, wo_a)  # [b,s,g,o_lora]
        wo_b = _deq(W["oproj"]["wo_b"]["weight"], mx.bfloat16)
        return (lat.reshape(b, s, g * self.cfg.o_lora_rank) @ wo_b.T)

    def _window_idx(self, seqlen: int):
        """Causal window indices for each query position. [1,s,w]."""
        w = self.cfg.window_size
        t = mx.arange(seqlen)
        lo = mx.maximum(t - w + 1, 0)
        idx = lo[:, None] + mx.arange(w)[None, :]
        idx = mx.where(idx > t[:, None], -1, idx)
        return idx[None].astype(mx.int32)

    def forward(self, token_ids: mx.array) -> mx.array:
        """token_ids: [b,s] int. Returns logits [b,s,vocab]."""
        cfg = self.cfg
        b, s = token_ids.shape
        W = self.w

        # embed
        h = mx.take(W["embed"]["weight"].astype(mx.bfloat16), token_ids.astype(mx.int32), axis=0)
        # expand to hc_mult copies
        h = mx.broadcast_to(h[:, :, None, :], (b, s, cfg.hc_mult, cfg.dim))

        # initial pre_mix: one-hot at copy 0
        pre_mix = mx.concatenate([
            mx.ones((b, s, 1), dtype=mx.float32),
            mx.zeros((b, s, cfg.hc_mult - 1), dtype=mx.float32),
        ], axis=-1)

        state = LayerState()

        # RoPE slices
        attn_cos = self.attn_cos[:s][None]  # [1,s,rd/2]
        attn_sin = self.attn_sin[:s][None]

        for L in range(cfg.n_layers):
            WL = W["layers"][L]

            # Engram injection
            if L in W["engrams"]:
                hash_ids = self._engram_hash(L, token_ids)
                eW = W["engrams"][L]
                # Use SSD on-demand lookup if available (avoid whole-table mmap)
                if L in self.ssd_lookups:
                    # SSD lookup returns [b,s,n_hash_cols,ed] directly
                    ssd_emb = self.ssd_lookups[L].lookup(hash_ids)
                    h = engram_forward_with_embed(h, hash_ids, ssd_emb, eW, cfg)
                else:
                    h = engram_forward(h, hash_ids, eW, cfg)

            # ---- attention sublayer ----
            residual = h
            attn_pre, attn_post, attn_comb = hc_mixes(
                h, WL["hc"]["attn_fn"], WL["hc"]["attn_scale"], WL["hc"]["attn_base"],
                cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
            a = hc_pre(h, pre_mix)
            a = _rms_norm(a, WL["attn_norm"]["weight"], cfg.norm_eps)

            # attention body
            q, qr = self._qproj(a, WL)
            q = apply_rope_real(q, attn_cos[0], attn_sin[0])
            kv = self._kvproj(a, WL)  # [b,s,head_dim] MQA
            kv = apply_rope_real(kv, attn_cos[0], attn_sin[0])

            ratio = cfg.compress_ratios[L]
            comp_kv_out = None
            comp_idx_out = None

            if ratio > 0:
                # is this layer a kv source?
                if L in cfg.kv_source_layers:
                    latent = compressor_forward(a, WL["compressor"], ratio, cfg)
                    # rotate compressed KV at group positions
                    g_cos = self.comp_cos[::ratio][:latent.shape[1]][None]
                    g_sin = self.comp_sin[::ratio][:latent.shape[1]][None]
                    rot_lat = apply_rope_real(latent, g_cos[0], g_sin[0])
                    state.compress_kv = rot_lat
                    # index key
                    if L in cfg.index_source_layers:
                        ik_w = WL["index_key"]
                        ik_wk = _deq(ik_w["wk"]["weight"], mx.bfloat16)
                        ik = _rms_norm(latent @ ik_wk.T, ik_w["k_norm"]["weight"], cfg.norm_eps)
                        ik = apply_rope_real(ik, g_cos[0], g_sin[0])
                        state.index_k = ik

                # is this layer an index source?
                if L in cfg.index_source_layers and state.index_k is not None:
                    idx_w = IndexerWeights(
                        wq_b=WL["indexer"]["wq_b"]["weight"],
                        weights_proj=WL["indexer"]["weights_proj"]["weight"],
                    )
                    comp_idx_out = indexer_forward(a, qr, state.index_k, attn_cos[0], attn_sin[0],
                                                   idx_w, cfg)
                    state.topk_idxs = comp_idx_out

                # reuse published
                if state.compress_kv is not None:
                    comp_kv_out = state.compress_kv

            # window indices
            win_idx = self._window_idx(s)  # [1,s,w]

            # concatenate window KV + compressed KV for gathering
            if comp_kv_out is not None:
                full_kv = mx.concatenate([kv, comp_kv_out], axis=1)  # [b, s+n_comp, d]
                # comp_idx_out indexes into the compressed portion; offset by s
                if comp_idx_out is not None:
                    comp_gather = mx.where(comp_idx_out >= 0, comp_idx_out + s, -1)
                else:
                    comp_gather = None
            else:
                full_kv = kv
                comp_gather = None

            o = sparse_attention(q, full_kv, WL["attn_sink"], win_idx, comp_gather,
                                 cfg.head_dim ** -0.5, cfg.attn_logit_softcap)
            # inverse RoPE on output
            o = apply_rope_real(o, attn_cos[0], attn_sin[0], inverse=True)
            a_out = self._oproj(o, WL)

            h = hc_post(a_out, residual, attn_post, attn_comb)

            # ---- FFN sublayer ----
            residual = h
            ffn_pre, ffn_post, ffn_comb = hc_mixes(
                h, WL["hc"]["ffn_fn"], WL["hc"]["ffn_scale"], WL["hc"]["ffn_base"],
                cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
            f = hc_pre(h, attn_pre)
            f = _rms_norm(f, WL["ffn_norm"]["weight"], cfg.norm_eps)
            f = moe_forward(f, WL["ffn"], cfg)
            h = hc_post(f, residual, ffn_post, ffn_comb)

            pre_mix = ffn_pre

        # final collapse + norm + head
        h = hc_pre(h, pre_mix)
        h = _rms_norm(h, W["norm"]["weight"], cfg.norm_eps)
        logits = h.astype(mx.bfloat16) @ W["head"]["weight"].astype(mx.bfloat16).T
        return logits

    # -- engram hash (prefill) --
    def _engram_hash(self, layer_id: int, token_ids: mx.array) -> mx.array:
        """Compute n-gram hash ids for the engram at this layer."""
        if self.engram_hash is not None:
            # Real n-gram hash: [b, s, n_layers, n_hash_cols]
            all_ids = self.engram_hash(token_ids, start_pos=0)
            lidx = self.cfg.engram_layer_ids.index(layer_id)
            return all_ids[:, :, lidx, :]  # [b, s, n_hash_cols]
        # Fallback placeholder (small model without tokenizer)
        cfg = self.cfg
        n_hash_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
        b, s = token_ids.shape
        num_emb = cfg.engram_num_embeddings[cfg.engram_layer_ids.index(layer_id)]
        base = token_ids.astype(mx.int32) % num_emb
        return mx.broadcast_to(base[:, :, None], (b, s, n_hash_cols))
