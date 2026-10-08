"""KV-cached generation for MLX v42 backend.
Implements prefill + single-token decode with KV cache.
Uses bf16-active-sparse MoE with LRU expert cache.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
import mlx.core as mx

from .model import _rms_norm
from .sparse_moe import sparse_moe_forward


@dataclass
class KVGenStats:
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    prefill_tokens: int = 0
    decode_tokens: int = 0

    @property
    def decode_tps(self):
        return self.decode_tokens / (self.decode_ms / 1000) if self.decode_ms > 0 else 0


class KVGenerator:
    """KV-cached generation: prefill once, then decode single tokens.
    Uses streaming weights + sparse MoE.
    """
    def __init__(self, model, streaming_weights):
        self.model = model
        self.sw = streaming_weights
        self.cfg = model.cfg
        self._kv_cache = {}  # layer_id -> (k_cache, v_cache)
        self._engram_history = []  # full token history for n-gram hash
        self.stats = KVGenStats()

    def _init_cache(self, max_seq_len=4096):
        L = self.cfg.n_layers
        n_heads = self.cfg.n_heads
        head_dim = self.cfg.head_dim
        for l in range(L):
            self._kv_cache[l] = (
                mx.zeros((1, max_seq_len, n_heads, head_dim), dtype=mx.bfloat16),
                mx.zeros((1, max_seq_len, n_heads, head_dim), dtype=mx.bfloat16),
            )

    def prefill(self, tokens):
        """Run full forward on prompt tokens, populate KV cache."""
        if not self._kv_cache:
            self._init_cache()

        t0 = time.time()
        b, s = tokens.shape

        # Embed
        # Use streaming embed weight
        sw = self.sw
        embed_w = sw.get_embed()
        h = embed_w[tokens]  # [b,s,d]

        self._engram_history = list(tokens[0])

        for L in range(self.cfg.n_layers):
            WL = sw.get_layer(L)
            residual = h

            # Engram injection
            engram_embed = None
            if L in self.model.ssd_lookups:
                engram_embed = self._engram_inject(tokens, L)
                if engram_embed is not None:
                    h = h + engram_embed

            # Attention
            h_norm = _rms_norm(h, WL['attn_norm']['weight'], self.cfg.norm_eps)
            qkv = h_norm @ WL['attn']['wqkv']['weight'].T
            q, k, v = qkv.split([self.cfg.rope_head_dim * self.cfg.n_heads,
                                  self.cfg.rope_head_dim * self.cfg.n_heads,
                                  self.cfg.rope_head_dim * self.cfg.n_heads], axis=-1)
            q = q.reshape(b, s, self.cfg.n_heads, self.cfg.rope_head_dim)
            k = k.reshape(b, s, self.cfg.n_heads, self.cfg.rope_head_dim)
            v = v.reshape(b, s, self.cfg.n_heads, self.cfg.rope_head_dim)

            # RoPE
            from v41f.mlx.model import apply_rope_real
            positions = mx.arange(s)
            cos = self.model.attn_cos[positions]
            sin = self.model.attn_sin[positions]
            q = apply_rope_real(q, cos, sin)
            k = apply_rope_real(k, cos, sin)

            # Store in KV cache
            self._kv_cache[L] = (
                self._kv_cache[L][0].at[:, :s].add(k),
                self._kv_cache[L][1].at[:, :s].add(v),
            )

            # Attention (full causal for prefill)
            # Simplified: just use q @ k^T with causal mask
            scale = self.cfg.head_dim ** -0.5
            scores = mx.einsum('bnhd,bmhd->bnhm', q, k) * scale
            # Causal mask
            mask = mx.triu(mx.ones((s, s), dtype=mx.bool_), k=1)
            scores = mx.where(mask[None, None, :, :], -1e9, scores)
            probs = mx.softmax(scores, axis=-1)
            attn_out = mx.einsum('bnhm,bmhd->bnhd', probs, v)
            attn_out = attn_out.reshape(b, s, -1)

            o = attn_out @ WL['attn']['wo']['weight'].T

            # HyperConn
            hc_out = residual + o
            h = hc_out

            # MoE
            ffn_norm = _rms_norm(h, WL['ffn_norm']['weight'], self.cfg.norm_eps)
            moe_out = sparse_moe_forward(ffn_norm,
                                         WL['ffn']['gate']['weight'],
                                         WL['ffn']['gate']['bias'],
                                         WL['ffn']['shared']['w1']['weight'],
                                         WL['ffn']['shared']['w3']['weight'],
                                         WL['ffn']['shared']['w2']['weight'],
                                         'ckpt_local/sft_mlx_bf16/expert',
                                         L, self.cfg)
            h = h + moe_out

            sw.release_layer(L)

        # Head
        head_w = sw.get_head()
        logits = h @ head_w.T

        self.stats.prefill_ms = (time.time() - t0) * 1000
        self.stats.prefill_tokens = s
        return logits

    def decode_step(self, token_id, pos):
        """Single-token decode with KV cache."""
        t0 = time.time()

        self._engram_history.append(int(token_id))
        tokens = mx.array([[token_id]])

        sw = self.sw
        embed_w = sw.get_embed()
        h = embed_w[tokens]  # [1,1,d]

        for L in range(self.cfg.n_layers):
            WL = sw.get_layer(L)
            residual = h

            # Engram
            engram_embed = None
            if L in self.model.ssd_lookups:
                engram_embed = self._engram_inject(mx.array([self._engram_history]), L)
                if engram_embed is not None:
                    h = h + engram_embed[:, -1:, :]

            # Attention
            h_norm = _rms_norm(h, WL['attn_norm']['weight'], self.cfg.norm_eps)
            qkv = h_norm @ WL['attn']['wqkv']['weight'].T
            q, k, v = qkv.split([self.cfg.rope_head_dim * self.cfg.n_heads,
                                  self.cfg.rope_head_dim * self.cfg.n_heads,
                                  self.cfg.rope_head_dim * self.cfg.n_heads], axis=-1)
            q = q.reshape(1, 1, self.cfg.n_heads, self.cfg.rope_head_dim)
            k = k.reshape(1, 1, self.cfg.n_heads, self.cfg.rope_head_dim)
            v = v.reshape(1, 1, self.cfg.n_heads, self.cfg.rope_head_dim)

            # RoPE
            from v41f.mlx.model import apply_rope_real
            cos = self.model.attn_cos[pos:pos+1]
            sin = self.model.attn_sin[pos:pos+1]
            q = apply_rope_real(q, cos, sin)
            k = apply_rope_real(k, cos, sin)

            # Write to KV cache
            k_cache, v_cache = self._kv_cache[L]
            k_cache = k_cache.at[:, pos:pos+1].add(k - k_cache[:, pos:pos+1])
            v_cache = v_cache.at[:, pos:pos+1].add(v - v_cache[:, pos:pos+1])
            self._kv_cache[L] = (k_cache, v_cache)

            # Attention: q vs all cached KV
            k_all = k_cache[:, :pos+1, :, :]
            v_all = v_cache[:, :pos+1, :, :]
            scale = self.cfg.head_dim ** -0.5
            scores = mx.einsum('bnhd,bmhd->bnhm', q, k_all) * scale
            probs = mx.softmax(scores, axis=-1)
            attn_out = mx.einsum('bnhm,bmhd->bnhd', probs, v_all)
            attn_out = attn_out.reshape(1, 1, -1)

            o = attn_out @ WL['attn']['wo']['weight'].T
            h = residual + o

            # MoE
            ffn_norm = _rms_norm(h, WL['ffn_norm']['weight'], self.cfg.norm_eps)
            moe_out = sparse_moe_forward(ffn_norm,
                                         WL['ffn']['gate']['weight'],
                                         WL['ffn']['gate']['bias'],
                                         WL['ffn']['shared']['w1']['weight'],
                                         WL['ffn']['shared']['w3']['weight'],
                                         WL['ffn']['shared']['w2']['weight'],
                                         'ckpt_local/sft_mlx_bf16/expert',
                                         L, self.cfg)
            h = h + moe_out

            sw.release_layer(L)

        head_w = sw.get_head()
        logits = h @ head_w.T

        self.stats.decode_ms += (time.time() - t0) * 1000
        self.stats.decode_tokens += 1
        return logits

    def _engram_inject(self, tokens, L):
        """Simple Engram injection using SSD lookup."""
        try:
            lookup = self.model.ssd_lookups[L]
            hash_result = self.model.engram_hash(tokens, L)
            if hash_result is None:
                return None
            # hash_result: [b,s,n_heads] indices into SSD table
            b, s, nh = hash_result.shape
            emb = lookup.lookup(hash_result.reshape(-1))  # [b*s*nh, head_dim]
            return emb.reshape(b, s, -1)
        except Exception:
            return None
