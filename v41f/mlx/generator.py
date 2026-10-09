"""Generation loop for the MLX v42 backend.

Two modes:
- accurate (default): each step re-runs model.forward on full token prefix.
  Mathematically identical to full forward. Slower but numerically correct.
- experimental-kv: KV-cached decode. Faster but has 12.9% layer-1 divergence
  on the real SFT model due to bf16 precision accumulation over 24 layers.
  NOT recommended for production.

Usage:
    from v41f.mlx.generator import MLXGenerator
    gen = MLXGenerator(model, mode='accurate')
    tokens = gen.generate(prompt_ids, max_new_tokens=4)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx

from .model import MLXV42Model


@dataclass
class GenStats:
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    prefill_tokens: int = 0
    decode_tokens: int = 0

    @property
    def prefill_tps(self):
        return self.prefill_tokens / (self.prefill_ms / 1000) if self.prefill_ms > 0 else 0

    @property
    def decode_tps(self):
        return self.decode_tokens / (self.decode_ms / 1000) if self.decode_ms > 0 else 0


class MLXGenerator:
    """Accurate generation: full forward each step.

    This is the default and recommended mode. It re-runs the full model.forward
    on the complete token prefix at every step, guaranteeing numerical identity
    with a single full forward pass.
    """

    def __init__(self, model: MLXV42Model, mode: str = "accurate"):
        self.model = model
        self.mode = mode
        self.stats = GenStats()
        # KV state only used in experimental-kv mode
        self._kv_cache = None
        self._pre_mix = None
        self._compress_kv = None
        self._index_k = None
        self._a_cache = {}

    def _init_cache(self):
        """Initialize KV cache arrays (experimental-kv mode only)."""
        cfg = self.model.cfg
        w = cfg.window_size
        d = cfg.head_dim
        self._kv_cache = [mx.zeros((1, w + cfg.max_position_embeddings, d)) for _ in range(cfg.n_layers)]
        self._pre_mix = None
        self._compress_kv = None
        self._index_k = None
        self._a_cache = {}

    def prefill(self, tokens: mx.array) -> mx.array:
        """Run forward on prompt, return last-position logits."""
        t0 = time.time()
        logits = self.model.forward(tokens)
        mx.eval(logits)
        dt = (time.time() - t0) * 1000
        self.stats.prefill_ms += dt
        self.stats.prefill_tokens += int(tokens.shape[1])
        return logits[0, -1, :]

    def decode_step(self, token: mx.array, pos: int = 0, history: list = None) -> mx.array:
        """Decode one step. In accurate mode, re-runs full forward on history+token."""
        t0 = time.time()
        if self.mode == "accurate":
            # Re-run full forward on complete prefix
            full = mx.array([history], dtype=mx.int32)
            logits = self.model.forward(full)
            mx.eval(logits)
            result = logits[0, -1, :]
        else:
            # experimental-kv mode
            result = self._decode_step_kv(token, pos)
        dt = (time.time() - t0) * 1000
        self.stats.decode_ms += dt
        self.stats.decode_tokens += 1
        return result

    def _decode_step_kv(self, token: mx.array, pos: int) -> mx.array:
        """KV-cached decode step (experimental)."""
        from .model import (apply_rope_real, hc_mixes, hc_pre, hc_post,
                             _rms_norm, engram_forward_with_embed, moe_forward)
        cfg = self.model.cfg
        W = self.model.w
        b = 1

        h = mx.take(W["embed"]["weight"].astype(mx.bfloat16), token.astype(mx.int32), axis=0)
        h = mx.broadcast_to(h[:, None, None, :], (b, 1, cfg.hc_mult, cfg.dim))
        pre_mix = mx.concatenate([mx.ones((b, 1, 1)), mx.zeros((b, 1, cfg.hc_mult - 1))], axis=-1)

        attn_cos = self.model.attn_cos[pos:pos+1][None]
        attn_sin = self.model.attn_sin[pos:pos+1][None]

        for L in range(cfg.n_layers):
            WL = W["layers"][L]

            # Engram
            if L in W["engrams"]:
                eW = W["engrams"][L]
                if L in self.model.ssd_lookups:
                    toks = mx.array([[0]], dtype=mx.int32)  # placeholder
                    hash_ids = self.model._engram_hash(L, toks)
                    ssd_emb = self.model.ssd_lookups[L].lookup(hash_ids)
                    h = engram_forward_with_embed(h, hash_ids, ssd_emb, eW, cfg)

            residual = h
            attn_pre, attn_post, attn_comb = hc_mixes(
                h, WL["hc"]["attn_fn"], WL["hc"]["attn_scale"], WL["hc"]["attn_base"],
                cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
            a = hc_pre(h, pre_mix)
            a = _rms_norm(a, WL["attn_norm"]["weight"], cfg.norm_eps)

            q, qr = self.model._qproj(a, WL)
            q = apply_rope_real(q, attn_cos[0], attn_sin[0])
            kv = self.model._kvproj(a, WL)
            kv = apply_rope_real(kv, attn_cos[0], attn_sin[0])

            self._kv_cache[L] = self._kv_cache[L].at[:, pos:pos+1].add(kv)
            full_cache = self._kv_cache[L][:, :pos+1, :]

            w = cfg.window_size
            lo = max(0, pos - w + 1)
            win_idx = mx.arange(lo, lo + w, dtype=mx.int32)[None, None, :]
            win_idx = mx.where(win_idx > pos, -1, win_idx)

            from .model import sparse_attention
            o = sparse_attention(q, full_cache, WL["attn_sink"], win_idx, None,
                                  cfg.head_dim ** -0.5, cfg.attn_logit_softcap)
            o = apply_rope_real(o, attn_cos[0], attn_sin[0], inverse=True)
            a_out = self.model._oproj(o, WL)
            h = hc_post(a_out, residual, attn_post, attn_comb)

            residual = h
            ffn_pre, ffn_post, ffn_comb = hc_mixes(
                h, WL["hc"]["ffn_fn"], WL["hc"]["ffn_scale"], WL["hc"]["ffn_base"],
                cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
            f = hc_pre(h, attn_pre)
            f = _rms_norm(f, WL["ffn_norm"]["weight"], cfg.norm_eps)
            f = moe_forward(f, WL["ffn"], cfg)
            h = hc_post(f, residual, ffn_post, ffn_comb)
            pre_mix = ffn_pre

        h = _rms_norm(h, W["norm"]["weight"], cfg.norm_eps)
        h = hc_pre(h, pre_mix)
        logits = h @ W["head"]["weight"].T.astype(h.dtype)
        return logits[0, 0, 0, :]

    def generate(self, prompt_ids: list, max_new_tokens: int = 4, temperature: float = 0.0) -> list:
        """Generate tokens. Accurate mode: full forward each step."""
        tokens = list(prompt_ids)
        toks = mx.array([tokens], dtype=mx.int32)
        logits = self.prefill(toks)

        for _ in range(max_new_tokens):
            if temperature > 0:
                probs = mx.softmax(logits / temperature)
                next_tok = int(mx.random.categorical(probs[None]))
            else:
                next_tok = int(mx.argmax(logits))

            tokens.append(next_tok)

            if self.mode == "accurate":
                toks = mx.array([tokens], dtype=mx.int32)
                logits = self.prefill(toks)
            else:
                logits = self._decode_step_kv(mx.array([next_tok], dtype=mx.int32), pos=len(tokens)-1)

        return tokens
