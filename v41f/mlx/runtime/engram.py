"""Engram: n-gram injection with SSD / memory hierarchy.

Design §2 / §6.1:
  * TokenBuffer holds the FULL token history; Engram only reads the last few
    tokens of context (n-gram hash is computed on the full history).
  * SSD layer  -- mmap'd fp8 rows, bounded LRU (cold lookups).
  * Memory layer -- recently hit n-gram embeddings resident (hot lookups).
  * SSD miss falls back to memory / zero vector and is counted.
  * Reports hit / miss / bytes / latency for the /metrics endpoint.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from ..engram_ssd import EngramSSDLookup, EngramSSDStats
from ..engram_hash import MLXNgramHash


@dataclass
class EngramBankStats:
    ssd: EngramSSDStats = field(default_factory=EngramSSDStats)
    mem_hits: int = 0
    zero_fallbacks: int = 0

    @property
    def hit_rate(self) -> float:
        tot = self.ssd.lookups
        return self.ssd.hits / max(1, tot)


class EngramBank:
    """Per-layer Engram tables: SSD cold layer + resident memory hot layer."""

    def __init__(self, tokenizer, cfg, ssd_dir="ckpt_local/engram_ssd",
                 cache_rows=4096, mem_capacity=1 << 16):
        self.cfg = cfg
        self.hasher = MLXNgramHash(
            tokenizer, layer_ids=tuple(cfg.engram_layer_ids),
            max_ngram_size=cfg.engram_max_ngram_size,
            n_heads=cfg.engram_n_heads,
            engram_vocab_size=cfg.engram_compressed_vocab_size,
            pad_id=cfg.engram_pad_id)
        from .weights import ENGRAM_ROWS
        self.ssd = {
            L: EngramSSDLookup(f"{ssd_dir}/embed_L{L}.bin", ENGRAM_ROWS[L],
                               head_dim=cfg.engram_head_dim, cache_rows=cache_rows)
            for L in cfg.engram_layer_ids}
        self.mem: dict[int, dict[int, mx.array]] = {L: {} for L in cfg.engram_layer_ids}
        self.mem_capacity = mem_capacity
        self.stats = EngramBankStats()

    def hash_ids(self, layer_id: int, all_tokens: mx.array) -> mx.array:
        """n-gram hashes over the full token history. all_tokens: [1,T] -> [1,T,cols]."""
        ids = self.hasher(all_tokens, start_pos=0)  # [1, T, n_layers, cols]
        lidx = self.cfg.engram_layer_ids.index(layer_id)
        return ids[:, :, lidx, :]  # [1, T, cols]

    def hash_chunk(self, hist: np.ndarray, start: int, slen: int) -> np.ndarray:
        """CPU hash of hist[:, start:start+slen]. Returns [B, slen, n_layers, cols] int64.

        Same integers as ``MLXNgramHash`` on the full history, sliced to this chunk.
        The decode step uses this so the forward graph does not sync at each Engram layer.
        """
        hs = self.hasher
        hist = np.asarray(hist, dtype=np.int64)
        if start + slen != hist.shape[1]:
            raise ValueError(f"engram hist length {hist.shape[1]} != {start}+{slen}")
        comp = hs.token_map[hist]
        pos = np.arange(start, start + slen)
        blocked = np.zeros((hist.shape[0], slen), dtype=np.bool_)
        toks = []
        for shift in range(hs.max_ngram):
            src = comp[:, np.clip(pos - shift, 0, None)]
            blocked = blocked | ((pos < shift)[None, :] | (src == hs.DEAD))
            toks.append(np.where(blocked, hs.pad_id, src))
        tokens = np.stack(toks, axis=-1).astype(np.int64)
        products = tokens[:, :, None, :] * hs.multipliers[None, None, :, :]
        rolling = products[..., 0]
        parts = []
        for i in range(1, hs.max_ngram):
            rolling = np.bitwise_xor(rolling, products[..., i])
            parts.append(np.mod(rolling[:, :, :, None], hs.primes_np[:, i - 1][None, None, :, :]))
        result = np.concatenate(parts, axis=-1)
        offsets = np.stack([np.asarray(o, dtype=np.int64) for o in hs.offsets])
        return result + offsets[None, None, :, :]

    def embed_chunk(self, hist: np.ndarray, start: int, slen: int):
        """Hashes and SSD rows for one forward chunk.

        Returns (emb, hash_ids). emb[layer] is [B, slen, cols, head] float32.
        hash_ids[layer] is [B, slen, cols] int32. Both are MLX arrays with no
        dependence on the token graph.
        """
        hashes = self.hash_chunk(hist, start, slen)
        emb, hash_mx = {}, {}
        for li, layer_id in enumerate(self.cfg.engram_layer_ids):
            h = hashes[:, :, li, :]
            rows = self.ssd[layer_id].rows_f32(h)
            emb[layer_id] = mx.array(np.ascontiguousarray(rows))
            hash_mx[layer_id] = mx.array(np.ascontiguousarray(h.astype(np.int32)))
        return emb, hash_mx

    def lookup(self, layer_id: int, hashes: mx.array) -> mx.array:
        """Gather embeddings for hashes [..., cols] -> [..., cols, ed]."""
        flat = np.array(hashes).reshape(-1)
        mem = self.mem[layer_id]
        out = np.zeros((len(flat), self.cfg.engram_head_dim), dtype=np.float32)
        t0 = time.perf_counter()
        for i, h in enumerate(flat):
            h = int(h)
            if h in mem:
                self.stats.mem_hits += 1
                out[i] = np.array(mem[h], copy=False)
                continue
            emb = self.ssd[layer_id].lookup(mx.array([h]))
            v = np.array(emb)[0]
            out[i] = v
            if len(mem) < self.mem_capacity:
                mem[h] = mx.array(v)
        self.stats.ssd.miss_latency_ms += (time.perf_counter() - t0) * 1000
        shape = list(np.array(hashes).shape) + [self.cfg.engram_head_dim]
        return mx.array(out).reshape(shape)
