"""SSD-backed Engram table with on-demand mmap row mapping + bounded LRU cache.

Reads fp8_e4m3fn uint8 rows from a raw .bin file and dequantizes on access.
No whole-table loading. Reports hit/miss/bytes/latency per batch.

The .bin layout: [num_embeddings, head_dim] fp8 uint8, row-major.
"""
from __future__ import annotations

import mmap
import os
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class EngramSSDStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    bytes_loaded: int = 0
    miss_latency_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.lookups)


def _fp8_u8_to_f32(u8: np.ndarray) -> np.ndarray:
    """Decode fp8_e4m3fn uint8 array to float32."""
    u32 = u8.astype(np.uint32) << 16
    return u32.view(np.float32).copy()


class EngramSSDLookup:
    """Memory-mapped fp8 engram table with bounded LRU row cache.

    Args:
        path: path to raw fp8 uint8 .bin [num_embeddings, head_dim].
        num_embeddings: table rows.
        head_dim: embedding dim.
        cache_rows: max rows in LRU cache.
    """

    def __init__(self, path: str, num_embeddings: int, head_dim: int,
                 cache_rows: int = 65536):
        self.num_embeddings = num_embeddings
        self.head_dim = head_dim
        self.row_bytes = head_dim  # fp8 = 1 byte/element
        self.path = path
        self.stats = EngramSSDStats()

        fd = os.open(path, os.O_RDONLY)
        self._mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        os.close(fd)

        self._cache: dict[int, mx.array] = {}
        self._cache_order: list[int] = []
        self.cache_rows = cache_rows
        self._table = np.ndarray(
            (num_embeddings, head_dim), dtype=np.uint8, buffer=self._mm)
        self._seen = np.zeros(num_embeddings, dtype=np.bool_)

    def rows_f32(self, indices: np.ndarray) -> np.ndarray:
        """Gather fp8 rows and decode them. indices: any shape, int."""
        idx = np.asarray(indices, dtype=np.int64)
        raw = self._table[idx]
        seen = self._seen[idx]
        self.stats.lookups += int(idx.size)
        self.stats.hits += int(np.count_nonzero(seen))
        self.stats.misses += int(idx.size - np.count_nonzero(seen))
        self._seen[idx] = True
        return _fp8_u8_to_f32(raw)

    def lookup(self, indices: mx.array) -> mx.array:
        """Gather embeddings for integer indices. Returns [..., head_dim] float32."""
        idx_np = np.array(indices).reshape(-1)
        out = np.zeros((len(idx_np), self.head_dim), dtype=np.float32)
        t0 = time.perf_counter()
        for i, ridx in enumerate(idx_np):
            ridx = int(ridx)
            self.stats.lookups += 1
            if ridx in self._cache:
                self.stats.hits += 1
                out[i] = np.array(self._cache[ridx], copy=False)
            else:
                self.stats.misses += 1
                offset = ridx * self.row_bytes
                raw = self._mm[offset:offset + self.row_bytes]
                u8 = np.frombuffer(raw, dtype=np.uint8)
                f32 = _fp8_u8_to_f32(u8)
                out[i] = f32
                self.stats.bytes_loaded += self.row_bytes
                self._cache[ridx] = mx.array(f32)
                self._cache_order.append(ridx)
                while len(self._cache_order) > self.cache_rows:
                    old = self._cache_order.pop(0)
                    self._cache.pop(old, None)
        self.stats.miss_latency_ms += (time.perf_counter() - t0) * 1000
        shape = list(np.array(indices).shape) + [self.head_dim]
        return mx.array(out).reshape(shape)

    def close(self):
        self._mm.close()
