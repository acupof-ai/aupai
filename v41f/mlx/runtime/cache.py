"""Cache objects for the v42 runtime, conforming to the mlx-lm cache protocol.

DESIGN_RUNTIME_V1.md §6:
  * Token state      -- every request keeps the FULL token sequence; Engram only
                        reads the last few tokens of context.
  * Window KV        -- one MQA KV cache per layer (256-dim BF16), mlx-lm KVCache.
  * Compressed state -- 3 stages, source layers 2/8/12. Each stage holds
                        compress_kv, index_k, the unfinished ratio-group pending_x
                        and the current-query top-k. Non-index layers never reuse
                        an earlier query's top-k.
  * HyperConnection  -- pre_mix carried as scalar state across decode steps.

Cache objects must expose ``state`` / ``merge`` / ``extract`` / ``filter`` /
``prepare`` / ``finalize`` / ``nbytes`` so they can feed mlx-lm's BatchGenerator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache, TokenBuffer

SOURCE_STAGES = (2, 8, 12)


@dataclass
class CompressedStage:
    """One compressed-attention stage for a source layer."""

    compress_kv: mx.array | None = None
    index_k: mx.array | None = None
    topk: mx.array | None = None
    pending_x: mx.array | None = None  # unfinished ratio-group hidden states

    def tuples(self):
        return (self.compress_kv, self.index_k, self.topk, self.pending_x)

    @classmethod
    def from_tuples(cls, t):
        return cls(*t)


class GlobalState:
    """Request-global token + compressed + HyperConnection state (cache slot 0).

    Slot 0 must NOT be mistaken for a normal KVCache by mlx-lm trimming: it is
    non-trimmable and carries the full token history.
    """

    def __init__(self):
        self.tokens: TokenBuffer = TokenBuffer()
        self.offset: int = 0
        self.cpu_hist: np.ndarray | None = None
        self.stages: dict[int, CompressedStage] = {s: CompressedStage() for s in SOURCE_STAGES}
        # HyperConnection pre_mix carried across decode steps (design §4): the
        # first block uses one-hot identity; subsequent decode steps reuse the
        # previous step's final ffn_pre.
        self.pre_mix: mx.array | None = None

    def append_tokens(self, tokens: mx.array) -> np.ndarray:
        """Append a chunk. Offset grows by the sequence length, not batch*length.

        A batch must keep one shared length. The CPU copy feeds the Engram hash
        without a device sync inside the layer loop.
        """
        if tokens.ndim == 1:
            tokens = tokens.reshape(1, -1)
        b, s = int(tokens.shape[0]), int(tokens.shape[1])
        ids = np.array(tokens, copy=True).astype(np.int64).reshape(b, s)
        if self.cpu_hist is None:
            self.cpu_hist = ids
        else:
            if self.cpu_hist.shape[0] != b:
                raise ValueError(
                    f"batch changed {self.cpu_hist.shape[0]} -> {b}; equal-length batch only")
            self.cpu_hist = np.concatenate([self.cpu_hist, ids], axis=1)
        if b == 1:
            self.tokens.update_and_fetch(tokens.reshape(-1))
        self.offset += s
        return self.cpu_hist

    @property
    def all_tokens(self) -> mx.array:
        return self.tokens.tokens

    # -- mlx-lm protocol ----------------------------------------------------
    @property
    def state(self):
        values: list[Any] = [self.all_tokens, self.offset, self.pre_mix]
        for s in SOURCE_STAGES:
            values.extend(self.stages[s].tuples())
        return tuple(values)

    @state.setter
    def state(self, values):
        tokens, self.offset, self.pre_mix, *tail = values
        if tokens is not None:
            self.tokens = TokenBuffer(tokens.tolist() if tokens is not None else [])
        self.stages = {}
        for i, s in enumerate(SOURCE_STAGES):
            j = i * 4
            self.stages[s] = CompressedStage.from_tuples(tail[j:j + 4])

    @property
    def nbytes(self) -> int:
        total = 0
        if isinstance(self.all_tokens, mx.array):
            total += self.all_tokens.nbytes
        for s in self.stages.values():
            for x in s.tuples():
                if isinstance(x, mx.array):
                    total += x.nbytes
        if isinstance(self.pre_mix, mx.array):
            total += self.pre_mix.nbytes
        return total

    def empty(self) -> bool:
        return self.offset == 0

    def is_trimmable(self) -> bool:
        return False

    # -- batch protocol (used by BatchGenerator) ----------------------------
    @classmethod
    def merge(cls, caches):
        # v42 keeps one GlobalState per sequence; batched forward runs each
        # sequence through its own cache list. Merge is identity on the first
        # (single-stream) path; continuous batching sequences requests.
        g = cls()
        return g

    def extract(self, idx: int):
        return self

    def filter(self, batch_indices):
        return

    def prepare(self, **kwargs):
        return

    def finalize(self):
        return


class LayerKVCache:
    """Per-layer MQA window cache (mlx-lm KVCache, one KV head)."""

    def __init__(self):
        self.window = KVCache()

    @property
    def offset(self) -> int:
        return self.window.offset

    @property
    def state(self):
        return self.window.state

    @state.setter
    def state(self, v):
        self.window.state = v

    @property
    def nbytes(self) -> int:
        return self.window.nbytes

    def empty(self) -> bool:
        return self.window.empty()

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n):
        return self.window.trim(n)

    @classmethod
    def merge(cls, caches):
        return BatchLayerKVCache.merge(caches)

    def extract(self, idx: int):
        return self

    def filter(self, batch_indices):
        return

    def prepare(self, **kwargs):
        return

    def finalize(self):
        return


class BatchLayerKVCache:
    """Batched view over per-layer KV caches (placeholder for continuous batch)."""

    def __init__(self):
        self.window = None

    @classmethod
    def merge(cls, caches):
        return cls()

    @property
    def nbytes(self) -> int:
        return 0

    def empty(self) -> bool:
        return True

    def is_trimmable(self) -> bool:
        return False


def make_cache(n_layers: int):
    """[GlobalState] + [LayerKVCache]*n_layers  (cache length = 1 + n_layers)."""
    return [GlobalState()] + [LayerKVCache() for _ in range(n_layers)]
