"""mlx-lm compatibility layer for v42 incremental generation.

Reuses mlx-lm's generation loop, thread-local Metal stream and KV cache style.
The v42-specific forward handles HyperConnection, compressed sparse attention,
MoE routing and SSD-backed Engram.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache


@dataclass
class CompressedStage:
    compress_kv: mx.array | None = None
    index_k: mx.array | None = None
    topk: mx.array | None = None
    pending_x: mx.array | None = None


class V42GlobalCache:
    """Request-global token and compressed-attention state."""

    def __init__(self):
        self.tokens = None
        self.offset = 0
        # Source layers 2, 8 and 12 publish independent compressed stores.
        self.stages = {2: CompressedStage(), 8: CompressedStage(), 12: CompressedStage()}

    def append_tokens(self, tokens):
        self.tokens = tokens if self.tokens is None else mx.concatenate([self.tokens, tokens], axis=-1)
        self.offset += tokens.shape[-1]
        return self.tokens

    @property
    def state(self):
        values = [self.tokens, self.offset]
        for source in (2, 8, 12):
            s = self.stages[source]
            values.extend([s.compress_kv, s.index_k, s.topk, s.pending_x])
        return tuple(values)

    @state.setter
    def state(self, values):
        self.tokens, self.offset, *tail = values
        self.stages = {}
        for i, source in enumerate((2, 8, 12)):
            j = i * 4
            self.stages[source] = CompressedStage(*tail[j:j + 4])

    @property
    def nbytes(self):
        total = self.tokens.nbytes if isinstance(self.tokens, mx.array) else 0
        for s in self.stages.values():
            for x in (s.compress_kv, s.index_k, s.topk, s.pending_x):
                if isinstance(x, mx.array):
                    total += x.nbytes
        return total

    def empty(self):
        return self.offset == 0

    def is_trimmable(self):
        return False


class V42LayerCache:
    """Per-layer MQA window cache using mlx-lm's mature KVCache."""

    def __init__(self):
        self.window = KVCache()

    @property
    def offset(self):
        return self.window.offset

    @property
    def state(self):
        return self.window.state

    @state.setter
    def state(self, value):
        self.window.state = value

    @property
    def nbytes(self):
        return self.window.nbytes

    def empty(self):
        return self.window.empty()

    def is_trimmable(self):
        return self.window.is_trimmable()

    def trim(self, n):
        return self.window.trim(n)


class V42MLXLMModel(nn.Module):
    """Adapter for mlx-lm's ``model(inputs, cache=...)`` contract."""

    def __init__(self, core_model, streaming_weights, forward_fn):
        super().__init__()
        self.core_model = core_model
        self.streaming_weights = streaming_weights
        self.forward_fn = forward_fn
        self.args = type(
            "V42Args",
            (),
            {"max_position_embeddings": 4096, "vocab_size": 32768},
        )()
        self._layers = [None] * core_model.cfg.n_layers

    @property
    def layers(self):
        return self._layers

    def make_cache(self):
        return [V42GlobalCache()] + [V42LayerCache() for _ in self.layers]

    def __call__(self, inputs: mx.array, cache: list[Any] | None = None):
        if cache is None:
            cache = self.make_cache()
        return self.forward_fn(
            self.core_model,
            self.streaming_weights,
            inputs,
            cache,
        )
