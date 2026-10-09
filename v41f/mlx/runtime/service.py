"""Shared model service: build the runtime once, prewarm routed experts.

memguard discipline: we quantize routed experts Q8 lazily on first use, then
prewarm all layers so the first real request does not pay the one-time quant
cost in its TTFT.  Non-routed weights stay BF16 cached; Engram tables are
mmap'd SSD rows.  The whole resident footprint must stay well under 6 GB.
"""
from __future__ import annotations

import time

import mlx.core as mx
from tokenizers import Tokenizer

from ..config import MLXV42Config
from .weights import WeightStore
from .engram import EngramBank
from .model import V42RuntimeModel

WEIGHT_DIR = "ckpt_local/sft_mlx_bf16"
TOKENIZER = "ckpt_local/tok/tokenizer.json"


class RuntimeService:
    def __init__(self, bits: int = 8, prewarm: bool = True):
        t0 = time.time()
        self.ws = WeightStore(WEIGHT_DIR, bits=bits, verify=True)
        self.cfg = MLXV42Config.from_v42_cfg(self.ws.v42)
        self.tokenizer = Tokenizer.from_file(TOKENIZER)
        self.engram = EngramBank(self.tokenizer, self.cfg)
        self.model = V42RuntimeModel(self.cfg, self.ws, self.engram)
        self.build_s = time.time() - t0
        if prewarm:
            self.prewarm_s = self._prewarm()
        else:
            self.prewarm_s = 0.0

    def _prewarm(self) -> float:
        """Quantize routed experts and compile the Sinkhorn kernel before the first request."""
        t0 = time.time()
        for L in range(self.cfg.n_layers):
            self.ws.qstore.get_layer(L)
        self.model.prepare()
        mx.eval(list(self.ws.qstore._layers.keys())) if False else None
        mx.clear_cache()
        return time.time() - t0

    def rss_gb(self) -> float:
        import resource
        # macOS ru_maxrss is in bytes (Linux would be KB); divide accordingly.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 ** 3)
