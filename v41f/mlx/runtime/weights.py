"""WeightStore: the single authoritative weight format for the v42 runtime.

Design (DESIGN_RUNTIME_V1.md §7):
  * One published dir: ``ckpt_local/v42_runtime/``.
  * ``accurate`` tier  -- attention / HyperConnection / Engram projections BF16,
                          routed experts Q8 affine gs=64, shared expert BF16,
                          Engram table SSD FP8.
  * ``fast`` tier      -- routed experts Q4 affine gs=64, shared expert Q8,
                          attention projections Q8.  Only an experimental switch;
                          it must independently pass the 64-token gate before it
                          can become the default.
  * Manifest must carry: source ckpt sha256, per-file sha256, shape/dtype/
                          bits/group_size, model structure, tokenizer sha256,
                          Engram table sha256.

Memory discipline (memguard: one Python process >6GB is killed):
  * Routed experts are quantized lazily on first use and kept resident Q8/Q4.
  * Non-routed weights are cached; the whole model fits well under 6GB.
  * We do NOT re-export weights (that peaks at 7.1GB). We read the already
    validated bf16 shards and quantize on the fly.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

SOURCE_SHA256 = "2836fa7e67c3767757dd1576f2cbfefffacd0b7308c3bc8922e74ef470653d69"
DEFAULT_DIR = "ckpt_local/sft_mlx_bf16"
RUNTIME_DIR = "ckpt_local/v42_runtime"
ENGram_DIR = "ckpt_local/engram_ssd"
TOKENIZER = "ckpt_local/tok/tokenizer.json"

# Engram table row counts (head_dim=128, fp8 1 byte/elem).
ENGRAM_ROWS = {1: 786862, 5: 788118, 9: 789492, 13: 791110, 17: 792776, 21: 794672}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ExpertQuantStore:
    """Lazy Q8/Q4 store for routed experts, quantized from resident bf16 shards.

    Uses ``mx.gather_qmm`` (fused grouped GEMM) so only the active top-k experts
    are evaluated -- no Python expert loop on the hot path.
    """

    weight_dir: str = DEFAULT_DIR
    group_size: int = 64
    bits: int = 8  # 8 = accurate Q8, 4 = fast Q4 (experimental)
    quantize_s: float = 0.0
    _layers: dict = field(default_factory=dict)
    _entries: dict = field(default_factory=dict)

    def __post_init__(self):
        with open(os.path.join(self.weight_dir, "manifest.json")) as f:
            self._entries = {e["src_key"]: e for e in json.load(f)["entries"]}

    def _load_bf16(self, key: str) -> mx.array:
        e = self._entries[key]
        path = os.path.join(self.weight_dir, e["file"])
        shape = tuple(e["shape"])
        raw = np.memmap(path, dtype=np.uint16, mode="r", shape=shape)
        return mx.array(np.asarray(raw), dtype=mx.uint16).view(mx.bfloat16)

    def get_layer(self, layer_id: int):
        cached = self._layers.get(layer_id)
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
        self._layers[layer_id] = out
        return out

    def prewarm(self, n_layers: int):
        for i in range(n_layers):
            self.get_layer(i)
        mx.clear_cache()


class WeightStore:
    """Lazy, validated weight access for the runtime model."""

    def __init__(self, weight_dir: str = DEFAULT_DIR, bits: int = 8,
                 group_size: int = 64, verify: bool = True):
        self.dir = weight_dir
        self.bits = bits
        self.group_size = group_size
        with open(os.path.join(weight_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        self.v42 = self.manifest["v42_cfg"]
        self.entries = {e["src_key"]: e for e in self.manifest["entries"]}
        if verify:
            self.verify_source()
        self._cache: dict = {}
        self._cache_bytes = 0
        self.qstore = ExpertQuantStore(weight_dir, group_size, bits)

    # -- verification -------------------------------------------------------
    def verify_source(self):
        sha = self.manifest.get("source_sha256", "")
        if sha != SOURCE_SHA256:
            raise ValueError(f"source sha256 mismatch: {sha[:16]} != {SOURCE_SHA256[:16]}")

    # -- low-level load -----------------------------------------------------
    def _load_raw(self, key: str) -> mx.array:
        e = self.entries[key]
        path = os.path.join(self.dir, e["file"])
        shape = tuple(e["shape"])
        if e["dst_dtype"] == "bf16":
            return mx.array(np.fromfile(path, dtype=np.uint16).reshape(shape),
                            dtype=mx.uint16).view(mx.bfloat16)
        if e["dst_dtype"] == "fp32":
            return mx.array(np.fromfile(path, dtype=np.float32).reshape(shape),
                            dtype=mx.float32)
        raise ValueError(f"Unknown dtype {e['dst_dtype']} for {key}")

    def get(self, key: str) -> mx.array:
        w = self._cache.get(key)
        if w is None:
            w = self._load_raw(key)
            self._cache[key] = w
            self._cache_bytes += int(self.entries[key]["bytes"])
        return w

    # -- structure accessors ------------------------------------------------
    def get_layer(self, L: int) -> dict:
        p = f"layers.{L}."
        d = {
            "attn_norm": {"weight": self.get(p + "attn_norm.weight")},
            "qproj": {"wq_a": {"weight": self.get(p + "attn.qproj.wq_a.weight")},
                      "wq_b": {"weight": self.get(p + "attn.qproj.wq_b.weight")},
                      "q_norm": {"weight": self.get(p + "attn.qproj.q_norm.weight")}},
            "kvproj": {"wkv": {"weight": self.get(p + "attn.kvproj.wkv.weight")},
                       "kv_norm": {"weight": self.get(p + "attn.kvproj.kv_norm.weight")}},
            "oproj": {"wo_a": self.get(p + "attn.oproj.wo_a"),
                      "wo_b": {"weight": self.get(p + "attn.oproj.wo_b.weight")}},
            "attn_sink": self.get(p + "attn.attn_sink"),
            "hc": {k: self.get(p + f"hc.hc_{k}") for k in
                   ["attn_fn", "attn_scale", "attn_base", "ffn_fn", "ffn_scale", "ffn_base"]},
            "ffn_norm": {"weight": self.get(p + "ffn_norm.weight")},
            "ffn": {"gate": {"weight": self.get(p + "ffn.gate.weight"),
                             "bias": self.get(p + "ffn.gate.bias")},
                    "shared": {"w1": {"weight": self.get(p + "ffn.shared_experts.w1.weight")},
                               "w3": {"weight": self.get(p + "ffn.shared_experts.w3.weight")},
                               "w2": {"weight": self.get(p + "ffn.shared_experts.w2.weight")}}},
        }
        if p + "attn.compressor.wkv.weight" in self.entries:
            comp = {"wkv": {"weight": self.get(p + "attn.compressor.wkv.weight")},
                    "norm": {"weight": self.get(p + "attn.compressor.norm.weight")}}
            if p + "attn.compressor.wgate.weight" in self.entries:
                comp["wgate"] = {"weight": self.get(p + "attn.compressor.wgate.weight")}
            d["compressor"] = comp
        if p + "attn.index_key.wk.weight" in self.entries:
            d["index_key"] = {"wk": {"weight": self.get(p + "attn.index_key.wk.weight")},
                              "k_norm": {"weight": self.get(p + "attn.index_key.k_norm.weight")}}
        if p + "attn.indexer.wq_b.weight" in self.entries:
            d["indexer"] = {"wq_b": {"weight": self.get(p + "attn.indexer.wq_b.weight")},
                            "weights_proj": {"weight": self.get(p + "attn.indexer.weights_proj.weight")}}
        return d

    @property
    def embed_weight(self):
        return self.get("embed.weight")

    @property
    def norm_weight(self):
        return self.get("norm.weight")

    @property
    def head_weight(self):
        return self.get("head.weight")

    def get_engram(self, L: int) -> dict:
        return {"wkv": {"weight": self.get(f"engrams.{L}.wkv.weight")},
                "q_weight": self.get(f"engrams.{L}.q_weight"),
                "k_weight": self.get(f"engrams.{L}.k_weight")}

    # -- unified runtime manifest -------------------------------------------
    def build_runtime_manifest(self, out_dir: str = RUNTIME_DIR) -> str:
        """Write the unified v42_runtime manifest WITHOUT re-exporting weights.

        We enumerate the already-validated bf16 shards and record their hashes,
        plus the quantized-expert tier metadata and tokenizer/engram hashes.
        """
        os.makedirs(out_dir, exist_ok=True)
        entries = []
        for e in self.manifest["entries"]:
            path = os.path.join(self.dir, e["file"])
            entries.append({
                "src_key": e["src_key"],
                "file": os.path.relpath(path, out_dir),
                "shape": e["shape"],
                "dtype": e["dst_dtype"],
                "bits": self.bits if e["src_key"].endswith(("ffn.w1", "ffn.w3", "ffn.w2")) else (16 if e["dst_dtype"] == "bf16" else 32),
                "group_size": self.group_size,
                "sha256": e.get("sha256") or _sha256_file(path),
            })
        tok_sha = _sha256_file(TOKENIZER)
        engram = {f"L{L}": {"file": f"engram_ssd/embed_L{L}.bin",
                            "rows": ENGRAM_ROWS[L], "head_dim": 128,
                            "dtype": "fp8_e4m3",
                            "sha256": _sha256_file(os.path.join(ENGram_DIR, f"embed_L{L}.bin"))}
                  for L in ENGRAM_ROWS}
        man = {
            "source_ckpt": self.manifest.get("source_ckpt", "ckpt_v42_sft_run.pt"),
            "source_sha256": SOURCE_SHA256,
            "v42_cfg": self.v42,
            "tier": "accurate" if self.bits == 8 else "fast",
            "quantization": {"expert_bits": self.bits, "group_size": self.group_size,
                             "mode": "affine"},
            "tokenizer": {"file": TOKENIZER, "sha256": tok_sha},
            "engram": engram,
            "entries": entries,
        }
        out = os.path.join(out_dir, "manifest.json")
        with open(out, "w") as f:
            json.dump(man, f, ensure_ascii=False, indent=2)
        return out
