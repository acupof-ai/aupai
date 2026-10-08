"""Load a torch v42 checkpoint into an MLX weight dict.

The checkpoint stores most Linear weights as ``torch.float8_e4m3fn`` (one byte
per value) and norms/biases as bf16. MLX has no native fp8 dtype; its
``mx.from_fp8`` consumes raw e4m3 bytes stored as uint8. We therefore keep fp8
weights as uint8 MLX arrays and dequantise on the fly at matmul time, so the
resident weight memory stays at ~1 byte/param instead of 2.

Memory discipline (memguard: single Python process >6GB is killed):
  * Routed-expert stacks stay uint8 (fp8). Only the ~8 active experts per token
    are dequantised to bf16 in a small temporary buffer.
  * Non-expert weights (attention projections, embed, head, hc tables, norms)
    are small enough to live as bf16.
  * Engram embedding tables stay uint8; see engram_ssd.py for the optional
    mmap/LRU path.

This module reads the checkpoint read-only. It never writes into the training
tree.
"""

from __future__ import annotations


import numpy as np


def _torch_fp8_to_uint8(t) -> np.ndarray:
    """View a torch float8_e4m3fn tensor as a uint8 numpy array (zero-copy)."""
    # torch fp8 tensors expose .view(torch.uint8); on CPU this is a view.
    return t.contiguous().view(t.uint8.dtype).numpy()


def _torch_bf16_to_bf16(t) -> np.ndarray:
    """View a torch bf16 tensor as a numpy array (MLX reads bf16 natively)."""
    return t.contiguous().numpy()


def load_ckpt_cfg(ckpt_path: str) -> dict:
    """Return (v42_cfg_dict, top_level_meta) without building the model."""
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    cfg = ck["cfg"]
    v42_cfg = cfg["v42_cfg"]
    meta = {
        "step": ck.get("step"),
        "vocab_id": ck.get("vocab_id"),
        "arch": cfg.get("arch"),
        "v42_engram": cfg.get("v42_engram"),
    }
    return v42_cfg, meta, ck["model"]


def build_mlx_weights(
    state_dict: dict,
    engram_mmap: bool = False,
    engram_dir: str | None = None,
) -> dict:
    """Convert a torch state_dict into a nested MLX-weight dict.

    Returns a dict with keys the model reads directly:
      {
        'embed': {'weight': mx.array(bf16)},
        'norm':  {'weight': mx.array(bf16)},
        'head':  {'weight': mx.array(uint8-fp8)},
        'layers': [ { per-layer weight dict } x N ],
        'engrams': { layer_id: { 'embed': uint8, 'wkv': bf16, 'q_weight': bf16, 'k_weight': bf16 } },
      }

    fp8 Linear weights are kept as uint8; the model calls mx.from_fp8 at matmul.
    bf16 norms/biases are kept as bf16 directly.
    """
    import mlx.core as mx

    weights = {"layers": [], "engrams": {}}
    n_layers = max(
        int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")
    ) + 1

    # top-level
    weights["embed"] = {"weight": mx.array(_torch_bf16_to_bf16(state_dict["embed.weight"]))}
    weights["norm"] = {"weight": mx.array(_torch_bf16_to_bf16(state_dict["norm.weight"]))}
    # head is fp8 in the ckpt but the reference keeps it fp32 at runtime; we
    # dequantise to bf16 once (it is only 33.5M params = 67MB).
    weights["head"] = {
        "weight": mx.from_fp8(
            mx.array(_torch_fp8_to_uint8(state_dict["head.weight"]), dtype=mx.uint8),
            mx.bfloat16,
        )
    }

    for L in range(n_layers):
        W: dict = {}
        # attention projections
        W["qproj"] = {
            "wq_a": {"weight": mx.array(_torch_fp8_to_uint8(state_dict[f"layers.{L}.attn.qproj.wq_a.weight"]), dtype=mx.uint8)},
            "q_norm": {"weight": mx.array(_torch_bf16_to_bf16(state_dict[f"layers.{L}.attn.qproj.q_norm.weight"]))},
            "wq_b": {"weight": mx.array(_torch_fp8_to_uint8(state_dict[f"layers.{L}.attn.qproj.wq_b.weight"]), dtype=mx.uint8)},
        }
        W["kvproj"] = {
            "wkv": {"weight": mx.array(_torch_fp8_to_uint8(state_dict[f"layers.{L}.attn.kvproj.wkv.weight"]), dtype=mx.uint8)},
            "kv_norm": {"weight": mx.array(_torch_bf16_to_bf16(state_dict[f"layers.{L}.attn.kvproj.kv_norm.weight"]))},
        }
        W["oproj"] = {
            "wo_a": mx.array(_torch_fp8_to_uint8(state_dict[f"layers.{L}.attn.oproj.wo_a"]), dtype=mx.uint8),
            "wo_b": {"weight": mx.array(_torch_fp8_to_uint8(state_dict[f"layers.{L}.attn.oproj.wo_b.weight"]), dtype=mx.uint8)},
        }
        W["attn_sink"] = mx.array(_torch_bf16_to_bf16(state_dict[f"layers.{L}.attn.attn_sink"]))
        W["attn_norm"] = {"weight": mx.array(_torch_bf16_to_bf16(state_dict[f"layers.{L}.attn_norm.weight"]))}

        # compressor (only on kv_source layers)
        ck_comp_wkv = f"layers.{L}.attn.compressor.wkv.weight"
        if ck_comp_wkv in state_dict:
            comp = {"wkv": {"weight": mx.array(_torch_fp8_to_uint8(state_dict[ck_comp_wkv]), dtype=mx.uint8)}}
            ck_gate = f"layers.{L}.attn.compressor.wgate.weight"
            if ck_gate in state_dict:
                comp["wgate"] = {"weight": mx.array(_torch_fp8_to_uint8(state_dict[ck_gate]), dtype=mx.uint8)}
            comp["norm"] = {"weight": mx.array(_torch_bf16_to_bf16(state_dict[f"layers.{L}.attn.compressor.norm.weight"]))}
            W["compressor"] = comp

        # index_key (only on layers that own the index key = kv_source AND index_source)
        ck_ik = f"layers.{L}.attn.index_key.wk.weight"
        if ck_ik in state_dict:
            W["index_key"] = {
                "wk": {"weight": mx.array(_torch_fp8_to_uint8(state_dict[ck_ik]), dtype=mx.uint8)},
                "k_norm": {"weight": mx.array(_torch_bf16_to_bf16(state_dict[f"layers.{L}.attn.index_key.k_norm.weight"]))},
            }

        # indexer (only on index_source layers)
        ck_idx = f"layers.{L}.attn.indexer.wq_b.weight"
        if ck_idx in state_dict:
            W["indexer"] = {
                "wq_b": {"weight": mx.array(_torch_fp8_to_uint8(state_dict[ck_idx]), dtype=mx.uint8)},
                "weights_proj": {"weight": mx.array(_torch_fp8_to_uint8(state_dict[f"layers.{L}.attn.indexer.weights_proj.weight"]), dtype=mx.uint8)},
            }

        # MoE
        W["ffn"] = {
            "w1": mx.array(_torch_fp8_to_uint8(state_dict[f"layers.{L}.ffn.w1"]), dtype=mx.uint8),
            "w3": mx.array(_torch_fp8_to_uint8(state_dict[f"layers.{L}.ffn.w3"]), dtype=mx.uint8),
            "w2": mx.array(_torch_fp8_to_uint8(state_dict[f"layers.{L}.ffn.w2"]), dtype=mx.uint8),
            "gate": {
                "weight": mx.array(_torch_fp8_to_uint8(state_dict[f"layers.{L}.ffn.gate.weight"]), dtype=mx.uint8),
                "bias": mx.array(_torch_bf16_to_bf16(state_dict[f"layers.{L}.ffn.gate.bias"])),
            },
            "shared": {
                "w1": {"weight": mx.array(_torch_fp8_to_uint8(state_dict[f"layers.{L}.ffn.shared_experts.w1.weight"]), dtype=mx.uint8)},
                "w3": {"weight": mx.array(_torch_fp8_to_uint8(state_dict[f"layers.{L}.ffn.shared_experts.w3.weight"]), dtype=mx.uint8)},
                "w2": {"weight": mx.array(_torch_fp8_to_uint8(state_dict[f"layers.{L}.ffn.shared_experts.w2.weight"]), dtype=mx.uint8)},
            },
        }
        W["ffn_norm"] = {"weight": mx.array(_torch_bf16_to_bf16(state_dict[f"layers.{L}.ffn_norm.weight"]))}

        # HyperConn tables (fp32 in ref, but saved as fp8 in ckpt; dequant to bf16)
        W["hc"] = {
            "attn_fn": mx.from_fp8(mx.array(_torch_fp8_to_uint8(state_dict[f"layers.{L}.hc.hc_attn_fn"]), dtype=mx.uint8), mx.bfloat16),
            "ffn_fn": mx.from_fp8(mx.array(_torch_fp8_to_uint8(state_dict[f"layers.{L}.hc.hc_ffn_fn"]), dtype=mx.uint8), mx.bfloat16),
            "attn_base": mx.array(_torch_bf16_to_bf16(state_dict[f"layers.{L}.hc.hc_attn_base"])),
            "ffn_base": mx.array(_torch_bf16_to_bf16(state_dict[f"layers.{L}.hc.hc_ffn_base"])),
            "attn_scale": mx.array(_torch_bf16_to_bf16(state_dict[f"layers.{L}.hc.hc_attn_scale"])),
            "ffn_scale": mx.array(_torch_bf16_to_bf16(state_dict[f"layers.{L}.hc.hc_ffn_scale"])),
        }
        weights["layers"].append(W)

    # Engram tables
    for L in sorted(
        {int(k.split(".")[1]) for k in state_dict if k.startswith("engrams.")}
    ):
        e = {}
        e["embed"] = mx.array(_torch_fp8_to_uint8(state_dict[f"engrams.{L}.embed.weight"]), dtype=mx.uint8)
        e["wkv"] = {"weight": mx.array(_torch_fp8_to_uint8(state_dict[f"engrams.{L}.wkv.weight"]), dtype=mx.uint8)}
        # q_weight/k_weight are saved fp8 but the ref keeps them bf16 and casts
        # to fp32 in the gate; dequant once.
        e["q_weight"] = mx.from_fp8(mx.array(_torch_fp8_to_uint8(state_dict[f"engrams.{L}.q_weight"]), dtype=mx.uint8), mx.bfloat16)
        e["k_weight"] = mx.from_fp8(mx.array(_torch_fp8_to_uint8(state_dict[f"engrams.{L}.k_weight"]), dtype=mx.uint8), mx.bfloat16)
        weights["engrams"][L] = e

    return weights
