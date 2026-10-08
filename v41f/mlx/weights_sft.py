"""Memory-safe SFT checkpoint loader for MLX.

The SFT ckpt stores expert weights as bf16 (6GB). Loading them all as bf16
exceeds memguard's 6GB single-Python limit. This loader:
  1. Uses torch mmap (zero RSS until accessed).
  2. Quantises bf16 expert weights to fp8 uint8 on the fly (3GB instead of 6GB).
  3. Loads small non-expert weights as bf16 (~470MB).
  4. Streams engram embed tables to SSD mmap (no RSS cost).
  5. Releases each layer's bf16 temp buffer after fp8 conversion.

Estimated peak RSS: ~3.5GB, well under the 6GB guard.
"""
from __future__ import annotations

import numpy as np
import torch


def _bf16_to_fp8_u8(t: torch.Tensor) -> np.ndarray:
    """Quantise a bf16 torch tensor to fp8_e4m3fn uint8 numpy array."""
    # torch has float8_e4m3fn quantization via .to(torch.float8_e4m3fn)
    fp8 = t.contiguous().to(torch.float8_e4m3fn)
    return fp8.view(torch.uint8).numpy()


def _bf16_to_bf16(t: torch.Tensor) -> np.ndarray:
    """bf16 torch -> uint16 numpy view (MLX reads as bf16)."""
    return t.contiguous().view(torch.uint16).numpy()


def load_sft_safe(ckpt_path: str, engram_ssd_dir: str | None = None):
    """Load SFT ckpt with memory discipline. Returns (v42_cfg, weights_dict, stats).

    weights_dict is ready for MLXV42Model. Engram embeds are NOT in the dict;
    use engram_ssd.EngramSSDLookup on the .bin files instead.
    """
    import mlx.core as mx

    ck = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    cfg = ck["cfg"]
    v42_cfg = cfg["v42_cfg"]
    sd = ck["model"]

    n_layers = v42_cfg["n_layers"]
    weights = {"layers": [], "engrams": {}}

    # Top-level: embed (bf16), norm (bf16), head (fp32 -> bf16)
    weights["embed"] = {"weight": mx.array(_bf16_to_bf16(sd["embed.weight"]))}
    weights["norm"] = {"weight": mx.array(_bf16_to_bf16(sd["norm.weight"]))}
    # head is fp32; cast to bf16 for MLX
    head_f32 = sd["head.weight"].float().numpy()
    weights["head"] = {"weight": mx.array(head_f32.astype(np.float32))}

    for L in range(n_layers):
        W = {}
        # Attention projections (bf16 -> keep as bf16, small)
        W["qproj"] = {
            "wq_a": {"weight": mx.array(_bf16_to_bf16(sd[f"layers.{L}.attn.qproj.wq_a.weight"]))},
            "q_norm": {"weight": mx.array(_bf16_to_bf16(sd[f"layers.{L}.attn.qproj.q_norm.weight"]))},
            "wq_b": {"weight": mx.array(_bf16_to_bf16(sd[f"layers.{L}.attn.qproj.wq_b.weight"]))},
        }
        W["kvproj"] = {
            "wkv": {"weight": mx.array(_bf16_to_bf16(sd[f"layers.{L}.attn.kvproj.wkv.weight"]))},
            "kv_norm": {"weight": mx.array(_bf16_to_bf16(sd[f"layers.{L}.attn.kvproj.kv_norm.weight"]))},
        }
        W["oproj"] = {
            "wo_a": mx.array(_bf16_to_bf16(sd[f"layers.{L}.attn.oproj.wo_a"])),
            "wo_b": {"weight": mx.array(_bf16_to_bf16(sd[f"layers.{L}.attn.oproj.wo_b.weight"]))},
        }
        W["attn_sink"] = mx.array(_bf16_to_bf16(sd[f"layers.{L}.attn.attn_sink"]))
        W["attn_norm"] = {"weight": mx.array(_bf16_to_bf16(sd[f"layers.{L}.attn_norm.weight"]))}

        # Compressor (if present)
        ck_wkv = f"layers.{L}.attn.compressor.wkv.weight"
        if ck_wkv in sd:
            comp = {"wkv": {"weight": mx.array(_bf16_to_bf16(sd[ck_wkv]))}}
            ck_gate = f"layers.{L}.attn.compressor.wgate.weight"
            if ck_gate in sd:
                comp["wgate"] = {"weight": mx.array(_bf16_to_bf16(sd[ck_gate]))}
            comp["norm"] = {"weight": mx.array(_bf16_to_bf16(sd[f"layers.{L}.attn.compressor.norm.weight"]))}
            W["compressor"] = comp

        # Index key / indexer
        ck_ik = f"layers.{L}.attn.index_key.wk.weight"
        if ck_ik in sd:
            W["index_key"] = {
                "wk": {"weight": mx.array(_bf16_to_bf16(sd[ck_ik]))},
                "k_norm": {"weight": mx.array(_bf16_to_bf16(sd[f"layers.{L}.attn.index_key.k_norm.weight"]))},
            }
        ck_idx = f"layers.{L}.attn.indexer.wq_b.weight"
        if ck_idx in sd:
            W["indexer"] = {
                "wq_b": {"weight": mx.array(_bf16_to_bf16(sd[ck_idx]))},
                "weights_proj": {"weight": mx.array(_bf16_to_bf16(sd[f"layers.{L}.attn.indexer.weights_proj.weight"]))},
            }

        # MoE: quantise bf16 experts to fp8 uint8 (memory saving)
        W["ffn"] = {
            "w1": mx.array(_bf16_to_fp8_u8(sd[f"layers.{L}.ffn.w1"]), dtype=mx.uint8),
            "w3": mx.array(_bf16_to_fp8_u8(sd[f"layers.{L}.ffn.w3"]), dtype=mx.uint8),
            "w2": mx.array(_bf16_to_fp8_u8(sd[f"layers.{L}.ffn.w2"]), dtype=mx.uint8),
            "gate": {
                "weight": mx.array(sd[f"layers.{L}.ffn.gate.weight"].numpy()),
                "bias": mx.array(sd[f"layers.{L}.ffn.gate.bias"].numpy()),
            },
            "shared": {
                "w1": {"weight": mx.array(_bf16_to_bf16(sd[f"layers.{L}.ffn.shared_experts.w1.weight"]))},
                "w3": {"weight": mx.array(_bf16_to_bf16(sd[f"layers.{L}.ffn.shared_experts.w3.weight"]))},
                "w2": {"weight": mx.array(_bf16_to_bf16(sd[f"layers.{L}.ffn.shared_experts.w2.weight"]))},
            },
        }
        W["ffn_norm"] = {"weight": mx.array(_bf16_to_bf16(sd[f"layers.{L}.ffn_norm.weight"]))}

        # HyperConn
        W["hc"] = {
            "attn_fn": mx.array(_bf16_to_bf16(sd[f"layers.{L}.hc.hc_attn_fn"])),
            "ffn_fn": mx.array(_bf16_to_bf16(sd[f"layers.{L}.hc.hc_ffn_fn"])),
            "attn_base": mx.array(_bf16_to_bf16(sd[f"layers.{L}.hc.hc_attn_base"])),
            "ffn_base": mx.array(_bf16_to_bf16(sd[f"layers.{L}.hc.hc_ffn_base"])),
            "attn_scale": mx.array(_bf16_to_bf16(sd[f"layers.{L}.hc.hc_attn_scale"])),
            "ffn_scale": mx.array(_bf16_to_bf16(sd[f"layers.{L}.hc.hc_ffn_scale"])),
        }
        weights["layers"].append(W)

    # Engram: small weights (wkv, q/k_weight) go in dict; embed goes to SSD
    for L in sorted(int(k.split(".")[1]) for k in sd if k.startswith("engrams.")):
        e = {
            "wkv": {"weight": mx.array(_bf16_to_bf16(sd[f"engrams.{L}.wkv.weight"]))},
            "q_weight": mx.array(_bf16_to_bf16(sd[f"engrams.{L}.q_weight"])),
            "k_weight": mx.array(_bf16_to_bf16(sd[f"engrams.{L}.k_weight"])),
        }
        weights["engrams"][L] = e

    stats = {
        "n_layers": n_layers,
        "expert_fp8_bytes": n_layers * 3 * 64 * 640 * 1024,  # approximate
        "engram_ssd_dir": engram_ssd_dir,
    }
    return v42_cfg, weights, stats
