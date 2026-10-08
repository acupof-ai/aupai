"""Numerical comparison: PyTorch v41f reference vs native MLX forward.

Builds the small v41f config on CPU, seeds deterministic weights, runs both
backends on the same token ids, and reports max absolute logit difference.

Run:  .venv/bin/python -m v41f.mlx.tests.compare_small
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

import mlx.core as mx

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from v41f.config import v41f_small
from v41f.lm import V42LM


def _seed_all(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def _a(t):
    """torch tensor -> mlx array (float32)."""
    return mx.array(t.detach().float().numpy())


def convert_to_mlx(model, cfg):
    """Map PyTorch state_dict to the MLX forward's expected weight dict."""
    sd = model.state_dict()

    def g(key):
        return _a(sd[key])

    W = {
        "embed": {"weight": g("embed.weight")},
        "norm": {"weight": g("norm.weight")},
        "head": {"weight": g("head.weight")},
        "layers": {},
        "engrams": {},
    }

    for L in range(cfg.n_layers):
        p = f"layers.{L}."
        LW = {
            "attn_norm": {"weight": g(p + "attn_norm.weight")},
            "qproj": {
                "wq_a": {"weight": g(p + "attn.qproj.wq_a.weight")},
                "q_norm": {"weight": g(p + "attn.qproj.q_norm.weight")},
                "wq_b": {"weight": g(p + "attn.qproj.wq_b.weight")},
            },
            "kvproj": {
                "wkv": {"weight": g(p + "attn.kvproj.wkv.weight")},
                "kv_norm": {"weight": g(p + "attn.kvproj.kv_norm.weight")},
            },
            "oproj": {
                "wo_a": g(p + "attn.oproj.wo_a"),
                "wo_b": {"weight": g(p + "attn.oproj.wo_b.weight")},
            },
            "attn_sink": g(p + "attn.attn_sink"),
            "hc": {
                "attn_fn": g(p + "hc.hc_attn_fn"),
                "attn_scale": g(p + "hc.hc_attn_scale"),
                "attn_base": g(p + "hc.hc_attn_base"),
                "ffn_fn": g(p + "hc.hc_ffn_fn"),
                "ffn_scale": g(p + "hc.hc_ffn_scale"),
                "ffn_base": g(p + "hc.hc_ffn_base"),
            },
            "ffn_norm": {"weight": g(p + "ffn_norm.weight")},
            "ffn": {
                "gate": {
                    "weight": g(p + "ffn.gate.weight"),
                    "bias": g(p + "ffn.gate.bias"),
                },
                "shared": {
                    "w1": {"weight": g(p + "ffn.shared_experts.w1.weight")},
                    "w3": {"weight": g(p + "ffn.shared_experts.w3.weight")},
                    "w2": {"weight": g(p + "ffn.shared_experts.w2.weight")},
                },
            },
        }

        # experts: stack per-expert weights
        E = cfg.n_routed_experts
        w1 = np.stack([sd[f"layers.{L}.ffn.experts.{e}.w1.weight"].detach().float().numpy() for e in range(E)])
        w3 = np.stack([sd[f"layers.{L}.ffn.experts.{e}.w3.weight"].detach().float().numpy() for e in range(E)])
        w2 = np.stack([sd[f"layers.{L}.ffn.experts.{e}.w2.weight"].detach().float().numpy() for e in range(E)])
        LW["ffn"]["w1"] = mx.array(w1)
        LW["ffn"]["w3"] = mx.array(w3)
        LW["ffn"]["w2"] = mx.array(w2)

        # compressor: only on kv-source layers (not merely ratio>0)
        if L in cfg.kv_source_layers:
            comp = {
                "wkv": {"weight": g(p + "attn.compressor.wkv.weight")},
                "norm": {"weight": g(p + "attn.compressor.norm.weight")},
            }
            # wgate exists only when ratio > 1 (gated pooling); ratio==1 has no gate
            if cfg.compress_ratios[L] > 1:
                comp["wgate"] = {"weight": g(p + "attn.compressor.wgate.weight")}
            LW["compressor"] = comp
        # index_key (if kv source)
        if L in cfg.kv_source_layers:
            LW["index_key"] = {
                "wk": {"weight": g(p + "attn.index_key.wk.weight")},
                "k_norm": {"weight": g(p + "attn.index_key.k_norm.weight")},
            }
        # indexer (if index source)
        if L in cfg.index_source_layers:
            LW["indexer"] = {
                "wq_b": {"weight": g(p + "attn.indexer.wq_b.weight")},
                "weights_proj": {"weight": g(p + "attn.indexer.weights_proj.weight")},
            }

        W["layers"][L] = LW
    return W


def mlx_forward(W, cfg, tokens):
    from v41f.mlx.model import MLXV42Model
    from v41f.mlx.config import MLXV42Config
    mcfg = MLXV42Config.from_v41f_config(cfg)
    model = MLXV42Model(mcfg, W)
    x = mx.array(np.array(tokens, dtype=np.int32))
    logits = model.forward(x)
    return np.array(logits.astype(mx.float32))


def main():
    _seed_all(0)
    print("Building PyTorch small model...")
    cfg = v41f_small()
    pt_model = V42LM(cfg).float().eval()
    n_params = sum(p.numel() for p in pt_model.parameters())
    print(f"  layers={cfg.n_layers} dim={cfg.dim} n_params={n_params/1e6:.1f}M")

    tokens = [[1, 5, 9, 13, 2, 7, 3, 0]]
    print(f"Running PyTorch forward on {tokens}...")
    with torch.no_grad():
        x = torch.tensor(tokens, dtype=torch.long)
        pt_logits = pt_model(x, no_head=False)[0].float().numpy()
    print(f"  PT logits shape: {pt_logits.shape}, max={pt_logits.max():.4f}")

    print("Converting weights to MLX...")
    W = convert_to_mlx(pt_model, cfg)
    print("Running MLX forward...")
    mx_logits = mlx_forward(W, cfg, tokens)
    print(f"  MX logits shape: {mx_logits.shape}, max={mx_logits.max():.4f}")

    diff = np.abs(pt_logits - mx_logits)
    print("\n=== COMPARISON ===")
    print(f"  max abs diff:  {diff.max():.6f}")
    print(f"  mean abs diff: {diff.mean():.6f}")
    print(f"  PT argmax[0,0]: {pt_logits[0,0].argmax()}  MX argmax[0,0]: {mx_logits[0,0].argmax()}")
    ok = diff.max() < 1.0 and pt_logits[0,0].argmax() == mx_logits[0,0].argmax()
    if ok:
        print("  PASS")
        return 0
    else:
        print("  FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
