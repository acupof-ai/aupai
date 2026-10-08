"""Known answers for the decode-path changes. No checkpoint load.

  .venv/bin/python -m v41f.mlx.runtime.opt_selftest
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from ..model import sinkhorn_loop, sparse_attention
from .hc_kernel import sinkhorn_fused


def _check_sinkhorn():
    rng = np.random.default_rng(0)
    raw = mx.array(rng.normal(size=(2, 3, 4, 4)).astype(np.float32))
    comb = mx.softmax(raw, axis=-1) + 1e-6
    ref = sinkhorn_loop(comb, 20, 1e-6)
    got = sinkhorn_fused(comb, 20, 1e-6)
    mx.eval(ref, got)
    err = float(mx.max(mx.abs(ref - got)).item())
    if err > 1e-5:
        raise SystemExit(f"sinkhorn max abs {err}")
    print(f"sinkhorn max_abs={err:.3e}")


def _check_attention_batch1():
    rng = np.random.default_rng(1)
    q = mx.array(rng.normal(size=(1, 2, 2, 4)).astype(np.float32))
    kv = mx.array(rng.normal(size=(1, 6, 4)).astype(np.float32))
    window = mx.array([[[0, 1, -1], [2, 3, 4]]], dtype=mx.int32)
    comp = mx.array([[[5, -1], [1, 0]]], dtype=mx.int32)
    sink = mx.zeros((2,), dtype=mx.float32)
    got = sparse_attention(q, kv, sink, window, comp, 0.5, 0.0)
    # Reference gather is the previous batch-0 take. Invalid slots stay masked.
    win_kv = mx.take(kv[0], mx.maximum(window[0], 0), axis=0)[None]
    comp_kv = mx.take(kv[0], mx.maximum(comp[0], 0), axis=0)[None]
    scale = 0.5
    win_scores = mx.einsum("bshd,bswd->bshw", q, win_kv) * scale
    win_scores = mx.where((window[0] >= 0)[None, :, None, :], win_scores, -float("inf"))
    comp_scores = mx.einsum("bshd,bskd->bshk", q, comp_kv) * scale
    comp_scores = mx.where((comp[0] >= 0)[None, :, None, :], comp_scores, -float("inf"))
    scores = mx.concatenate([win_scores, comp_scores], axis=-1)
    row_max = mx.max(scores, axis=-1, keepdims=True)
    exp_s = mx.exp(scores - row_max)
    denom = mx.sum(exp_s, axis=-1) + mx.exp(-row_max[..., 0])
    probs = exp_s / denom[..., None]
    ref = (
        mx.einsum("bshw,bswd->bshd", probs[..., :3], win_kv)
        + mx.einsum("bshw,bswd->bshd", probs[..., 3:], comp_kv)
    )
    mx.eval(got, ref)
    err = float(mx.max(mx.abs(got - ref)).item())
    if err > 1e-5:
        raise SystemExit(f"attention max abs {err}")
    print(f"attention max_abs={err:.3e}")


def _check_hash():
    from tokenizers import Tokenizer

    from ..config import MLXV42Config
    from .engram import EngramBank

    tok = Tokenizer.from_file("ckpt_local/tok/tokenizer.json")
    layer_ids = (1, 5, 9, 13, 17, 21)
    cfg = MLXV42Config.from_v42_cfg({
        "vocab_size": 32768, "dim": 1024, "n_layers": 24, "n_heads": 16, "head_dim": 256,
        "rope_head_dim": 64, "q_lora_rank": 256, "o_groups": 8, "o_lora_rank": 256,
        "window_size": 128, "rope_theta": 10000.0,
        "compress_ratios": (0, 0) + (2,) * 10 + (1,) * 12,
        "kv_source_layers": (2, 8, 12),
        "index_source_layers": (2, 8, 12, 16, 20),
        "compress_rope_theta": 160000.0,
        "index_n_heads": 8, "index_head_dim": 128, "index_topk": 512,
        "n_routed_experts": 64, "n_shared_experts": 1, "n_activated_experts": 8,
        "moe_inter_dim": 640, "score_func": "sqrtsoftplus", "gate_temp": 1.0,
        "norm_topk_prob": True, "route_scale": 1.5, "swiglu_limit": 10.0, "norm_eps": 1e-20,
        "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
        "engram_layer_ids": layer_ids, "engram_max_ngram_size": 4, "engram_n_heads": 4,
        "engram_head_dim": 128, "engram_vocab_size": 65536, "engram_pad_id": 2,
        "engram_num_embeddings": (1,) * 6, "engram_compressed_vocab_size": 19357,
    })
    bank = EngramBank(tok, cfg)
    hs = bank.hasher
    rng = np.random.default_rng(2)
    hist = rng.integers(0, 30000, size=(2, 24), dtype=np.int64)
    ref = np.array(hs(mx.array(hist), start_pos=0))
    got = bank.hash_chunk(hist, 19, 5)
    if not np.array_equal(got, ref[:, 19:]):
        raise SystemExit(f"hash mismatch max {(got - ref[:, 19:]).max()}")
    print("hash chunk matches full-history slice")


def main():
    _check_sinkhorn()
    _check_attention_batch1()
    _check_hash()
    print("opt_selftest ok")


if __name__ == "__main__":
    main()
