"""Smoke test: load the real checkpoint config, build tiny random weights,
run a forward on a short token sequence. Verifies shapes and that MLX ops work.
"""
import sys
sys.path.insert(0, "/Users/bytedance/code/aupai")

import mlx.core as mx

from v41f.mlx.config import MLXV42Config
from v41f.mlx.weights import load_ckpt_cfg

CKPT = "/Users/bytedance/code/aupai/ckpt_local/ckpt_v42s2.pt.step54000.fp8"

def main():
    v42_cfg, meta, sd = load_ckpt_cfg(CKPT)
    cfg = MLXV42Config.from_v42_cfg(v42_cfg)
    print("Config loaded:")
    print(f"  layers={cfg.n_layers} dim={cfg.dim} heads={cfg.n_heads} head_dim={cfg.head_dim}")
    print(f"  engram_layers={cfg.engram_layer_ids}")
    print(f"  kv_source={cfg.kv_source_layers} index_source={cfg.index_source_layers}")
    print(f"  compress_ratios={cfg.compress_ratios}")

    # Build tiny fake weights by reading shapes from the real state dict
    # and filling small random values (NOT loading the real 3.9GB).
    fake_w = {"layers": [], "engrams": {}}
    # embed/norm/head
    fake_w["embed"] = {"weight": mx.random.normal((cfg.vocab_size, cfg.dim)) * 0.02}
    fake_w["norm"] = {"weight": mx.ones(cfg.dim)}
    fake_w["head"] = {"weight": mx.random.normal((cfg.vocab_size, cfg.dim)) * 0.02}

    for L in range(cfg.n_layers):
        W = {}
        # use real shapes but random data (as bf16, skip fp8 dequant for smoke)
        def bf16(shape):
            return mx.random.normal(shape).astype(mx.bfloat16) * 0.02
        W["qproj"] = {
            "wq_a": {"weight": bf16((cfg.q_lora_rank, cfg.dim))},
            "q_norm": {"weight": mx.ones(cfg.q_lora_rank)},
            "wq_b": {"weight": bf16((cfg.n_heads * cfg.head_dim, cfg.q_lora_rank))},
        }
        W["kvproj"] = {
            "wkv": {"weight": bf16((cfg.head_dim, cfg.dim))},
            "kv_norm": {"weight": mx.ones(cfg.head_dim)},
        }
        W["oproj"] = {
            "wo_a": bf16((cfg.o_groups, cfg.o_lora_rank, (cfg.n_heads // cfg.o_groups) * cfg.head_dim)),
            "wo_b": {"weight": bf16((cfg.dim, cfg.o_groups * cfg.o_lora_rank))},
        }
        W["attn_sink"] = mx.zeros(cfg.n_heads)
        W["attn_norm"] = {"weight": mx.ones(cfg.dim)}
        if L in cfg.kv_source_layers:
            comp = {"wkv": {"weight": bf16((cfg.head_dim, cfg.dim))},
                    "norm": {"weight": mx.ones(cfg.head_dim)}}
            if cfg.compress_ratios[L] > 1:
                comp["wgate"] = {"weight": bf16((cfg.head_dim, cfg.dim))}
            W["compressor"] = comp
        if L in cfg.index_source_layers and L in cfg.kv_source_layers:
            W["index_key"] = {"wk": {"weight": bf16((cfg.index_head_dim, cfg.head_dim))},
                              "k_norm": {"weight": mx.ones(cfg.index_head_dim)}}
        if L in cfg.index_source_layers:
            W["indexer"] = {"wq_b": {"weight": bf16((cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank))},
                            "weights_proj": {"weight": bf16((cfg.index_n_heads, cfg.dim))}}
        W["ffn"] = {
            "w1": bf16((cfg.n_routed_experts, cfg.moe_inter_dim, cfg.dim)),
            "w3": bf16((cfg.n_routed_experts, cfg.moe_inter_dim, cfg.dim)),
            "w2": bf16((cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim)),
            "gate": {"weight": bf16((cfg.n_routed_experts, cfg.dim)), "bias": mx.zeros(cfg.n_routed_experts)},
            "shared": {"w1": {"weight": bf16((cfg.moe_inter_dim, cfg.dim))},
                       "w3": {"weight": bf16((cfg.moe_inter_dim, cfg.dim))},
                       "w2": {"weight": bf16((cfg.dim, cfg.moe_inter_dim))}},
        }
        W["ffn_norm"] = {"weight": mx.ones(cfg.dim)}
        mh = (2 + cfg.hc_mult) * cfg.hc_mult
        W["hc"] = {
            "attn_fn": bf16((mh, cfg.hc_mult * cfg.dim)),
            "ffn_fn": bf16((mh, cfg.hc_mult * cfg.dim)),
            "attn_base": mx.zeros(mh), "ffn_base": mx.zeros(mh),
            "attn_scale": mx.ones(3) * 0.01, "ffn_scale": mx.ones(3) * 0.01,
        }
        fake_w["layers"].append(W)

    # engram: skip for smoke (placeholder hash is fine)
    print("Running forward on 1x16 tokens...")
    from v41f.mlx.model import MLXV42Model
    model = MLXV42Model(cfg, fake_w)
    tokens = mx.random.randint(0, cfg.vocab_size, (1, 16))
    logits = model.forward(tokens)
    mx.eval(logits)
    print(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
    print(f"Logits sample: {logits[0, 0, :5]}")
    print("SMOKE OK")

if __name__ == "__main__":
    main()
