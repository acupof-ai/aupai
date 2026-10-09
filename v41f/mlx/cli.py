"""Bounded interactive CLI for MLX v42 backend.

Modes:
  --sft         Load SFT model with bf16-active-sparse MoE.

Examples:
  .venv/bin/python -m v41f.mlx.cli --sft --prompt "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n" --max-new-tokens 64
"""
from __future__ import annotations

import argparse, time, resource
import mlx.core as mx


def build_sft_model():
    """Build SFT model with streaming weights + sparse MoE."""
    from v41f.mlx.weights_streaming import StreamingWeights
    from v41f.mlx.config import MLXV42Config
    from v41f.mlx.model import MLXV42Model
    from v41f.mlx.engram_hash import MLXNgramHash
    from v41f.mlx.engram_ssd import EngramSSDLookup
    from tokenizers import Tokenizer

    sw = StreamingWeights()
    mcfg = MLXV42Config.from_v42_cfg(sw.v42)
    model = MLXV42Model.__new__(MLXV42Model)
    model.cfg = mcfg
    model.engram_hash = None
    model.ssd_lookups = {}

    head_dim = mcfg.rope_head_dim
    t = mx.arange(8192, dtype=mx.float32)
    # Attention RoPE uses rope_theta; compressed-KV / index RoPE uses compress_rope_theta
    # (160000). The two tables MUST differ -- sharing the 10000 table diverges from the
    # PyTorch gold on the very first token (v41f/attention.py:145).
    attn_inv = 1.0 / (mcfg.rope_theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    attn_freqs = mx.outer(t, attn_inv)
    model.attn_cos = mx.cos(attn_freqs); model.attn_sin = mx.sin(attn_freqs)
    comp_inv = 1.0 / (mcfg.compress_rope_theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    comp_freqs = mx.outer(t, comp_inv)
    model.comp_cos = mx.cos(comp_freqs); model.comp_sin = mx.sin(comp_freqs)

    tok = Tokenizer.from_file('ckpt_local/tok/tokenizer.json')
    model.engram_hash = MLXNgramHash(tok, layer_ids=tuple(sw.v42['engram_layer_ids']),
                                     max_ngram_size=sw.v42['engram_max_ngram_size'],
                                     n_heads=sw.v42['engram_n_heads'],
                                     engram_vocab_size=sw.v42['engram_compressed_vocab_size'],
                                     pad_id=sw.v42.get('engram_pad_id', 2))
    engram_sizes = {1:786862, 5:788118, 9:789492, 13:791110, 17:792776, 21:794672}
    for L in sw.v42['engram_layer_ids']:
        model.ssd_lookups[L] = EngramSSDLookup(f'ckpt_local/engram_ssd/embed_L{L}.bin',
                                                engram_sizes[L], head_dim=128, cache_rows=4096)
    return model, sw, tok


def main():
    ap = argparse.ArgumentParser(description="MLX v42 CLI (bf16-active-sparse)")
    ap.add_argument("--sft", action="store_true", help="Load SFT model")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    print("Loading SFT model (bf16-active-sparse)...")
    t0 = time.perf_counter()
    model, sw, tok = build_sft_model()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    prompt_ids = tok.encode(args.prompt).ids
    print(f"Prompt: {len(prompt_ids)} tokens")

    from v41f.mlx.forward_streaming import forward_streaming
    tokens = list(prompt_ids)
    generated = []

    print(f"Generating {args.max_new_tokens} tokens...")
    gen_start = time.time()

    for i in range(args.max_new_tokens):
        t1 = time.time()
        toks = mx.array([tokens], dtype=mx.int32)
        logits = forward_streaming(model, sw, toks)
        mx.eval(logits)
        next_tok = int(mx.argmax(logits[0, -1, :]))
        step_ms = (time.time() - t1) * 1000

        tokens.append(next_tok)
        generated.append(next_tok)

        if next_tok == 32764:  # im_end
            print(f"  im_end at step {i}")
            break

        if i % 10 == 0:
            print(f"  step {i}: tok={next_tok}, {step_ms:.0f}ms")

    total_ms = (time.time() - gen_start) * 1000
    text = tok.decode(generated)
    print("\n--- Result ---")
    print(f"Generated {len(generated)} tokens in {total_ms:.0f}ms ({len(generated)/(total_ms/1000):.1f} tok/s)")
    print(f"Text: {text[:200]}")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"Peak RSS: {rss:.0f}MB")


if __name__ == "__main__":
    main()
