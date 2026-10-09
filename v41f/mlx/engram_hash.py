"""Real n-gram Engram hash in MLX, bit-exact mirror of v41f/engram.py NgramHashState.

The hash is a pure integer function: compressed token map -> XOR rolling over
lookback shifts -> prime-modulo per (n-gram size, head) -> offset per layer.

Usage:
    from v41f.mlx.engram_hash import MLXNgramHash
    hs = MLXNgramHash(tokenizer, engram_layer_ids=(1,5,9,13,17,21), ...)
    ids = hs(input_ids, start_pos=0)  # [b, s, n_hash_cols] int
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
from sympy import isprime


def _find_next_prime(start, seen):
    c = start + 1
    while not isprime(c) or c in seen:
        c += 1
    return c


def build_compressed_token_map(tokenizer):
    """Mirror v41f.engram.build_compressed_token_map. Returns (lookup, compressed_vocab_size)."""
    from tokenizers import Regex, normalizers
    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    nv = tokenizer.get_vocab_size(with_added_tokens=True) if hasattr(tokenizer, "get_vocab_size") else len(tokenizer)
    key_to_new = {}
    lookup = [0] * nv
    for tid in range(nv):
        text = backend.decode([tid], skip_special_tokens=False)
        if "" in text:
            key = backend.id_to_token(tid)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[tid] = new_id
    return lookup, len(key_to_new)


def compute_primes(layer_ids, max_ngram_size, n_heads, vocab_size):
    """Mirror EngramLayout.from_args. Returns primes[layer][ngram_shift][head]."""
    primes = []
    seen = set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes = []
            current = vocab_size - 1
            for _ in range(n_heads):
                current = _find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))
    return primes


def compute_multipliers(layer_ids, max_ngram_size, tokenizer_vocab_size):
    """Mirror compute_hash_multipliers. Returns [n_layers, max_ngram] int64."""
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for lid in layer_ids:
        gen = np.random.default_rng(10007 * lid)
        vals = gen.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(vals * 2 + 1)
    return np.stack(rows)


class MLXNgramHash:
    """Mirrors NgramHashState.forward. int64 arithmetic in MLX.

    Uses the PyTorch build_compressed_token_map for the token fold (one-time setup),
    and replicates the XOR/prime/multiplier arithmetic exactly.
    """

    DEAD = -1

    def __init__(self, tokenizer, layer_ids, max_ngram_size=4, n_heads=4,
                 engram_vocab_size=65536, pad_id=2):
        # Reuse the PyTorch token fold (NFKC/NFD/strip/lowercase) for exactness.
        from v41f.engram import build_compressed_token_map, compute_hash_multipliers, EngramLayout
        lookup, comp_vocab = build_compressed_token_map(tokenizer)
        self.token_map = np.array(lookup, dtype=np.int64)
        self.comp_vocab = comp_vocab
        self.pad_id = int(self.token_map[pad_id])
        # primes from EngramLayout (same algorithm)
        layout = EngramLayout.from_args(type("A", (), {
            "engram_layer_ids": layer_ids,
            "engram_max_ngram_size": max_ngram_size,
            "engram_n_heads": n_heads,
            "engram_vocab_size": engram_vocab_size,
            "engram_num_embeddings": (0,) * len(layer_ids),
            "engram_head_dim": 128,
        })())
        primes = layout.primes
        flat = [[p for ng in layer for p in ng] for layer in primes]
        self.offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.multipliers = compute_hash_multipliers(layer_ids, max_ngram_size, comp_vocab).numpy()
        self.primes_np = np.array(primes)
        self.layer_ids = layer_ids
        self.max_ngram = max_ngram_size
        self.n_heads = n_heads
        self.multipliers_mx = mx.array(self.multipliers)
        self.primes_mx = mx.array(self.primes_np)
        self.token_map_mx = mx.array(self.token_map)

    def n_hash_cols(self):
        return (self.max_ngram - 1) * self.n_heads

    def __call__(self, input_ids: mx.array, start_pos: int = 0) -> mx.array:
        """input_ids: [b, s] int32/int64. Returns [b, s, n_layers, n_hash_cols] int64."""
        b, s = input_ids.shape
        # compress tokens
        compressed = mx.take(self.token_map_mx, input_ids.astype(mx.int64), axis=0)  # [b,s]
        # rolling lookback: for shift 0..max_ngram-1, gather (pos-shift).clamp_min(0)
        # We compute directly without a persistent cache (prefill path).
        positions = mx.arange(start_pos, start_pos + s)[None, :]  # [1,s]
        tokens_stack = []
        blocked = mx.zeros((b, s), dtype=mx.bool_)
        for shift in range(self.max_ngram):
            src_pos = mx.clip(positions - shift, 0, None)  # [1,s]
            src = compressed[:, src_pos[0]]  # [b, s]
            block = (positions < shift) | (src == self.DEAD)
            blocked = blocked | mx.broadcast_to(block, (b, s))
            tok = mx.where(blocked, self.pad_id, src)
            tokens_stack.append(tok)
        tokens = mx.stack(tokens_stack, axis=-1)  # [b,s,max_ngram]

        # products: tokens.unsqueeze(2) * multipliers
        # tokens [b,s,max_ngram] -> [b,s,1,max_ngram]; multipliers [n_layers, max_ngram]
        products = tokens[:, :, None, :].astype(mx.int64) * self.multipliers_mx[None, None, :, :]
        rolling = products[..., 0]  # [b,s,n_layers]
        hashes = []
        for i in range(1, self.max_ngram):
            rolling = mx.bitwise_xor(rolling, products[..., i])
            # primes[:, i-1] is [n_layers, n_heads]
            h = rolling[:, :, :, None] % self.primes_mx[:, i - 1][None, None, :, :]
            hashes.append(h)
        result = mx.concatenate(hashes, axis=-1)  # [b,s,n_layers,n_hash_cols]
        # add offsets per layer
        offsets_mx = mx.array(np.array(self.offsets))  # [n_layers, n_hash_cols]
        result = result + offsets_mx[None, None, :, :]
        return result
