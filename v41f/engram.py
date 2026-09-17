"""Engram n-gram hash state and injection gate, faithful to upstream.

Hash state: third_party/deepseek_v41_ref/engram_ref.py.ref (EngramLayout,
NgramHashState). Ids are compressed by whitespace/case/accent folding, then each
position is XOR-hashed with the preceding 1..max_ngram_size-1 tokens, one
prime-sized disjoint bucket range per (n-gram size, head). Look-back stops at the
sequence start and at any DEAD (image) span.

Gate: model_ref.py.ref Engram. n_hash_cols rows are looked up, flattened and
projected by wkv into hc_mult keys plus one shared value; the gate is a
normalized dot of the stream against each key, signed-sqrt then sigmoid.

This is the single-process CPU/training form: the table is a plain nn.Embedding
(upstream ParallelEngramEmbedding shards fp8 rows over a TP world). The hash and
gate math is unchanged; P0 pins both bit-exact (hashes) and allclose (gate).
"""

from dataclasses import dataclass

import numpy as np
import torch
from sympy import isprime
from torch import nn


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    candidate = start + 1
    while not isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map each token id onto a smaller space where ids that normalize alike collapse.

    A private-use sentinel keeps a lone space alive through Strip(); a partial UTF-8
    byte token (decodes to U+FFFD) is keyed by its raw piece, never normalized.
    """
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """One odd multiplier per (layer, lookback), from a per-layer RNG; bounded against int64 overflow."""
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """[layer][n-gram size][head] prime bucket moduli; ranges are disjoint within a layer."""

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int

    @classmethod
    def from_args(cls, args) -> "EngramLayout | None":
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = args.engram_max_ngram_size, args.engram_n_heads
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], args.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(args.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=args.engram_head_dim,
        )


class NgramHashState(nn.Module):
    """Maps each position to the hash ids of the 2..max_ngram_size grams ending there.

    Returns [B, L, n_engram_layers, n_hash_cols]. The int64 cache carries compressed ids
    across prefill/decode; a DEAD slot (image span) blocks every n-gram crossing it.
    """

    DEAD = -1

    def __init__(self, args, layout: EngramLayout, tokenizer):
        super().__init__()
        self.layout = layout
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        assert vocab_size == args.engram_compressed_vocab_size, (
            vocab_size,
            args.engram_compressed_vocab_size,
        )
        self.pad_id = token_map[args.engram_pad_id]
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size)
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(offsets)), persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)
        self.register_buffer(
            "cache",
            torch.empty(args.max_batch_size, args.max_seq_len, dtype=torch.int64),
            persistent=False,
        )

    # NO `torch.inference_mode()` HERE. It returns INFERENCE tensors, and the caller feeds
    # them to `Engram.forward`'s `self.embed(...)` inside an autograd-tracked graph, which
    # raises "Inference tensors cannot be saved for backward" -- so an engram-on config could
    # not complete a training step at all (the default config sets engram_layer_ids=(1,), so
    # this blocked the production shape, not an exotic one).
    #
    # `no_grad()` is what this wants: the hash ids are a pure function of `input_ids` with no
    # grad-worthy input, so no graph should be built over them -- but the RESULT must be an
    # ordinary tensor the caller can use in a graph. `no_grad` leaves the same values and the
    # same memory behaviour without the inference flag. (A `.clone()` at the return would also
    # work and is more invasive: it copies a [B, L, n_hashes] int64 tensor every call.)
    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, start_pos: int, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, self.DEAD)
        self.cache[:batch, start_pos : start_pos + seqlen] = compressed

        positions = torch.arange(start_pos, start_pos + seqlen, device=input_ids.device).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            source = self.cache[:batch].gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)  # [B, L, max_ngram_size]

        products = tokens.unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


class Engram(nn.Module):
    """Writes the n-gram lookup into the residual stream, gated by stream-key match."""

    def __init__(self, args, layer_id: int, layout: EngramLayout):
        super().__init__()
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.clamp_value = 1e-6
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads

        self.embed = nn.Embedding(layout.num_embeddings[self.layer_hash_index], layout.head_dim)
        self.wkv = nn.Linear(n_hash_cols * layout.head_dim, args.dim * (args.hc_mult + 1), bias=False)
        self.eps = args.norm_eps
        # EXPLICIT bf16, matching the ref, which builds these with a bare torch.ones INSIDE
        # no set_dtype block while the process default is bf16 (model_ref :345-346; the one
        # construction-time set_dtype(float32) at :940 wraps the six HC tables only). The
        # faithful value is therefore bf16, and the forward's `q_weight.float() * k_weight.float()`
        # (:348) is where the ref lifts it back. Spelled out rather than inherited from the
        # ambient default so a caller that sets a different default dtype cannot silently
        # move a checkpointed parameter's dtype.
        self.q_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim, dtype=torch.bfloat16))
        self.k_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim, dtype=torch.bfloat16))

    def forward(
        self, x: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x: [B, L, hc_mult, dim]; hash_ids: [B, L, n_hash_cols]; token_mask [B, L] False
        shuts the gate so those positions pass through untouched."""
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
