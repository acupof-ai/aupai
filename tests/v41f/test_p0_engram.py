"""P0: Engram n-gram hash state and injection gate match the vendored reference.

Hash ids are INTEGER and must be bit-identical (bucket primes, per-layer offsets,
2/3/4-gram position mapping, XOR rolling, DEAD/pad blocking, prefill/decode cache).
The injection gate is floating math and must allclose at 1e-5 fp32.

A tiny synthetic tokenizer with a hand-known normalization answer is used instead of
the real 32k vocab: case/accent/whitespace folding and the U+FFFD partial-byte branch
are all exercised, and the compressed map is asserted against a literal expected list.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from allclose import cmp
from ref_oracle import load_reference

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f import engram as ours

# raw token id -> decoded text. Hand-built so normalization collapse is known:
#   0 "a" / 1 "A"           -> "a"
#   2 "y"                   -> pad source
#   3 " The" 4 "the" 5 "THE" 6 "\tThe\n" -> "the"
#   7 "cafe" 8 "café"  -> "cafe" (NFD + StripAccents)
#   9 " " 10 "\n"           -> " " (lone-space sentinel survives Strip)
#   11 partial UTF-8 byte   -> keyed by the raw piece, never normalized
PIECES = ["a", "A", "y", " The", "the", "THE", "\tThe\n", "cafe", "café", " ", "\n", "�"]
EXPECTED_MAP = [0, 0, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5]
EXPECTED_COMPRESSED_VOCAB = 6


class _Backend:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces[i] for i in ids)

    def id_to_token(self, i):
        return f"<raw{i}>"


class _Tok:
    def __init__(self, pieces):
        self.backend_tokenizer = _Backend(pieces)

    def __len__(self):
        return len(PIECES)


def _args():
    # vocab_size 20 -> primes > 19; two layers * 3 ngram sizes * 2 heads = 12 primes.
    return SimpleNamespace(
        engram_layer_ids=(0, 1),
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=8,
        engram_vocab_size=20,
        engram_pad_id=2,
        engram_compressed_vocab_size=EXPECTED_COMPRESSED_VOCAB,
        engram_num_embeddings=(204, 358),
        max_batch_size=4,
        max_seq_len=64,
        dim=16,
        hc_mult=2,
        norm_eps=1e-6,
    )


def test_compressed_token_map_known_answer():
    tok = _Tok(PIECES)
    lookup, size = ours.build_compressed_token_map(tok)
    assert size == EXPECTED_COMPRESSED_VOCAB
    assert lookup == EXPECTED_MAP
    # semantic groupings the literal list encodes
    assert lookup[0] == lookup[1]  # case fold
    assert lookup[3] == lookup[4] == lookup[5] == lookup[6]  # case + strip + whitespace
    assert lookup[7] == lookup[8]  # accent strip
    assert lookup[9] == lookup[10]  # lone whitespace, sentinel kept it non-empty
    assert lookup[11] not in EXPECTED_MAP[:11]  # U+FFFD keyed by raw piece


def test_layout_primes_offsets_exact():
    args = _args()
    ref_layout = load_reference()[1].EngramLayout.from_args(args)
    our_layout = ours.EngramLayout.from_args(args)
    # first 12 primes above 19, drawn in order and never reused across layers/rows
    assert ref_layout.primes == (
        ((23, 29), (31, 37), (41, 43)),
        ((47, 53), (59, 61), (67, 71)),
    )
    assert our_layout.primes == ref_layout.primes
    n_cols = (args.engram_max_ngram_size - 1) * args.engram_n_heads
    assert n_cols == 6
    assert our_layout.num_embeddings == (204, 358)
    # num_embeddings equals the sum of the per-layer prime bucket ranges
    for li, layer_primes in enumerate(our_layout.primes):
        assert our_layout.num_embeddings[li] == sum(p for grp in layer_primes for p in grp)


def _scalar_hashes(seq, mult, primes, offsets, pad, n_heads, max_n=4):
    """Independent pure-python reimplementation of NgramHashState.forward on one
    1-D compressed sequence (entries are compressed ids or DEAD=-1), prefill start 0.
    Mirrors cache-gather, cumulative blocked flag, pad substitution and the per-group
    prime modulus/offset, returning [L][n_hash_cols] plain ints."""
    out = []
    L = len(seq)
    for pos in range(L):
        blocked = False
        prods = []
        for shift in range(max_n):
            source = seq[max(0, pos - shift)]
            blocked = blocked or (pos < shift) or (source == ours.NgramHashState.DEAD)
            tok = pad if blocked else source
            prods.append(int(tok) * int(mult[shift]))
        cols = []
        for g in range(max_n - 1):  # 2-gram .. max_n-gram
            roll = prods[0]
            for k in range(1, g + 2):
                roll ^= prods[k]
            for head in range(n_heads):  # heads are the inner dimension
                cols.append(roll % int(primes[g, head]) + int(offsets[g * n_heads + head]))
        out.append(cols)
    return out


def test_multipliers_and_hash_buffers_bit_exact():
    _, engram_ref = load_reference()
    args = _args()
    tok = _Tok(PIECES)
    layout = engram_ref.EngramLayout.from_args(args)
    ref_h = engram_ref.NgramHashState(args, layout, tok)
    our_h = ours.NgramHashState(args, layout, tok)
    assert torch.equal(our_h.primes, ref_h.primes)
    assert torch.equal(our_h.offsets, ref_h.offsets)
    assert torch.equal(our_h.multipliers, ref_h.multipliers)
    assert torch.equal(our_h.token_map, ref_h.token_map)
    assert our_h.pad_id == ref_h.pad_id == EXPECTED_MAP[2]
    # per-layer RNG gives distinct multiplier rows
    assert not torch.equal(our_h.multipliers[0], our_h.multipliers[1])
    # multipliers are odd (the overflow-safe construction doubles and adds 1)
    assert (our_h.multipliers % 2 == 1).all()


def test_hash_ids_bit_exact_and_manual():
    _, engram_ref = load_reference()
    args = _args()
    tok = _Tok(PIECES)
    layout = engram_ref.EngramLayout.from_args(args)
    ref_h = engram_ref.NgramHashState(args, layout, tok)
    our_h = ours.NgramHashState(args, layout, tok)

    torch.manual_seed(0)
    ids = torch.randint(0, len(PIECES), (2, 9))
    with torch.inference_mode():
        ref_out = ref_h(ids.clone(), 0)
        our_out = our_h(ids.clone(), 0)
    assert our_out.shape == (2, 9, 2, 6)
    assert torch.equal(our_out, ref_out), "hash ids must be bit-identical"

    # layer 0 of every row/position matches the independent scalar reimplementation
    comp = our_h.token_map[ids]
    mult = our_h.multipliers[0]
    primes = our_h.primes[0]
    offsets = our_h.offsets[0]
    for b in range(2):
        seq = comp[b].tolist()
        want = _scalar_hashes(seq, mult, primes, offsets, our_h.pad_id, args.engram_n_heads)
        assert our_out[b, :, 0].tolist() == want


def test_dead_token_blocks_crossing_ngrams():
    _, engram_ref = load_reference()
    args = _args()
    tok = _Tok(PIECES)
    layout = engram_ref.EngramLayout.from_args(args)
    ref_h = engram_ref.NgramHashState(args, layout, tok)
    our_h = ours.NgramHashState(args, layout, tok)

    ids = torch.randint(0, len(PIECES), (1, 8))
    mask = torch.ones(1, 8, dtype=torch.bool)
    mask[0, 3] = False  # DEAD image span at position 3
    with torch.inference_mode():
        ref_out = ref_h(ids.clone(), 0, mask)
        our_out = our_h(ids.clone(), 0, mask)
    assert torch.equal(our_out, ref_out)

    # independent scalar reference over the DEAD-substituted sequence, layer 0
    dead_seq = torch.where(mask, our_h.token_map[ids], ours.NgramHashState.DEAD)[0].tolist()
    want = _scalar_hashes(
        dead_seq, our_h.multipliers[0], our_h.primes[0], our_h.offsets[0], our_h.pad_id, args.engram_n_heads
    )
    assert our_out[0, :, 0].tolist() == want


def test_prefill_decode_cache_uses_prefix():
    _, engram_ref = load_reference()
    args = _args()
    tok = _Tok(PIECES)
    layout = engram_ref.EngramLayout.from_args(args)

    def run(state_cls):
        h = state_cls(args, layout, tok)
        with torch.inference_mode():
            prefix = torch.tensor([[0, 1, 2, 3, 4]])
            h(prefix, 0)
            suffix = torch.tensor([[5, 6, 7]])
            return h(suffix, 5).clone()

    assert torch.equal(run(ours.NgramHashState), run(engram_ref.NgramHashState))

    # the suffix's 3/4-gram columns actually depend on the cached prefix
    h = ours.NgramHashState(args, layout, tok)
    with torch.inference_mode():
        h(torch.tensor([[0, 1, 2, 3, 4]]), 0)
        with_prefix = h(torch.tensor([[5, 6, 7]]), 5).clone()
        h2 = ours.NgramHashState(args, layout, tok)
        h2(torch.tensor([[11, 11, 11, 11, 11]]), 0)
        other_prefix = h2(torch.tensor([[5, 6, 7]]), 5).clone()
    assert not torch.equal(with_prefix, other_prefix)


def _dense_pair(args, layout, layer_id=0):
    """Build ref + ours Engram; replace the ref's fp8 sharded embed/projection with the
    exact dense weights ours uses, so the comparison isolates the gate math."""
    model, engram_ref = load_reference()
    ref_args = SimpleNamespace(dim=args.dim, hc_mult=args.hc_mult, norm_eps=args.norm_eps)
    our_e = ours.Engram(args, layer_id, layout)
    ref_e = model.Engram(ref_args, layer_id, layout)

    n_cols = (args.engram_max_ngram_size - 1) * args.engram_n_heads
    table_rows = layout.num_embeddings[ref_e.layer_hash_index]
    ref_embed = nn.Embedding(table_rows, layout.head_dim)
    ref_embed.weight.data.copy_(our_e.embed.weight.data)
    ref_e.embed = ref_embed

    ref_wkv = nn.Linear(n_cols * layout.head_dim, args.dim * (args.hc_mult + 1), bias=False)
    ref_wkv.weight.data.copy_(our_e.wkv.weight.data)
    ref_e.wkv = ref_wkv

    ref_e.q_weight.data.copy_(our_e.q_weight.data)
    ref_e.k_weight.data.copy_(our_e.k_weight.data)
    return ref_e, our_e


def test_engram_gate_allclose():
    _, engram_ref = load_reference()
    args = _args()
    layout = engram_ref.EngramLayout.from_args(args)

    torch.manual_seed(2)
    x = (torch.randn(2, 7, args.hc_mult, args.dim) * 2.0).float()
    h_state = ours.NgramHashState(args, layout, _Tok(PIECES))
    with torch.inference_mode():
        all_ids = h_state(torch.randint(0, len(PIECES), (2, 7)), 0)  # [B,L,n_layers,n_cols]

    # the production caller indexes [B,L,layer_hash_index] before the gate, so ids fed to an
    # Engram are that one layer's [B,L,n_cols]. Verify both engram layers (tables 204, 358).
    for layer_id in (0, 1):
        ref_e, our_e = _dense_pair(args, layout, layer_id=layer_id)
        hash_ids = all_ids[:, :, layer_id]
        with torch.inference_mode():
            ref_out = ref_e(x, hash_ids)
            our_out = our_e(x, hash_ids)
        max_abs, _ = cmp(f"Engram gate L{layer_id} fp32", our_out, ref_out, atol=1e-5)
        assert max_abs <= 1e-5
        with torch.inference_mode():
            ref_b = ref_e(x.bfloat16(), hash_ids)
            our_b = our_e(x.bfloat16(), hash_ids)
        cmp(f"Engram gate L{layer_id} bf16", our_b, ref_b, atol=2e-2)


def test_engram_gate_sign_and_mask():
    _, engram_ref = load_reference()
    args = _args()
    layout = engram_ref.EngramLayout.from_args(args)
    _, our_e = _dense_pair(args, layout)
    n_cols = (args.engram_max_ngram_size - 1) * args.engram_n_heads
    # zero stream/keys: dot clamps to 1e-6, signed sqrt -> sigmoid(0.001) ~ 0.50025,
    # so the gate is provably bounded away from 0 on both signs of the dot.
    x = torch.zeros(1, 1, args.hc_mult, args.dim)
    hash_ids = torch.zeros(1, 1, n_cols, dtype=torch.long)
    with torch.inference_mode():
        out_lo = our_e(-torch.ones_like(x) * 3.0, hash_ids)
        out_hi = our_e(torch.ones_like(x) * 3.0, hash_ids)
        x3 = torch.randn(1, 4, args.hc_mult, args.dim)
        h3 = torch.zeros(1, 4, n_cols, dtype=torch.long)
        mask = torch.tensor([[True, False, True, False]])
        out_masked = our_e(x3, h3, mask)
    assert (out_hi > 0).any() and (out_lo < 0).any()
    # masked positions pass through untouched
    assert torch.equal(out_masked[:, [1, 3]], x3[:, [1, 3]])
    assert not torch.equal(out_masked[:, [0, 2]], x3[:, [0, 2]])
