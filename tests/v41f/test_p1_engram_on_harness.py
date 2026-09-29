"""A4 prerequisite harness: the vendored ref Transformer must build with the Engram
MECHANISM ON, on CPU, disk-free, through a synthetic tokenizer.

Before this harness the only ref path the suite built forced engram_layer_ids=()
(ref_oracle.bf16_args); turning a layer on died two different, opaque ways deep in the ref:
a NoneType AttributeError (Transformer got tokenizer=None) masking an IndexError (empty
engram_num_embeddings). engram_on_args / build_engram_transformer derive the table rows
from the bucket primes, measure the compressed vocab off the tokenizer, and validate the
contract BEFORE construction.

Gates here:
- ON (one layer) builds and the assembled ref engram_hash/engram modules match the
  v41f leaf modules bit-for-bit/fp32 on the same synthetic tokenizer and ids;
- OFF builds with tokenizer=None unchanged (the default whole-model path is bit-identical);
- passing None tokenizer to an ON construction raises a CLEAR ValueError before the ref's
  opaque AttributeError (the mutant must fail loudly, not stack-trace from NoneType).

Nothing reads data/tokenizer.json; v41f/model.py is untouched (that wiring is assembly
step A3, #468).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from allclose import cmp
from ref_oracle import (
    bf16_args,
    build_engram_transformer,
    engram_on_args,
    load_reference,
    synthetic_tokenizer,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f import engram as ours_engram

# small CPU shape with room for the engram primes; one engram layer (id 1).
_SHAPE = dict(
    n_layers=3,
    dim=16,
    vocab_size=12,
    hc_mult=2,
    q_lora_rank=16,
    n_heads=4,
    o_groups=4,
    o_lora_rank=16,
    head_dim=32,
    rope_head_dim=16,
    compress_ratios=(0, 0, 0),
    kv_source_layers=(),
    index_source_layers=(),
    n_routed_experts=8,
    n_activated_experts=2,
    moe_inter_dim=64,
)
ENGRAM_LAYER = 1


def _on_built():
    model, engram_mod = load_reference()
    tok = synthetic_tokenizer()
    args, num_emb, csize = engram_on_args(
        model, engram_mod, engram_layer_ids=(ENGRAM_LAYER,), tokenizer=tok, **_SHAPE
    )
    ref = build_engram_transformer(model, args, tok)
    return model, engram_mod, args, tok, num_emb, csize, ref


def test_on_path_builds_with_engram_on_one_layer():
    model, engram_mod, args, tok, num_emb, csize, ref = _on_built()
    # the hash state exists, exactly one layer carries an Engram module
    assert ref.engram_hash is not None
    assert [l.engram is not None for l in ref.layers] == [False, True, False]
    # compressed vocab measured off the tokenizer
    assert csize == 6  # the synthetic known-answer pieces fold to 6 classes
    # table rows must EQUAL that layer's bucket-prime sum, not merely be positive. Hash ids
    # land in [0, prime_sum), so extra rows are unreachable dead weight that only surfaces
    # later as a wrong parameter count -- pin the exact value here (de's required gate).
    layout = engram_mod.EngramLayout.from_args(args)
    prime_sum = sum(p for ngram in layout.primes[0] for p in ngram)
    assert num_emb == (prime_sum,), (
        f"table rows {num_emb} != bucket-prime sum ({prime_sum},): extra rows are unreachable "
        "dead weight")


def test_num_embeddings_off_by_one_goes_red():
    """Mutant: make the derivation return prime_sum + 1. The five functional tests still
    pass (a hash id lands in [0, prime_sum), so the extra table row is never indexed -- pure
    dead weight visible only later as a wrong param count). The build gate's exact-value
    assertion must reject it by NAME. Patch the derivation the way a +1 typo would and run
    the real build path."""
    import ref_oracle

    model, engram_mod = load_reference()
    tok = synthetic_tokenizer()
    real_derive = ref_oracle._engram_num_embeddings

    def plus_one(engram_mod_arg, over):
        rows = real_derive(engram_mod_arg, over)
        return tuple(r + 1 for r in rows)

    ref_oracle._engram_num_embeddings = plus_one
    try:
        args, bad_num, _ = engram_on_args(
            model, engram_mod, engram_layer_ids=(ENGRAM_LAYER,), tokenizer=tok, **_SHAPE)
        layout = engram_mod.EngramLayout.from_args(args)
        prime_sum = sum(p for ngram in layout.primes[0] for p in ngram)
        # this is the exact assertion test_on_path_builds uses; it MUST fire on the mutant
        try:
            assert bad_num == (prime_sum,), (
                f"table rows {bad_num} != bucket-prime sum ({prime_sum},): extra rows are "
                "unreachable dead weight")
        except AssertionError as e:
            assert "bucket-prime sum" in str(e) and "dead weight" in str(e)
        else:
            raise AssertionError("+1 num_embeddings mutant was not rejected by the named gate")
    finally:
        ref_oracle._engram_num_embeddings = real_derive


def test_on_hash_state_matches_ours():
    """The ref Transformer's assembled NgramHashState and the v41f NgramHashState produce
    bit-identical hash ids on the same tokenizer / input (the leaf P0 gate, re-proven on
    the instance the assembled Transformer actually holds)."""
    model, engram_mod, args, tok, num_emb, csize, ref = _on_built()
    layout = engram_mod.EngramLayout.from_args(args)
    our_h = ours_engram.NgramHashState(args, layout, tok)
    # copy buffers ref -> ours so only the forward math is compared (primes/offsets/token_map
    # are derived identically, but pin them explicitly)
    ref_h = ref.engram_hash
    for name in ("primes", "offsets", "multipliers", "token_map"):
        assert torch.equal(getattr(ref_h, name), getattr(our_h, name)), name
    ids = torch.tensor([[0, 3, 4, 7, 9, 2, 11], [1, 6, 5, 8, 10, 2, 3]])
    with torch.inference_mode():
        got = our_h(ids, 0)
        want = ref_h(ids, 0)
    assert got.shape == want.shape
    cmp("assembled engram hash ids", got, want, atol=0.0, rtol=0.0)  # integer, exact


def test_on_injection_gate_matches_ours():
    """The Engram module the assembled ref Transformer attaches on layer 1 matches the v41f
    Engram gate output fp32, after the P0 dense-weight substitution (ref shards/fp8 the
    engram embedding; the harness runs single-process dense)."""
    from torch import nn

    model, engram_mod, args, tok, num_emb, csize, ref = _on_built()
    layout = engram_mod.EngramLayout.from_args(args)
    torch.manual_seed(2)
    our_e = ours_engram.Engram(args, ENGRAM_LAYER, layout)
    ref_e = ref.layers[ENGRAM_LAYER].engram
    assert ref_e.layer_hash_index == layout.layer_ids.index(ENGRAM_LAYER)

    n_cols = (args.engram_max_ngram_size - 1) * args.engram_n_heads
    rows = layout.num_embeddings[ref_e.layer_hash_index]
    dense_embed = nn.Embedding(rows, layout.head_dim)
    dense_embed.weight.data.copy_(our_e.embed.weight.data)
    ref_e.embed = dense_embed
    dense_wkv = nn.Linear(n_cols * layout.head_dim, args.dim * (args.hc_mult + 1), bias=False)
    dense_wkv.weight.data.copy_(our_e.wkv.weight.data)
    ref_e.wkv = dense_wkv
    ref_e.q_weight.data.copy_(our_e.q_weight.data)
    ref_e.k_weight.data.copy_(our_e.k_weight.data)

    h_state = ours_engram.NgramHashState(args, layout, tok)
    x = (torch.randn(2, 7, args.hc_mult, args.dim) * 2.0).float()
    with torch.inference_mode():
        hash_ids = h_state(torch.randint(0, 12, (2, 7)), 0)[:, :, ref_e.layer_hash_index]
        ref_out = ref_e(x.float(), hash_ids)
        our_out = our_e(x.float(), hash_ids)
    cmp("assembled engram injection", our_out, ref_out, atol=1e-5)


def test_off_path_unchanged_with_tokenizer_none():
    """Default OFF path: tokenizer=None still builds, no hash, no engram modules, identical
    to bf16_args with engram_layer_ids=(). The ON helper must not perturb the OFF path."""
    model, _ = load_reference()
    args_off_default = bf16_args(model, **_SHAPE)
    a = build_engram_transformer(model, args_off_default, None)
    args_off_explicit = bf16_args(model, engram_layer_ids=(), **_SHAPE)
    b = build_engram_transformer(model, args_off_explicit, None)
    assert a.engram_hash is None and b.engram_hash is None
    assert all(l.engram is None for l in a.layers)
    assert all(l.engram is None for l in b.layers)
    # identical constructor config -> same set of parameter/buffer names (bit-stable path)
    assert set(dict(a.named_parameters())) == set(dict(b.named_parameters()))


def test_none_tokenizer_on_construction_raises_clear_error():
    """MUTANT: an ON construction handed None must fail with a clear ValueError naming the
    tokenizer BEFORE the ref's opaque 'NoneType has no attribute backend_tokenizer'. This is
    the diagnostic-order gate: it must not surface as AttributeError from deep in the ref."""
    model, engram_mod = load_reference()
    # helper level
    try:
        engram_on_args(model, engram_mod, engram_layer_ids=(1,), tokenizer=None, **_SHAPE)
    except ValueError as e:
        assert "tokenizer" in str(e) and "engram" in str(e).lower()
    else:
        raise AssertionError("engram_on_args accepted tokenizer=None with a layer on")
    # builder level: even with otherwise-valid args, None is refused before Transformer
    tok = synthetic_tokenizer()
    args, _, _ = engram_on_args(model, engram_mod, engram_layer_ids=(1,), tokenizer=tok, **_SHAPE)
    try:
        build_engram_transformer(model, args, None)
    except ValueError as e:
        assert "tokenizer" in str(e)
    else:
        raise AssertionError("build_engram_transformer accepted None on an ON config")
    # and the RAW ref path is exactly what we are protecting against: prove the opaque error
    # still exists when the guard is bypassed, so the guard is doing real work, not matching
    # a world that already raised a good error.
    raised = False
    try:
        model.Transformer(args, None)
    except AttributeError as e:
        raised = "backend_tokenizer" in str(e)
    assert raised, "raw ref no longer raises the opaque NoneType error -- guard may be dead"
