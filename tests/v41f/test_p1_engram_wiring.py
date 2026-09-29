"""P1 step A: Engram wired into V41FModel, against the vendored reference.

The assembly plan (#468) splits the wiring into steps with a PAIR of gates per step: one
that the whole-network logits are unchanged with the mechanism OFF, one that they match the
reference with it ON. Both halves are needed and neither implies the other -- v41f_small
leaves all three subsystems off, so the existing whole-model gate cannot fail on engram
wiring by construction, and an ON-only gate cannot show that the OFF path was left alone.

A synthetic tokenizer (the P0 known-answer piece set) is used throughout: it is the disk-free
CPU double the #475 harness provides, so no test here needs data/tokenizer.json and
v41f/model.py never reads one.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from allclose import cmp
from ref_oracle import build_engram_transformer, load_reference, synthetic_tokenizer
from test_p0_attention import _patched_reference
from test_p1_block import _split_ref_3d
from test_p1_model import _build_pair, _copy_layer, _model_args, _run_full

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.config import v41f_small
from v41f.engram import EngramLayout
from v41f.model import V41FModel

# the ref's engram small shape (_DRAFT_SMALL/test_p0_engram use the same numbers)
_ENGRAM = dict(engram_layer_ids=(1,), engram_max_ngram_size=4, engram_n_heads=2,
               engram_head_dim=8, engram_vocab_size=20, engram_pad_id=2)
_ENGRAM_LAYER = 1
_B, _S = 2, 8


def _on_cfg():
    """v41f_small with the engram on and its table rows derived from the layout.

    Derived, never a literal: the rows are the sum of a layer's bucket primes and move with
    engram_vocab_size / engram_n_heads / engram_max_ngram_size.

    vocab_size is pinned to the synthetic tokenizer's piece count. The engram path indexes
    `token_map[input_ids]`, and that map has one row per tokenizer id -- so a config whose
    vocab_size exceeds its tokenizer's length indexes off the end (measured: id 12670 against
    a 12-row map). The ref reads the same table, so this is a property of the pair, not of
    one side.
    """
    tok = synthetic_tokenizer()
    cfg = v41f_small(vocab_size=len(tok), tokenizer=tok, **_ENGRAM)
    return cfg


def _build_on_pair(seed=41):
    """(ref Transformer ON, ours V41FModel ON, cfg, tokenizer) with finite matching weights.

    Both sides are built from ONE config (the single-shape-source rule) and the engram
    parameters are copied weight-for-weight, so a mismatch is a wiring difference and not a
    different random draw.
    """
    # the pure-torch CPU patches _build_pair installs (quant identities, the fp32 Sinkhorn
    # port, the sparse_attn dtype boundary). Without them the ref Block cannot run on CPU at
    # all -- the ON pair must be patched exactly as the OFF pair is, or the two sides are not
    # the same reference.
    model = _patched_reference()
    model.hc_split_sinkhorn = _split_ref_3d
    tok = synthetic_tokenizer()
    cfg = _on_cfg()
    # ONE shape source: _model_args derives ModelArgs from the SAME V41FConfig by field
    # intersection, so every shape reaches both sides. Handing the ref only the engram
    # fields would leave it on ModelArgs' own defaults while ours follows the config -- the
    # exact class of drift this rule exists to prevent.
    _, args = _model_args(model, cfg)
    assert tuple(args.engram_layer_ids) == tuple(cfg.engram_layer_ids), (
        "the engram layer list did not survive the config -> ModelArgs intersection")
    assert tuple(args.engram_num_embeddings) == tuple(cfg.engram_num_embeddings), (
        "engram_num_embeddings did not survive the intersection: the ref would build its "
        "table from a default rather than from the derived value")
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        ref = build_engram_transformer(model, args, tok)
        ours = V41FModel(cfg, max_batch_size=_B, max_seq_len=64, tokenizer=tok).eval()
    finally:
        torch.set_default_dtype(prev)

    g = torch.Generator().manual_seed(seed)

    def finite(t, scale=0.1):
        return (torch.randn(t.shape, generator=g, dtype=torch.float32) * scale).to(t.dtype)

    with torch.no_grad():
        assert len(ref.layers) == len(ours.layers)
        for rl, ol in zip(ref.layers, ours.layers, strict=True):
            _copy_layer(rl, ol, g)
        ref.embed.weight.copy_(finite(ref.embed.weight))
        ours.embed.weight.copy_(ref.embed.weight)
        ref.norm.weight.copy_(finite(ref.norm.weight))
        ours.norm.weight.copy_(ref.norm.weight)
        ref.head.weight.copy_(finite(ref.head.weight))
        ours.head.weight.copy_(ref.head.weight)

        # The engram module the reference attached. The ref shards the table through
        # ParallelEngramEmbedding, which stores rows in fp8 and DEQUANTIZES TO BF16 ON LOOKUP
        # (model_ref :313-317) -- so the substituted dense table must be bf16 too, or the ref
        # feeds a different dtype into wkv than ours does and the comparison measures the
        # substitution rather than the wiring (measured: fp32 substitution gives 5.0e-4 at
        # the module output on identical inputs).
        #
        # Weights flow ours -> ref, as everywhere else in this harness: generate finite values
        # once, put them in our module, then copy into the reference's substitutes. Copying
        # the other way would make the two sides equal by construction wherever they differ.
        from torch import nn

        ref_e = ref.layers[_ENGRAM_LAYER].engram
        our_e = ours.engrams[_ENGRAM_LAYER]
        assert ref_e is not None and our_e is not None
        n_cols = (_ENGRAM["engram_max_ngram_size"] - 1) * _ENGRAM["engram_n_heads"]
        with torch.no_grad():
            our_e.embed.weight.copy_(finite(our_e.embed.weight))
            our_e.wkv.weight.copy_(finite(our_e.wkv.weight))
            for name in ("q_weight", "k_weight"):
                getattr(our_e, name).copy_(finite(getattr(our_e, name)))
            dense = nn.Embedding(our_e.embed.num_embeddings, our_e.embed.embedding_dim,
                                 dtype=our_e.embed.weight.dtype)
            dense.weight.copy_(our_e.embed.weight)
            ref_e.embed = dense
            dense_wkv = nn.Linear(n_cols * _ENGRAM["engram_head_dim"],
                                  cfg.dim * (cfg.hc_mult + 1), bias=False,
                                  dtype=our_e.wkv.weight.dtype)
            dense_wkv.weight.copy_(our_e.wkv.weight)
            ref_e.wkv = dense_wkv
            for name in ("q_weight", "k_weight"):
                getattr(ref_e, name).data.copy_(getattr(our_e, name))
    return ref, ours, cfg, tok


def _run_ref_full(ref, ids):
    return _run_full(ref, ids)


def test_engram_on_whole_model_allclose():
    """G-A3: with the engram ON, whole-network logits match the vendored Transformer.

    The mechanism-ON half. bf16 full sequence, the production dtype; the mutation at the end
    proves the equality is non-vacuous (a moved engram weight must break it).
    """
    ref, ours, cfg, tok = _build_on_pair()
    torch.manual_seed(123)
    ids = torch.randint(0, cfg.vocab_size, (_B, _S), dtype=torch.long)
    rlogits, _ = _run_ref_full(ref, ids)
    with torch.no_grad():
        ologits, oeng = ours(ids)
    assert oeng is None  # dspark_target_layer_ids is () on this config
    assert rlogits.shape == (_B, _S, cfg.vocab_size) == ologits.shape
    m, _ = cmp("engram-ON bf16 full-sequence logits", ologits, rlogits, atol=5e-2)

    # mutation: moving OUR engram table must break the equality (the ON path is live)
    with torch.no_grad():
        ours.engrams[_ENGRAM_LAYER].embed.weight.add_(0.5)
        mut, _ = ours(ids)
        ours.engrams[_ENGRAM_LAYER].embed.weight.sub_(0.5)
    d = (mut - rlogits).abs().max().item()
    assert d > 1e-2, f"engram table mutation did not move the logits ({d}) -- path is dead"
    print(f"  engram-ON bf16 max_abs={m:.4e}; engram-table mutation={d:.4f} (red if ~0)")


def test_engram_off_path_bit_identical():
    """G-A2: the mechanism-OFF whole-network path is not disturbed by the wiring.

    THE REGRESSION HALF, and the one the assembly plan calls the red line. v41f_small leaves
    the engram off, so this could not fail on an engram wiring error *by construction* --
    what it can do is catch the wiring changing the OFF path, which is the other way the
    step can go wrong. Two things are asserted: the OFF config still builds to today's
    structure (no hash state, every slot None), and its logits are the same as the reference
    at the documented tolerance.
    """
    ref, ours, cfg = _build_pair(cfg=v41f_small())
    assert ours.engram_hash is None and all(s is None for s in ours.engrams), (
        "the OFF config now builds engram machinery: engram_layer_ids=() must leave every "
        "slot None and no hash state")
    torch.manual_seed(123)
    ids = torch.randint(0, cfg.vocab_size, (_B, _S), dtype=torch.long)
    rlogits, _ = _run_ref_full(ref, ids)
    with torch.no_grad():
        ologits, _ = ours(ids)
    cmp("engram-OFF bf16 full-sequence logits", ologits, rlogits, atol=5e-2)


def test_engram_injection_site_is_consumed():
    """G-A4: the engram injection actually reaches the forward.

    G-A3 alone cannot show this. A call site that is wired to the WRONG position -- or one
    that runs but whose result is dropped -- can still match a reference that places it the
    same way. The test that separates them: run the model normally, then compare against a
    variant that skips the injection for ONE layer. Dead wiring gives identical logits.
    """
    ref, ours, cfg, tok = _build_on_pair()
    torch.manual_seed(9)
    ids = torch.randint(0, cfg.vocab_size, (_B, _S), dtype=torch.long)
    with torch.no_grad():
        normal, _ = ours(ids)

        # skip the injection on the engram layer only, leaving everything else identical
        saved = ours.engrams[_ENGRAM_LAYER]
        ours.engrams[_ENGRAM_LAYER] = None
        try:
            skipped, _ = ours(ids)
        finally:
            ours.engrams[_ENGRAM_LAYER] = saved
    d = (normal.float() - skipped.float()).abs().max().item()
    assert d > 1e-3, (
        f"dropping the layer-{_ENGRAM_LAYER} engram changed the logits by only {d}: the "
        f"injection is computed but not consumed, or it is injected somewhere inert")
    print(f"  engram injection site gap={d:.4f} (0 would mean the call site is dead)")


def test_engram_param_delta_is_exactly_the_engram():
    """G-A5: the ON/OFF parameter difference is exactly the engram, asserted by NAME SET.

    A count would pass if the wiring added one tensor and dropped another. The names are the
    contract: four engram parameters on the one engram layer, all bf16, and the hash state's
    buffers are non-persistent so they never enter a checkpoint.
    """
    tok = synthetic_tokenizer()
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        off = V41FModel(v41f_small(), max_batch_size=_B)
        on = V41FModel(_on_cfg(), max_batch_size=_B, max_seq_len=64, tokenizer=tok)
    finally:
        torch.set_default_dtype(prev)

    off_p, on_p = dict(off.named_parameters()), dict(on.named_parameters())
    added = set(on_p) - set(off_p)
    removed = set(off_p) - set(on_p)
    want = {f"engrams.{_ENGRAM_LAYER}.{n}" for n in ("embed.weight", "wkv.weight", "q_weight", "k_weight")}
    assert added == want, f"added params {sorted(added)} != the engram set {sorted(want)}"
    assert not removed, f"the wiring removed parameters: {sorted(removed)}"
    for n in sorted(added):
        assert on_p[n].dtype == torch.bfloat16, f"{n} is {on_p[n].dtype}, expected bfloat16"

    # the hash state carries no trainable parameter, and its buffers are rebuilt not saved
    assert not [n for n in on_p if "engram_hash" in n], "the hash state must have no parameters"
    on_sd = set(on.state_dict())
    hash_bufs = {n for n, _ in on.engram_hash.named_buffers()}
    assert hash_bufs, "the hash state has no buffers -- the persistence check is vacuous"
    assert not {f"engram_hash.{n}" for n in hash_bufs} & on_sd, (
        "the hash state's buffers entered state_dict; they are derived from the tokenizer and "
        "must stay persistent=False")
    print(f"  engram param delta = {sorted(added)} (all bf16); hash buffers {sorted(hash_bufs)} not persisted")


def test_off_config_has_no_engram_state_keys():
    """G-A6: the OFF config's state_dict is exactly what it was before the wiring.

    The other half of the regression claim: adding ModuleList slots must not emit keys, so an
    existing checkpoint still loads with strict=True.
    """
    off = V41FModel(v41f_small(), max_batch_size=_B)
    assert not [k for k in off.state_dict() if "engram" in k], "OFF config emits engram keys"
    assert off.engram_hash is None
    assert list(off.engrams) == [None] * off.cfg.n_layers


def test_none_tokenizer_with_engram_raises_clear_error():
    """MUTANT-adjacent: an ON config handed no tokenizer must fail with a clear message
    naming the tokenizer, not an AttributeError from deep inside the hash state."""
    try:
        V41FModel(_on_cfg(), max_batch_size=_B)
    except ValueError as e:
        assert "tokenizer" in str(e), e
    else:
        raise AssertionError("an ON config built with tokenizer=None")


def test_mutant_wrong_engram_call_site_goes_red():
    """A call site that injects on the WRONG layer must not pass the ON gate.

    The reference injects at `layer.engram` for the layers in `engram_layer_ids`. Moving ours
    to a different layer keeps every parameter and every shape, so only a value comparison
    separates them -- which is exactly what G-A3 is for. This test proves G-A3 can see it.
    """
    ref, ours, cfg, tok = _build_on_pair()
    torch.manual_seed(123)
    ids = torch.randint(0, cfg.vocab_size, (_B, _S), dtype=torch.long)
    with torch.no_grad():
        good, _ = ours(ids)
        # move the module to a layer that should NOT carry one
        moved = ours.engrams[_ENGRAM_LAYER]
        ours.engrams[_ENGRAM_LAYER] = None
        ours.engrams[0] = moved
        try:
            bad, _ = ours(ids)
        finally:
            ours.engrams[0] = None
            ours.engrams[_ENGRAM_LAYER] = moved
    d = (good.float() - bad.float()).abs().max().item()
    assert d > 1e-3, f"injecting at the wrong layer changed the logits by only {d}"
    print(f"  wrong-layer injection gap={d:.4f} (G-A3 can see a moved call site)")


def test_engram_qk_weight_dtype_is_explicit_not_ambient():
    """The engram gate weights are EXPLICITLY bf16, not whatever the ambient default is.

    The reference builds them with a bare `torch.ones(...)` inside no `set_dtype` block while
    the process default is bf16 (model_ref :345-346; the only construction-time
    `set_dtype(float32)` at :940 wraps the six HC tables), so bf16 is the faithful value and
    the forward's `q_weight.float() * k_weight.float()` (:348) is where it is lifted back.

    The point of spelling it out is that a checkpointed parameter's dtype must not move with
    a caller's ambient setting. That is only observable under a DIFFERENT default, which is
    why the build below runs under fp32 and a plain bf16-default build could not fail here.
    """
    tok = synthetic_tokenizer()
    cfg = _on_cfg()
    for default, label in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
        prev = torch.get_default_dtype()
        torch.set_default_dtype(default)
        try:
            m = V41FModel(cfg, max_batch_size=_B, max_seq_len=64, tokenizer=tok)
        finally:
            torch.set_default_dtype(prev)
        e = m.engrams[_ENGRAM_LAYER]
        for name in ("q_weight", "k_weight"):
            dt = getattr(e, name).dtype
            assert dt == torch.bfloat16, (
                f"built under a {label} default, engrams.{_ENGRAM_LAYER}.{name} is {dt}: the "
                f"dtype is following the ambient default instead of being pinned, so the same "
                f"config would checkpoint a different dtype depending on the caller")
    print("  engram q/k_weight stay bf16 under both bf16 and fp32 ambient defaults")


def test_engram_table_rows_equal_bucket_prime_sum():
    """The derived table row count EQUALS the layer's bucket-prime sum -- exactly.

    Hash ids live in [0, prime_sum), so a table with extra rows is never indexed out of range
    and the surplus is unreachable dead weight. No index check, no allclose and no gate in
    this file can see the difference; the only observable is the parameter count, which
    surfaces much later as a mismatch against a separately-derived number. Pin the exact value
    here so an off-by-one in the derivation fails at the point that caused it.

    (Carried from 0e's #479, which was still open when this landed -- measured: with the
    derivation forced to +1, every other test in this file passes and only this one fails.)
    """
    from ref_oracle import load_reference as _lr

    model, engram_mod = _lr()
    cfg = _on_cfg()
    layout = engram_mod.EngramLayout.from_args(_model_args(model, cfg)[1])
    assert len(layout.primes) == len(cfg.engram_layer_ids)
    for i, layer_primes in enumerate(layout.primes):
        prime_sum = sum(p for ngram in layer_primes for p in ngram)
        assert cfg.engram_num_embeddings[i] == prime_sum, (
            f"layer {cfg.engram_layer_ids[i]}: derived rows {cfg.engram_num_embeddings[i]} != "
            f"bucket-prime sum {prime_sum} -- extra rows are unreachable dead weight that no "
            f"index check can see")
    print(f"  engram rows == bucket-prime sums {cfg.engram_num_embeddings}")


def test_mutant_num_embeddings_plus_one_goes_red():
    """MUTANT: force the derivation to +1 (the off-by-one the row-count gate exists for).

    Proves the assertion above is load-bearing rather than a restatement of the value it
    checks: every other test in this file passes under this mutation.
    """
    from v41f.config import V41FConfig as _C

    tok = synthetic_tokenizer()
    cfg = _on_cfg()
    real = _C.derived_engram_num_embeddings
    _C.derived_engram_num_embeddings = lambda self: tuple(r + 1 for r in real(self))
    try:
        bad = cfg.with_derived_engram(tokenizer=tok)
        layout = EngramLayout.from_args(_model_args(load_reference()[0], bad)[1])
        prime_sum = sum(p for ngram in layout.primes[0] for p in ngram)
        assert bad.engram_num_embeddings[0] != prime_sum, (
            "the +1 mutation did not move the row count -- the gate has no subject")
        # and the model still builds, which is exactly why only a value assertion catches it
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            V41FModel(bad, max_batch_size=_B, max_seq_len=64, tokenizer=tok)
        finally:
            torch.set_default_dtype(prev)
    finally:
        _C.derived_engram_num_embeddings = real
    print("  +1 mutation survives every shape check; only the row-count assertion rejects it")
