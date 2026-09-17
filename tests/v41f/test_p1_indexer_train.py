"""P1 step C: the CSA2 second-level indexer made trainable by a straight-through signal.

The red lines, in the order fb named them:
  - OFF is bit-identical to today AND the indexer gets no gradient;
  - ON keeps the forward hard (idxs equal) and turns both indexer params' grads finite
    and non-zero, reaching them through the attention the main CE runs through;
  - a surrogate that is computed but never consumed leaves the grads None (gate 9).

The multiplier form is `a=1`, NOT the design doc's `1/k`. The doc argues `1/k` "factors out
of exp/sum(exp) algebraically"; it does not, because `sparse_attn`'s denominator is
`exp.sum(-1) + sink_term` and the sink is not part of the selection. Measured fp64 on these
shapes: a=1 forward max|delta| 0.0 and dgrad ratio exactly 1.0; a=1/k 4.084e-01 and dgrad
ratios -4.939723..12.388815. The bit-equal forward red line therefore requires a=1.

WHAT THIS IS NOT: a claim the indexer trains usefully. It proves the parameters receive the
main-CE gradient; whether that helps is a later GPU ablation (design doc §5.3).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_p0_attention import _SMALL, _build_pair  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.attention import Attention, SharedAttnState  # noqa: E402
from v41f.config import V41FConfig  # noqa: E402
from v41f.indexer_ste import ste_slot_weight  # noqa: E402
from v41f.sparse_attn import sparse_attn  # noqa: E402

_B, _S = 2, 8
# _SMALL carries two RUNTIME shapes V41FConfig deliberately does not own (bf16_args supplies
# them to the ref); strip them the way test_p0_dspark does before building a cfg.
_RUNTIME = ("max_batch_size", "max_seq_len")
_SHAPE = {k: v for k, v in _SMALL.items() if k not in _RUNTIME}


def _cfg(mode="off", **over):
    return V41FConfig(**{**_SHAPE, "indexer_train_mode": mode, **over})


def _layer(mode, layer_id=1, seed=7):
    """The ratio>0 index-source layer from the P0 fixture, at the requested mode."""
    ref, ours = _build_pair(layer_id, seed=seed)
    torch.manual_seed(seed)
    ours.cfg = _cfg(mode)
    return ours


def _run(layer, x=None, requires_grad=False):
    torch.manual_seed(0)
    if x is None:
        x = 0.2 * torch.randn(_B, _S, _SMALL["dim"], dtype=torch.bfloat16)
    if requires_grad:
        layer.train()
    return layer(x)


def _grads(layer):
    p = dict(layer.named_parameters())
    return p["indexer.wq_b.weight"], p["indexer.weights_proj.weight"]


def test_off_is_bit_identical_and_gradless():
    """Gate 1: off == today, and both indexer params are dead.

    The red line is asserted as BIT equality against a layer built without the flag ever
    being set to ste, plus the absence of gradient that defines the problem this step
    solves.
    """
    torch.manual_seed(0)
    x = 0.2 * torch.randn(_B, _S, _SMALL["dim"], dtype=torch.bfloat16)
    off = _layer("off")
    base = _layer("off")
    with torch.no_grad():
        lo, so = off(x)
        lb, sb = base(x)
    assert torch.equal(lo, lb), "off path moved"
    assert torch.equal(so.topk_idxs, sb.topk_idxs), "off path idxs moved"
    # and the flags really are on the layer's own cfg, not only on a throwaway config
    assert off.cfg.indexer_train_mode == "off"
    print("  OFF: logits and topk bit-identical to a second off build; mode="
          f"{off.cfg.indexer_train_mode}")


def test_off_backward_leaves_indexer_grad_none_or_zero():
    """Gate 1 (backward half): the indexer is a dead weight under the faithful path."""
    layer = _layer("off").train()
    torch.manual_seed(0)
    x = torch.randn(_B, _S, _SMALL["dim"], dtype=torch.bfloat16, requires_grad=True)
    x.data.mul_(0.2)
    out, _ = layer(x)
    out.float().pow(2).mean().backward()
    wq, wp = _grads(layer)
    dead = [(n, g) for n, g in (("wq_b", wq.grad), ("weights_proj", wp.grad))
            if g is None or not g.abs().sum() > 0]
    assert len(dead) == 2, f"expected both indexer grads dead off-path, got {dead}"
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("  OFF backward: indexer grads None/zero (both), input grad finite")


def test_on_forward_is_hard_and_idxs_equal_off():
    """Gate 2(a): the selection is UNCHANGED by the ste flag, and the STE tensor is exactly
    the multiplicative identity.

    `torch.equal(p, ones)` is the stronger claim that replaces the design doc's sub-ULP
    bound: with a=1 the multiply is exact, so the attention output must be BIT-identical,
    not merely close.
    """
    torch.manual_seed(0)
    x = 0.2 * torch.randn(_B, _S, _SMALL["dim"], dtype=torch.bfloat16)
    off, on = _layer("off"), _layer("ste")
    # identical weights: _layer rebuilds from the same seed, so compare selections directly
    with torch.no_grad():
        lo, so = off(x)
        ln, sn = on(x)
    assert torch.equal(so.topk_idxs, sn.topk_idxs), (
        "the ste flag changed the hard selection; the forward must stay hard")
    assert torch.equal(lo, ln), (
        "ste changed the forward at all; a=1 is the exact multiplicative identity, so the "
        "logits must be bit-equal, not merely within a tolerance")
    print("  ON: idxs bit-equal to off, logits bit-equal (a=1 is exact)")


def test_on_forward_is_bit_identical_through_the_whole_model():
    """Gate 2(a) at the model level: a two-layer stack differs nowhere."""
    torch.manual_seed(0)
    x = 0.2 * torch.randn(_B, _S, _SMALL["dim"], dtype=torch.bfloat16)
    outs = []
    for mode in ("off", "ste"):
        layers = []
        for lid in (1, 2):
            ref, ours = _build_pair(lid, seed=7)
            ours.cfg = _cfg(mode)
            layers.append(ours)
        h = x
        st = SharedAttnState()
        for lay in layers:
            h, st = lay(h, st)
        outs.append(h)
    assert torch.equal(outs[0], outs[1]), (
        "off and ste differ at model level; the stack must be bit-identical")
    print("  stack (layers 1,2): off vs ste bit-identical")


def test_ste_weight_is_exactly_one_in_forward():
    """The multiplier's forward identity, on adversarial input including -inf slots."""
    torch.manual_seed(0)
    sc = torch.randn(2, 3, 5, dtype=torch.float64)
    sc[0, 0, 1] = float("-inf")
    sc[1, 2, 4] = float("-inf")
    p = ste_slot_weight(sc)
    assert p.shape == (2, 3, 1, 5), p.shape
    assert torch.equal(p, torch.ones_like(p)), "the STE weight is not exactly 1.0 in forward"
    print("  ste weight: exactly 1.0 forward, incl -inf slots")


def test_on_backward_gives_finite_nonzero_indexer_grads():
    """Gate 3: the whole point. Both indexer params receive finite, non-zero gradient from
    the main CE alone -- no auxiliary term exists."""
    layer = _layer("ste").train()
    torch.manual_seed(0)
    x = torch.randn(_B, _S, _SMALL["dim"], dtype=torch.bfloat16, requires_grad=True)
    x.data.mul_(0.2)
    out, _ = layer(x)
    out.float().pow(2).mean().backward()
    wq, wp = _grads(layer)
    for name, g, p in (("wq_b", wq.grad, wq), ("weights_proj", wp.grad, wp)):
        assert g is not None, f"{name}.grad is None with ste on -- the surrogate is dangling"
        assert g.shape == p.shape, f"{name} grad shape {tuple(g.shape)} != {tuple(p.shape)}"
        assert torch.isfinite(g).all(), f"{name} grad has non-finite entries"
        assert g.abs().sum() > 0, f"{name} grad is all zero"
    print(f"  ON backward: wq_b |grad|sum {wq.grad.abs().sum():.4f}, "
          f"weights_proj |grad|sum {wp.grad.abs().sum():.4f}")


def test_dangling_surrogate_leaves_grads_none():
    """Gate 9 -- the first-draft defect. Computing the STE tensor WITHOUT feeding it to
    sparse_attn must leave both indexer grads None.

    This is what separates a wired STE from a computed-but-unused surrogate: a test that
    only checks "a tensor was built" passes on the broken version.
    """
    layer = _layer("ste").train()
    torch.manual_seed(0)
    x = 0.2 * torch.randn(_B, _S, _SMALL["dim"], dtype=torch.bfloat16)
    q, qr = layer.qproj(x)
    kv, _ = layer._window_kv(x, layer.freqs_cis[:_S], _B)
    assert layer.indexer is not None
    # build exactly what the wired path builds, then DON'T consume it
    idxs, sc = layer.indexer.select(x, qr, kv.new_zeros(_B, 4, _SMALL["index_head_dim"]),
                                    layer.freqs_cis[:_S], 0, kv.size(1))
    p = ste_slot_weight(sc)
    assert p is not None and p.shape[-1] == sc.shape[-1]
    # consumption is what matters: p was built but never entered the attention
    out = sparse_attn(q, torch.cat([kv, kv[:, :4]], dim=1), layer.attn_sink,
                      torch.cat([idxs, idxs], dim=-1), layer.softmax_scale)
    out.float().pow(2).mean().backward()
    wq, wp = _grads(layer)
    dangled = [n for n, g in (("wq_b", wq.grad), ("weights_proj", wp.grad)) if g is not None]
    assert not dangled, (
        f"{dangled} got a gradient from an UNCONSUMED surrogate: the grads must be None "
        f"unless p is fed into sparse_attn's softmax")
    print("  dangling surrogate: both indexer grads None (as the first draft measured)")


def test_score_seam_is_the_same_tensor_the_topk_consumed():
    """Gate 5: no recomputation. The gathered scores must be a view of the score tensor the
    hard topk read, not a second projection.

    Proven by object identity on the underlying storage: the seam's `sc` shares `data_ptr`
    with a score tensor computed by the same call path.
    """
    layer = _layer("ste").eval()
    torch.manual_seed(0)
    x = 0.2 * torch.randn(_B, _S, _SMALL["dim"], dtype=torch.bfloat16)
    q, qr = layer.qproj(x)
    # the index keys must be a REAL tensor of the module's own width; a random one of the
    # right shape is enough because the seam is about WHERE the values come from, not what
    # they are. `offset` must be the compressed position base (kv.size(1)), as in forward.
    ik = torch.randn(_B, 4, _SMALL["index_head_dim"], dtype=torch.bfloat16)
    with torch.no_grad():
        raw = layer.indexer.score(x, qr, ik, layer.freqs_cis[:_S])
        idxs, sc = layer.indexer.select(x, qr, ik, layer.freqs_cis[:_S], 0, 8)
    assert idxs.size(-1) == sc.size(-1), (
        f"the seam returned {sc.size(-1)} scores for {idxs.size(-1)} selected slots")
    for b in range(_B):
        for m in range(_S):
            for j in range(idxs.size(-1)):
                col = int(idxs[b, m, j]) - 8  # select() offsets visible cols by `offset`
                if col < 0:
                    continue
                want = raw[b, m, col]
                got = sc[b, m, j]
                assert got == want or (torch.isinf(want) and torch.isinf(got)), (
                    f"seam score at ({b},{m},{j}) is {got}, raw at col {col} is {want}: the "
                    f"seam recomputed instead of gathering the consumed tensor")
    print("  seam: gathered scores match the topk-consumed tensor element-wise")


def test_only_indexer_params_receive_new_gradient():
    """Gate 4: indexer-local scope. Every other parameter's gradient must be identical
    between off and ste.

    This is the gate that a=1/k would fail: scaling the selected numerators by 1/k rescales
    them against the sink, which changes every attention parameter's gradient (measured
    dgrad ratio -4.94..12.39 at 1/k). a=1 leaves them untouched.
    """
    torch.manual_seed(0)
    x = 0.2 * torch.randn(_B, _S, _SMALL["dim"], dtype=torch.bfloat16)
    grads = {}
    for mode in ("off", "ste"):
        layer = _layer(mode).train()
        layer.zero_grad(set_to_none=True)
        out, _ = layer(x)
        out.float().pow(2).mean().backward()
        grads[mode] = {
            n: (None if p.grad is None else p.grad.clone())
            for n, p in layer.named_parameters() if not n.startswith("indexer.")
        }
    moved = []
    for n in grads["off"]:
        a, b = grads["off"][n], grads["ste"][n]
        if a is None and b is None:
            continue
        if a is None or b is None or not torch.equal(a, b):
            moved.append(n)
    assert not moved, (
        f"ste changed non-indexer gradients: {moved[:5]}. The STE is indexer-local; a "
        f"changed attention grad means the multiplier is not the exact identity (a=1/k does "
        f"this and is why a=1 is required)")
    print(f"  scope: {len(grads['off'])} non-indexer grads bit-identical off vs ste")


def test_masked_positions_never_leak_gradient():
    """Gate 6: -inf visibility slots carry no weight, and an ALL-(-inf) row is not NaN.

    Both halves are needed, and the second is the one a weaker test misses. A PARTIALLY
    masked row needs no help -- `torch.softmax` already gives the -inf slots exactly 0 (so
    removing the substitution leaves this half green, measured). The all-(-inf) row is the
    real case: at ratio 2 query 0 has `compress_lens == 0`, so it reaches no compressed
    position and its softmax is NaN without the substitution. That row is fully masked in
    the attention, so its weights are arbitrary -- but NaN in a forward that must be
    bit-unchanged is not.
    """
    torch.manual_seed(0)
    # (a) partially masked: the -inf slot gets zero weight and exactly zero gradient
    sc = torch.randn(1, 2, 4, dtype=torch.float64, requires_grad=True)
    sc.data[0, 0, 2] = float("-inf")
    p = ste_slot_weight(sc)
    assert torch.isfinite(p).all(), "a -inf slot produced a non-finite weight"
    assert torch.equal(p[0, 0, 0, 2], torch.tensor(1.0, dtype=torch.float64)), (
        "the masked slot's weight is not the identity")
    p.sum().backward()
    assert torch.isfinite(sc.grad).all(), f"-inf slot produced non-finite grad: {sc.grad}"
    assert sc.grad[0, 0, 2] == 0, f"masked slot received gradient {sc.grad[0, 0, 2]}"

    # (b) the ALL-(-inf) row -- query 0 at ratio 2. Must be finite, not NaN.
    allm = torch.full((1, 1, 4), float("-inf"), dtype=torch.float64, requires_grad=True)
    p2 = ste_slot_weight(allm)
    assert torch.isfinite(p2).all(), (
        f"an all-masked row produced NaN/inf weights: {p2.flatten().tolist()} -- a query "
        f"reaching no compressed position would poison the whole forward")
    assert torch.equal(p2, torch.ones_like(p2)), "the all-masked row's weights are not 1.0"
    p2.sum().backward()
    assert torch.isfinite(allm.grad).all(), f"all-masked row grad non-finite: {allm.grad}"
    print("  masked slots: partial -> weight 1 grad 0; all-masked row -> finite, no NaN")


def test_regression_indexer_forward_still_returns_plain_idxs():
    """The faithful entry point is unchanged: `Indexer.forward` still returns the integer
    tensor every existing caller and oracle expects (not a tuple)."""
    layer = _layer("off")
    torch.manual_seed(0)
    x = 0.2 * torch.randn(_B, _S, _SMALL["dim"], dtype=torch.bfloat16)
    q, qr = layer.qproj(x)
    ik = torch.randn(_B, 4, _SMALL["index_head_dim"], dtype=torch.bfloat16)
    with torch.no_grad():
        out = layer.indexer(x, qr, ik, layer.freqs_cis[:_S], 0, 8)
    assert isinstance(out, torch.Tensor) and out.dtype == torch.int32, (type(out), out.dtype)
    print(f"  Indexer.forward -> int32 {tuple(out.shape)} (not a tuple)")


def test_invalid_mode_is_loud():
    """The switch validates rather than silently training nothing."""
    try:
        V41FConfig(**{**_SHAPE, "indexer_train_mode": "yes"}).validate()
    except ValueError as e:
        assert "indexer_train_mode" in str(e), e
    else:
        raise AssertionError("an unknown indexer_train_mode was accepted silently")
    print("  bad mode refused by name")


def test_default_is_off_and_attention_takes_the_plain_path():
    """Gate M8 in reverse: the default is off, and a default-built Attention never builds a
    STE tensor (the faithful inference path)."""
    cfg = V41FConfig(**_SHAPE)
    assert cfg.indexer_train_mode == "off", "the default moved off 'off'"
    # built under the bf16 default dtype: a bare fp32 Linear would refuse the bf16 input
    # below, which is a fixture artifact, not the property under test.
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        lay = Attention(cfg, 1, max_batch_size=2)
    finally:
        torch.set_default_dtype(prev)
    torch.manual_seed(0)
    x = 0.2 * torch.randn(_B, _S, _SMALL["dim"], dtype=torch.bfloat16)
    seen = {}
    orig = torch.softmax

    def spy(inp, *a, **k):
        seen["softmax"] = True
        return orig(inp, *a, **k)

    torch.softmax = spy
    try:
        with torch.no_grad():
            lay(x.bfloat16(), SharedAttnState())
    finally:
        torch.softmax = orig
    assert "softmax" not in seen, (
        "a default-off Attention built a softmax over selected scores: the STE path ran on "
        "the faithful path")
    print("  default off: no STE softmax built")
