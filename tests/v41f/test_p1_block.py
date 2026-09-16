"""P1: the assembled Block (Hyper-Connections around Attention + MoE) matches the
vendored upstream model_ref.Block end to end.

One v41f_small layer, CPU bf16, prefill (start_pos=0). The residual stream is hc_mult
copies [b,s,hc,d]; a block collapses onto the prior layer's pre_mix for attention and
onto its own attention pre for the FFN, then returns the new stream plus that FFN pre for
the next block. We compare BOTH the expanded residual stream and the hand-off pre.

Weights are made finite in the reference (upstream allocates torch.empty) and copied
weight-for-weight into ours: attention via the same suffix map the P0 attention test
uses, MoE by structural name (gate/experts/shared), norms and the six fp32 HC tables
directly. A single mis-wired tensor or wrong collapse order moves the output, so this is
the connection-level allclose the leaf tests could not give.

Precision scope: the gate here is CPU bf16 (whole-Block residual ~2.3e-2, atol 5e-2),
the production precision. Forcing attention to fp32 leaves a ~4.6e-3 residual on a path
upstream does not support (attn wo_a hardcoded bfloat16, act_quant in bf16); rope and
sparse_attn are exact there, so it is benign boundary accumulation, not a logic error —
see issue #433. Comparison hazard: vendored apply_rotary_emb mutates its input in place
(y.copy_) while ours returns a new tensor; compare the RETURNED tensors on contiguous
clones, never a ref-rotated view against a discarded ours return (spurious O(1) "diff").
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from allclose import cmp
from ref_oracle import bf16_args
from test_p0_attention import _SMALL, _patched_reference

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.block import Block, make_identity_pre_mix
from v41f.config import V41FConfig

# ref attention suffix -> ours, applied under the "attn." prefix
_ATTN_SUFFIX = {
    "attn_sink": "attn_sink",
    "wq_a.weight": "qproj.wq_a.weight",
    "q_norm.weight": "qproj.q_norm.weight",
    "wq_b.weight": "qproj.wq_b.weight",
    "wkv.weight": "kvproj.wkv.weight",
    "kv_norm.weight": "kvproj.kv_norm.weight",
    "wo_b.weight": "oproj.wo_b.weight",
    "compressor.norm.weight": "compressor.norm.weight",
    "compressor.wkv.weight": "compressor.wkv.weight",
    "compressor.wgate.weight": "compressor.wgate.weight",
    "indexer.wq_b.weight": "indexer.wq_b.weight",
    "indexer.weights_proj.weight": "indexer.weights_proj.weight",
    "indexer.wk.weight": "index_key.wk.weight",
    "indexer.k_norm.weight": "index_key.k_norm.weight",
}
_HC = ["hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale"]


def _split_ref_3d(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """3D wrapper over ref_oracle's 2D sinkhorn port (same math, the real kernel reshapes
    [b,s,mix]->[b*s,mix] and back). Ref Block.hc_mixes feeds 3D mixes, so the 2D-only stub
    must be wrapped exactly as the P0 hyperconn allclose does."""
    import sys as _sys

    kernel = _sys.modules.get("kernel")
    b, s, _ = mixes.shape
    flat = mixes.reshape(b * s, -1)
    pre, post, comb = kernel.hc_split_sinkhorn(flat, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
    return (pre.view(b, s, hc_mult), post.view(b, s, hc_mult), comb.view(b, s, hc_mult, hc_mult))


def _build_block_pair(layer_id, seed=7, small=None):
    model = _patched_reference()
    model.hc_split_sinkhorn = _split_ref_3d
    # hyper-connections need hc_mult pinned (ref ModelArgs default is 4; the attention-only
    # _SMALL config did not carry it). The reference args default to hc_mult=4 already.
    # MoE: _SMALL left experts at ModelArgs defaults (8 routed / top2 / inter 1024); pin the
    # same into the v41f config whose production default is the 48-expert Flash-S shape.
    small = small or {**_SMALL, "hc_mult": 4, "n_routed_experts": 8,
                      "n_activated_experts": 2, "moe_inter_dim": 1024}
    args = bf16_args(model, **small)
    cfg = V41FConfig(**{k: v for k, v in small.items() if k not in ("max_batch_size", "max_seq_len")})
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        ref = model.Block(layer_id, args, None).eval()
        ours = Block(cfg, layer_id, max_batch_size=small["max_batch_size"]).eval()
    finally:
        torch.set_default_dtype(prev)

    ref_p, ours_p = dict(ref.named_parameters()), dict(ours.named_parameters())
    g = torch.Generator().manual_seed(seed)

    def finite(t, scale=0.1):
        return (torch.randn(t.shape, generator=g, dtype=torch.float32) * scale).to(t.dtype)

    plan = {}  # ref full name -> ours full name
    for rsuf, osuf in _ATTN_SUFFIX.items():
        rn, on = f"attn.{rsuf}", f"attn.{osuf}"
        if rn in ref_p and on in ours_p:  # compressor/indexer only on a source layer
            plan[rn] = on
    # grouped wo_a: ref flat param vs ours grouped reshaped view
    plan["attn.wo_a.weight"] = "attn.oproj.wo_a"
    # norms
    plan["attn_norm.weight"] = "attn_norm.weight"
    plan["ffn_norm.weight"] = "ffn_norm.weight"
    # hyper-connection fp32 tables live under ours self.hc
    for h in _HC:
        plan[h] = f"hc.{h}"
    # MoE: gate weight + every routed expert + the shared expert align structurally.
    # gate.bias is a Parameter upstream but a persistent zero BUFFER in ours (no VL); copy
    # the finite ref bias into it the same way the P0 MoE allclose does, else routing differs.
    plan["ffn.gate.weight"] = "ffn.gate.weight"
    ffn_keys = [k for k in ref_p if k.startswith("ffn.experts.") or k.startswith("ffn.shared_experts.")]
    for k in ffn_keys:
        plan[k] = k

    missing = [k for k in plan if k not in ref_p]
    assert not missing, f"ref params not found: {missing}"
    with torch.no_grad():
        for rn, on in plan.items():
            rw, ow = ref_p[rn], ours_p[on]
            assert rw.shape == ow.shape or rw.numel() == ow.numel(), (rn, rw.shape, ow.shape)
            rw.copy_(finite(rw))
            ow.copy_(rw if rw.shape == ow.shape else rw.view_as(ow))
        # selection-only gate bias: upstream fp32 Parameter -> ours persistent buffer
        rb, ob = ref.ffn.gate.bias, ours.ffn.gate.bias
        rb.copy_(finite(rb, scale=0.5))
        ob.data.copy_(rb.float())

    # every ours parameter must have been filled (an unconnected leaf would stay at init)
    unmapped = set(ours_p) - set(plan.values())
    assert not unmapped, f"ours params left at default init: {sorted(unmapped)}"
    return ref, ours, cfg


def _run(layer_id, seed, tag):
    torch.manual_seed(0)
    ref, ours, cfg = _build_block_pair(layer_id, seed=seed)
    b, s = 2, 8
    x = (0.2 * torch.randn(b, s, cfg.dim)).bfloat16()
    xr = x.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
    pre_mix = make_identity_pre_mix(xr, cfg.hc_mult)
    with torch.no_grad():
        rx, r_pre = ref(xr, 0, pre_mix, None)
        ox, o_pre, _ = ours(xr, 0, pre_mix, None)
    max_abs, _ = cmp(f"Block L{layer_id} residual {tag}", ox, rx, atol=5e-2)
    cmp(f"Block L{layer_id} ffn_pre hand-off {tag}", o_pre, r_pre, atol=5e-2)
    assert ox.shape == (b, s, cfg.hc_mult, cfg.dim)
    assert o_pre.shape == (b, s, cfg.hc_mult)
    print(f"  block L{layer_id} {tag} max_abs={max_abs:.4e}")


def test_block_ratio0_window_only():
    _run(0, 11, "window-only")


def test_block_ratio2_compressed():
    _run(1, 7, "compressed-source")


# all window-only layers (no cross-layer KV): a stack isolates the Hyper-Connection
# coefficient hand-off, whose timing only accumulates across blocks.
_CHAIN_SMALL = {
    **_SMALL,
    "compress_ratios": (0, 0, 0), "kv_source_layers": (), "index_source_layers": (),
    "hc_mult": 4, "n_routed_experts": 8, "n_activated_experts": 2, "moe_inter_dim": 1024,
}


def _run_chain(xr, pre_mix, blocks):
    h, pm = xr, pre_mix
    for blk in blocks:
        if hasattr(blk, "hc"):  # v41f Block threads the SharedAttnState
            h, pm, _ = blk(h, 0, pm, None)
        else:  # vendored ref Block
            h, pm = blk(h, 0, pm, None)
    return h


def test_two_block_chain_hc_hand_off():
    """Two stacked blocks must match upstream when each Block returns its FFN pre to the
    NEXT block's attention. A single block is nearly invariant to the coefficient source
    (one RMSNorm erases it), so the timing is only observable in a chain: collapse the
    second block's FFN on the INCOMING pre instead of its own attention pre and the stack
    must diverge. This is the mutation the leaf/one-block tests cannot see."""
    torch.manual_seed(0)
    refs, ours, cfgs = [], [], []
    for i in range(2):
        r, o, cfg = _build_block_pair(i, seed=100 + i, small=_CHAIN_SMALL)
        refs.append(r)
        ours.append(o)
        cfgs.append(cfg)
    cfg = cfgs[0]
    b, s = 2, 8
    x = (0.2 * torch.randn(b, s, cfg.dim)).bfloat16()
    xr = x.unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
    pre = make_identity_pre_mix(xr, cfg.hc_mult)
    with torch.no_grad():
        ref_h = _run_chain(xr, pre, refs)
        got_h = _run_chain(xr, pre, ours)
    max_abs, _ = cmp("two-Block chain residual", got_h, ref_h, atol=5e-2)
    print(f"  chain max_abs={max_abs:.4e}")

    # Mutation sensitivity at the actual hand-off contract. Block 1 must feed its attention
    # the input collapsed on block 0's RETURNED ffn_pre (the coefficient carried between
    # blocks) -- not the incoming pre_mix. RMSNorm later damps the difference to ~bf16 noise
    # in the final stream, so asserting on the end state alone is blind; capture the precise
    # attention input by wrapping the instance forward (a function attr, not a child module).
    import types

    captured = {}
    attn1 = ours[1].attn
    real_forward = attn1.forward

    def spy_forward(self, inp, state=None):
        captured["inp"] = inp.detach().clone()
        return real_forward(inp, state)

    attn1.forward = types.MethodType(spy_forward, attn1)
    try:
        with torch.no_grad():
            h0, pm0, _ = ours[0](xr, 0, pre, None)
            _ = ours[1](h0, 0, pm0, None)
        correct_in = captured["inp"]
        # Block applies attn_norm before self.attn, so the spy sees the normed collapse.
        # Exact equality pins the wiring: block1's attention receives precisely the collapse
        # of h0 on the pre block0 returned (not a recomputed/wrong coefficient).
        want = ours[1].attn_norm(ours[1].hc.hc_pre(h0, pm0))
        cmp("block1 attention collapses on block0 ffn_pre", correct_in, want, atol=1e-6)
        # anti-blindness: pre-norm, collapsing on block0's returned pre is genuinely different
        # from collapsing on the incoming identity pre, so the exact equality above is not a
        # tautology between two identical tensors (RMSNorm hides the gap only post-norm).
        prenorm_correct = ours[1].hc.hc_pre(h0, pm0)
        prenorm_identity = ours[1].hc.hc_pre(h0, pre)
        gap = (prenorm_correct.float() - prenorm_identity.float()).abs().max().item()
        scale = prenorm_correct.float().abs().mean().item()
    finally:
        attn1.forward = real_forward
    # 0.64 abs against ~0.3 mean activation (~2x): block0's returned coefficient materially
    # changes block1's attention input, so the exact-equality assertion is not a tautology.
    assert gap > 0.1 and gap > scale, (
        f"block0's returned pre must materially change block1's (pre-norm) attention input "
        f"vs the incoming identity pre; gap={gap:.4f} mean|x|={scale:.4f} -- hand-off unobserved")
    print(f"  block1 attn hand-off pre-norm gap={gap:.4f} vs mean|x|={scale:.4f} (exact post-norm match)")
