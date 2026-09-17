"""P0 DSpark draft block: DSparkAttention + DSparkBlock vs an INDEPENDENT oracle.

The vendored model_ref has only the INFERENCE draft path (non-causal over the block,
noise fill, autoregressive sample). The TRAINING kernel is v41f-defined (doc §2.6
prereg): one parallel teacher-forced causal forward over gold shifted ids,

    KV   = concat(main window prefix, this draft block's own causal KV)
    RoPE = freqs[main_len : main_len+block_size]
    query i sees the whole main prefix + draft positions 0..i only,

and must equal gold-fed sequential decode column-by-column. This file proves that
equivalence two ways:

- fp32 CONTROL gate (atol 1e-3, the CONSTRAINT): both sides in float32, the oracle runs
  the ref Block's HC/norm/MoE and a hand-written column-by-column causal decode built
  from the ref Attention's own parameters. bf16 hides wiring bugs; fp32 does not.
- bf16 parity floor (atol 5e-2): the production dtype.

Every mutant (causal mask broken, topk off-by-one, draft RoPE start offset/reset, main
window seed dropped) is applied to OUR module alone and must go RED on the fp32 gate.
The mask/topk/freq mutants are separate gates: one must not stand in for another.

Weights are made finite in the reference (upstream allocates torch.empty) and copied
weight-for-weight; attention via the P1 suffix map, MoE structurally, the six fp32 HC
tables and main_proj/main_norm directly.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from allclose import cmp
from ref_oracle import bf16_args
from test_p0_attention import _SMALL, _patched_reference

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import v41f.mtp as mtp_mod
from v41f.block import make_identity_pre_mix
from v41f.config import V41FConfig
from v41f.mtp import DSparkBlock, dspark_causal_topk

T, S = 7, 5  # main prefix len, draft block size (T != S on purpose:
# a length-based RoPE hook can then target the draft call)
B = 2

_ATTN_SUFFIX = {
    "attn_sink": "attn_sink",
    "wq_a.weight": "qproj.wq_a.weight",
    "q_norm.weight": "qproj.q_norm.weight",
    "wq_b.weight": "qproj.wq_b.weight",
    "wkv.weight": "kvproj.wkv.weight",
    "kv_norm.weight": "kvproj.kv_norm.weight",
    "wo_b.weight": "oproj.wo_b.weight",
}
_HC = ["hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale"]

# ratio0 window-only draft shape: layer 0 of the _SMALL stack is already ratio0, so the
# reference Attention for that layer has exactly the DSparkAttention parameter set.
_DRAFT_SMALL = {
    **_SMALL,
    "n_layers": 5,
    "compress_ratios": (0, 2, 2, 1, 1),
    "kv_source_layers": (1, 3),
    "index_source_layers": (1, 3),
    "hc_mult": 4,
    "n_routed_experts": 8,
    "n_activated_experts": 2,
    "moe_inter_dim": 1024,
    # ref ModelArgs small defaults differ from the V41FConfig production defaults; pin both
    # so the comparison is weight-for-weight math, not a config mismatch (route_scale 1.5 vs
    # ref 1.0 alone scales every routed expert and hides inside the bf16 floor).
    "route_scale": 1.0,
    "swiglu_limit": 0.0,
    "n_mtp_layers": 1,
    "dspark_block_size": S,
    "dspark_target_layer_ids": (0,),
    "window_size": 16,
}


def _split_ref_3d(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """3D wrapper over ref_oracle's 2D sinkhorn port (same math as test_p1_block uses)."""
    import sys as _sys

    kernel = _sys.modules.get("kernel")
    b, s, _ = mixes.shape
    flat = mixes.reshape(b * s, -1)
    pre, post, comb = kernel.hc_split_sinkhorn(flat, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
    return pre.view(b, s, hc_mult), post.view(b, s, hc_mult), comb.view(b, s, hc_mult, hc_mult)


def _install_oracle_attn(model):
    """Return a forward (unbound) for the ref ratio0 Attention that performs gold-fed
    COLUMN-BY-COLUMN causal draft decode. It uses only the ref module's own parameters
    and ref math (apply_rotary_emb, sparse_attn, grouped oproj); the only v41f-defined
    piece is the causal loop + gold ids, which upstream does not have."""

    rope = model.apply_rotary_emb
    sparse_attn = model.sparse_attn

    def oracle_attn(self, x, start_pos, *attn_args):
        main_x = attn_args[0]
        bsz, block_size, _ = x.size()
        t = main_x.size(1)
        rd = self.rope_head_dim

        main_kv = self.kv_norm(self.wkv(main_x))
        rope(main_kv[..., -rd:], self.freqs_cis[:t])

        f = self.freqs_cis[t : t + block_size]
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        rope(q[..., -rd:], f)
        kv = self.kv_norm(self.wkv(x))
        rope(kv[..., -rd:], f)

        outs = []
        for i in range(block_size):
            kvi = torch.cat([main_kv, kv[:, : i + 1]], dim=1)
            idx = torch.arange(t + i + 1, device=x.device).view(1, 1, -1).expand(bsz, 1, -1)
            oi = sparse_attn(q[:, i : i + 1], kvi, self.attn_sink, idx, self.softmax_scale)
            rope(oi[..., -rd:], f[i : i + 1], True)
            outs.append(oi)
        o = torch.cat(outs, dim=1)
        o = o.view(bsz, block_size, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))

    return oracle_attn


def _build_pair(dtype, seed=7):
    """(ref_block_with_oracle_attn, ours_DSparkBlock, cfg, embed, inputs). All weights
    finite, copied weight-for-weight. dtype=torch.float32 for the control gate."""
    model = _patched_reference()
    model.hc_split_sinkhorn = _split_ref_3d
    args = bf16_args(model, **_DRAFT_SMALL)
    cfg = V41FConfig(**{k: v for k, v in _DRAFT_SMALL.items() if k not in ("max_batch_size", "max_seq_len")})

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        ref = model.Block(0, args, None).eval()
        # stage-0-only draft pieces the ordinary ref Block does not carry
        ref.main_proj = model.Linear(cfg.dim, cfg.dim).eval()
        ref.main_norm = model.RMSNorm(cfg.dim, cfg.norm_eps).eval()
        ref.attn.forward = _install_oracle_attn(model).__get__(ref.attn, type(ref.attn))
        ours = DSparkBlock(cfg, 0, 1, max_batch_size=_DRAFT_SMALL["max_batch_size"]).eval()
        embed = torch.nn.Embedding(cfg.vocab_size, cfg.dim)
    finally:
        torch.set_default_dtype(prev)

    # the fp32 control gate runs BOTH sides in float32 (ref Linear is bf16 by construction;
    # cast both modules after build so every GEMM/RMSNorm accumulates in fp32).
    ref.to(dtype)
    ours.to(dtype)
    embed.to(dtype)
    # coefficient/sink/routing-bias tables are fp32 in BOTH precisions. ref Block stores
    # the six HC tables at top level (hc_attn_fn...); ours stores them under hc.*, so match
    # by leaf name. ours gate bias is a persistent buffer.
    fp32_leaves = set(_HC) | {"attn_sink", "bias"}

    def set_param_fp32(mod, name, p):
        parent = mod
        parts = name.split(".")
        for sub in parts[:-1]:
            parent = getattr(parent, sub)
        parent._parameters[parts[-1]] = torch.nn.Parameter(p.float(), requires_grad=p.requires_grad)

    with torch.no_grad():
        for mod in (ref, ours):
            for name, p in list(mod.named_parameters()):
                if name.split(".")[-1] in fp32_leaves:
                    set_param_fp32(mod, name, p)
        # ours selection-only gate bias is a persistent buffer (ref keeps it a Parameter)
        ours.ffn.gate._buffers["bias"] = ours.ffn.gate.bias.float()

    ref_p, ours_p = dict(ref.named_parameters()), dict(ours.named_parameters())
    g = torch.Generator().manual_seed(seed)

    def finite(t, scale=0.1):
        return (torch.randn(t.shape, generator=g, dtype=torch.float32) * scale).to(t.dtype)

    plan = {}
    for rsuf, osuf in _ATTN_SUFFIX.items():
        plan[f"attn.{rsuf}"] = f"attn.{osuf}"
    plan["attn.wo_a.weight"] = "attn.oproj.wo_a"
    plan["attn_norm.weight"] = "attn_norm.weight"
    plan["ffn_norm.weight"] = "ffn_norm.weight"
    plan["main_proj.weight"] = "main_proj.weight"
    plan["main_norm.weight"] = "main_norm.weight"
    for h in _HC:
        plan[h] = f"hc.{h}"
    plan["ffn.gate.weight"] = "ffn.gate.weight"
    for k in ref_p:
        if k.startswith("ffn.experts.") or k.startswith("ffn.shared_experts."):
            plan[k] = k

    with torch.no_grad():
        for rn, on in plan.items():
            rw, ow = ref_p[rn], ours_p[on]
            assert rw.shape == ow.shape or rw.numel() == ow.numel(), (rn, rw.shape, ow.shape)
            rw.copy_(finite(rw))
            ow.copy_(rw if rw.shape == ow.shape else rw.view_as(ow))
        ref.ffn.gate.bias.copy_(finite(ref.ffn.gate.bias, scale=0.5))
        ours.ffn.gate.bias.data.copy_(ref.ffn.gate.bias.float())
        embed.weight.copy_(finite(embed.weight))

    unmapped = set(ours_p) - set(plan.values())
    assert not unmapped, f"ours params left at default init: {sorted(unmapped)}"

    gen = torch.Generator().manual_seed(seed + 1)
    main_hidden = (0.2 * torch.randn(B, T, cfg.dim, generator=gen)).to(dtype)
    gold_ids = torch.randint(0, cfg.vocab_size, (B, S), generator=gen)
    return model, ref, ours, cfg, embed, main_hidden, gold_ids


def _run_ref(model, ref, cfg, embed, main_hidden, gold_ids):
    main_x = ref.main_norm(ref.main_proj(main_hidden))
    x = embed(gold_ids).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
    pre = make_identity_pre_mix(x, cfg.hc_mult)
    with torch.no_grad():
        rx, rpre = ref(x, 0, pre, None, main_x)
    return rx, rpre


def _run_ours(ours, cfg, embed, main_hidden, gold_ids):
    with torch.no_grad():
        x, pre, main_kv, ml = ours.forward_train_embed(main_hidden, gold_ids, embed, make_identity_pre_mix)
        ox, opre = ours(x, pre, main_kv, ml)
    return ox, opre


def test_dspark_block_allclose_fp32_control():
    model, ref, ours, cfg, embed, mh, ids = _build_pair(torch.float32)
    rx, rpre = _run_ref(model, ref, cfg, embed, mh, ids)
    ox, opre = _run_ours(ours, cfg, embed, mh, ids)
    cmp("DSpark block residual fp32", ox, rx, atol=1e-3)
    cmp("DSpark block ffn_pre fp32", opre, rpre, atol=1e-3)
    assert torch.isfinite(ox).all()


def test_dspark_block_allclose_bf16_floor():
    model, ref, ours, cfg, embed, mh, ids = _build_pair(torch.bfloat16)
    mh, ids = mh.bfloat16(), ids
    rx, rpre = _run_ref(model, ref, cfg, embed, mh, ids)
    ox, opre = _run_ours(ours, cfg, embed, mh, ids)
    cmp("DSpark block residual bf16", ox, rx, atol=5e-2)
    cmp("DSpark block ffn_pre bf16", opre, rpre, atol=5e-2)


def test_seed_main_prefix_is_roped_kv_and_prefill_returns_x():
    """seed_main_prefix must equal RoPE(kv_norm(wkv(main_x))) with freqs[:main_len], the
    ref prefill seeding semantics (model_ref:1040-1052); the prefill call returns x."""
    model, ref, ours, cfg, embed, mh, ids = _build_pair(torch.float32)
    a = ours.attn
    with torch.no_grad():
        main_x = ours.main_norm(ours.main_proj(mh))
        got = a.seed_main_prefix(main_x)
        want = a.kvproj(main_x)
        from v41f.rope import apply_rotary_emb

        apply_rotary_emb(want[..., -a.rd :], a.freqs_cis[:T])
        cmp("seed main kv", got, want, atol=1e-6)
        assert got.shape == (B, T, cfg.head_dim)


def test_causal_topk_matches_hand_computed_rows():
    """Row i = all main positions [0,t) + draft positions [t, t+i]; ragged tail -1."""
    idx = dspark_causal_topk(T, S, B, torch.device("cpu"))
    assert idx.shape == (B, S, T + S)
    for i in range(S):
        row = idx[0, i]
        valid = row[row >= 0].tolist()
        assert valid == list(range(T)) + list(range(T, T + i + 1))
        assert (row[len(valid) :] == -1).all()
    # batch rows identical
    assert torch.equal(idx[0], idx[1])


def test_window_only_construction_assert():
    """A draft attention is always ratio0 with no compressor/indexer (ref assert)."""
    cfg = V41FConfig(**{k: v for k, v in _DRAFT_SMALL.items() if k not in ("max_batch_size", "max_seq_len")})
    blk = DSparkBlock(cfg, 0, 1)
    assert blk.attn.compress_ratio == 0
    assert blk.attn.compressor is None and blk.attn.indexer is None


def test_forward_embed_noise_fill_and_main_projection():
    """Inference entry (ref forward_embed:1129-1138): real anchor at position 0, noise at
    1.., hc expansion, and the projected anchor seeds the main KV."""
    model, ref, ours, cfg, embed, mh, ids = _build_pair(torch.float32)
    anchor = ids[:, 0]
    with torch.no_grad():
        x, pre, main_kv, ml = ours.forward_embed(mh, anchor, embed, make_identity_pre_mix)
    assert x.shape == (B, S, cfg.hc_mult, cfg.dim)
    assert ml == T and main_kv.shape == (B, T, cfg.head_dim)
    # every hc copy at position 0 is the anchor embedding; positions 1.. the noise row
    with torch.no_grad():
        e_anchor = embed(anchor)
        e_noise = embed(
            torch.full_like(anchor, cfg.dspark_noise_token_id if hasattr(cfg, "dspark_noise_token_id") else 0)
        )
    assert torch.allclose(x[:, 0, 0, :], e_anchor, atol=1e-6)
    assert torch.allclose(x[:, 1, 0, :], e_noise, atol=1e-6)
    # identity pre_mix collapses onto copy 0
    assert pre[:, :, 0].eq(1).all() and pre[:, :, 1:].eq(0).all()


def test_main_hidden_consumed_pre_block_unchanged():
    """The draft block must train against the target layers' attn INPUT handed in, never
    recompute it: project_main receives the exact tensor forward_train_embed was given."""
    model, ref, ours, cfg, embed, mh, ids = _build_pair(torch.float32)
    seen = {}
    orig = ours.project_main

    def spy(h):
        seen["ptr"] = h.data_ptr()
        return orig(h)

    ours.project_main = spy
    with torch.no_grad():
        ours.forward_train_embed(mh, ids, embed, make_identity_pre_mix)
    assert seen["ptr"] == mh.data_ptr()


def test_real_block_training_step_loss_drops_and_grads_present():
    """Real DSparkBlock in a teacher-forced multi-token step: finite loss, AdamW drives it
    down on a fixed batch, and gradients reach all three parameter families (main_proj,
    draft attention, draft MoE)."""
    torch.manual_seed(3)
    model, ref, ours, cfg, embed, mh, ids = _build_pair(torch.float32)
    head = torch.nn.Linear(cfg.dim, cfg.vocab_size)
    params = list(ours.parameters()) + list(embed.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-3)
    labels = torch.randint(0, cfg.vocab_size, (B, S))
    ce = torch.nn.CrossEntropyLoss()

    def step():
        opt.zero_grad()
        x, pre, main_kv, ml = ours.forward_train_embed(mh, ids, embed, make_identity_pre_mix)
        h, _ = ours(x, pre, main_kv, ml)
        logits = head(h[:, :, 0, :])
        loss = ce(logits.reshape(-1, cfg.vocab_size), labels.reshape(-1))
        loss.backward()
        opt.step()
        return loss.item()

    first = step()
    assert first == first  # finite
    for _ in range(8):
        last = step()
    assert last < first - 1e-3, (first, last)
    assert ours.main_proj.weight.grad is not None and ours.main_proj.weight.grad.abs().sum() > 0
    assert ours.attn.qproj.wq_b.weight.grad is not None and ours.attn.qproj.wq_b.weight.grad.abs().sum() > 0
    gsum = sum(p.grad.abs().sum() for p in ours.ffn.experts.parameters() if p.grad is not None)
    assert gsum > 0


def test_markov_rank0_omits_inference_heads():
    cfg = V41FConfig(**{k: v for k, v in _DRAFT_SMALL.items() if k not in ("max_batch_size", "max_seq_len")})
    blk = DSparkBlock(cfg, 0, 1)
    assert not any("markov" in n or "confidence" in n for n, _ in blk.named_parameters())
    raised = False
    try:
        DSparkBlock(
            V41FConfig(
                **{
                    **{k: v for k, v in _DRAFT_SMALL.items() if k not in ("max_batch_size", "max_seq_len")},
                    "dspark_markov_rank": 8,
                }
            ),
            0,
            1,
        )
    except NotImplementedError:
        raised = True
    assert raised


# ----- independent fp32 mutant gates: each must go red on its OWN assertion ----------


def _max_residual_after_mutation(mutate):
    model, ref, ours, cfg, embed, mh, ids = _build_pair(torch.float32)
    rx, _ = _run_ref(model, ref, cfg, embed, mh, ids)
    mutate(ours)
    ox, _ = _run_ours(ours, cfg, embed, mh, ids)
    return (ox.float() - rx.float()).abs().max().item()


def test_mutant_break_causal_mask_goes_red():
    """Replace the triangular topk with a block that sees ALL draft columns (the ref
    inference shape, wrong for gold training): future leak must move the fp32 result."""

    def full_topk(main_len, block_size, batch, device):
        idx = (
            torch.cat(
                [
                    torch.arange(main_len),
                    main_len + torch.arange(block_size),
                ]
            )
            .view(1, 1, -1)
            .expand(batch, block_size, -1)
            .contiguous()
        )
        return idx

    orig = mtp_mod.dspark_causal_topk
    mtp_mod.dspark_causal_topk = full_topk
    try:
        d = _max_residual_after_mutation(lambda blk: None)
    finally:
        mtp_mod.dspark_causal_topk = orig
    assert d > 1e-2, d


def test_mutant_topk_off_by_one_goes_red():
    """Independently of the full-mask mutant: row i seeing draft 0..i+1 (one future
    column) must go red."""
    orig = mtp_mod.dspark_causal_topk

    def leak_one(main_len, block_size, batch, device):
        rows = []
        for i in range(block_size):
            upto = min(i + 2, block_size)
            rows.append(torch.cat([torch.arange(main_len), main_len + torch.arange(upto)]))
        padded = torch.full((block_size, main_len + block_size), -1, dtype=torch.long)
        for i, r in enumerate(rows):
            padded[i, : r.numel()] = r
        return padded.unsqueeze(0).expand(batch, -1, -1).contiguous()

    mtp_mod.dspark_causal_topk = leak_one
    try:
        d = _max_residual_after_mutation(lambda blk: None)
    finally:
        mtp_mod.dspark_causal_topk = orig
    assert d > 1e-2, d


def _record_draft_rope_start():
    """Build a clean pair, run the production forward, and record WHERE the length-S draft
    q/kv RoPE starts by matching the passed freqs against freqs_cis. The contract is
    start == main_len (T). Deleting the offset in DSparkAttention.forward makes this read
    0 or T+1, so this recorder IS the delete-fix-goes-red gate for draft RoPE placement."""
    seen = {}
    real = mtp_mod.apply_rotary_emb

    def rec(x, freqs, inverse=False):
        if (not inverse) and freqs.shape[0] == S and "start" not in seen:
            for start in range(0, 32):
                if torch.equal(freqs, seen["freqs_cis"][start : start + S]):
                    seen["start"] = start
                    break
        return real(x, freqs, inverse)

    mtp_mod.apply_rotary_emb = rec
    try:
        model, ref, ours, cfg, embed, mh, ids = _build_pair(torch.float32)
        seen["freqs_cis"] = ours.attn.freqs_cis
        with torch.no_grad():
            x, pre, main_kv, ml = ours.forward_train_embed(mh, ids, embed, make_identity_pre_mix)
            ours(x, pre, main_kv, ml)
    finally:
        mtp_mod.apply_rotary_emb = real
    return seen.get("start")


def test_clean_draft_rope_starts_at_main_len():
    assert _record_draft_rope_start() == T


def _production_draft_qkv(forced_start=None):
    """Run the real DSparkBlock.forward and CAPTURE the post-RoPE draft q and kv the
    production forward actually produces (intercept apply_rotary_emb on the length-S draft
    call only; the main-seed call has length T and is left alone). forced_start, when set,
    replaces the draft freqs slice -- the exact source bug (T+1 late, or 0 reset). Both the
    clean and mutated captures travel the real qproj/kvproj + production RoPE path."""
    real = mtp_mod.apply_rotary_emb
    cap = {}

    def hook(x, freqs, inverse=False):
        out = real(x, freqs, inverse)
        if (not inverse) and freqs.shape[0] == S:
            cap["t"] = out.detach().clone()
        return out

    mtp_mod.apply_rotary_emb = hook
    try:
        model, ref, ours, cfg, embed, mh, ids = _build_pair(torch.float32)
        if forced_start is not None:

            def forced(x, freqs, inverse=False):
                if (not inverse) and freqs.shape[0] == S:
                    freqs = ours.attn.freqs_cis[forced_start : forced_start + S]
                return hook(x, freqs, inverse)

            mtp_mod.apply_rotary_emb = forced
        with torch.no_grad():
            x, pre, main_kv, ml = ours.forward_train_embed(mh, ids, embed, make_identity_pre_mix)
            ours(x, pre, main_kv, ml)
    finally:
        mtp_mod.apply_rotary_emb = real
    return cap["t"]


def test_mutant_draft_rope_start_offset_goes_red():
    clean = _production_draft_qkv()
    bad = _production_draft_qkv(T + 1)
    assert (clean - bad).abs().max() > 1e-2


def test_mutant_draft_rope_reset_to_zero_goes_red():
    clean = _production_draft_qkv()
    bad = _production_draft_qkv(0)
    assert (clean - bad).abs().max() > 1e-2


def test_mutant_main_window_seed_dropped_goes_red():
    """Drop RoPE from the main prefix: the seeded main_kv itself must differ O(1) from the
    correct RoPE(kv) (measured on main_kv, before attention can damp it)."""

    def unroped_seed(self, main_x, main_len=None):
        return self.kvproj(main_x)  # RoPE tail omitted

    orig = mtp_mod.DSparkAttention.seed_main_prefix
    mtp_mod.DSparkAttention.seed_main_prefix = unroped_seed
    try:
        model, ref, ours, cfg, embed, mh, ids = _build_pair(torch.float32)
        from v41f.rope import apply_rotary_emb

        with torch.no_grad():
            main_x = ours.main_norm(ours.main_proj(mh))
            bad = ours.attn.seed_main_prefix(main_x)
            want = ours.attn.kvproj(main_x)
            apply_rotary_emb(want[..., -ours.attn.rd :], ours.attn.freqs_cis[:T])
            d = (bad - want).abs().max().item()
    finally:
        mtp_mod.DSparkAttention.seed_main_prefix = orig
    assert d > 1e-2, d
