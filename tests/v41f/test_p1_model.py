"""P1: the assembled V41FModel (embed + Block stack + norm + fp32 head) matches the
vendored model_ref.Transformer end to end on v41f_small (engram/MTP/vision off).

CPU bf16, prefill (start_pos 0), full-sequence logits. Weights are made finite in the
reference (it allocates torch.empty) and copied weight-for-weight: every layer via the same
plan as the one-Block allclose (test_p1_block), plus embed / final norm / fp32 head. The
reference head defaults to the last inference position, so it is wrapped to return the
full sequence; the sampled output_ids are ignored.

This proves the model-level joins the Block test could not: the embedding expands to
hc_mult copies, one SharedAttnState carries cross-layer KV through the whole stack, the
final hc_pre -> RMSNorm -> head order, and the full parameter set reconciles to
scripts/v41f_param_count.py. It is still NOT trainable: no loss, optimizer or checkpoint.
"""

import dataclasses
import importlib.util
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from allclose import cmp
from ref_oracle import bf16_args
from test_p0_attention import _patched_reference
from test_p1_block import _ATTN_SUFFIX, _HC, _split_ref_3d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.config import v41f_s, v41f_small
from v41f.model import V41FModel

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]


def _model_args(model, cfg=None):
    """ModelArgs derived from the SAME V41FConfig the v41f model uses, so there is one
    shape source (a field changed in the config reaches both sides). Only the ModelArgs
    fields the config does not carry (dtype/vision/engram, batch/seq) are forced by
    bf16_args; max_batch/seq are test harness sizes, not model shapes."""
    cfg = cfg or v41f_small()
    valid = {f.name for f in dataclasses.fields(model.ModelArgs)}
    over = {k: v for k, v in dataclasses.asdict(cfg).items() if k in valid}
    over.update(max_batch_size=2, max_seq_len=64)
    return cfg, bf16_args(model, **over)


def _copy_layer(ref_layer, ours_layer, g):
    """Fill one ref Block finite and copy into one v41f Block using the P1 plan."""
    ref_p, ours_p = dict(ref_layer.named_parameters()), dict(ours_layer.named_parameters())

    def finite(t, scale=0.1):
        return (torch.randn(t.shape, generator=g, dtype=torch.float32) * scale).to(t.dtype)

    plan = {}
    for rsuf, osuf in _ATTN_SUFFIX.items():
        rn, on = f"attn.{rsuf}", f"attn.{osuf}"
        if rn in ref_p and on in ours_p:
            plan[rn] = on
    plan["attn.wo_a.weight"] = "attn.oproj.wo_a"
    plan["attn_norm.weight"] = "attn_norm.weight"
    plan["ffn_norm.weight"] = "ffn_norm.weight"
    for h in _HC:
        plan[h] = f"hc.{h}"
    plan["ffn.gate.weight"] = "ffn.gate.weight"
    for k in [k for k in ref_p if k.startswith("ffn.experts.") or k.startswith("ffn.shared_experts.")]:
        plan[k] = k

    assert all(k in ref_p for k in plan)
    with torch.no_grad():
        for rn, on in plan.items():
            rw, ow = ref_p[rn], ours_p[on]
            rw.copy_(finite(rw))
            ow.copy_(rw if rw.shape == ow.shape else rw.view_as(ow))
        # selection-only gate bias: ref fp32 Parameter -> ours persistent buffer
        rb = ref_layer.ffn.gate.bias
        rb.copy_(finite(rb, scale=0.5))
        ours_layer.ffn.gate.bias.data.copy_(rb.float())
    unmapped = set(ours_p) - set(plan.values())
    assert not unmapped, f"ours layer params left at init: {sorted(unmapped)}"


def _build_pair(seed=21, cfg=None):
    model = _patched_reference()
    model.hc_split_sinkhorn = _split_ref_3d
    cfg, args = _model_args(model, cfg)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        ref = model.Transformer(args, None).eval()
        ours = V41FModel(cfg, max_batch_size=2).eval()
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
        ref.head.weight.copy_(finite(ref.head.weight))  # fp32 on both
        ours.head.weight.copy_(ref.head.weight)
    return ref, ours, cfg


def _run_full(model, ids):
    """Full-sequence logits from a reference Transformer (its head defaults to last-only)."""
    orig = model.head.forward
    model.head.forward = lambda x, full_logits=False: orig(x, True)
    try:
        with torch.no_grad():
            _, logits, main = model(ids, 0)
    finally:
        model.head.forward = orig
    return logits, main


def _to_fp32(*mods):
    for mod in mods:
        for p in mod.parameters():
            p.data = p.data.float()
        for bu in mod.buffers():
            if bu.is_floating_point():
                bu.data = bu.data.float()


def test_whole_model_logits_allclose():
    """Whole-network logits match the vendored Transformer end to end on v41f_small.

    Primary gate is CPU bf16 FULL-sequence logits (production dtype). Deterministic CPU
    bf16 is bit-identical once every ModelArgs/V41FConfig field aligns (the args derive
    from one config, see _model_args), so this is 0.0 across seeds/seq lens and atol 5e-2
    leaves margin without rubber-stamping. An fp32 control attributes precision (the #433
    non-native path) and a head-weight mutation proves the equality is non-vacuous.
    """
    b, s = 2, 8
    ref, ours, cfg = _build_pair()
    torch.manual_seed(123)
    ids = torch.randint(0, cfg.vocab_size, (b, s), dtype=torch.long)
    rlogits, rmain = _run_full(ref, ids)
    ologits, omain = ours(ids)
    assert rmain is None and omain is None
    assert rlogits.shape == (b, s, cfg.vocab_size) == ologits.shape
    m16, _ = cmp("bf16 full-sequence logits", ologits, rlogits, atol=5e-2)

    # non-native fp32 path: same wiring, the precision-attribution control (#433)
    r32, o32, _ = _build_pair()
    _to_fp32(r32, o32)
    rl32, _ = _run_full(r32, ids)
    ol32, _ = o32(ids)
    m32, _ = cmp("fp32 full-sequence logits (control)", ol32, rl32, atol=5e-2)

    # mutation: perturbing OUR head weight must break the bf16 equality (it is not vacuous)
    with torch.no_grad():
        ours.head.weight.add_(0.5)
        mut, _ = ours(ids)
        ours.head.weight.sub_(0.5)
    m_mut = (mut - rlogits).abs().max().item()
    assert m_mut > 1.0, m_mut
    print(f"  bf16 max_abs={m16:.4e}; fp32 max_abs={m32:.4e}; head-weight mutation={m_mut:.3f} (red if ~0)")


def test_dspark_target_hidden_is_pre_block_attn_input():
    """Pin WHERE the DSpark/MTP target hidden is read. ref Transformer.forward appends
    h.mean(dim=2) for i in target_layer_ids BEFORE the block runs (the MTP head reads that
    layer's attention INPUT, ref :1264-1267). v41f_small leaves the target list empty so
    this line has zero coverage there -- moving the append AFTER the block kept every test
    green. Use a one-target config and prove the recorded hidden is the pre-block stream
    mean, bit-matches the reference at that same site, and is genuinely NOT the block
    output (anti-tautology)."""
    import types

    k = 2
    cfg = v41f_small(dspark_target_layer_ids=(k,))  # MTP module still off (dspark_block_size=0)
    ref, ours, cfg = _build_pair(seed=31, cfg=cfg)
    b, s = 2, 8
    torch.manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (b, s), dtype=torch.long)

    # capture the expanded stream immediately before/after our target block
    pre, post = {}, {}
    real_fwd = ours.layers[k].forward

    def spy(self, h, start_pos, pre_mix, state):
        pre["h"] = h.detach().clone()
        out, pm, st = real_fwd(h, start_pos, pre_mix, state)
        post["h"] = out.detach().clone()
        return out, pm, st

    ours.layers[k].forward = types.MethodType(spy, ours.layers[k])
    orig = ref.head.forward
    ref.head.forward = lambda x, full_logits=False: orig(x, True)
    try:
        with torch.no_grad():
            _, _, rmain = ref(ids, 0)
            _, omain = ours(ids)
    finally:
        ref.head.forward = orig
        ours.layers[k].forward = real_fwd

    assert rmain is not None and omain is not None, "target=(k,) must record a hidden"
    # (1) it is the same tensor the reference records at its pre-block site
    cmp("DSpark target hidden vs ref", omain, rmain, atol=5e-2)
    # (2) it equals the PRE-block stream mean exactly (the attention input), not recomputed
    pre_mean = pre["h"].mean(dim=2)
    cmp("DSpark target is pre-block attn input", omain, pre_mean, atol=1e-6)
    # (3) anti-tautology: pre-block attn input materially differs from the block OUTPUT
    post_mean = post["h"].mean(dim=2)
    gap = (pre_mean.float() - post_mean.float()).abs().max().item()
    assert not torch.allclose(omain.float(), post_mean.float(), atol=1e-2), (
        "target hidden indistinguishable from block output -- position unobserved"
    )
    print(f"  target@{k} hidden==pre-block mean exact; pre-vs-post-block gap={gap:.4f} (must be large)")


def _count(cfg):
    spec = importlib.util.spec_from_file_location("vpc", str(_ROOT / "scripts" / "v41f_param_count.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.count(cfg)


def test_param_count_reconciles_to_script():
    _, ours, cfg = _build_pair()
    r = _count(cfg)

    def numel(mod):
        return sum(p.numel() for p in mod.parameters())

    embed = numel(ours.embed)
    head = numel(ours.head)
    final_norm = numel(ours.norm)
    # backbone: Block params PLUS the persistent gate-bias buffer (upstream gate.bias is a
    # trainable Parameter; v41f stores it as a persistent zero buffer, so parameters() alone
    # under-counts it by n_layers*n_routed).
    backbone_params = sum(numel(l) for l in ours.layers)
    gate_bias_buf = sum(
        l.ffn.gate.bias.numel() for l in ours.layers if not isinstance(l.ffn.gate.bias, torch.nn.Parameter)
    )
    backbone = backbone_params + gate_bias_buf

    assert embed == r["embedding"] == cfg.vocab_size * cfg.dim
    assert head == r["lm_head"] == cfg.vocab_size * cfg.dim
    assert final_norm == r["final_norm"] == cfg.dim
    assert backbone == r["backbone_total"], (backbone, r["backbone_total"])
    total = backbone + embed + head + final_norm
    assert total == r["total_params"] == 180_262_534, (total, r["total_params"])
    # the fp32 head weight is the only non-bf16 parameter
    assert ours.head.weight.dtype == torch.float32
    print(
        f"  v41f_small total={total:,} (backbone {backbone:,} + embed {embed:,} + "
        f"head {head:,} + norm {final_norm:,}); gate-bias buffer={gate_bias_buf}"
    )

    # production v41f_s: static reconciliation even though this PR never instantiates it
    rs = _count(v41f_s())
    assert rs["total_params"] == 904_583_784, rs["total_params"]
    assert rs["active_params_per_token"] == 210_950_760, rs["active_params_per_token"]
    print(f"  v41f_s static total={rs['total_params']:,} active={rs['active_params_per_token']:,}")
