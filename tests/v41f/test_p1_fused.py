"""Track P parity gates: the fused attention path, the real-valued RoPE, the stacked MoE, and
the compile-cleanliness of the v42 forward.

  - attn_impl "fused" == "chunked" on the same packed batch, logits and every gradient, with
    the KL indexer loss on (the fused path must hand it the same lse);
  - rope_impl "real" == "complex" to bf16 rounding on the same model;
  - moe_stacked: loop path equals the per-expert layout bit-for-bit after the loader remap,
    and the remap round-trips both ways;
  - torch._dynamo.explain over the tiny V42LM forward: the graph-break count is printed and
    bounded (a regression that adds a break turns this red).

Run:  python3 tests/v41f/test_p1_fused.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
sys.path.insert(0, str(Path(__file__).absolute().parent))
from allclose import grad_rtol  # noqa: E402
from test_p1_docpack import _batch, _cfg, _grads, _model  # noqa: E402
from v41f import docpack  # noqa: E402
from v41f.moe import MoE  # noqa: E402
from v41f.rope import apply_rotary_emb, apply_rotary_real, precompute_freqs_cis, rope_cos_sin  # noqa: E402

MAX_GRAPH_BREAKS = 0


def _compare(a, b, ids, cu, tag, tol=1e-4):
    la, ga = _grads(a, ids, cu)
    lb, gb = _grads(b, ids, cu)
    d = (la - lb).abs().max().item()
    assert d < tol, f"{tag}: logits differ by {d:.3e}"
    assert ga.keys() == gb.keys(), f"{tag}: grad sets differ {sorted(ga.keys() ^ gb.keys())}"
    worst = 0.0
    for n in ga:
        rel = ((ga[n] - gb[n]).norm() / ga[n].norm().clamp_min(1e-12)).item()
        worst = max(worst, rel)
        ntol = max(tol, grad_rtol(n))
        assert rel < ntol, f"{tag}: grad {n} differs rel {rel:.3e} (tol {ntol:.0e})"
    return d, worst, len(ga)


def test_fused_equals_chunked():
    ids, cu = _batch()
    old = docpack.ATTN_CHUNK
    docpack.ATTN_CHUNK = 5
    try:
        for mode in ("off", "kl"):
            chk = _model(_cfg(attn_impl="chunked", indexer_train_mode=mode))
            fus = _model(_cfg(attn_impl="fused", indexer_train_mode=mode))
            fus.load_state_dict(chk.state_dict())
            for c in (cu, None):
                d, w, n = _compare(chk, fus, ids, c, f"fused vs chunked ({mode}, cu={'packed' if c is not None else None})")
            if mode == "kl":
                chk(ids, cu=cu)
                fus(ids, cu=cu)
                a, b = float(chk.indexer_loss), float(fus.indexer_loss)
                assert abs(a - b) < 1e-5, f"indexer KL differs: {a} vs {b}"
            print(f"  fused == chunked ({mode}): logits {d:.1e}, worst grad rel {w:.1e}, {n} grads")
        ck = _model(_cfg(attn_impl="fused", indexer_train_mode="kl", block_ckpt=True))
        ck.load_state_dict(fus.state_dict())
        d, w, n = _compare(fus, ck, ids, cu, "block_ckpt vs none", tol=1e-6)
        print(f"  block_ckpt == none: logits {d:.1e}, worst grad rel {w:.1e}")
    finally:
        docpack.ATTN_CHUNK = old


def test_rope_real_equals_complex():
    torch.manual_seed(0)
    fc = precompute_freqs_cis(16, 40)
    for shape in ((2, 40, 16), (2, 40, 4, 16)):
        x = torch.randn(*shape)
        a = apply_rotary_emb(x.clone(), fc)
        cos, sin = rope_cos_sin(fc)
        b = apply_rotary_real(x.clone(), cos, sin)
        assert torch.allclose(a, b, atol=1e-5), f"rope real vs complex {shape}: {(a - b).abs().max()}"
        ai = apply_rotary_emb(a.clone(), fc, inverse=True)
        bi = apply_rotary_real(b.clone(), cos, sin, inverse=True)
        assert torch.allclose(ai, x, atol=1e-5) and torch.allclose(bi, x, atol=1e-5), "inverse does not undo"
    # per-token positions [b,s,d/2], the packed-row form
    pos = torch.randint(0, 40, (2, 40))
    fcp = fc[pos]
    x = torch.randn(2, 40, 4, 16)
    a = apply_rotary_emb(x.clone(), fcp)
    b = apply_rotary_real(x.clone(), *rope_cos_sin(fcp))
    assert torch.allclose(a, b, atol=1e-5)
    # whole model, bf16-rounded reference: the two impls must agree at the bf16 floor
    ids, cu = _batch()
    ref = _model(_cfg(attn_impl="chunked"))
    real = _model(_cfg(attn_impl="chunked", rope_impl="real"))
    real.load_state_dict(ref.state_dict())
    d, w, n = _compare(ref, real, ids, cu, "rope real vs complex (model)", tol=2e-4)
    print(f"  rope real == complex: model logits {d:.1e}, worst grad rel {w:.1e}")


def test_stacked_moe_equals_loop_and_remaps():
    torch.manual_seed(0)
    kw = dict(dim=32, n_routed_experts=6, n_activated_experts=2, moe_inter_dim=24, swiglu_limit=10.0)
    loop = MoE(**kw)
    stk = MoE(**kw, stacked=True)
    missing = stk.load_state_dict(loop.state_dict(), strict=True)  # remap per-expert -> stacked
    assert not missing.missing_keys and not missing.unexpected_keys, missing
    assert torch.equal(stk.w1[3], loop.experts[3].w1.weight)
    x = torch.randn(2, 7, 32)
    a, b = loop(x), stk(x)
    assert torch.equal(a, b), f"stacked vs loop differ by {(a - b).abs().max()}"
    # gradients agree too
    a.square().sum().backward()
    b.square().sum().backward()
    assert torch.allclose(stk.w2.grad[1], loop.experts[1].w2.weight.grad, atol=1e-6)
    # round trip back
    loop2 = MoE(**kw)
    loop2.load_state_dict(stk.state_dict(), strict=True)
    assert torch.equal(loop2.experts[5].w3.weight, stk.w3[5])
    keys = set(stk.state_dict())
    assert {"w1", "w3", "w2"} <= keys and not any(k.startswith("experts.") for k in keys), sorted(keys)
    # the fused activation is the reference clamp order
    g, u, w = torch.randn(5, 8) * 20, torch.randn(5, 8) * 20, torch.rand(5, 1)
    from v41f.moe import swiglu_clamp
    ref = w * (F.silu(g.clamp(max=10.0)) * u.clamp(-10.0, 10.0))
    assert torch.equal(swiglu_clamp(g, u, w, 10.0), ref)
    # the grouped_mm dispatch (GPU path) against the loop, on CPU: bf16 GEMMs vs fp32 linears
    MoE.grouped_on_cpu = True
    try:
        xb = x.to(torch.bfloat16)
        stk_b, loop_b = stk.to(torch.bfloat16), loop.to(torch.bfloat16)
        c = stk_b(xb).float()
    finally:
        MoE.grouped_on_cpu = False
    r = loop_b(xb).float()
    rel = ((c - r).norm() / r.norm()).item()
    assert rel < 2e-2, f"grouped vs loop rel {rel:.3e}"
    print(f"  stacked MoE == loop (bit-equal), remap both ways, swiglu_clamp == reference; grouped_mm rel {rel:.1e}")


def test_deepgemm_layout_and_emulated_fp8():
    """moe_gemm deepgemm (torch emulation off CUDA): the padded contiguous layout and the fp8
    quantize/dequantize path agree with grouped_mm bf16 within fp8 tolerance, forward and grads."""
    from v41f import deepgemm_moe as dg

    torch.manual_seed(0)
    counts = torch.tensor([3, 0, 130, 128, 1])
    m_pad = dg.padded_m(int(counts.sum()), counts.numel())
    assert m_pad >= 128 + 0 + 256 + 128 + 128 and m_pad % 128 == 0, m_pad  # a shape, >= the used length
    pos, mi = dg.padded_layout(counts, m_pad)
    assert mi.numel() == m_pad and pos.numel() == counts.sum()
    # segments: e0 rows 0-127 (3 real), e1 none, e2 128-383 (130 real), e3 384-511, e4 512-639 (1 real)
    assert (mi[:3] == 0).all() and (mi[3:128] == -1).all()
    assert (mi[128:258] == 2).all() and pos[3].item() == 128 and (mi[384:512] == 3).all() and mi[512].item() == 4
    assert (mi[640:] == -1).all(), "the unused tail is padding"
    assert pos.unique().numel() == pos.numel() and (mi[pos] >= 0).all(), "every real row has its own padded row"
    # fixed padded length across routings: the same shape for any counts of the same total
    assert dg.padded_m(262, 5) == m_pad and dg.padded_layout(torch.tensor([262, 0, 0, 0, 0]), m_pad)[1].numel() == m_pad
    kw = dict(dim=256, n_routed_experts=4, n_activated_experts=2, moe_inter_dim=128, swiglu_limit=10.0)
    ref = MoE(**kw, stacked=True).to(torch.bfloat16)
    dgm = MoE(**kw, stacked=True, moe_gemm="deepgemm").to(torch.bfloat16)
    dgm.load_state_dict(ref.state_dict())
    x = (torch.randn(3, 70, 256) * 0.5).to(torch.bfloat16)
    MoE.grouped_on_cpu = True
    try:
        xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
        ya, yb = ref(xa).float(), dgm(xb).float()
        ya.square().sum().backward()
        yb.square().sum().backward()
    finally:
        MoE.grouped_on_cpu = False
    rel = ((ya - yb).norm() / ya.norm()).item()
    grel = ((xa.grad.float() - xb.grad.float()).norm() / xa.grad.float().norm()).item()
    wrel = max(((getattr(ref, n).grad.float() - getattr(dgm, n).grad.float()).norm()
                / getattr(ref, n).grad.float().norm()).item() for n in ("w1", "w3", "w2"))
    print(f"  deepgemm (emulated fp8) vs grouped_mm bf16: out rel {rel:.2e}, dx rel {grel:.2e}, dw rel {wrel:.2e}")
    assert rel < 5e-2 and grel < 8e-2 and wrel < 8e-2, (rel, grel, wrel)


def test_qk_norm_and_softcap_switches():
    """Both deviations default off and reproduce the reference path bit-for-bit; on, they change the
    logits; a checkpoint saved without qk_norm loads into a qk_norm model (identity weight) and the
    softcap at C=1e6 is the uncapped softmax to fp32 rounding."""
    ids, cu = _batch()
    base = _model(_cfg(attn_impl="fused", indexer_train_mode="kl"))
    off = _model(_cfg(attn_impl="fused", indexer_train_mode="kl", qk_norm=False, attn_logit_softcap=0.0))
    off.load_state_dict(base.state_dict())
    d, w, _ = _compare(base, off, ids, cu, "switches off == reference path", tol=1e-12)
    assert d == 0.0 and w == 0.0, (d, w)
    # softcap: huge C == off; small C changes the logits and every attention path agrees fused vs chunked
    hi = _model(_cfg(attn_impl="fused", indexer_train_mode="kl", attn_logit_softcap=1e6))
    hi.load_state_dict(base.state_dict())
    d, w, _ = _compare(base, hi, ids, cu, "softcap 1e6 == off", tol=1e-4)
    capf = _model(_cfg(attn_impl="fused", indexer_train_mode="kl", attn_logit_softcap=0.5))
    capc = _model(_cfg(attn_impl="chunked", indexer_train_mode="kl", attn_logit_softcap=0.5))
    capf.load_state_dict(base.state_dict())
    capc.load_state_dict(base.state_dict())
    _compare(capf, capc, ids, cu, "softcap 0.5 fused == chunked", tol=1e-3)  # tanh at C=0.5 amplifies fp32 rounding in the LSE merge
    lb, _ = _grads(base, ids, cu)
    lc, _ = _grads(capf, ids, cu)
    assert (lb - lc).abs().max() > 1e-3, "softcap 0.5 must change the logits"
    # qk_norm: old checkpoint round-trips (missing head norms filled with ones), output changes
    qk = _model(_cfg(attn_impl="fused", indexer_train_mode="kl", qk_norm=True))
    res = qk.load_state_dict(base.state_dict(), strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    assert torch.equal(qk.layers[0].attn.qproj.head_norm.weight, torch.ones(qk.layers[0].attn.head_dim))
    names = set(qk.state_dict())
    assert any(n.endswith("qproj.head_norm.weight") for n in names) and any(n.endswith("indexer.q_head_norm.weight") for n in names)
    lq, _ = _grads(qk, ids, cu)
    assert (lb - lq).abs().max() > 1e-3, "qk_norm must change the logits"
    # and the qk_norm model's own state dict round-trips into a fresh qk_norm model
    qk2 = _model(_cfg(attn_impl="fused", indexer_train_mode="kl", qk_norm=True))
    qk2.load_state_dict(qk.state_dict(), strict=True)
    _compare(qk, qk2, ids, cu, "qk_norm state_dict round trip", tol=1e-12)
    print("  switches off == reference (bit-exact); softcap 1e6 == off; softcap fused == chunked; qk_norm loads an old ckpt")


def test_forward_compiles_without_graph_breaks():
    import torch._dynamo as dynamo

    from v41f.lm import V42LM

    ids, cu = _batch()
    cfg = _cfg(attn_impl="fused", rope_impl="real", indexer_train_mode="kl", moe_stacked=True,
               qk_norm=True, attn_logit_softcap=30.0)  # both deviations on: they must not break the graph
    torch.manual_seed(0)
    m = V42LM(cfg, max_batch_size=2).float().train()
    m.record_qk_stats = True  # the health probe's side effects must not break the graph
    torch._dynamo.mark_dynamic(cu, 0)
    dynamo.reset()
    MoE.grouped_on_cpu = True  # trace the GPU dispatch path, not the CPU per-expert loop
    try:
        ex = dynamo.explain(m)(ids, None, cu, None)
    finally:
        MoE.grouped_on_cpu = False
    reasons = [str(r.reason)[:90] for r in ex.break_reasons]
    print(f"  dynamo.explain: {ex.graph_count} graph(s), {ex.graph_break_count} break(s)")
    for r in reasons:
        print(f"    break: {r}")
    assert ex.graph_break_count <= MAX_GRAPH_BREAKS, f"graph breaks: {reasons}"
    assert m.qk_stats is not None and len(m.qk_stats) == cfg.n_layers, "qk stats not replayed under dynamo"


TESTS = [test_fused_equals_chunked, test_rope_real_equals_complex, test_stacked_moe_equals_loop_and_remaps,
         test_deepgemm_layout_and_emulated_fp8, test_qk_norm_and_softcap_switches,
         test_forward_compiles_without_graph_breaks]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"ok   {t.__name__}")
    print(f"fused gates: {len(TESTS)}/{len(TESTS)} passed")
