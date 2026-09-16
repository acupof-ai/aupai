"""P0: the assembled Attention block matches the vendored upstream Attention.

Two layer shapes at the v41f_small config, CPU bf16, prefill (start_pos=0):
- layer 0, compress_ratio=0: pure sliding-window, no Compressor/Indexer built,
- layer 1, compress_ratio=2 with it as the kv+index source: window KV concatenated with
  the compressed KV and indexer topk, shared sink, inverse-RoPE tail, grouped wo_a.

Plus an isolation test for the explicit SharedAttnState container that replaces the
upstream process-global `shared_attn`: two independent passes (each its own container)
forwarded interleaved and backwarded in reverse order must give each pass exactly the
gradient it gets alone. A process global would have pass B overwrite pass A's published
compressed KV before A's backward, corrupting A's gradient; per-pass containers cannot.

The upstream decode ring buffer and the fp8/fp4 act_quant calls are out of P0 scope:
training is prefill-only, and on CPU bf16 those quant calls are identities. ref_oracle
stubs act_quant/fp4_act_quant to raise, so here they (and the sparse_attn pure-torch port)
are patched on the loaded reference module to their CPU-bf16 identity/dtype semantics.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).absolute().parent))
from allclose import cmp
from ref_oracle import bf16_args, load_reference

sys.path.insert(0, str(Path(__file__).absolute().parents[2]))
from v41f.attention import Attention, SharedAttnState
from v41f.config import V41FConfig

_SMALL = dict(
    dim=64,
    n_layers=3,
    head_dim=32,
    rope_head_dim=16,
    n_heads=4,
    o_groups=4,
    o_lora_rank=16,
    q_lora_rank=16,
    index_n_heads=2,
    index_head_dim=16,
    index_topk=4,
    compress_ratios=(0, 2, 2),
    kv_source_layers=(1,),
    index_source_layers=(1,),
    max_batch_size=2,
    max_seq_len=64,
    window_size=4,
)


def _patched_reference():
    """Load the vendored model with the prefill-only CPU bf16 quant identities installed.

    The reference binds `from kernel import act_quant, fp4_act_quant, sparse_attn`, so the
    names must be replaced on the model module itself, not on the stubbed kernel module.
    sparse_attn accumulates against the fp32 attn_sink but the real kernel stores
    empty_like(q) (bf16); the ref_oracle pure-torch port omits that output cast, so the
    patched version adds it -- it is the true kernel's activation-dtype boundary.
    """
    model, _ = load_reference()
    model.act_quant = lambda x, *a, **k: (x, None)
    model.fp4_act_quant = lambda x, *a, **k: x

    def _sparse_attn(q, kv, attn_sink, topk_idxs, scale):
        b, m, h, _d = q.shape
        valid = topk_idxs >= 0
        safe = topk_idxs.clamp_min(0)
        gathered = kv[torch.arange(b)[:, None, None], safe]
        scores = torch.einsum("bmhd,bmtd->bmht", q, gathered) * scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        row_max = torch.nan_to_num(scores.amax(dim=-1, keepdim=True), neginf=0.0)
        exp = torch.nan_to_num(torch.exp(scores - row_max), nan=0.0, posinf=0.0, neginf=0.0)
        sink = torch.exp(attn_sink.view(1, 1, h) - row_max.squeeze(-1))
        denom = exp.sum(dim=-1) + sink
        out = torch.einsum("bmht,bmtd->bmhd", exp, gathered) / denom.unsqueeze(-1)
        return out.to(q.dtype)

    model.sparse_attn = _sparse_attn
    model.default_dtype = torch.bfloat16
    return model


def _build_pair(layer_id, seed=7):
    """Identical-weight reference and v41f Attention for one layer, built under the bf16
    default dtype the real Transformer sets (RMSNorm weights and plain buffers are bf16;
    Compressor ratio>1 projections and attn_sink stay fp32 by explicit dtype)."""
    model = _patched_reference()
    args = bf16_args(model, **_SMALL)
    cfg = V41FConfig(**{k: v for k, v in _SMALL.items() if k not in ("max_batch_size", "max_seq_len")})
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        ref = model.Attention(layer_id, args).eval()
        ours = Attention(cfg, layer_id, max_batch_size=_SMALL["max_batch_size"]).eval()
    finally:
        torch.set_default_dtype(prev)

    # ref name -> ours name; the compressor/indexer entries exist only on a ratio>0 source
    name_map = {
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
    ref_p, ours_p = dict(ref.named_parameters()), dict(ours.named_parameters())
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        # fill the REFERENCE weights finite first: upstream allocates most with torch.empty
        # and the checkpoint loader would fill them; uninitialized bf16 can be NaN.
        for rn, on in name_map.items():
            if rn not in ref_p:
                continue
            rw, ow = ref_p[rn], ours_p[on]
            assert rw.dtype == ow.dtype, (rn, rw.dtype, ow.dtype)
            rw.copy_(0.1 * torch.randn(rw.shape, generator=g, dtype=rw.dtype))
            ow.copy_(rw)
        # upstream wo_a is one flat [n_groups*o_lora_rank, in]; ours keeps it grouped
        rw, ow = ref_p["wo_a.weight"], ours_p["oproj.wo_a"]
        assert rw.dtype == ow.dtype and rw.numel() == ow.numel()
        rw.copy_(0.1 * torch.randn(rw.shape, generator=g, dtype=rw.dtype))
        ow.copy_(rw.view_as(ow))
    return ref, ours


def test_attention_ratio2_compressed_prefill():
    torch.manual_seed(0)
    ref, ours = _build_pair(1)
    x = 0.2 * torch.randn(2, 8, _SMALL["dim"])
    with torch.no_grad():
        r = ref(x.bfloat16(), 0)
        want = r[0] if isinstance(r, tuple) else r
        got, state = ours(x.bfloat16())
    max_abs, _ = cmp("Attention ratio2 prefill", got, want, atol=5e-2)
    assert want.shape == (2, 8, _SMALL["dim"])
    # source layer published all three cross-layer tensors
    assert state.compress_kv.shape == (2, 4, _SMALL["head_dim"])
    assert state.index_k.shape == (2, 4, _SMALL["index_head_dim"])
    assert state.topk_idxs.shape == (2, 8, _SMALL["index_topk"])
    print(f"  ratio2 max_abs={max_abs:.4e}")


def test_attention_ratio0_window_only_prefill():
    torch.manual_seed(1)
    ref, ours = _build_pair(0, seed=11)
    assert ours.compressor is None and ours.indexer is None and ours.compress_ratio == 0
    x = 0.2 * torch.randn(2, 8, _SMALL["dim"])
    with torch.no_grad():
        r = ref(x.bfloat16(), 0)
        want = r[0] if isinstance(r, tuple) else r
        got, state = ours(x.bfloat16())
    max_abs, _ = cmp("Attention ratio0 prefill", got, want, atol=5e-2)
    # nothing to publish on a window-only layer
    assert state.compress_kv is None and state.topk_idxs is None
    print(f"  ratio0 max_abs={max_abs:.4e}")


def test_shared_state_two_backwards_do_not_alias():
    """Two prefill passes through source L1 + reuse L2, each with its own SharedAttnState,
    forwarded interleaved and backwarded in reverse order, must produce on the shared L1
    weights the SUM of each pass-alone gradient; and pass B must not overwrite the KV pass
    A published (the upstream global shared_attn failure)."""
    cfg = V41FConfig(**{k: v for k, v in _SMALL.items() if k not in ("max_batch_size", "max_seq_len")})
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        src = Attention(cfg, 1, max_batch_size=2)  # kv + index source
        reuse = Attention(cfg, 2, max_batch_size=2)  # ratio 2, reuses published KV/topk
    finally:
        torch.set_default_dtype(prev)

    xa = (0.2 * torch.randn(2, 8, cfg.dim)).bfloat16()
    xb = (0.3 * torch.randn(2, 8, cfg.dim)).bfloat16()

    def run(x):
        st = SharedAttnState()
        oa, st = src(x, st)
        ob, _ = reuse(x, st)
        return oa.float().square().mean() + ob.float().square().mean(), st

    def grad_alone(x):
        src.zero_grad(set_to_none=True)
        reuse.zero_grad(set_to_none=True)
        loss, _ = run(x)
        loss.backward()
        return {n: p.grad.detach().clone() for n, p in src.named_parameters() if p.grad is not None}

    g_a, g_b = grad_alone(xa), grad_alone(xb)

    # interleaved: forward A, forward B (fresh container), then backward B then A
    src.zero_grad(set_to_none=True)
    reuse.zero_grad(set_to_none=True)
    loss_a, state_a = run(xa)
    comp_a_snapshot = state_a.compress_kv.detach().clone()
    idx_a_snapshot = state_a.index_k.detach().clone()
    loss_b, state_b = run(xb)
    # distinct containers published distinct storage; B did not touch A's published KV
    assert state_a.compress_kv is not state_b.compress_kv
    assert state_a.topk_idxs is not state_b.topk_idxs
    cmp(
        "state A compress_kv not overwritten by pass B",
        state_a.compress_kv.float(),
        comp_a_snapshot.float(),
        atol=0.0,
        rtol=0.0,
    )
    cmp(
        "state A index_k not overwritten by pass B",
        state_a.index_k.float(),
        idx_a_snapshot.float(),
        atol=0.0,
        rtol=0.0,
    )
    loss_b.backward()
    loss_a.backward()

    worst = 0.0
    for n, p in src.named_parameters():
        if p.grad is None:
            continue
        # the interleaved gradient equals the sum of the two independent pass gradients
        d = (p.grad.float() - (g_a[n].float() + g_b[n].float())).abs().max().item()
        worst = max(worst, d)
    assert worst < 1e-3, f"two-backward gradient contamination, max diff {worst}"
    print(f"  two-backward no-alias max diff={worst:.3e}")
