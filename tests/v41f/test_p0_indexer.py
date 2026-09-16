"""P0: CSA2 Indexer level-two scoring/selection and level-one candidate-block
coarse filter match the vendored upstream reference, CPU, no quant/dist/global.

Selection indices must be bitwise equal; scores match to 1e-4. The reference
Indexer hard-calls the SM100 fp4 activation quant, which the oracle stubs to
raise, so for these pure-math paths we temporarily substitute the identity (on
CPU bf16 the quant is numerically the identity). We use non-kv-owning layers and
feed the already-built index keys + RoPE table, exactly as the task scopes.
"""
import contextlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from allclose import cmp
from ref_oracle import bf16_args, load_reference

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch.nn.functional as F

from v41f.indexer import Indexer, select_candidate_blocks


@contextlib.contextmanager
def _identity_fp4(model):
    """Reference forward quantises q/k in place; on CPU bf16 that is a no-op."""
    saved = model.fp4_act_quant
    model.fp4_act_quant = lambda t, *_a, **_k: t
    try:
        yield
    finally:
        model.fp4_act_quant = saved


def _args(model, **over):
    # small single-process shapes; non-kv-owning layers fed index_k directly
    base = dict(
        dim=48, q_lora_rank=32, index_n_heads=4, index_head_dim=16,
        rope_head_dim=8, index_topk=6, n_layers=3,
        compress_ratios=(2, 2, 2), kv_source_layers=(),
        candidate_source_layer=-1, candidate_topk_blocks=0, candidate_block_size=0,
        max_seq_len=64, max_batch_size=3,
    )
    base.update(over)
    return bf16_args(model, **base)


def _freqs(model, rope_head_dim, seqlen, args):
    return model.precompute_freqs_cis(
        rope_head_dim, seqlen, args.original_seq_len, args.rope_theta,
        args.rope_factor, args.beta_fast, args.beta_slow)


def _pair(model, args, layer_id):
    model.default_dtype = torch.bfloat16  # ColumnParallelLinear default storage -> plain F.linear
    ref = model.Indexer(args, layer_id)
    # upstream leaves Indexer weights as torch.empty (checkpoint-filled); seed finite
    # values on the reference, then copy into ours so both share identical weights.
    with torch.no_grad():
        torch.manual_seed(1234)
        ref.wq_b.weight.copy_(torch.randn_like(ref.wq_b.weight) * 0.05)
        ref.weights_proj.weight.copy_(torch.randn_like(ref.weights_proj.weight) * 0.05)
    ours = Indexer(args.dim, args.q_lora_rank, args.index_n_heads,
                   args.index_head_dim, args.rope_head_dim, args.index_topk,
                   args.compress_ratios[layer_id]).bfloat16()
    ours.wq_b.weight.data.copy_(ref.wq_b.weight.data)
    ours.weights_proj.weight.data.copy_(ref.weights_proj.weight.data)
    return ref, ours


def _expected_score(model, ref, x, qr, index_k, qfreq):
    """Reference arithmetic assembled from reference primitives (the oracle)."""
    q = F.linear(qr, ref.wq_b.weight).unflatten(-1, (ref.n_heads, ref.index_head_dim))
    model.apply_rotary_emb(q[..., -ref.rope_head_dim:], qfreq)
    weights = F.linear(x, ref.weights_proj.weight) * (ref.softmax_scale * ref.n_heads ** -0.5)
    sc = torch.einsum("bshd,btd->bsht", q, index_k)
    return (sc.relu_() * weights.unsqueeze(-1)).sum(dim=2)


def test_indexer_prefill_visibility_and_topk():
    model, _ = load_reference()
    torch.manual_seed(0)
    args = _args(model)
    layer_id = 1  # non-owner (kv_source_layers empty), ratio 2
    ref, ours = _pair(model, args, layer_id)
    b, s = 3, 16
    x = (torch.randn(b, s, args.dim) * 0.5).bfloat16()
    qr = (torch.randn(b, s, args.q_lora_rank) * 0.5).bfloat16()
    t = s // args.compress_ratios[layer_id]
    index_k = (torch.randn(b, t, args.index_head_dim) * 0.5).bfloat16()
    freqs = _freqs(model, args.rope_head_dim, s, args)

    with _identity_fp4(model):
        ref.freqs_cis = freqs
        model.shared_attn.index_k = index_k
        ref_out = ref(x, qr, None, 0, 0)
    our_out = ours(x, qr, index_k, freqs, start_pos=0, offset=0)

    # numeric scoring vs reference-assembled oracle
    exp = _expected_score(model, ref, x, qr, index_k, freqs)
    got = ours.score(x, qr, index_k, freqs).float()
    cmp("indexer score prefill", got, exp, atol=1e-4)
    assert torch.equal(our_out, ref_out), "prefill selected indices differ"

    # visibility: query 0 sees 0 reachable positions -> its whole row is -1
    assert (ref_out[:, 0] == -1).all(), "first query must reach 0 compressed positions"
    # every returned visible index is within the per-query causal reach
    ratio = args.compress_ratios[layer_id]
    reach = torch.arange(1, s + 1) // ratio
    for qi in range(s):
        for bi in range(b):
            vis = ref_out[bi, qi]
            assert (vis[vis >= 0] < reach[qi]).all(), f"b{bi} q{qi} sees a future pos"
    # ascending (position re-sort) within each query's valid entries
    for bi in range(b):
        for qi in range(s):
            v = ref_out[bi, qi]
            v = v[v >= 0]
            assert torch.equal(v, v.sort().values), "topk not re-sorted into position order"


def test_indexer_decode_offset():
    model, _ = load_reference()
    torch.manual_seed(2)
    args = _args(model)
    layer_id = 1
    ref, ours = _pair(model, args, layer_id)
    ratio = args.compress_ratios[layer_id]
    start, s, b = 8, 4, 2
    end = start + s
    x = (torch.randn(b, s, args.dim) * 0.5).bfloat16()
    qr = (torch.randn(b, s, args.q_lora_rank) * 0.5).bfloat16()
    t = end // ratio
    index_k = (torch.randn(b, t, args.index_head_dim) * 0.5).bfloat16()
    full = _freqs(model, args.rope_head_dim, end, args)
    qfreq = full[start:end]
    offset = 0

    with _identity_fp4(model):
        ref.freqs_cis = full
        model.shared_attn.index_k = index_k
        ref_out = ref(x, qr, None, start, offset)
    our_out = ours(x, qr, index_k, qfreq, start_pos=start, offset=offset)
    cmp("indexer score decode",
        ours.score(x, qr, index_k, qfreq).float(),
        _expected_score(model, ref, x, qr, index_k, qfreq).float(), atol=1e-4)
    assert torch.equal(our_out, ref_out), "decode indices differ"
    # topk smaller than reachable -> no -1 padding in decode
    assert (ref_out >= 0).all()


def test_select_candidate_blocks_matches_ref():
    model, _ = load_reference()
    torch.manual_seed(3)
    block, topk_blocks = 4, 3
    # prefill-style: [b, query, positions]; per-query visible-length lens [q,1]
    b, q, width = 2, 9, 20
    logits = torch.randn(b, q, width)
    lens = ((torch.arange(1, q + 1) // 2).clamp_min(1)).unsqueeze(-1)  # [q,1]
    logits = logits.masked_fill(torch.arange(width) >= lens, -torch.inf)
    got = select_candidate_blocks(logits, lens, topk_blocks, block)
    want = model.select_candidate_blocks(logits, lens, topk_blocks, block)
    assert got.dtype == torch.bool and got.shape == want.shape
    assert torch.equal(got, want), "candidate mask (prefill lens) differs"

    # decode-style: scalar compress_lens
    l2 = torch.randn(q, width)
    assert torch.equal(select_candidate_blocks(l2, width, topk_blocks, block),
                       model.select_candidate_blocks(l2, width, topk_blocks, block))

    # semantics: each query's newest reachable block is pinned in even if outscored.
    # (the mask covers the whole block incl. positions past the current visible length;
    # those score -inf from the visibility mask anyway, so they never get selected.)
    newest_block = (lens.squeeze(-1) - 1) // block  # [q]
    for qi in range(q):
        p = int(newest_block[qi]) * block
        assert got[:, qi, p:p + block].any(-1).all(), f"q{qi} newest block not pinned"

    # only one block reachable (lens=1): later -inf blocks dropped, block0 pinned
    far = torch.full((1, 1, width), -torch.inf)
    far[..., 0] = 1.0
    m = model.select_candidate_blocks(far, 1, topk_blocks, block)
    assert m[..., :block].any() and not m[..., block:].any(), \
        "unreachable -inf blocks must be dropped"


def test_two_level_source_and_consumer():
    model, _ = load_reference()
    torch.manual_seed(4)
    # source layer 0, consumer layer 2; both non-kv-owning
    args = _args(model, candidate_source_layer=0,
                 candidate_topk_blocks=3, candidate_block_size=4)
    b, s = 3, 24
    ratio = 2
    t = s // ratio

    def build(layer_id):
        ref, ours = _pair(model, args, layer_id)
        x = (torch.randn(b, s, args.dim) * 0.4).bfloat16()
        qr = (torch.randn(b, s, args.q_lora_rank) * 0.4).bfloat16()
        index_k = (torch.randn(b, t, args.index_head_dim) * 0.4).bfloat16()
        freqs = _freqs(model, args.rope_head_dim, s, args)
        return ref, ours, x, qr, index_k, freqs

    # --- source: computes candidates, does not mask its own topk ---
    sref, sour, x, qr, index_k, freqs = build(0)
    assert sref.is_candidate_source and not sref.uses_candidates
    with _identity_fp4(model):
        sref.freqs_cis = freqs
        model.shared_attn.index_k = index_k
        src_out = sref(x, qr, None, 0, 0)
    cand = model.shared_attn.candidates
    exp_cand = model.select_candidate_blocks(
        _expected_score(model, sref, x, qr, index_k, freqs).masked_fill(
            torch.arange(t) >= (torch.arange(1, s + 1) // ratio).unsqueeze(-1), -torch.inf),
        (torch.arange(1, s + 1) // ratio).unsqueeze(-1), 3, 4)
    assert torch.equal(cand, exp_cand), "source candidate mask differs"
    src_ours = sour(x, qr, index_k, freqs, start_pos=0, offset=0)
    assert torch.equal(src_ours, src_out), "source selection differs"

    # --- consumer: masks to source candidates ---
    cref, cour, x2, qr2, index_k2, freqs2 = build(2)
    assert cref.uses_candidates and not cref.is_candidate_source
    # give the consumer deterministic candidates (a few positions per query)
    fixed_cand = torch.zeros(b, s, t, dtype=torch.bool)
    for qi in range(s):
        fixed_cand[:, qi, max(0, qi // ratio - 2): qi // ratio + 1] = True
    model.shared_attn.candidates = fixed_cand
    with _identity_fp4(model):
        cref.freqs_cis = freqs2
        model.shared_attn.index_k = index_k2
        c_ref_out = cref(x2, qr2, None, 0, 0)
    c_our_out = cour(x2, qr2, index_k2, freqs2, start_pos=0, offset=0,
                     candidates=fixed_cand)
    assert torch.equal(c_our_out, c_ref_out), "consumer candidate-masked selection differs"
    # Faithful semantics: the consumer only *prefers* candidate positions (others are
    # -inf). For a query with >= index_topk reachable candidate positions, every pick
    # must be a candidate; with fewer, the spare topk slots are arbitrary reachable
    # (-inf-filled) positions, which upstream still returns by idxs<lens.
    topk = args.index_topk
    for bi in range(b):
        for qi in range(s):
            reach_n = (qi + 1) // ratio  # == forward's compress_lens at this query
            cand_reachable = fixed_cand[bi, qi, :reach_n].sum().item()
            picks = c_ref_out[bi, qi]
            picks = picks[picks >= 0]
            if cand_reachable >= topk:
                assert all(fixed_cand[bi, qi, p] for p in picks.tolist()), \
                    f"b{bi} q{qi}: picked a non-candidate despite enough candidates"
            # every pick is always a reachable position
            assert (picks < reach_n).all(), f"b{bi} q{qi}: picked an unreachable pos"
