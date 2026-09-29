"""Packed-row training path for v41f: document layout, chunked exact sparse attention, and
the indexer's KL alignment loss.

The reference forward assumes one document per row. Training packs several documents into
each 4096-token row, and V4.1 trains with sample-level attention masking (tech report
§4.2.2), so every position-dependent piece restarts at a document start: RoPE positions,
the sliding window, the compressor's m-token groups, and which compressed entries a query
may see. `doc_layout` turns train.doc_cu_seqlens into the per-token (doc, pos, doclen)
the attention reads.

`windowed_sparse_attn` computes the reference single softmax over [window keys ; selected
compressed entries ; sink] one query chunk at a time. The reference `sparse_attn` gathers
[b,s,k,d] per layer (5.4 GB at b4 s4096 k640 d256, bf16) and autograd keeps it; here each
chunk runs under activation checkpointing, so a layer keeps only its inputs for backward
and the per-chunk scores are rebuilt in backward. Within a chunk the compressed entries are
sliced to the causal prefix any query of the chunk can see (entries are position-ordered,
so an entry whose last token precedes the chunk end is among the first c1//m), which
halves the entry work on average.

`indexer_kl` is the DSA-lineage indexer objective (DeepSeek-V3.2 §2.1): the target is the
main attention's probability mass on each selected entry, summed over heads and
L1-normalised over the selected set; the indexer distribution is its softmax over the same
set; the loss is KL(target || indexer). Indexer inputs are detached, so this loss trains
only the indexer projections and the index-key projection, and the main CE never reaches
them.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ATTN_CHUNK = 512  # query rows per chunk; tests shrink it to exercise the chunk seams


def doc_layout(cu, b: int, s: int, device):
    """cu_seqlens over the flattened b*s stream (None = one document per row) ->
    (doc [b,s] global document id, pos [b,s] position inside its document, doclen [b,s],
    cu_docs int32 the merged cu_seqlens the flash window branch takes).

    CONTRACT: a given `cu` is sorted, unique, starts at 0, ends at b*s and contains every row
    start -- exactly what train.doc_cu_seqlens produces (it unions the row starts itself). It is
    trusted here rather than re-merged: `unique` has a data-dependent output shape and was one
    torch.compile graph break per forward."""
    rows = torch.arange(0, b * s + 1, s, device=device, dtype=torch.long)
    cu = rows if cu is None else cu.to(device=device, dtype=torch.long)
    i = torch.arange(b * s, device=device)
    d = torch.searchsorted(cu, i, right=True) - 1
    pos = i - cu[d]
    doclen = cu[d + 1] - cu[d]
    return d.view(b, s), pos.view(b, s), doclen.view(b, s), cu.to(torch.int32)


def entry_visibility(e_valid, e_doc, e_last, doc):
    """[b,n] entry metadata x [b,s] query docs -> [b,s,n] bool: the entry is complete, in
    the query's document, and its last token is at or before the query (the reference
    `(t+1)//m` rule, per document)."""
    t = torch.arange(doc.size(1), device=doc.device)
    return (e_valid[:, None, :] & (e_doc[:, None, :] == doc[:, :, None])
            & (e_last[:, None, :] <= t[None, :, None]))


def _sel_mask(idx, nv: int):
    """[b,q,k] entry indices (-1 = empty) -> [b,q,nv] bool. Empty slots write into a spare
    column that is dropped, so they can never clear a real selection of entry 0."""
    spare = torch.where(idx >= 0, idx.long(), nv)
    m = torch.zeros(*idx.shape[:2], nv + 1, dtype=torch.bool, device=idx.device)
    return m.scatter_(-1, spare, True)[..., :nv]


def cap(sc, softcap: float):
    """cfg.attn_logit_softcap: C * tanh(sc / C) on scaled scores, identity at 0. A deviation from the
    reference softmax, applied to every score path but never to the sink."""
    return softcap * torch.tanh(sc / softcap) if softcap else sc


def _attn_chunk(qc, kvw, compc, idxc, sink, docq, dock, c0: int, k0: int, window: int, scale: float,
                softcap: float = 0.0):
    b, q, h, _ = qc.shape
    t = torch.arange(c0, c0 + q, device=qc.device)
    j = torch.arange(k0, k0 + kvw.size(1), device=qc.device)
    mw = (j[None, :] <= t[:, None]) & (t[:, None] - j[None, :] < window)
    mw = mw[None] & (docq[:, :, None] == dock[:, None, :])  # [b,q,kw]
    parts = [cap(torch.einsum("bqhd,bkd->bhqk", qc, kvw).float() * scale, softcap).masked_fill(~mw[:, None], float("-inf"))]
    if compc is not None:
        sel = _sel_mask(idxc, compc.size(1))
        parts.append(cap(torch.einsum("bqhd,bnd->bhqn", qc, compc).float() * scale, softcap)
                     .masked_fill(~sel[:, None], float("-inf")))
    parts.append(sink.float().view(1, h, 1, 1).expand(b, h, q, 1))
    sc = torch.cat(parts, -1)
    lse = torch.logsumexp(sc, -1)
    p = torch.exp(sc - lse[..., None])
    kw = kvw.size(1)
    o = torch.einsum("bhqk,bkd->bqhd", p[..., :kw].to(kvw.dtype), kvw)
    if compc is not None:
        o = o + torch.einsum("bhqn,bnd->bqhd", p[..., kw:-1].to(compc.dtype), compc)
    return o, lse


def windowed_sparse_attn(q, kv, comp, comp_idx, sink, scale: float, doc, window: int, m: int, softcap: float = 0.0):
    """q [b,s,h,d]; kv [b,s,d] this layer's window KV (MQA); comp [b,n,d] RoPE'd compressed
    entries or None; comp_idx [b,s,k] selected entry indices (-1 = empty) or None; sink [h];
    doc [b,s]; m the entries' compression ratio. Returns (o [b,s,h,d], lse [b,h,s] detached).

    Exactly the reference single softmax: the sink sits in the denominator with no value."""
    b, s, h, d = q.shape
    outs, lses = [], []
    for c0 in range(0, s, ATTN_CHUNK):
        c1 = min(s, c0 + ATTN_CHUNK)
        k0 = max(0, c0 - window + 1)
        compc = idxc = None
        if comp is not None:
            nv = min(comp.size(1), c1 // m)
            if nv > 0:
                compc, idxc = comp[:, :nv], comp_idx[:, c0:c1]
        args = (q[:, c0:c1], kv[:, k0:c1], compc, idxc, sink, doc[:, c0:c1], doc[:, k0:c1])
        if torch.is_grad_enabled():
            o, lse = checkpoint(_attn_chunk, *args, c0, k0, window, scale, softcap, use_reentrant=False)
        else:
            o, lse = _attn_chunk(*args, c0, k0, window, scale, softcap)
        outs.append(o)
        lses.append(lse)
    return torch.cat(outs, 1), torch.cat(lses, -1).detach()


@torch.no_grad()
def select_visible(indexer, x, qr, index_k, freqs, visible, m: int):
    """Hard top-k per query over the visible entries, chunked and without a graph.
    Returns [b,s,K] int32 entry indices in ascending order, -1 where fewer than K entries
    are visible; K = min(index_topk, n)."""
    b, s, _ = x.shape
    n = index_k.size(1)
    kk = min(indexer.index_topk, n)
    out = torch.full((b, s, kk), -1, dtype=torch.int32, device=x.device)
    for c0 in range(0, s, ATTN_CHUNK):
        c1 = min(s, c0 + ATTN_CHUNK)
        nv = min(n, c1 // m)
        if nv == 0:
            continue
        vis = visible[:, c0:c1, :nv]
        sc = indexer.score(x[:, c0:c1], qr[:, c0:c1], index_k[:, :nv], freqs[:, c0:c1])
        sc = sc.masked_fill(~vis, float("-inf"))
        k = min(kk, nv)
        idx = sc.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values
        out[:, c0:c1, :k] = torch.where(vis.gather(-1, idx), idx, -1).int()
    return out


def _kl_chunk(xc, qrc, ikc, fc, tgt, sel, has, indexer):
    ps = indexer.score(xc, qrc, ikc, fc).masked_fill(~sel, float("-inf"))
    ps = torch.where(has[..., None], ps, torch.zeros_like(ps))  # empty rows: no NaN in backward
    logq = F.log_softmax(ps, -1)
    use = sel & (tgt > 0)
    kl = torch.where(use, tgt * (tgt.clamp_min(1e-30).log() - logq), torch.zeros_like(logq))
    return (kl.sum(-1) * has).sum()


def indexer_kl(indexer, x, qr, index_k, freqs, q, comp, comp_idx, lse, scale: float, m: int, softcap: float = 0.0):
    """Sum over queries of KL(main-attention target || indexer) on the selected entries, and
    the number of queries that had a target. Only `indexer` and `index_k`'s producer get
    gradient: x, qr, q, comp and lse are used detached."""
    b, s, h, _ = q.shape
    n = comp.size(1)
    tot = x.new_zeros((), dtype=torch.float32)
    cnt = x.new_zeros((), dtype=torch.float32)
    for c0 in range(0, s, ATTN_CHUNK):
        c1 = min(s, c0 + ATTN_CHUNK)
        nv = min(n, c1 // m)
        if nv == 0:
            continue
        sel = _sel_mask(comp_idx[:, c0:c1], nv)
        with torch.no_grad():
            sc = cap(torch.einsum("bqhd,bnd->bhqn", q[:, c0:c1].detach(), comp[:, :nv].detach()).float() * scale, softcap)
            p = torch.exp(sc - lse[:, :, c0:c1, None]).masked_fill(~sel[:, None], 0.0).sum(1)
            tsum = p.sum(-1)
            has = tsum > 0
            tgt = p / tsum.clamp_min(1e-30)[..., None]
        args = (x[:, c0:c1].detach(), qr[:, c0:c1].detach(), index_k[:, :nv], freqs[:, c0:c1], tgt, sel, has)
        if torch.is_grad_enabled():
            tot = tot + checkpoint(_kl_chunk, *args, indexer, use_reentrant=False)
        else:
            tot = tot + _kl_chunk(*args, indexer)
        cnt = cnt + has.sum()
    return tot, cnt


# ------------------------------------------------------------------ fused path (attn_impl "fused")
# The window branch is one flash-attn varlen call over the whole packed row (causal, window
# 128, GQA h:1, document boundaries from cu_docs); the selected-entry branch stays chunked and
# checkpointed; the two are joined with the sink by fp32 log-sum-exp, which is the reference
# single softmax exactly (softmax over a union of sets = LSE-weighted mix of the parts).

try:
    from flash_attn.cute import interface as _fa  # flash-attn 4, image-baked on the pod
    HAS_FA_CUTE = True
except ImportError:  # CPU: the pure-torch window below carries the parity tests
    _fa = None
    HAS_FA_CUTE = False


class _FlashWindow(torch.autograd.Function):
    """flash_attn.cute varlen forward/backward with return_lse; q [N,h,d], kv [N,1,d] used as
    both K and V (the model's MQA latent is one vector). lse comes back [h, N]."""

    @staticmethod
    def forward(ctx, q, kv, cu, max_len, window, scale, softcap=0.0):
        # flash_attn.cute's softcap is the same C*tanh(scaled score / C) as docpack.cap
        o, lse = _fa._flash_attn_fwd(
            q, kv, kv, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max_len, max_seqlen_k=max_len,
            causal=True, window_size_left=window - 1, window_size_right=0,
            softmax_scale=scale, softcap=softcap or None, return_lse=True)
        ctx.save_for_backward(q, kv, o, lse)
        ctx.meta = (cu, max_len, window, scale, softcap)
        return o, lse

    @staticmethod
    def backward(ctx, go, glse):
        q, kv, o, lse = ctx.saved_tensors
        cu, max_len, window, scale, softcap = ctx.meta
        dq, dk, dv = _fa._flash_attn_bwd(
            q, kv, kv, o, go.to(o.dtype), lse, dlse=glse.contiguous() if glse is not None else None,
            softmax_scale=scale, softcap=float(softcap), causal=True, window_size_left=window - 1, window_size_right=0,
            cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=max_len, max_seqlen_k=max_len)
        return dq, dk + dv, None, None, None, None, None


def _window_torch(q, kv, doc, window, scale, softcap: float = 0.0):
    """Pure-torch window branch (CPU / no flash): chunked masked softmax without the sink.
    Returns o [b,s,h,d] fp32-accumulated in q's dtype and lse [b,h,s] fp32."""
    b, s, h, _ = q.shape
    outs, lses = [], []
    for c0 in range(0, s, ATTN_CHUNK):
        c1 = min(s, c0 + ATTN_CHUNK)
        k0 = max(0, c0 - window + 1)
        qc, kvw = q[:, c0:c1], kv[:, k0:c1]
        t = torch.arange(c0, c1, device=q.device)
        j = torch.arange(k0, c1, device=q.device)
        mw = (j[None, :] <= t[:, None]) & (t[:, None] - j[None, :] < window)
        mw = mw[None] & (doc[:, c0:c1, None] == doc[:, None, k0:c1])
        sc = cap(torch.einsum("bqhd,bkd->bhqk", qc, kvw).float() * scale, softcap).masked_fill(~mw[:, None], float("-inf"))
        lse = torch.logsumexp(sc, -1)
        p = torch.exp(sc - lse[..., None])
        outs.append(torch.einsum("bhqk,bkd->bqhd", p.to(kvw.dtype), kvw))
        lses.append(lse)
    return torch.cat(outs, 1), torch.cat(lses, -1)


def _entry_chunk(qc, compc, idxc, scale: float, softcap: float = 0.0):
    """Selected-entry branch for one query chunk: normalized output and its lse (no sink)."""
    sel = _sel_mask(idxc, compc.size(1))
    sc = cap(torch.einsum("bqhd,bnd->bhqn", qc, compc).float() * scale, softcap).masked_fill(~sel[:, None], float("-inf"))
    lse = torch.logsumexp(sc, -1)                       # -inf where the query selected nothing
    p = torch.exp(sc - lse[..., None].nan_to_num(neginf=0.0))
    p = torch.where(torch.isfinite(lse)[..., None], p, torch.zeros_like(p))
    return torch.einsum("bhqn,bnd->bqhd", p.to(compc.dtype), compc), lse


def fused_sparse_attn(q, kv, comp, comp_idx, sink, scale: float, doc, cu_docs, window: int, m: int,
                      softcap: float = 0.0):
    """Same contract as windowed_sparse_attn (o [b,s,h,d], lse [b,h,s] detached), computed as
    window(flash) + entries(chunked) + sink, joined by LSE. `cu_docs` int32 over the flattened
    b*s stream, from doc_layout."""
    b, s, h, d = q.shape
    if HAS_FA_CUTE and q.is_cuda:
        ow, lw = _FlashWindow.apply(q.reshape(b * s, h, d), kv.reshape(b * s, 1, d), cu_docs, s, window, scale, softcap)
        ow = ow.view(b, s, h, d)
        lw = lw.view(h, b, s).permute(1, 0, 2).float()      # [b,h,s]
    else:
        ow, lw = _window_torch(q, kv, doc, window, scale, softcap)
    le = torch.full_like(lw, float("-inf"))
    oe = torch.zeros_like(ow)
    if comp is not None:
        outs, lses = [], []
        for c0 in range(0, s, ATTN_CHUNK):
            c1 = min(s, c0 + ATTN_CHUNK)
            nv = min(comp.size(1), c1 // m)
            if nv == 0:
                outs.append(torch.zeros_like(ow[:, c0:c1]))
                lses.append(torch.full_like(lw[:, :, c0:c1], float("-inf")))
                continue
            args = (q[:, c0:c1], comp[:, :nv], comp_idx[:, c0:c1], scale, softcap)
            if torch.is_grad_enabled():
                o, lse = checkpoint(_entry_chunk, *args, use_reentrant=False)
            else:
                o, lse = _entry_chunk(*args)
            outs.append(o)
            lses.append(lse)
        oe, le = torch.cat(outs, 1), torch.cat(lses, -1)
    ls = sink.float().view(1, h, 1).expand(b, h, s)
    total = torch.logsumexp(torch.stack([lw, le, ls], 0), 0)   # [b,h,s]
    ww = torch.exp(lw - total)
    we = torch.where(torch.isfinite(le), torch.exp(le - total), torch.zeros_like(le))
    o = ow.float() * ww.permute(0, 2, 1)[..., None] + oe.float() * we.permute(0, 2, 1)[..., None]
    return o.to(q.dtype), total.detach()
