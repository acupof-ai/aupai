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
    (doc [b,s] global document id, pos [b,s] position inside its document, doclen [b,s]).

    Row starts are merged into cu, so a document never spans two rows even if the caller's
    cu omitted a row boundary."""
    rows = torch.arange(0, b * s + 1, s, device=device, dtype=torch.long)
    cu = rows if cu is None else torch.cat([cu.to(device=device, dtype=torch.long), rows]).unique()
    i = torch.arange(b * s, device=device)
    d = torch.searchsorted(cu, i, right=True) - 1
    pos = i - cu[d]
    doclen = cu[d + 1] - cu[d]
    return d.view(b, s), pos.view(b, s), doclen.view(b, s)


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


def _attn_chunk(qc, kvw, compc, idxc, sink, docq, dock, c0: int, k0: int, window: int, scale: float):
    b, q, h, _ = qc.shape
    t = torch.arange(c0, c0 + q, device=qc.device)
    j = torch.arange(k0, k0 + kvw.size(1), device=qc.device)
    mw = (j[None, :] <= t[:, None]) & (t[:, None] - j[None, :] < window)
    mw = mw[None] & (docq[:, :, None] == dock[:, None, :])  # [b,q,kw]
    parts = [(torch.einsum("bqhd,bkd->bhqk", qc, kvw).float() * scale).masked_fill(~mw[:, None], float("-inf"))]
    if compc is not None:
        sel = _sel_mask(idxc, compc.size(1))
        parts.append((torch.einsum("bqhd,bnd->bhqn", qc, compc).float() * scale)
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


def windowed_sparse_attn(q, kv, comp, comp_idx, sink, scale: float, doc, window: int, m: int):
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
            o, lse = checkpoint(_attn_chunk, *args, c0, k0, window, scale, use_reentrant=False)
        else:
            o, lse = _attn_chunk(*args, c0, k0, window, scale)
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


def indexer_kl(indexer, x, qr, index_k, freqs, q, comp, comp_idx, lse, scale: float, m: int):
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
            sc = torch.einsum("bqhd,bnd->bhqn", q[:, c0:c1].detach(), comp[:, :nv].detach()).float() * scale
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
