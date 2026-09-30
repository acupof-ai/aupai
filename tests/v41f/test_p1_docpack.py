"""Packed-row path (v41f/docpack.py): document isolation, chunked attention, indexer KL.

Known answers, each against a path that shares no packed-row code:
  - packed [A;B] through the doc path == A and B run alone through the untouched reference
    path (cu=None), at a config with m=0/2/1 layers, kv sources, Reindex and Reuse layers
    and document lengths not divisible by m;
  - attn_impl "chunked" == "ref" on the same packed batch, logits and every gradient;
  - under "kl" the indexer loss reaches exactly the indexer projections and index keys, the
    main CE reaches none of them, and a few indexer-only steps lower the loss.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from allclose import grad_rtol  # noqa: E402
from v41f import docpack  # noqa: E402
from v41f.config import v41f_small  # noqa: E402
from v41f.model import V41FModel  # noqa: E402

_LENS = [(13, 10), (7, 9, 7)]  # rows of documents; 13, 9 and 7 are not multiples of m=2


# 16 index heads: an index score is exactly 0 when every head's ReLU is 0, and with 2 heads a
# quarter of the scores tie at 0, so top-k's tie order (which depends on row width) decides
# the selection and packed/separate differ by 6e-2 for that reason alone.
def _cfg(**over):
    base = dict(
        vocab_size=97, dim=64, n_heads=4, head_dim=32, rope_head_dim=16, q_lora_rank=32,
        o_groups=2, o_lora_rank=16, window_size=4, compress_ratios=(0, 2, 2, 1, 1),
        kv_source_layers=(1, 3), index_source_layers=(1, 3, 4), index_n_heads=16,
        index_head_dim=32, index_topk=3, n_routed_experts=4, n_activated_experts=2,
        moe_inter_dim=32, hc_mult=2)
    base.update(over)
    return v41f_small(**base)


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        m = V41FModel(cfg, max_batch_size=2)
    finally:
        torch.set_default_dtype(prev)
    m = m.float()
    with torch.no_grad():  # a nonzero sink, so the sink term is exercised
        for layer in m.layers:
            layer.attn.attn_sink.normal_()
    return m


def _batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    rows = [torch.randint(2, 97, (sum(r),), generator=g) for r in _LENS]
    s = max(len(r) for r in rows)
    assert all(len(r) == s for r in rows)
    ids = torch.stack(rows)
    cu = [0]
    for i, r in enumerate(_LENS):
        for n in r:
            cu.append(cu[-1] + n)
    return ids, torch.tensor(cu, dtype=torch.int32)


def test_packed_equals_separate_documents():
    cfg = _cfg()
    m = _model(cfg)
    ids, cu = _batch()
    with torch.no_grad():
        packed, _ = m(ids, cu=cu)
        worst = 0.0
        for row, lens in enumerate(_LENS):
            off = 0
            for n in lens:
                alone, _ = m(ids[row:row + 1, off:off + n])
                worst = max(worst, (packed[row, off:off + n] - alone[0]).abs().max().item())
                off += n
    assert worst < 1e-4, f"packed vs separate documents differ by {worst:.3e}"
    print(f"  packed [A;B] == separate docs: max|d| {worst:.2e}")


def test_single_doc_cu_equals_no_cu():
    m = _model(_cfg())
    ids, _ = _batch()
    s = ids.size(1)
    with torch.no_grad():
        a, _ = m(ids)
        b, _ = m(ids, cu=torch.arange(0, 2 * s + 1, s, dtype=torch.int32))
    d = (a - b).abs().max().item()
    assert d < 1e-4, f"one-doc-per-row cu vs cu=None differ by {d:.3e}"


def _grads(m, ids, cu):
    m.zero_grad(set_to_none=True)
    logits, _ = m(ids, cu=cu)
    (logits.float().square().mean()).backward()
    return logits.detach(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def test_chunked_equals_ref():
    ids, cu = _batch()
    ref = _model(_cfg())
    chk = _model(_cfg(attn_impl="chunked"))
    chk.load_state_dict(ref.state_dict())
    old = docpack.ATTN_CHUNK
    docpack.ATTN_CHUNK = 5  # several chunks per row, seams inside documents and windows
    try:
        for c in (cu, None):
            la, ga = _grads(ref, ids, c)
            lb, gb = _grads(chk, ids, c)
            d = (la - lb).abs().max().item()
            assert d < 1e-4, f"chunked vs ref logits differ by {d:.3e} (cu={'packed' if c is not None else None})"
            assert ga.keys() == gb.keys(), f"grad sets differ: {sorted(ga.keys() ^ gb.keys())}"
            for n in ga:
                rel = ((ga[n] - gb[n]).norm() / ga[n].norm().clamp_min(1e-12)).item()
                assert rel < grad_rtol(n), f"grad {n} differs rel {rel:.3e} (tol {grad_rtol(n):.0e})"
    finally:
        docpack.ATTN_CHUNK = old
    print(f"  chunked == ref: logits and {len(ga)} grads")


def _indexer_names(cfg):
    names = {f"layers.{i}.attn.indexer.{leaf}.weight" for i in cfg.index_source_layers
             for leaf in ("wq_b", "weights_proj")}
    names |= {f"layers.{i}.attn.index_key.{leaf}.weight" for i in cfg.kv_source_layers
              if i in cfg.index_source_layers for leaf in ("wk", "k_norm")}
    return names


def test_kl_routes_only_to_the_indexer():
    cfg = _cfg(attn_impl="chunked", indexer_train_mode="kl")
    m = _model(cfg)
    ids, cu = _batch()
    want = _indexer_names(cfg)
    assert want <= {n for n, p in m.named_parameters() if p.requires_grad}, "kl left an indexer leaf frozen"
    logits, _ = m(ids, cu=cu)
    kl = m.indexer_loss
    assert kl is not None and torch.isfinite(kl) and kl.item() >= 0, f"bad indexer loss {kl}"
    ce = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
    got_ce = {n for n, g in zip(dict(m.named_parameters()),
                                torch.autograd.grad(ce, list(m.parameters()), retain_graph=True, allow_unused=True))
              if g is not None and g.abs().sum() > 0}
    assert not (got_ce & want), f"main CE reached indexer leaves: {sorted(got_ce & want)}"
    got_kl = {n for n, g in zip(dict(m.named_parameters()),
                                torch.autograd.grad(kl, list(m.parameters()), allow_unused=True))
              if g is not None and g.abs().sum() > 0}
    assert got_kl == want, f"KL grads: missing {sorted(want - got_kl)}, extra {sorted(got_kl - want)}"
    print(f"  kl: loss {kl.item():.4f}, reaches exactly {len(want)} indexer leaves, CE reaches none")


def test_kl_descends():
    cfg = _cfg(attn_impl="chunked", indexer_train_mode="kl")
    m = _model(cfg)
    ids, cu = _batch()
    params = [p for n, p in m.named_parameters() if n in _indexer_names(cfg)]
    opt = torch.optim.Adam(params, lr=3e-2)
    first = None
    for _ in range(30):
        m(ids, cu=cu)
        loss = m.indexer_loss
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    m(ids, cu=cu)
    last = m.indexer_loss.item()
    assert last < 0.5 * first, f"indexer KL did not descend: {first:.4f} -> {last:.4f}"
    print(f"  kl descends under indexer-only steps: {first:.4f} -> {last:.4f}")


def _broken_isolation():
    """Mutant: RoPE positions do not restart per document (absolute row positions). The
    packed test must go red on it."""
    import v41f.attention as att
    src = att.Attention._forward_docs

    def fwd(self, x, state):
        saved = state.pos
        state.pos = torch.arange(saved.size(1)).expand_as(saved)
        try:
            return src(self, x, state)
        finally:
            state.pos = saved
    att.Attention._forward_docs = fwd
    try:
        test_packed_equals_separate_documents()
    except AssertionError:
        return True
    finally:
        att.Attention._forward_docs = src
    return False


if __name__ == "__main__":
    tests = [test_packed_equals_separate_documents, test_single_doc_cu_equals_no_cu,
             test_chunked_equals_ref, test_kl_routes_only_to_the_indexer, test_kl_descends]
    for t in tests:
        t()
    assert _broken_isolation(), "mutant (positions not restarted per document) survived the packed test"
    print(f"docpack: {len(tests)}/{len(tests)} passed, isolation mutant red")
