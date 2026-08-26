"""CPU self-check for train.py architecture changes (no GPU deps needed).

Checks: AttnRes fwd/bwd (Full, Block, grad_ckpt), zero-init == uniform mean, and legacy checkpoint
round-trip: old-key state_dict -> load (remap) -> save -> load, identical key set and outputs.
Run: python scripts/test_arch_compat.py
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train  # noqa: E402

if train.chunk_kda is None:  # no fla on this machine: shape-preserving stand-in
    train.chunk_kda = lambda q, k, v, **kw: (q * 0 + v, None)
from train import Cfg, HybridLM, build_optimizers  # noqa: E402

Cfg.d, Cfg.heads, Cfg.layers, Cfg.ffn_hidden, Cfg.vocab, Cfg.seq = 64, 2, 4, 128, 100, 16
x = torch.randint(0, 100, (2, 16))
y = torch.randint(0, 100, (2, 16))

for blocks, ckpt, dyn in [
    (0, False, False),
    (3, False, False),
    (5, False, False),
    (0, True, False),
    (0, False, True),
]:
    Cfg.attn_res, Cfg.attn_res_blocks, Cfg.grad_ckpt, Cfg.attn_res_dyn_q = True, blocks, ckpt, dyn
    m = HybridLM(Cfg)
    h, _ = m(x, y)
    h.sum().backward()
    assert m.final_ar.q.grad is not None and torch.isfinite(h).all()
    assert blocks == 0 or len(m.ar_block_ends) == blocks, "Block AttnRes must produce exactly N blocks"
    assert len(build_optimizers(m, Cfg)) == 4

srcs = [train.Source.of(torch.randn(1, 3, 8)) for _ in range(5)]
mean = sum(s.v for s in srcs) / 5
assert torch.allclose(train.AttnRes(8)(srcs), mean, atol=1e-6), "zero-init must equal uniform mean"
dq = train.AttnRes(8, dyn_q=True)
nn.init.zeros_(dq.dyn[1].weight)
assert torch.allclose(dq(srcs), mean, atol=1e-6), "dyn_q zero-init must equal uniform mean"
Cfg.attn_res_dyn_q = False

# legacy checkpoint: split fused weights back into old keys
Cfg.attn_res, Cfg.grad_ckpt = False, False
old = HybridLM(Cfg)
legacy = {}
for k, v in old.state_dict().items():
    if k.endswith(".gb.weight"):  # gate|beta|pad -> gate_proj, beta_proj
        legacy[k.replace("gb", "gate_proj")] = v[: Cfg.d]
        legacy[k.replace("gb", "beta_proj")] = v[Cfg.d : Cfg.d + Cfg.heads]
        continue
    for fused, (a, b) in {"w13": ("w1", "w3"), "kv_up": ("k_up", "v_up"), "qg": ("q", "gate")}.items():
        if k.endswith(f".{fused}.weight"):
            va, vb = v.chunk(2)
            legacy[k.replace(fused, a)] = va
            legacy[k.replace(fused, b)] = vb
            break
    else:
        legacy[k] = v
Cfg.attn_res = True
new = HybridLM(Cfg)
new.load_state_dict(legacy)
assert new.attn_res is False and Cfg.attn_res is False, "old ckpt must disable AttnRes"
assert set(new.state_dict()) == set(old.state_dict()), "round-trip key set changed"
again = HybridLM(Cfg)
again.load_state_dict(new.state_dict())
with torch.no_grad():
    assert torch.allclose(old(x)[0], new(x)[0]) and torch.allclose(old(x)[0], again(x)[0])
# optimizer plumbing: schedule gates wd decay to Muon, snapshot is a real copy, conv kernels off the scalar group
Cfg.attn_res = False
m = HybridLM(Cfg)
opts = build_optimizers(m, Cfg)
assert all(p.ndim != 3 for p in opts[2].param_groups[0]["params"]), "conv kernels must not be in scalar group"
train.set_schedule(opts, 0, 100, Cfg)
assert opts[1].param_groups[0]["weight_decay"] == Cfg.embed_wd, "embedding wd must not be overwritten"
assert opts[0].param_groups[0]["weight_decay"] == Cfg.muon_wd
train.set_schedule(opts, 100, 100, Cfg)
assert opts[0].param_groups[0]["weight_decay"] == 0.0
assert train.lr_mult(10**6, 100, Cfg) == Cfg.final_lr_frac, "lr must stay at the floor past total (resume)"
m(x, y)[0].sum().backward()
for o in opts:
    o.step()
snap = train.opt_snapshot(opts)
before = next(v for st in snap[1]["state"].values() for v in st.values() if torch.is_tensor(v)).clone()
m(x, y)[0].sum().backward()
for o in opts:
    o.step()
after = next(v for st in snap[1]["state"].values() for v in st.values() if torch.is_tensor(v))
assert torch.equal(before, after), "snapshot must not alias live optimizer state"
# KDA decay init: mean retention exp(-softplus(dt_bias)) ~ 0.9 (was 0.5 with zero init at g=0)
dt_bias = m.blocks[0].mixer.dt_bias
ret = torch.exp(-torch.nn.functional.softplus(dt_bias)).mean().item()
assert 0.85 < ret < 0.99, ret
# doc boundaries: row starts + positions after <eos>, over the flattened stream
idx = torch.tensor([[5, 1, 7, 7], [1, 1, 3, 3]])
cu = train.doc_cu_seqlens(idx, eos_id=1)
assert cu.tolist() == [0, 2, 4, 5, 6, 8] and cu.dtype == torch.int32, cu
m = HybridLM(Cfg)
assert m(x, y, train.doc_cu_seqlens(x, 1))[0].shape == (2, 16, Cfg.d)
print("test_arch_compat OK")

# --- mix schedule: the plan is sharded per rank, so what one rank holds is 1/world of it ---
import json  # noqa: E402
import tempfile  # noqa: E402

_POOL = {"web": 400, "math": 40, "chat": 10}
_orig_domain_seqs = train._domain_seqs
try:
    train.Cfg.seq, train.Cfg.val_frac, train.Cfg.val_rows_max = 8, 0.05, 3
    train.Cfg.anneal_frac = 0.10
    train._domain_seqs = lambda domain, tok, is_main, ddp: (
        torch.arange(_POOL[domain], dtype=torch.int32)
        .repeat_interleave(train.Cfg.seq + 1)
        .view(_POOL[domain], -1)
        + list(_POOL).index(domain) * 1000
    )
    mix = {
        "total_tokens": 400 * 8,
        "domains": {
            "web": {"weight": 0.80, "epochs": 2, "anneal": 0.40},
            "math": {"weight": 0.15, "epochs": 5, "anneal": 0.40},
            "chat": {"weight": 0.05, "epochs": 6, "anneal": 0.20},
        },
    }
    mp = os.path.join(tempfile.mkdtemp(), "mix.json")
    json.dump(mix, open(mp, "w"))
    W = 4
    shards = [train.build_mix(mp, None, False, False, rank=i, world=W) for i in range(W)]
    tr = [s[0] for s in shards]
    assert len({len(t) for t in tr}) == 1, f"ranks got different row counts: {[len(t) for t in tr]}"
    glob = torch.empty((len(tr[0]) * W, train.Cfg.seq + 1), dtype=torch.int32)
    for i, t in enumerate(tr):
        glob[i::W] = t  # rank i holds rows i, i+W, i+2W, ...
    assert (glob == glob[:, :1]).all(), "a scheduled row was assembled from two different pool rows"
    dom = glob[:, 0] // 1000
    main_n = int(len(dom) * (1 - train.Cfg.anneal_frac))
    main_math = (dom[:main_n] == 1).float().mean().item()
    ann_math = (dom[main_n:] == 1).float().mean().item()
    assert ann_math > main_math * 1.5, (
        f"anneal must upweight math: main {main_math:.2f} anneal {ann_math:.2f}"
    )
    for di, name in enumerate(_POOL):
        n_val = min(max(1, int(_POOL[name] * train.Cfg.val_frac)), train.Cfg.val_rows_max)
        used = int((dom == di).sum())
        assert used <= (_POOL[name] - n_val) * mix["domains"][name]["epochs"], f"{name} exceeded its cap"
        va = shards[0][1]
        vrows = {int(r[0]) % 1000 for r in va if int(r[0]) // 1000 == di}
        trows = {int(r[0]) % 1000 for r in glob if int(r[0]) // 1000 == di}
        assert not (vrows & trows), f"{name}: val and train share rows {sorted(vrows & trows)[:5]}"
finally:
    train._domain_seqs = _orig_domain_seqs
print("test_mix_schedule OK")


# --- AttnRes: the paper's form, the two exact rewrites, and the per-block activation cost ---
def _paper(ar, srcs):
    """The form before the rewrites: an explicit RMSNorm(v_i) with a learned gain, dotted with q_l."""

    def rmsnorm(v):
        return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * ar.g

    q = ar.q if ar.dyn is None else ar.q + ar.dyn(rmsnorm(srcs[-1]))
    a = torch.stack([(rmsnorm(v) * q).sum(-1) for v in srcs]).float().softmax(0).to(srcs[0].dtype)
    return sum(a[i].unsqueeze(-1) * srcs[i] for i in range(len(srcs)))


_srcs = [torch.randn(2, 3, 16) for _ in range(4)]
assert torch.allclose(
    train.AttnRes(16)([train.Source.of(v) for v in _srcs]), torch.stack(_srcs).mean(0), atol=1e-6
), "AttnRes must start as the mean"

for _dyn in (False, True):
    _ar = train.AttnRes(16, dyn_q=_dyn)
    with torch.no_grad():
        _ar.g.normal_(1.0, 0.3)
        _ar.q.normal_(0, 0.5)
        if _dyn:
            _ar.dyn[1].weight.normal_(0, 0.1)
    _srcs = [torch.randn(2, 3, 16, requires_grad=True) for _ in range(5)]
    _got = _ar([train.Source.of(v) for v in _srcs])
    _want = _paper(_ar, _srcs)
    assert torch.allclose(_got, _want, atol=1e-5), f"rewrite changed the forward (dyn_q={_dyn})"
    _got.square().sum().backward()
    _gq, _gg = _ar.q.grad.clone(), _ar.g.grad.clone()
    _ar.zero_grad()
    for _v in _srcs:
        _v.grad = None
    _want.square().sum().backward()
    # relative: float32 accumulation over a squared-sum loss, not an algebraic difference
    for _name, _a, _b in (("q", _gq, _ar.q.grad), ("g", _gg, _ar.g.grad)):
        assert (_a - _b).abs().max() <= 1e-3 * _b.abs().max(), (
            f"rewrite changed the backward wrt {_name} (dyn_q={_dyn})"
        )

# A source carries a [B,T,1] scale, not a [B,T,D] normalized copy: rsqrt(mean(v^2)) is a
# per-position scalar, so v_hat . gq == rsqrt(...) * (v . gq).
_v = torch.randn(2, 3, 16)
_s = train.Source.of(_v)
assert _s.scale.shape == (2, 3, 1), _s.scale.shape
_gq = torch.randn(16)
assert torch.allclose((_s.normed() * _gq).sum(-1), (_v * _gq).sum(-1) * _s.scale.squeeze(-1), atol=1e-5)

for _nb, _pairs in ((0, 325), (2, 61), (4, 85)):
    _cfg = type(
        "C",
        (train.Cfg,),
        {
            "layers": 12,
            "attn_res": True,
            "attn_res_blocks": _nb,
            "vocab": 128,
            "d": 32,
            "heads": 2,
            "ffn_hidden": 64,
            "seq": 8,
        },
    )
    _m = train.HybridLM(_cfg)
    _p, _blocks, _partial = 0, 1, 0
    for _n in range(1, 2 * _cfg.layers + 1):
        _p += _blocks + _partial
        _partial = 1
        if _n in _m.ar_block_ends:
            _blocks += 1
            _partial = 0
    _p += _blocks + _partial
    assert _p == _pairs, f"attn_res_blocks={_nb}: {_p} pairs, expected {_pairs}"
print("test_attn_res OK")
