# restartable: a CPU self-check, seconds end to end. Its only write is a mix.json fixture
# into tempfile.mkdtemp(), which nothing reads back -- an interrupt costs a rerun, not work.
"""Self-check for train.py architecture changes.

Runs on CPU where fla is absent (a shape-preserving stand-in replaces the Triton
kernel) and on CUDA where fla is present -- the real chunk_kda cannot take CPU
tensors, so a fla machine without a visible GPU exits loudly instead of silently
skipping. On CUDA the model runs under the same bf16 autocast training uses
(train.py:755): FlashAttention refuses fp32, so the CUDA path never executed
before that was added. Checks: AttnRes fwd/bwd (Full, Block, grad_ckpt), zero-init == uniform
mean, and legacy checkpoint round-trip: old-key state_dict -> load (remap) ->
save -> load, identical key set and outputs.
Run: python scripts/test_arch_compat.py
"""

import contextlib
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train  # noqa: E402

if train.chunk_kda is None:  # no fla on this machine: shape-preserving stand-in
    train.chunk_kda = lambda q, k, v, **kw: (q * 0 + v, None)
    DEV = "cpu"
elif torch.cuda.is_available():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        DEV = "cuda"
    else:  # no pin: land on the freest card, GPU0 may be busy
        _free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        DEV = f"cuda:{_free.index(max(_free))}"
else:
    sys.exit(
        "fla is installed but no CUDA device is visible: the real chunk_kda is a "
        "Triton kernel and cannot run on CPU tensors. Set CUDA_VISIBLE_DEVICES, or "
        "run on a machine without fla. Skipping silently would leave this gate dead "
        "on the only machine with the real kernel."
    )


@contextlib.contextmanager
def _amp():
    if DEV.startswith("cuda"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


from train import Cfg, HybridLM, build_optimizers  # noqa: E402

Cfg.d, Cfg.heads, Cfg.layers, Cfg.ffn_hidden, Cfg.vocab, Cfg.seq = 64, 2, 4, 128, 100, 16
x = torch.randint(0, 100, (2, 16), device=DEV)
y = torch.randint(0, 100, (2, 16), device=DEV)

for blocks, ckpt, dyn in [
    (0, False, False),
    (3, False, False),
    (5, False, False),
    (0, True, False),
    (0, False, True),
]:
    Cfg.attn_res, Cfg.attn_res_blocks, Cfg.grad_ckpt, Cfg.attn_res_dyn_q = True, blocks, ckpt, dyn
    m = HybridLM(Cfg).to(DEV)
    with _amp():
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
old = HybridLM(Cfg).to(DEV)
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
# remap_legacy_state_dict cats with fresh CPU beta/pad tensors, so the round-trip
# runs on CPU; the models move to DEV only for the forward comparison.
legacy = {k: v.cpu() for k, v in legacy.items()}
Cfg.attn_res = True
new = HybridLM(Cfg)
new.load_state_dict(legacy)
assert new.attn_res is False and Cfg.attn_res is False, "old ckpt must disable AttnRes"
assert set(new.state_dict()) == set(old.state_dict()), "round-trip key set changed"
again = HybridLM(Cfg)
again.load_state_dict(new.state_dict())
new.to(DEV)
again.to(DEV)
with torch.no_grad(), _amp():
    assert torch.allclose(old(x)[0], new(x)[0]) and torch.allclose(old(x)[0], again(x)[0])
# GPU legacy load: the remap's pad tensor must be built on the state_dict's
# device -- a CPU pad made loading any legacy GPU checkpoint fail (mixed-device
# cat). This guard is the whole point of the fix; on CPU it is vacuous.
if DEV.startswith("cuda"):
    legacy_gpu = {k: v.to(DEV) for k, v in legacy.items()}
    gpu_m = HybridLM(Cfg).to(DEV)
    gpu_m.load_state_dict(legacy_gpu)
    with torch.no_grad(), _amp():
        assert torch.allclose(gpu_m(x)[0], new(x)[0], atol=1e-2), "GPU legacy load diverged"
# optimizer plumbing: schedule gates wd decay to Muon, snapshot is a real copy, conv kernels off the scalar group
Cfg.attn_res = False
m = HybridLM(Cfg).to(DEV)
opts = build_optimizers(m, Cfg)
assert all(p.ndim != 3 for p in opts[2].param_groups[0]["params"]), "conv kernels must not be in scalar group"
train.set_schedule(opts, 0, 100, Cfg)
assert opts[1].param_groups[0]["weight_decay"] == Cfg.embed_wd, "embedding wd must not be overwritten"
assert opts[0].param_groups[0]["weight_decay"] == Cfg.muon_wd
train.set_schedule(opts, 100, 100, Cfg)
assert opts[0].param_groups[0]["weight_decay"] == 0.0
assert train.lr_mult(10**6, 100, Cfg) == Cfg.final_lr_frac, "lr must stay at the floor past total (resume)"
with _amp():
    m(x, y)[0].sum().backward()
for o in opts:
    o.step()
snap = train.opt_snapshot(opts)
before = next(v for st in snap[1]["state"].values() for v in st.values() if torch.is_tensor(v)).clone()
with _amp():
    m(x, y)[0].sum().backward()
for o in opts:
    o.step()
after = next(v for st in snap[1]["state"].values() for v in st.values() if torch.is_tensor(v))
assert torch.equal(before, after), "snapshot must not alias live optimizer state"
# KDA decay init: mean retention exp(-softplus(dt_bias)) ~ 0.9
dt_bias = m.blocks[0].mixer.dt_bias
ret = torch.exp(-torch.nn.functional.softplus(dt_bias)).mean().item()
assert 0.85 < ret < 0.99, ret
# doc boundaries: row starts + positions after <eos>, over the flattened stream
idx = torch.tensor([[5, 1, 7, 7], [1, 1, 3, 3]])
cu = train.doc_cu_seqlens(idx, eos_id=1)
# 5 is gone against the pre-2026-08-30 expectation: flat[4:6] is <eos><eos>, one padding
# region, not two length-1 documents. Row starts (0 and 4) are unconditional -- dropping 4
# would let a document span two rows of the batch.
assert cu.tolist() == [0, 2, 4, 6, 8] and cu.dtype == torch.int32, cu
m = HybridLM(Cfg).to(DEV)
with _amp():
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


# --- FoNE: value-carrying [NUM] embedding + per-digit head ---
import fone  # noqa: E402

Cfg.d, Cfg.heads, Cfg.layers, Cfg.ffn_hidden, Cfg.seq = 64, 2, 4, 128, 16
Cfg.attn_res, Cfg.grad_ckpt = False, False
Cfg.fone, Cfg.num_id, Cfg.vocab = True, 100, 101  # [NUM] one past the base vocab
_m = HybridLM(Cfg).to(DEV)
_x = torch.randint(0, 100, (2, 16), device=DEV)
_x[0, 3] = _x[1, 5] = Cfg.num_id
_v = torch.zeros(2, 16, device=DEV)
_v[0, 3], _v[1, 5] = 152.0, 1640.0

with _amp():
    _h, _ = _m(_x, torch.zeros(1, device=DEV), num_vals=_v)
_nl = _m.num_logits(_h)
assert _nl.shape == (2, 16, fone.INT_DIGITS + fone.FRAC_DIGITS, 10), _nl.shape

# Untrained per-digit loss must sit at the ten-way chance level, not somewhere odd.
_mask = _x == Cfg.num_id
_tgt = fone.digits_of(_v[_mask].tolist()).to(DEV)
_loss = torch.nn.functional.cross_entropy(_nl[_mask].reshape(-1, 10), _tgt.reshape(-1))
assert 1.9 < _loss.item() < 2.9, f"digit loss {_loss.item()} far from ln(10)=2.303"

# Both FoNE parameters must actually receive gradient.
_loss.backward()
assert _m.num_head.weight.grad is not None and _m.num_head.weight.grad.abs().sum() > 0
assert _m.num_proj.weight.grad is not None and _m.num_proj.weight.grad.abs().sum() > 0

# The value must reach the hidden state -- same ids, different numbers, different output.
with _amp():
    _h2, _ = _m(_x, torch.zeros(1, device=DEV), num_vals=_v * 0 + 7.0)
assert not torch.allclose(_h, _h2), "num_vals does not affect the forward pass"

# A [NUM] id the model could never predict would make the token useless.
assert Cfg.num_id < Cfg.vocab, "[NUM] must be inside the logit slice"

# Opt-out is a no-op: no new parameters, forward works without num_vals.
Cfg.fone, Cfg.vocab = False, 100
_m0 = HybridLM(Cfg).to(DEV)
assert not hasattr(_m0, "num_proj") and not hasattr(_m0, "num_head")
with _amp():
    _h0, _ = _m0(_x.clamp(max=99), torch.zeros(1, device=DEV))
assert torch.isfinite(_h0).all()
Cfg.fone = False
print("test_fone OK")


# --- FoNE data path: text -> ids + compact values -> dense per-position values ---
from tokenizers import Tokenizer  # noqa: E402

_tok_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tokenizer.json"
)
if os.path.exists(_tok_path):
    _tk = Tokenizer.from_file(_tok_path)
    if _tk.token_to_id(fone.NUM_TOKEN) is not None:
        train.Cfg.fone, train.Cfg.seq = True, 63
        _texts = ["原价200元打8折是160元", "没有数字", "答案是140，余数6，超大数 12345678 不动"] * 20
        _ids, _vals = train.encode(_texts, _tk, chunk=50)
        _n = len(_ids) // (train.Cfg.seq + 1)
        _rows = _ids[: _n * (train.Cfg.seq + 1)].view(-1, train.Cfg.seq + 1)
        _dense = train.scatter_values(_rows, _vals, train.Cfg.num_id)
        _mask = _rows == train.Cfg.num_id
        # Values must land in row-major order, one per [NUM], and nowhere else.
        assert torch.equal(_dense[_mask], _vals[: int(_mask.sum())].float()), "scatter misaligned"
        assert (_dense[~_mask] == 0).all(), "value leaked onto a non-[NUM] position"
        # A number too large for the Fourier code must keep its ordinary tokens.
        assert not (_vals >= 10**fone.INT_DIGITS).any(), "oversized value entered the stream"
        train.Cfg.fone, train.Cfg.seq = False, 16
        print("test_fone_data OK")

        # --- SFT packing carries the same values -------------------------------
        import tempfile

        from prepare_sft import pack_and_save

        _num_id = _tk.token_to_id(fone.NUM_TOKEN)
        _pairs = [("问：原价200元打8折？\n答：", "160元") for _ in range(8)]
        with tempfile.TemporaryDirectory() as _td:
            _out = os.path.join(_td, "p.pt")
            pack_and_save(_pairs, _tk, _tk.token_to_id("<eos>"), _out, 63, num_id=_num_id)
            _d = torch.load(_out, weights_only=True)
        _m = _d["input_ids"] == _num_id
        assert _m.any(), "packer produced no [NUM]"
        assert (_d["values"][~_m] == 0).all(), "packed value outside a [NUM] position"
        # 200, 8 and 160 are the numbers in every example, so those are the values.
        assert set(_d["values"][_m].tolist()) <= {200.0, 8.0, 160.0}, _d["values"][_m].unique()
        # Prompt positions stay masked; the answer's 160 must still be a loss target.
        assert (_d["labels"][_m] == _num_id).any(), "no [NUM] survived as a loss target"
        print("test_fone_sft_pack OK")

        # --- text -> [NUM] + values -> text survives the round trip ------------
        for _t in ["原价200元打8折是160元", "余数6，商3.5", "没有数字", "超大数 12345678 不动"]:
            _i, _v = fone.encode_prompts([_t], _tk, _num_id)
            assert fone.decode_text(_i[0], [x for x in _v[0] if x], _tk, _num_id) == _t, _t
        assert fone.render(36.0) == "36" and fone.render(3.5) == "3.5" and fone.render(0.0) == "0"
        # return_hidden must not disturb the logits it sits beside.
        train.Cfg.fone, train.Cfg.num_id, train.Cfg.vocab = True, 100, 101
        _m2 = train.HybridLM(train.Cfg).to(DEV).eval()
        _x = torch.randint(0, 100, (2, 8), device=DEV)
        with torch.no_grad(), _amp():
            _l1, _n1 = _m2(_x)
            _l2, _h2 = _m2(_x, return_hidden=True)
        assert _n1 is None and _h2 is not None and torch.equal(_l1, _l2), "return_hidden changed the logits"
        # no_head skips the vocabulary head so a decoder can run it on the B positions it
        # actually reads instead of on B x T. It must be the SAME number: generate_batch now
        # takes this path, so a divergence here silently rewrites every generated token.
        with torch.no_grad(), _amp():
            _n3, _h3 = _m2(_x, no_head=True)
        assert _n3 is None, "no_head still returned logits"
        with _amp():
            assert torch.equal(_m2.lm_logits(_h3), _l1), "no_head + lm_logits != the full-head path"
        train.Cfg.fone, train.Cfg.vocab = False, 100
        print("test_fone_infer OK")
    else:
        print("test_fone_data SKIP (tokenizer has no [NUM]; run scripts/build_tokenizer.py)")
else:
    print("test_fone_data SKIP (no data/tokenizer.json)")

# infer_local.py keeps a Mac-local (no-fla) HybridLM copy so local inference runs without
# Triton/GPU. A future architecture change that touches train.HybridLM but forgets the copy
# turns on-the-Mac saves/loads into silent tensor-header scrambles. Pin them to the SAME
# state_dict key set here, so the mismatch fails in CI instead of on a laptop.
_base = train.Cfg
_base.attn_res, _base.attn_res_blocks, _base.grad_ckpt, _base.attn_res_dyn_q = False, 0, False, False
import infer_local  # noqa: E402  (Mac: pure-PyTorch stand-in, no fla import)

_keys_real = set(train.HybridLM(_base).state_dict())
_keys_local = set(infer_local.HybridLM(_base).state_dict())
assert _keys_local == _keys_real, (
    "infer_local.HybridLM state_dict diverged from train.HybridLM — a shared-key regression. "
    f"only-in-train={sorted(_keys_real - _keys_local)[:6]} only-in-local={sorted(_keys_local - _keys_real)[:6]}"
)
print("infer_local keys == train keys: OK")


# A FoNE run and a plain run must not share a token cache: --fone rewrites the token stream
# but leaves the vocabulary fingerprint untouched, so the freshness check cannot tell them
# apart, and the two directions fail differently with neither saying why.
_was_fone = train.Cfg.fone
try:
    train.Cfg.fone = False
    _plain = train._domain_cache_path("web_hq")
    train.Cfg.fone = True
    _fone = train._domain_cache_path("web_hq")
finally:
    train.Cfg.fone = _was_fone
assert _plain != _fone, f"both flags map to the same token cache: {_plain}"
assert "_fone" in _fone and "_fone" not in _plain, (_plain, _fone)
print("token cache namespaced by --fone: OK")


# --- the two vocab_fingerprint implementations must agree -------------------------
# train.py has one and scripts/loader.py has another, deliberately: loader must stay
# importable without torch. Nothing asserted they agree except test_e2e, which is
# GPU-only -- so a divergence would make every checkpoint unloadable and CI would be
# green. Checkpoints are stamped by train's copy and verified by loader's.
import train as _train  # noqa: E402
from loader import vocab_fingerprint as _loader_fp  # noqa: E402

_tok_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tokenizer.json"
)
if os.path.exists(_tok_path):
    from tokenizers import Tokenizer as _Tok  # noqa: E402

    _t = _Tok.from_file(_tok_path)
    assert _train.vocab_fingerprint(_t) == _loader_fp(_t), (
        f"train.vocab_fingerprint {_train.vocab_fingerprint(_t)} != "
        f"loader.vocab_fingerprint {_loader_fp(_t)}: every checkpoint would fail to load"
    )
    print("vocab_fingerprint: train == loader OK")
else:
    print("vocab_fingerprint SKIP (no data/tokenizer.json)")

# doc_cu_seqlens: a run of <eos> is padding and opens ONE document, not one per token.
# SFT rows are <eos>-padded to seq (mean 489 per 4097 row); one boundary per pad made every
# pad a length-1 document, fla's varlen grid is per-document, and batch 16 launched
# grid=(2, 78936, 1) against CUDA's gridDim.Y limit of 65535 -- surfacing as a bare
# `Triton Error [CUDA]: invalid argument` that read as a broken environment for an hour.
_E = 1
_packed = _train.doc_cu_seqlens(torch.tensor([[7, 8, _E, 9, 9], [6, 6, _E, 5, 5]]), _E).tolist()
assert _packed == [0, 3, 5, 8, 10], f"packed rows must be unchanged, got {_packed}"
_padded = _train.doc_cu_seqlens(torch.tensor([[7, 8, _E, _E, _E], [6, 6, 6, 6, _E]]), _E).tolist()
assert _padded == [0, 5, 10], f"an <eos> run must open one document, got {_padded}"
_rows = _train.doc_cu_seqlens(torch.tensor([[7, _E, _E, _E], [_E, _E, 3, 3]]), _E).tolist()
assert _rows == [0, 4, 6, 8], f"a row start survives even when its first token is <eos>, got {_rows}"
_wide = torch.cat([torch.tensor([[7, 8]]), torch.full((1, 4095), _E)], 1).repeat(8, 1)
_ndoc = len(_train.doc_cu_seqlens(_wide, _E)) - 1
assert _ndoc == 8, f"8 padded rows must be 8 documents, got {_ndoc} (grid overflows past 65535)"
print(f"doc_cu_seqlens: packed unchanged, {_ndoc} docs for 8 padded rows (was 32768) OK")

# MasterWeights must clear p.grad. The optimizer holds the fp32 copies, so its zero_grad()
# clears m.grad and nothing clears p.grad: backward() accumulated into the old one and the
# --fp32_master arm trained on a running sum (2.0, 4.0, 6.0 over three steps) while the
# control arm did not -- an A/B that would have blamed the difference on fp32 master weights.
_lin = torch.nn.Linear(2, 1, bias=False)
_mw = _train.MasterWeights(_lin)
_norms = []
for _ in range(3):
    (_lin(torch.ones(1, 2)) * 2).sum().backward()
    _mw.pull_grads()
    _norms.append(float(_lin.weight.grad.norm()) if _lin.weight.grad is not None else 0.0)
assert _norms == [0.0, 0.0, 0.0], f"pull_grads must leave p.grad cleared, got {_norms}"
_grads = [float(m.grad.abs().max()) for _, m in _mw.pairs]
assert all(abs(g - 2.0) < 1e-6 for g in _grads), f"each step's grad must be the step's own, got {_grads}"
print("MasterWeights: p.grad cleared every step, m.grad does not accumulate OK")
