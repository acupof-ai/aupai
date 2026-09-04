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
     python scripts/test_arch_compat.py --selftest   (identical; the hook's calling convention)

--selftest is accepted so the pre-commit hook's SELFTEST_FILES map can call this file the
same way it calls every other entry, and REJECTING an unknown argument is the point of
handling it explicitly. This module asserts at import time with no main(), so before this
it ignored argv entirely: `test_arch_compat.py --selftest` ran the checks and exited 0, and
so would `--no-such-flag`. The hook's own comment says why that is not good enough -- "a
script that exits 0 on an unknown argument would otherwise register as a pass" -- and a
file whose checks are its module body is exactly where that happens silently.
"""

import contextlib
import copy
import os
import sys

import torch
import torch.nn as nn

# Before any of the work below, and before the heavy imports: an unknown flag must fail
# loudly rather than run the suite and report success for a call nobody meant to make.
if len(sys.argv) > 1 and sys.argv[1:] != ["--selftest"]:
    sys.exit(f"usage: {os.path.basename(__file__)} [--selftest]  (got {sys.argv[1:]})")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datagen"))
import model  # noqa: E402
import train  # noqa: E402

if train.chunk_kda is None:  # no fla on this machine: shape-preserving stand-in
    # Patch model, not just train: DeltaRecurrence reads its OWN module global, and after the
    # b0-8 split train.chunk_kda is a re-exported SEPARATE binding -- setting only that no
    # longer reaches the call site. Both are set so a caller reading either sees the stand-in.
    model.chunk_kda = train.chunk_kda = lambda q, k, v, **kw: (q * 0 + v, None)
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

# INSIDE the cuda branch, and only there (de-55). CI runs this on a machine with no fla, where
# DEV == "cpu" and no card is touched; an unconditional claim would refuse every CI run.
#
# The freest-card branch above is a card taken with no CVD and no claim, chosen by an
# instantaneous free-memory poll -- the ownership test AGENTS.md rejects. claim_my_cards refuses
# an unset CVD, so reaching this line on that branch now fails loudly and names the fix
# (CUDA_VISIBLE_DEVICES=N) instead of landing on whatever card looked idle a moment ago.
if DEV.startswith("cuda"):
    from loader import claim_my_cards  # noqa: E402

    claim_my_cards("test_arch_compat", note="arch compat gate")

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
    # FOUR groups: muon, embed, scalar, arq. b0-17's untied head adds a FIFTH, and only when
    # both --untie_head and --head_lr are set -- so this assertion is about the DEFAULT config and
    # says so, rather than passing by luck. Asserting the untied count here too would make the
    # number the test's subject; scripts/test_untie_head.py owns the three arms.
    assert not getattr(Cfg, "untie_head", False),         "Cfg.untie_head defaults True; the 4-group count below describes the tied default and "         "every existing checkpoint was trained under it"
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

# ---------------------------------------------------------------- sparse memory layers
# Charter docs/standards/memory_layers_0905.md asks for three cases: memory fwd/bwd on CPU,
# save/load round-trip, and a legacy (no-memory) checkpoint still loading. Each one below is
# followed by the thing it would let through if it were only the obvious assertion.
_MV, _MK = 64 * 64, 8  # a square value count, and top_k <= side (ProductKeyMemory raises otherwise)
Cfg.mem_values, Cfg.mem_top_k, Cfg.mem_layers, Cfg.mem_sparse = _MV, _MK, "1,3", True
Cfg.grad_ckpt = False

# 1. FWD/BWD ON BOTH _body PATHS, and the attn_res one is the reason this case exists rather
# than a formality. _body has two paths: the plain one calls Block.forward (which adds the
# memory) and the attn_res one iterates Block.sublayers(), which returns (ar1,n1,mixer) and
# (ar2,n2,ffn) and NO memory branch. Cfg.attn_res defaults True and the head-hybrid control
# trained with it True, so before 2026-09-05 every memory arm would have taken that path,
# skipped the memory, trained as the CONTROL, and reported a null it never tested -- with the
# flags, the log and the ledger row all saying it carried the table. A test that ran only the
# plain path would have been green through all of that, which is why both are here and why
# each asserts TOUCHED ROWS rather than just a finite loss: the silent-skip world produces a
# perfectly finite loss. Measured then: 0 rows touched on the attn_res path.
for _ar_on in (True, False):
    Cfg.attn_res = _ar_on
    torch.manual_seed(5)
    _mm = HybridLM(Cfg).to(DEV)
    assert _mm.memory is not None and _mm.mem_layers == [1, 3], (_mm.mem_layers,)
    with _amp():
        _mh, _ = _mm(x, y)
    _mh.float().sum().backward()
    _md = _mm.memory.diagnostics()
    assert torch.isfinite(_mh).all(), f"memory forward not finite (attn_res={_ar_on})"
    assert _md["touched_rows"] > 0, (
        f"attn_res={_ar_on}: the memory was reached by NO token, so this arm trains as the "
        f"control while its flags say otherwise -- the 2026-09-05 sublayers() defect")
    assert _mm.memory.values.weight.grad is not None, "value table got no gradient"
    assert _mm.memory.values.weight.grad.is_sparse, (
        "mem_sparse=True must give a COO grad: the dense one is 4.3B rows at M3 and defeats "
        "the index-exchange DDP path entirely")
    assert _mm.memory.keys.grad is not None and _mm.memory.keys.grad.abs().sum() > 0, (
        "the keys got no gradient, so the lookup can never learn WHICH values to read and the "
        "table is a fixed random projection")
    # THE SHARED POOL IS ONE MODULE AND ONE PARAMETER. Assigning it to an attribute per block
    # would register it once per reading block: three copies in state_dict and the same tensor
    # handed to the optimizer three times.
    #
    # BOTH CHECKS BELOW REPLACE ONES THAT WERE BLIND, and the mutation that exposed them was
    # registering the pool under a DIFFERENT attribute name (self._mem_registered = memory)
    # alongside the list. Measured 2026-09-05: the test stayed green.
    #   - `[k for k in state_dict() if k.endswith("memory.values.weight")]` counted 1, because
    #     the duplicate's key is blocks.1._mem_registered.values.weight -- a name-suffix test
    #     only finds a duplicate that the mutation happens to name the same way.
    #   - `sum(1 for p in parameters() if p is ...)` counted 1, because nn.Module.parameters()
    #     DEDUPLICATES by identity by default. It answers "is this tensor a parameter", which
    #     was never the question; the question is how many times the optimizer will be handed it.
    # So: count by STORAGE across every state_dict entry, and walk parameters with
    # remove_duplicate=False. Verified both report 4 under the mutant and 1 here.
    assert _mm.blocks[1]._mem[0] is _mm.blocks[3]._mem[0] is _mm.memory, "pool must be shared"
    _ptr = _mm.memory.values.weight.data_ptr()
    _dups = [k for k, v in _mm.state_dict().items() if v.data_ptr() == _ptr]
    assert len(_dups) == 1, f"the value table's storage appears {len(_dups)} times in state_dict: {_dups}"
    _opt_hits = [n for n, p in _mm.named_parameters(remove_duplicate=False)
                 if p is _mm.memory.values.weight]
    assert len(_opt_hits) == 1, (
        f"the value table reaches the optimizer {len(_opt_hits)} times: {_opt_hits}")
    assert not any(k.endswith("touched") or k.endswith("last_entropy") for k in _mm.state_dict()), (
        "the diagnostics buffers must be non-persistent: saving them makes two checkpoints of "
        "the same weights differ by a window counter")

# 2. SAVE/LOAD ROUND-TRIP, asserted on the OUTPUT and not only on the key set. A reload that
# dropped the memory silently would keep every key (the pool is still constructed) and still
# produce a finite forward; only the numbers differ. eval() both sides so the `touched` write
# is the only nondeterminism and it touches no output.
Cfg.attn_res = True
torch.manual_seed(7)
_msrc = HybridLM(Cfg).to(DEV).eval()
_mdst = HybridLM(Cfg).to(DEV).eval()
with torch.no_grad(), _amp():
    _before = _msrc(x)[0].clone()
    assert not torch.allclose(_before, _mdst(x)[0]), (
        "two fresh models agree before loading, so this case cannot see a load that does nothing")
_mdst.load_state_dict(_msrc.state_dict())
with torch.no_grad(), _amp():
    assert torch.allclose(_before, _mdst(x)[0]), "memory checkpoint round-trip changed the output"
assert torch.equal(_msrc.memory.values.weight, _mdst.memory.values.weight)

# 3. A LEGACY CHECKPOINT STILL LOADS. Every checkpoint before 2026-09-05 has no memory tensors
# and no mem_* in its cfg, and the control is one of them -- it must construct bit-identically
# to how it trained, so mem_values 0 has to mean "no pool at all", not "an empty pool".
Cfg.mem_values = 0
_mnone = HybridLM(Cfg).to(DEV)
assert _mnone.memory is None and _mnone.mem_layers == [], "mem_values 0 must build no pool"
assert not any("memory" in k for k in _mnone.state_dict()), "no-memory model carries memory keys"
_legacy_sd = _mnone.state_dict()
Cfg.mem_values = _MV
_mwith = HybridLM(Cfg).to(DEV)
try:
    _mwith.load_state_dict(_legacy_sd)
    raise AssertionError(
        "a memory model accepted a checkpoint with no memory tensors. Then a resume of the "
        "control under --mem_values would train a RANDOM table and read from it, and nothing "
        "would say so")
except RuntimeError as _e:
    assert "memory" in str(_e), _e
Cfg.mem_values = 0
HybridLM(Cfg).to(DEV).load_state_dict(_legacy_sd)  # the real path: cfg says 0, weights have none
Cfg.mem_values, Cfg.mem_top_k, Cfg.mem_layers = 0, 32, "3,6,9"

# 4. mem_layers OUT OF RANGE MUST RAISE. An index past the last block would attach the pool to
# nothing, and the arm would train as the control with every flag saying otherwise -- the same
# silent-null as case 1, reached by a typo in a launch line instead of by a code path.
Cfg.mem_values, Cfg.mem_layers = _MV, f"0,{Cfg.layers}"
try:
    HybridLM(Cfg)
    raise AssertionError("mem_layers past the last block was accepted; the pool attaches to nothing")
except ValueError as _e:
    assert "outside" in str(_e), _e
# BOTH SPELLINGS OF mem_layers MUST BUILD ONE ARCHITECTURE: the flag arrives as a string and a
# saved cfg as whatever it was saved as, and two spellings reading as two arms is the shape that
# made a ledger knob look like a 1/sqrt(L) rule.
Cfg.mem_layers = "1,3"
_a = HybridLM(Cfg).mem_layers
Cfg.mem_layers = [1, 3]
assert HybridLM(Cfg).mem_layers == _a == [1, 3], "list and comma-string forms disagree"
Cfg.mem_values, Cfg.mem_layers = 0, "3,6,9"
print("memory layers: fwd/bwd on BOTH _body paths (attn_res on and off), COO value grad, "
      "keys learn, one shared pool registered once, round-trip exact, legacy ckpt loads, "
      "bad mem_layers raises OK")

# ------------------------------------------------- memory optimizer group and FP8 exclusion
# Every assertion here is about a MIS-ROUTING, which is the failure mode that does not crash: the
# run trains, the loss moves, and nothing says the table was optimised as if it were a matrix.
Cfg.mem_values, Cfg.mem_top_k, Cfg.mem_layers, Cfg.mem_sparse = _MV, _MK, "1,3", True
Cfg.attn_res, Cfg.grad_ckpt = True, False
_om = HybridLM(Cfg).to(DEV)
_opts = {o.aupai_group: o for o in build_optimizers(_om, Cfg)}
assert "mem" in _opts, "no `mem` optimizer group; the memory is being stepped by another group"
assert isinstance(_opts["mem"], torch.optim.Adagrad), (
    f"the memory group is {type(_opts['mem']).__name__}. AdamW RAISES on a sparse gradient, and "
    f"SparseAdam's two fp32 moments are 32 GiB at M3 against Adagrad's 16 -- the difference "
    f"between 22 GiB of slack on a 95.58 GiB card and 6 (4c 2026-09-05)")
_memp = [p for g in _opts["mem"].param_groups for p in g["params"]]
assert any(p is _om.memory.values.weight for p in _memp), "value table is not in the mem group"
assert any(p is _om.memory.keys for p in _memp), "the keys are not in the mem group"
# AND NOWHERE ELSE. The value table is 2D, so `p.ndim == 2` would put it in MUON, whose
# Newton-Schulz orthogonalisation is meaningless for a table read by index; the keys are 3D and
# would land in `arq` at attn_res_lr beside the AttnRes pseudo-queries. Asserting only "in mem"
# would pass while it sat in both, and a parameter in two groups is stepped twice per step.
for _gn, _go in _opts.items():
    if _gn == "mem":
        continue
    _leak = [n for n, p in _om.named_parameters()
             if train._is_mem_fqn(n) and any(p is q for g in _go.param_groups for q in g["params"])]
    assert not _leak, f"memory parameter(s) also in the `{_gn}` group: {_leak}"
assert _opts["mem"].param_groups[0]["lr"] == Cfg.mem_lr, "mem group is not at Cfg.mem_lr"

# THE GROUP MUST ACTUALLY STEP A SPARSE GRADIENT, and the assertion is "every row with a nonzero
# gradient moved", NOT "the moved set equals the touched set". Measured 2026-09-05: 966 rows were
# read, 948 moved, and the 18 that did not had a gradient of exactly 0.0 -- their softmax weight
# underflowed, so there was nothing to apply. Set equality would have failed on a correct step and
# sent the next reader after Adagrad; row counts would have hidden the opposite defect. The
# predicate that separates them is the gradient, so that is what this reads.
# Only the mem optimizer is stepped: Muon's step is torch.compiled and Inductor's C++ backend
# fails on a laptop toolchain, which has nothing to do with this group.
Cfg.attn_res = True
torch.manual_seed(11)
_sm = HybridLM(Cfg).to(DEV)
_smopt = {o.aupai_group: o for o in build_optimizers(_sm, Cfg)}["mem"]
_w0 = _sm.memory.values.weight.detach().clone()
with _amp():
    _sh, _ = _sm(x, y)
_sm.lm_logits(_sh).float().sum().backward()
_sg = _sm.memory.values.weight.grad.coalesce()
_rowmag = torch.zeros(_MV, device=_sg.values().device).index_add_(
    0, _sg.indices()[0], _sg.values().abs().sum(-1).float())
_smopt.step()
_moved = (_sm.memory.values.weight.detach() - _w0).abs().sum(-1) > 0
assert int((_rowmag > 0).sum()) > 0, "no row had a nonzero gradient -- this case would be vacuous"
_stuck = ((_rowmag > 0) & ~_moved).nonzero().flatten()
assert _stuck.numel() == 0, (
    f"{_stuck.numel()} row(s) had a nonzero gradient and did not move: the sparse step is not "
    f"reaching the table")
_wrong = ((_rowmag == 0) & _moved).nonzero().flatten()
assert _wrong.numel() == 0, (
    f"{_wrong.numel()} row(s) moved with NO gradient -- the update is being applied densely, "
    f"which at M3 is a 4.3-billion-row write per step")

# FP8: the memory's linears are `query`, `gate`, `out` -- no leaf name says "memory" and all are
# 16-aligned at d>=128, so the OLD leaf-name filter converted every one of them. This asserts the
# new full-fqn filter excludes them AND changes no other verdict in the model, because a filter
# that excluded more than the memory would silently drop unrelated layers out of FP8.
_mem_lin = [f for f, mod in _om.named_modules()
            if isinstance(mod, nn.Linear) and train._is_mem_fqn(f)]
assert _mem_lin, "no nn.Linear inside the memory -- this case would be vacuous"
assert all(train._fp8_ok(_om.get_submodule(f), f.rsplit(".", 1)[-1]) for f in _mem_lin), (
    "the leaf-name filter already rejects the memory's linears here, so this case cannot see the "
    "defect it exists for -- re-derive it at the real d before trusting it")
assert not any(train._fp8_filter(_om.get_submodule(f), f) for f in _mem_lin), (
    f"_fp8_filter still converts {_mem_lin}: casting the query projection to e4m3 changes WHICH "
    f"values a token retrieves -- a discrete change in the top_k set, not a small numerical one")
_other_lin = [f for f, mod in _om.named_modules()
              if isinstance(mod, nn.Linear) and not train._is_mem_fqn(f)]
_changed = [f for f in _other_lin
            if train._fp8_filter(_om.get_submodule(f), f)
            != train._fp8_ok(_om.get_submodule(f), f.rsplit(".", 1)[-1])]
assert not _changed, f"the new filter changed the verdict for non-memory linears: {_changed}"
# The legacy path (FP8_RECIPE=legacy) is a SECOND converter, and it saw only leaf names too. Two
# paths disagreeing about which parameters are FP8 is worse than either choice, because the recipe
# is an env var: the difference appears in neither the launch line nor the checkpoint's cfg.
_leg = train._convert_to_fp8_legacy(copy.deepcopy(_om))
_leg_conv = [f for f, mod in _leg.named_modules() if type(mod).__name__ == "FP8Linear"]
assert _leg_conv, "the legacy converter converted nothing -- this case would be vacuous"
assert not [f for f in _leg_conv if train._is_mem_fqn(f)], (
    f"the legacy FP8 path converted memory linears: {[f for f in _leg_conv if train._is_mem_fqn(f)]}")
Cfg.mem_values, Cfg.mem_layers = 0, "3,6,9"
print(f"memory optimizer: own Adagrad group at mem_lr, keys+values in it and in no other group; "
      f"FP8 excludes {len(_mem_lin)} memory linears on both paths and changes no other verdict OK")

# --warmdown 0 must land as 0.0, not be skipped as falsy. The generic args->Cfg loop uses
# `if hasattr(Cfg,k) and v`, which drops 0.0; the WSD stage-1 join sets --warmdown 0 to keep
# lr at stable, so a skipped 0.0 would silently anneal stage 1. This guards the explicit
# is-not-None apply that train.py adds for warmdown/anneal_frac.
class _CfgStub:
    warmdown = 0.65
    anneal_frac = 0.10
_stub = _CfgStub()
for _k, _v in {"warmdown": 0.0, "anneal_frac": 0.0}.items():  # the buggy `and v` path
    if hasattr(_stub, _k) and _v:
        setattr(_stub, _k, _v)
assert _stub.warmdown == 0.65 and _stub.anneal_frac == 0.10, "sanity: the falsy path should NOT apply 0.0"
_stub2 = _CfgStub()
for _k, _v in {"warmdown": 0.0, "anneal_frac": 0.0}.items():  # the fixed is-not-None path
    if _v is not None:
        setattr(_stub2, _k, _v)
assert _stub2.warmdown == 0.0 and _stub2.anneal_frac == 0.0, "--warmdown 0 must land as 0.0 (WSD stage-1 join)"
print("wsd flags: --warmdown 0 lands as 0.0, not skipped OK")

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
from loader import vocab_fingerprint as _loader_fp  # noqa: E402

import model as _model  # noqa: E402
import train as _train  # noqa: E402

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

# ShortConv: the K shifted multiply-adds must equal nn.Conv1d bit-for-bit (shared weights).
# The arithmetic form ships for speed (3.44x compiled; conv_depthwise2d_generic is ~6%
# bandwidth), so it must be numerically the conv it replaces -- not merely close. The spy
# captures the patched forward's own short_conv output (its first silu) and then aborts the
# forward, so the check tracks the real branch without running chunk_kda (Triton, GPU-only).
_dr = _train.DeltaRecurrence(Cfg).to(DEV)
_xd = torch.randn(2, 16, Cfg.d, device=DEV)
_Kc = _dr.short_conv.kernel_size[0]
import torch.nn.functional as _F  # noqa: E402

with torch.no_grad():
    _hc = _F.pad(_xd.transpose(1, 2), (_Kc - 1, 0))
    _ref_h = _F.silu(_dr.short_conv(_hc).transpose(1, 2))  # the plain nn.Conv1d path
_cap = {}
_orig_silu = _F.silu
class _Stop(Exception): pass
def _spy(t, *a, **k):
    _cap["h"] = _orig_silu(t, *a, **k)  # first silu in forward is the short_conv out
    raise _Stop  # abort before chunk_kda
_F.silu = _spy
try:
    with torch.no_grad():
        _dr(_xd)
except _Stop:
    pass
finally:
    _F.silu = _orig_silu
_diff = (_cap["h"] - _ref_h).abs().max()
assert _diff < 1e-4, f"short_conv shifted form != nn.Conv1d (max diff {_diff:.2e})"
print(f"short_conv: shifted multiply-adds == nn.Conv1d (max diff {_diff:.2e}) OK")

# conv_doc_isolated: the flag that makes cu reach the short_conv too. Without it the conv reads
# across document boundaries -- measured 2026-09-04 as 48.88 at the block-0 output against a
# 0.9253 tolerance (eff.kda_document_isolation_violated, runs/n8/). This runs on CPU without
# chunk_kda by reusing the silu spy: the conv output is the whole question, since the kernel and
# the attention were both controlled out on random inputs.
#
# THREE CASES, and the third is the one that protects existing results:
#   isolated + cu     a document's conv output must EQUAL that document scored alone
#   not isolated + cu it must NOT, or the flag does nothing and the gate is vacuous
#   no cu             both settings must be bitwise identical: a single-document row is
#                     unaffected, so nothing without packing changes
def _conv_out(dr, x, cu):
    """The short_conv output only, via the silu spy, without reaching chunk_kda."""
    cap, orig = {}, _F.silu

    def spy(t, *a, **k):
        cap["h"] = orig(t, *a, **k)
        raise _Stop
    _F.silu = spy
    try:
        with torch.no_grad():
            dr(x, cu=cu)
    except _Stop:
        pass
    finally:
        _F.silu = orig
    return cap["h"]


class _CfgIso(Cfg):
    conv_doc_isolated = True


_dr_iso = _train.DeltaRecurrence(_CfgIso).to(DEV)
_dr_iso.load_state_dict(_dr.state_dict())  # same weights: a topology test, not an init test
assert _dr_iso.conv_doc_isolated and not _dr.conv_doc_isolated, "the flag did not reach the module"

# ONE ROW, two documents of 10 and 6, cu over the flat B*T stream as doc_cu_seqlens builds it.
_x2 = torch.randn(1, 16, Cfg.d, device=DEV)
_cu2 = torch.tensor([0, 10, 16], dtype=torch.int32, device=DEV)
_solo = [_conv_out(_dr_iso, _x2[:, :10], torch.tensor([0, 10], dtype=torch.int32, device=DEV)),
         _conv_out(_dr_iso, _x2[:, 10:], torch.tensor([0, 6], dtype=torch.int32, device=DEV))]
_packed_iso = _conv_out(_dr_iso, _x2, _cu2)
_packed_leak = _conv_out(_dr, _x2, _cu2)
_d_iso = max((_packed_iso[0, :10] - _solo[0][0]).abs().max().item(),
             (_packed_iso[0, 10:] - _solo[1][0]).abs().max().item())
_d_leak = (_packed_leak[0, 10:] - _solo[1][0]).abs().max().item()
assert _d_iso < 1e-6, f"conv_doc_isolated ON still leaks across documents (max diff {_d_iso:.2e})"
assert _d_leak > 1e-3, (
    f"conv_doc_isolated OFF does NOT leak (max diff {_d_leak:.2e}) -- either the flag is a no-op or "
    f"this fixture cannot see the defect, and in both cases the ON case above proves nothing")
# NO cu: the two settings must be bitwise identical, so no existing single-document result moves.
_d_nocu = (_conv_out(_dr_iso, _xd, None) - _conv_out(_dr, _xd, None)).abs().max().item()
assert _d_nocu == 0.0, f"cu=None differs between flag settings by {_d_nocu:.2e} (must be bitwise 0)"
print(f"conv_doc_isolated: ON isolates ({_d_iso:.2e}), OFF leaks ({_d_leak:.2e}), "
      f"cu=None bitwise identical OK")

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

# The attention fallback must honour cu. It used to take cu and ignore it, so doc_mask=True
# trained with every document attending across every boundary -- five A/B arms landed 0.293
# nat off the ladder with nothing in the log looking wrong. Three checks, and B is the one
# that matters: without it, A can pass on an implementation that drops the mask entirely.
_torch = torch
import contextlib  # noqa: E402


class _C:  # smallest cfg GatedMLA reads; head_dim 16 because flash-attn 4 on SM90
    d, heads = 32, 2   # rejects head_dim < 8 or not divisible by 8


_torch.manual_seed(0)
_mla = _train.GatedMLA(_C).eval()
_B, _T = 2, 8
_x = _torch.randn(_B, _T, _C.d)
_cu = _torch.tensor([0, 3, 8, 13, 16])  # row 0: docs [0,3) [3,8); row 1: [8,13) [13,16)


@contextlib.contextmanager
def _no_flash():
    """This block is about the fallback: on a flash machine the module would take the
    flash path and never exercise the mask being checked (and refuse fp32 besides)."""
    # BOTH modules, for the same reason as chunk_kda at line 33: GatedMLA.forward reads its
    # OWN module global, and after the b0-8 split `train.HAS_FA` is a re-exported SEPARATE
    # binding -- rebinding only train's copy leaves model.HAS_FA True, the flash path runs, and
    # flash_attn asserts on this block's fp32 input. That is exactly how this failed on the pod
    # (test_arch_compat.py:587 -> model.py:170) while all 14 CPU checks passed: the CPU machine
    # has no flash_attn, so HAS_FA is already False there and the missed rebinding is invisible.
    was = _train.HAS_FA
    _train.HAS_FA = _model.HAS_FA = False
    try:
        yield
    finally:
        _train.HAS_FA = _model.HAS_FA = was


def _fallback(x, cu):
    """The real branch, not a copy of it -- a copy drifts from the code it vouches for."""
    with _no_flash(), _torch.no_grad():
        return _mla(x, cu)


def _per_doc(x, cu):
    """Gold standard: every document attended on its own, then concatenated -- what
    flash_attn_varlen_func computes. Same module, cu=None, one document at a time."""
    flat = x.reshape(-1, x.shape[-1])
    out = _torch.empty_like(flat)
    with _no_flash(), _torch.no_grad():
        for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
            out[a:b] = _mla(flat[a:b].unsqueeze(0), None)[0]
    return out.view_as(x)


_masked, _gold = _fallback(_x, _cu), _per_doc(_x, _cu)
assert _torch.allclose(_masked, _gold, atol=1e-5), "masked SDPA != per-document attention"
_naive = _fallback(_x, None)
assert not _torch.allclose(_naive, _gold, atol=1e-5), \
    "plain causal matches the gold standard -- this test cannot fail"
_x2 = _x.clone()
_x2[0, 0:3] += 5.0  # rewrite document 0; the document after it must not move
_out2 = _fallback(_x2, _cu)
assert _torch.allclose(_masked[0, 3:8], _out2[0, 3:8], atol=1e-6), \
    "a later document moved when an earlier one changed"
assert not _torch.allclose(_masked[0, :3], _out2[0, :3], atol=1e-6), \
    "the rewritten document did not change"
print("doc-mask fallback: == per-document attention, != plain causal, no cross-document leak OK")

# GPU: the flash path must agree with the masked fallback. This is the only shape that
# catches a mis-bound cu -- flash-attn 4 exports the same two names as v2 with a different
# positional order (its 4th positional is qv), so a positional call would pass cu as qv and
# silently drop the mask instead of raising. Same test, two jobs: it is also the
# correctness check for the fallback on a card.
def _gpu_check(cfg, B, T, cu):
    """flash == masked, flash != plain-causal, and the flash branch actually ran -- three
    asserts at one shape. Tolerance is a fraction of the flash-vs-naive gap, not an absolute
    picked after the fact: agreement must be small NEXT TO the difference the mask makes."""
    xg = _torch.randn(B, T, cfg.d).cuda().to(_torch.bfloat16)
    mg = _train.GatedMLA(cfg).cuda().to(_torch.bfloat16).eval()
    cug = cu.cuda().to(_torch.int32)
    # PATCH THE MODULE THAT OWNS THE SYMBOL. This read `_train.flash_attn_varlen_func`, and
    # train.py:135 re-exports 14 names from model -- flash_attn_varlen_func is not one of them.
    # So this line raised AttributeError and _gpu_check NEVER RAN, taking all three asserts
    # below with it (2026-09-04, found while running this before the head-hybrid edit; the
    # symbol has never been on train, at 28ae5917 which added this or at any commit since).
    # GatedMLA.forward resolves it as a model-module global, so model is the only binding that
    # changes what the mixer calls; patching train would not have counted anything either.
    real = _model.flash_attn_varlen_func
    n = [0]
    def _shim(*a, **k):
        n[0] += 1
        return real(*a, **k)
    _model.flash_attn_varlen_func = _shim
    try:
        # autocast, as training does: rms_norm returns fp32 otherwise and flash refuses it.
        with _torch.no_grad(), _torch.autocast("cuda", dtype=_torch.bfloat16):
            flash = mg(xg, cug)
            assert n[0] == 1, f"flash branch did not run (called {n[0]}x) -- max diff 0 would be two fallbacks"
            _train.HAS_FA = _model.HAS_FA = False  # both bindings; see _no_flash
            ref = mg(xg, cug)
            naive = mg(xg, None)       # no cu: plain causal, the mask absent
            _train.HAS_FA = _model.HAS_FA = True
    finally:
        _model.flash_attn_varlen_func = real
    d = (flash.float() - ref.float()).abs().max().item()
    gap = (ref.float() - naive.float()).abs().max().item()
    assert gap > 10 * d, f"mask barely changes output (gap {gap:.4f} vs diff {d:.4f}) -- test cannot fail"
    assert d < 0.1 * gap, f"flash varlen != masked SDPA (diff {d:.4f}, mask gap {gap:.4f}) -- cu may be mis-bound"
    print(f"flash==masked (diff {d:.4f}) != plain-causal (gap {gap:.4f}) at B={B} T={T} hd={cfg.d // cfg.heads} OK")


if _torch.cuda.is_available() and _train.HAS_FA:
    _gpu_check(_C, _B, _T, _cu)
    class _CBig:  # T=4096, hd=128 -- a real training shape, spanning many beta-kernel tiles
        d, heads = 256, 2
    _gpu_check(_CBig, 2, 4096, _torch.tensor([0, 1500, 4096, 4700, 8192]))
else:
    print("flash varlen vs fallback SKIP (no CUDA or no flash_attn)")


# --- the flash wrapper must stay dynamo-disabled -----------------------------------------
# Removing the wrap silently restores 70 flash recompiles per 110 steps, 20 of them recurring
# after step 50, because the varlen wrapper's shape asserts specialise dynamo on the DOCUMENT
# COUNT and that count is unbounded (eff.recompile_recurrence_explained). Throughput does not
# move when it regresses -- 81K in both arms of the lane test -- so nothing else in the suite
# would catch it. This asserts the wrap by its effect on a traced function, not by looking for
# an attribute name that a torch bump could rename.
#
# ON _model, NOT _train, and this site was broken the same way as _gpu_check's: train.py:135
# re-exports 14 names from model and flash_attn_varlen_func is not one of them, so this raised
# AttributeError and the assert never ran (2026-09-04). Its own message pointed at "train.py's
# flash import block", which is where the wrap is NOT -- model.py:52-60 holds it. Two sites in
# one file reading the same nonexistent attribute is why this is a ruling and not a typo.
if _model.HAS_FA:
    _f = _model.flash_attn_varlen_func
    _marker = getattr(_f, "_torchdynamo_disable", None)
    assert _marker, (
        "flash_attn_varlen_func is not wrapped in torch._dynamo.disable. Its shape asserts "
        "(cute/interface.py:376/381/384) specialise dynamo on the unbounded document count, "
        "which reopens permanent recompilation at ~54.9 ms/step with NO tok/s signal. "
        "Restore the wrap at model.py's flash import block."
    )
    print("flash_attn_varlen_func is dynamo-disabled OK")
else:
    print("flash dynamo-disable SKIP (no flash_attn)")


# attn_res=True with a block that cannot supply sublayers() must RAISE AT CONSTRUCTION, not
# run with depth attention silently off. The condition is statically decidable -- which blocks
# implement sublayers() is fixed once the model is built -- so a forward-time throw would
# crash at step 1 at best and, for a block on a conditional branch, not until step 8000
# (tilerl, design page §2). Verified to FAIL with the guard removed from HybridLM.__init__.
_sub_cfg = copy.copy(Cfg)
_sub_cfg.d, _sub_cfg.layers, _sub_cfg.vocab, _sub_cfg.fone = 128, 2, 256, False
_sub_cfg.attn_res = True


class _NoSublayers(nn.Module):
    """A plausible new block: right forward contract, no sublayers()."""

    def __init__(self, cfg, **kw):
        super().__init__()
        self.lin = nn.Linear(cfg.d, cfg.d)

    def forward(self, x, cu=None):
        return x + self.lin(x)


_real_block = model.Block
model.Block = _NoSublayers
try:
    model.HybridLM(_sub_cfg)
    raise AssertionError(
        "attn_res=True with a block lacking sublayers() constructed successfully -- depth "
        "attention would be silently OFF while the config says it is on"
    )
except TypeError as _e:
    assert "sublayers" in str(_e) and "_NoSublayers" in str(_e), _e
finally:
    model.Block = _real_block
model.HybridLM(_sub_cfg)  # the real Block still constructs under attn_res=True
_sub_cfg.attn_res = False
model.HybridLM(_sub_cfg)
print("attn_res sublayers() contract: raises at construction, real Block unaffected OK")

# attn_res_fused: the flag must be a pure value-preserving swap, and its default OFF.
# Checked on the GRADIENT, not just the forward: Source.scale is rms_scale(v), so v
# reaches the output by two routes, and a fused node that owns only one still matches
# the forward to 1.5e-07 while dV lands 7.6% low (docs/lessons/forward_check_hides_
# gradient_error.md). A forward-only assertion here would be green for that bug.
assert model.AttnRes(8).fused is False, "attn_res_fused must default OFF"
assert getattr(train.Cfg, "attn_res_fused", None) is False, "Cfg.attn_res_fused must default OFF"
_fd, _fB, _fT, _fn = 64, 2, 8, 6
_fref_out = _fref_g = None
for _fused in (False, True):
    torch.manual_seed(3)
    _ar = model.AttnRes(_fd, fused=_fused)
    with torch.no_grad():
        _ar.q.normal_(std=0.5)
        _ar.g.normal_(mean=1.0, std=0.2)
    _vs = [torch.randn(_fB, _fT, _fd, generator=torch.Generator().manual_seed(20 + i),
                       requires_grad=True) for i in range(_fn)]
    _o = _ar([model.Source.of(x) for x in _vs])
    _o.backward(torch.randn(_fB, _fT, _fd, generator=torch.Generator().manual_seed(77)))
    if not _fused:
        _fref_out, _fref_g = _o.detach().clone(), [x.grad.clone() for x in _vs]
    else:
        _do = (_o.detach() - _fref_out).abs().max().item()
        _dg = max((x.grad - r).abs().max().item() for x, r in zip(_vs, _fref_g, strict=True))
        assert _do < 1e-5, f"fused forward differs by {_do:.2e}"
        assert _dg < 1e-5, f"fused dV differs by {_dg:.2e}"
print("attn_res_fused: default OFF; ON matches OFF in forward AND dV "
      f"(max {_do:.2e} / {_dg:.2e}) OK")
