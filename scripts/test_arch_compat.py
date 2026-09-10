# restartable: a CPU self-check, seconds end to end. Its only write is a mix.json fixture
# into tempfile.mkdtemp(), which nothing reads back -- an interrupt costs a rerun, not work.
"""Self-check for train.py architecture changes.

Runs on CPU where fla is absent (a shape-preserving stand-in replaces the Triton
kernel) and on CUDA where fla is present -- the real chunk_kda cannot take CPU
tensors, so a fla machine without a visible GPU exits loudly instead of silently
skipping. On CUDA the model runs under the same bf16 autocast training uses
(train.py's train-loop autocast on amp_dtype): FlashAttention refuses fp32, so the CUDA path never executed
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
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

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
for _ar_on, _sp_on in ((True, True), (True, False), (False, True), (False, False)):
    Cfg.attn_res, Cfg.mem_sparse = _ar_on, _sp_on
    torch.manual_seed(5)
    _mm = HybridLM(Cfg).to(DEV)
    assert _mm.memory is not None and _mm.mem_layers == [1, 3], (_mm.mem_layers,)
    with _amp():
        _mh, _ = _mm(x, y)
    _mh.float().sum().backward()
    _md = _mm.memory.diagnostics()
    assert torch.isfinite(_mh).all(), f"memory forward not finite (attn_res={_ar_on} mem_sparse={_sp_on})"
    assert _md["touched_rows"] > 0, (
        f"attn_res={_ar_on} mem_sparse={_sp_on}: the memory was reached by NO token, so this arm trains as the "
        f"control while its flags say otherwise -- the 2026-09-05 sublayers() defect")
    assert _mm.memory.values.weight.grad is not None, "value table got no gradient"
    # THE GRAD KIND MUST FOLLOW THE FLAG, in both directions. Asserting is_sparse
    # unconditionally would have gone red the moment the arms switched to mem_sparse=False, and
    # asserting nothing would let the flag mean nothing. M1/M2 run FALSE (measured on the pod
    # 2026-09-05: NCCL raises "does not support all_reduce with sparse tensors" on a COO grad, so
    # DDP has no sparse path here); the sparse form stays supported for the single-process case
    # and for M3, which is measured both ways.
    assert _mm.memory.values.weight.grad.is_sparse == bool(Cfg.mem_sparse), (
        f"mem_sparse={Cfg.mem_sparse} but grad.is_sparse="
        f"{_mm.memory.values.weight.grad.is_sparse}: the flag does not describe what ran")
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
print("memory layers: fwd/bwd on all 4 (attn_res, mem_sparse) combinations, grad kind follows the "
      "flag, keys learn, one shared pool registered once, round-trip exact, legacy ckpt loads, "
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

# ------------------------------------------- mem_sel_lr and mem_query_norm (added 2026-09-05)
# WHY THESE FLAGS EXIST: M1/M2/M3 were stopped under readout 4 with key-usage collapse (M1
# pool_touched_frac 0.0945 at step 1000, key_gini 0.9192), and both flags are candidate fixes for
# the SELECTION path. Every case below is about a DEFAULT that must reproduce those arms exactly,
# because a flag whose default changes the architecture makes every measurement before it
# incomparable while all three arm names stay the same.

# 1. THE DEFAULTS ARE A BIT-FOR-BIT NO-OP. Asserted on the two things a change could move: the
# state dict's key set (a new parameter or buffer) and the optimizer grouping (a re-routed
# tensor). Comparing forward outputs would NOT catch a new parameter that happens to start at
# identity, and comparing group counts alone would not catch a tensor moving between two groups
# that both exist.
Cfg.mem_values, Cfg.mem_top_k, Cfg.mem_layers, Cfg.mem_sparse = _MV, _MK, "1,3", True
Cfg.attn_res, Cfg.grad_ckpt = True, False
assert Cfg.mem_sel_lr <= 0, f"Cfg.mem_sel_lr default {Cfg.mem_sel_lr} is not the no-split sentinel"
assert Cfg.mem_query_norm == "none", f"Cfg.mem_query_norm default is {Cfg.mem_query_norm!r}"
torch.manual_seed(7)
_dflt = HybridLM(Cfg).to(DEV)
_dflt_keys = set(_dflt.state_dict())
_dflt_groups = {o.aupai_group: [n for n, p in _dflt.named_parameters()
                                if any(p is q for g in o.param_groups for q in g["params"])]
                for o in build_optimizers(_dflt, Cfg)}
assert "mem_sel" not in _dflt_groups, (
    "the default built a mem_sel group. Then M1's relaunch would not be M1: the selector would "
    "train at a different lr with nothing in the launch line saying so")
assert set(_dflt_groups["mem"]) == {n for n, _ in _dflt.named_parameters() if train._is_mem_fqn(n)}, (
    "at the default, ALL memory parameters must be in the one `mem` group, as M1/M2/M3 ran")

# 2. THE SPLIT MOVES query+keys AND NOTHING ELSE, and the union is unchanged. A split that dropped
# a tensor would leave it un-optimised -- read every forward, never updated -- which is the §184
# shape: the diagnostics would still look healthy because `touched` counts reads.
Cfg.mem_sel_lr = 0.002
_spl = HybridLM(Cfg).to(DEV)
_spl_groups = {o.aupai_group: [n for n, p in _spl.named_parameters()
                               if any(p is q for g in o.param_groups for q in g["params"])]
               for o in build_optimizers(_spl, Cfg)}
assert set(_spl_groups["mem_sel"]) == {"memory.query.weight", "memory.keys"}, _spl_groups["mem_sel"]
assert set(_spl_groups["mem"]) | set(_spl_groups["mem_sel"]) == set(_dflt_groups["mem"]), (
    f"the split changed WHICH parameters the memory groups hold, not just how they are divided: "
    f"lost {set(_dflt_groups['mem']) - set(_spl_groups['mem']) - set(_spl_groups['mem_sel'])}, "
    f"gained {set(_spl_groups['mem']) | set(_spl_groups['mem_sel']) - set(_dflt_groups['mem'])}")
assert not set(_spl_groups["mem"]) & set(_spl_groups["mem_sel"]), (
    "a parameter is in BOTH memory groups, so it is stepped twice per step")
_sel_opt = {o.aupai_group: o for o in build_optimizers(_spl, Cfg)}
assert _sel_opt["mem_sel"].param_groups[0]["lr"] == 0.002, "mem_sel group is not at Cfg.mem_sel_lr"
assert _sel_opt["mem"].param_groups[0]["lr"] == Cfg.mem_lr, "the split changed the table's lr"
# AND THE NAMES STILL LINE UP WITH THE OPTIMIZERS. build_optimizers zips two lists, so a new
# optimizer appended without its name would shift every label after it and the step line would
# report the selector's lr as the table's -- the number would look plausible and be wrong.
for _o in build_optimizers(_spl, Cfg):
    assert getattr(_o, "aupai_group", None), f"an optimizer got no group name: {type(_o).__name__}"
Cfg.mem_sel_lr = -1.0

# 3. set_schedule MUST SCALE THE NEW GROUP TOO. It is a separate optimizer, and a group that
# never enters the warmup loop would run at full lr from step 1 while every printed lr said
# otherwise -- and at step 1 that is the largest lr the selector ever sees.
Cfg.mem_sel_lr = 0.002
_sched = {o.aupai_group: o for o in build_optimizers(HybridLM(Cfg).to(DEV), Cfg)}
train.set_schedule(list(_sched.values()), 1, 1000, Cfg)
_warm = _sched["mem_sel"].param_groups[0]["lr"]
assert 0 < _warm < 0.002, (
    f"mem_sel lr is {_warm} at step 1 of a 1000-step run with warmup {Cfg.warmup}: the group is "
    f"not being warmed up, so the selector takes its largest steps before anything is learned")
train.set_schedule(list(_sched.values()), Cfg.warmup, 1000, Cfg)
assert abs(_sched["mem_sel"].param_groups[0]["lr"] - 0.002) < 1e-9, (
    "mem_sel lr does not reach its initial_lr at the end of warmup")
Cfg.mem_sel_lr = -1.0

# 4. EACH query_norm VALUE CHANGES THE SELECTION, and that is the assertion -- not that the
# forward runs. A normalisation wired in but never reaching the scores would give a finite loss
# and an unchanged top-k, which is exactly the silent no-op these three cells are meant to
# distinguish. Same seed and same input for all three, so a difference is the flag.
#
# READ IN TRAIN MODE, and that is not a detail. BatchNorm in EVAL uses running stats, which at
# init are mean 0 var 1 -- an affine identity -- so an eval read of the bn arm selects EXACTLY the
# rows the none arm does (measured: identical touched buffers, 235 rows both). An eval-mode
# fixture would therefore report bn as a no-op when it is not, and would have sent the next reader
# after a defect in the wiring. The arms train in train mode; that is where the flag has an effect.
#
# THE SNAPSHOT IS TAKEN BEFORE diagnostics(reset=True), which zeroes `touched`. The first version
# of this case cloned the buffer AFTER the reset, so all three arms compared as all-zero and
# EQUAL -- the assertion below then fired on the l2 arm, which does differ. A fixture that reads
# the counter it just cleared cannot see any arm's selection.
_sel_sets = {}
for _qn in ("none", "l2", "bn"):
    Cfg.mem_query_norm = _qn
    torch.manual_seed(21)
    _qm = HybridLM(Cfg).to(DEV)
    _qm.train()
    with _amp():
        _qh, _ = _qm(x, y)
    assert torch.isfinite(_qh).all(), f"query_norm={_qn} forward not finite"
    _rows = _qm.memory.touched.clone()          # BEFORE the reset below
    _qh.float().sum().backward()
    _d = _qm.memory.diagnostics(reset=True)
    assert _d["touched_rows"] > 0, f"query_norm={_qn}: no row was read"
    assert _qm.memory.values.weight.grad is not None, f"query_norm={_qn}: table got no gradient"
    assert int(_rows.sum()) == _d["touched_rows"], (
        f"query_norm={_qn}: the snapshot holds {int(_rows.sum())} rows and diagnostics reports "
        f"{_d['touched_rows']} -- the clone is not the window this row describes")
    _sel_sets[_qn] = (_rows, _d["touched_rows"])
    # The new parameters exist only in the arm that declares them, or a checkpoint from one arm
    # would load into another and the flag would be recorded in cfg while the weights disagreed.
    _extra = set(_qm.state_dict()) - _dflt_keys
    if _qn == "none":
        assert not _extra, f"query_norm=none added state: {_extra}"
    elif _qn == "l2":
        assert _extra == {"memory.q_log_temp"}, _extra
    else:
        assert any(k.startswith("memory.q_bn.") for k in _extra), _extra
# THE THREE ARMS MUST NOT ALL SELECT THE SAME ROWS. Compared on the touched-row SET rather than a
# count: two arms can read the same NUMBER of distinct rows while reading different ones, and it
# is which rows that the collapse is about.
Cfg.mem_query_norm = "none"
_base_rows = _sel_sets["none"][0]
for _qn in ("l2", "bn"):
    assert not torch.equal(_sel_sets[_qn][0], _base_rows), (
        f"query_norm={_qn} selected exactly the same rows as none, so the branch is not reaching "
        f"the scores the top-k reads and this arm would be the control under another name")

# 4b. THE l2 ARM MUST NORMALISE THE KEYS, asserted THROUGH THE REAL FORWARD. Deleting the key
# side is the mutant that survived the first version of this case (2026-09-05): that version
# computed the comparison from the module's parts with its own einsum, so it never executed the
# branch it was testing -- a fixture built beside the implementation instead of on it.
#
# THE KNOWN ANSWER: normalising an already-unit vector is the identity. So a module whose keys are
# PRE-normalised must produce the same output as one whose keys are raw, because the correct
# forward normalises them itself. If the forward does not, the raw-key module reads different keys.
#
# JUDGED AGAINST THE none ARM'S OWN SENSITIVITY, not against a tolerance I choose. Exact equality
# fails on correct code: F.normalize is not bitwise idempotent, and under amp the two paths differ
# by ~5e-07. A hand-picked epsilon would be a number with no basis, so the reference is the SAME
# substitution on the none arm, which does not normalise and therefore shows what "the keys really
# changed" is worth on this input. Measured 2026-09-05: correct l2 5e-07, none arm 1e-02, the
# mutant that deletes the key line 1e-02 -- four orders between the two worlds, and the assertion
# is that l2 sits an order below the none arm rather than at it.
Cfg.mem_query_norm = "none"
torch.manual_seed(31)
_nna = HybridLM(Cfg).to(DEV).eval()
_nnb = copy.deepcopy(_nna)
with torch.no_grad():
    _nnb.memory.keys.copy_(F.normalize(_nnb.memory.keys, dim=-1))
    _yc, _ = _nna(x, y)
    _yd, _ = _nnb(x, y)
_none_delta = (_yc.float() - _yd.float()).abs().max().item()
assert _none_delta > 1e-4, (
    f"pre-normalising the keys moved the none arm's output by only {_none_delta:.3e}, so this "
    f"input cannot tell the two key sets apart and the comparison below is vacuous")
Cfg.mem_query_norm = "l2"
torch.manual_seed(31)
_l2a = HybridLM(Cfg).to(DEV).eval()
_l2b = copy.deepcopy(_l2a)
with torch.no_grad():
    _l2b.memory.keys.copy_(F.normalize(_l2b.memory.keys, dim=-1))
    _ya, _ = _l2a(x, y)
    _yb, _ = _l2b(x, y)
_l2_delta = (_ya.float() - _yb.float()).abs().max().item()
assert _l2_delta < _none_delta / 10.0, (
    f"the l2 arm does not normalise the keys inside forward: pre-normalising them moved its "
    f"output by {_l2_delta:.3e}, against {_none_delta:.3e} for the none arm doing the same "
    f"substitution -- i.e. it responded like a module that reads the keys raw. Query-side L2 "
    f"alone cannot reorder a half's top-k (every score in a row is scaled by the same positive "
    f"number), so without the key side the arm cannot change key concentration, which is what "
    f"key_gini measures and what collapsed on M1 (0.9192)")
# AND THE PRE-NORMALISATION IS NOT VACUOUS: if the keys were already unit vectors at init, both
# deltas would be ~0 and the ratio above would be meaningless.
assert (_l2a.memory.keys.norm(dim=-1) - 1.0).abs().max() > 1e-3, (
    "the keys are already unit-norm at init, so both deltas are zero and this case proves "
    "nothing -- re-derive it against the real init scale (key_dim ** -0.5)")

# 5. AN UNKNOWN VALUE MUST RAISE, not fall back to none. A typo that silently ran the control
# would report a clean null for a change it never applied -- the §177 shape, and the reason
# argparse also restricts this flag to a choice list.
Cfg.mem_query_norm = "layernorm"
try:
    HybridLM(Cfg)
    raise AssertionError("query_norm='layernorm' was accepted; the arm would run as the control")
except ValueError as _e:
    assert "query_norm" in str(_e), _e
Cfg.mem_query_norm = "none"
Cfg.mem_values, Cfg.mem_layers, Cfg.mem_top_k = 0, "3,6,9", 32
print("mem_sel_lr/mem_query_norm: defaults are a no-op (same state keys, same one mem group), "
      "the split moves exactly query+keys and preserves the union, set_schedule warms the new "
      "group, all three norms change which rows are selected, unknown value raises OK")
Cfg.mem_values, Cfg.mem_top_k, Cfg.mem_layers, Cfg.mem_sparse = _MV, _MK, "1,3", True

# READOUT 6 AND THE TABLE MASTER, as ONE known-answer case with both worlds, because either half
# alone proves nothing: a checksum probe that never fires cannot show the master works, and a
# master with no probe cannot show the bf16 table was broken.
#
# THE ARITHMETIC IS THE CASE, and it has to be derived rather than guessed. Weights are 1.0, where
# the bf16 ULP is 2^-8 and anything under half of it (2^-9 = 1.953e-3) rounds back to 1.0 on the
# add. Adagrad's step for a constant gradient is lr*g/sqrt(k*g^2) = lr/sqrt(k), so it DECAYS: the
# per-step update is lr and the sum over n steps is lr*sum(1/sqrt(k)), not n*lr. At lr = 2^-11 the
# per-step update is 4.88e-4, four times under the half-ULP, so every single step rounds away; the
# sum over 20 steps is 3.71e-3, 1.90 times OVER it, so an fp32 master that keeps the steps crosses
# the threshold and the bf16 weight moves. Both numbers are needed: a per-step update above the
# half-ULP would move the table without a master (no negative world) and a 20-step sum below it
# would move nothing with one (no positive world).
#
# NO MODEL, NO FORWARD, NO CARD: the mechanism under test is dtype arithmetic in the optimizer, and
# a real forward would make the gradient a function of the lookup, so a failure could not be
# attributed. The gradient is written by hand for exactly that reason.
_RP = torch.Generator().manual_seed(4242)
_probe_vec = torch.randn(8, generator=_RP, dtype=torch.float32)


def _row_sums(t):
    return t.detach().float() @ _probe_vec


def _rounding_world(with_master, steps=20, lr=2.0 ** -11):
    """(rows whose checksum moved, the table after `steps` Adagrad steps).

    THE REAL TableMaster, not a re-implementation of it. An earlier version of this case inlined
    `m.grad = g.float()` and `tbl.copy_(m)` by hand, and a mutant that emptied TableMaster.push
    left it GREEN: the case was testing its own fixture. Driving the class means the mutant kills
    it. The module is a stub named to satisfy _is_mem_fqn -- TableMaster selects by fqn, so the
    name is part of what is under test.
    """
    holder = nn.Module()
    holder.values = nn.Embedding(4, 8)
    holder.values.weight = nn.Parameter(torch.ones(4, 8, dtype=torch.bfloat16))
    root = nn.Module()
    root.memory = holder          # fqn becomes "memory.values.weight"
    tbl = root.memory.values.weight
    before = _row_sums(tbl)
    if with_master:
        tm = train.TableMaster(root)
        assert len(tm.pairs) == 1, "TableMaster did not select the stub's value table by fqn"
        opt = torch.optim.Adagrad([tm.pairs[0][1]], lr=lr)
    else:
        tm = None
        opt = torch.optim.Adagrad([tbl], lr=lr)
    for _ in range(steps):
        tbl.grad = torch.ones_like(tbl)        # same sign every step: updates accumulate
        if tm is not None:
            tm.pull_grads()
            opt.step()
            opt.zero_grad(set_to_none=True)
            tm.push()
        else:
            opt.step()
            opt.zero_grad(set_to_none=True)
    return int((_row_sums(tbl) != before).sum()), tbl


_neg_moved, _neg_tbl = _rounding_world(with_master=False)
_pos_moved, _pos_tbl = _rounding_world(with_master=True)
assert _neg_moved == 0, (
    f"NEGATIVE world: {_neg_moved} of 4 rows moved in a bf16 table stepped at lr 2^-11 on weights "
    f"of 1.0. Each Adagrad step is 4.88e-4 against a bf16 half-ULP of 1.95e-3, so every step must "
    f"round back to 1.0 -- if rows moved, this case can no longer demonstrate the rounding it "
    f"exists for and the lr must be re-derived before the positive half means anything")
assert _pos_moved == 4, (
    f"POSITIVE world: only {_pos_moved} of 4 rows moved WITH an fp32 master. 20 steps of lr/sqrt(k) "
    f"at lr 2^-11 sum to 3.71e-3, 1.90x the 1.95e-3 half-ULP, so the accumulation does cross it: "
    f"this failing means the master is not reaching the table (TableMaster.push), not that the "
    f"sum was too small")
# The two worlds must differ in the TABLE, not merely in the counter: a probe that reported a
# change no reader could see in the weights would be measuring itself.
assert not torch.equal(_pos_tbl.detach(), _neg_tbl.detach()), (
    "both worlds produced the identical bf16 table, so the checksum difference above cannot be "
    "coming from the master -- the probe is reporting something other than the weights")

# THE REAL METHODS ON THE REAL MODULE, so the hand-rolled loop above cannot pass while
# ProductKeyMemory's own accounting is wrong. First call arms and returns -1; a call with nothing
# changed returns 0; a call after touching one row returns 1.
Cfg.attn_res = True
torch.manual_seed(12)
_rm = HybridLM(Cfg).to(DEV).memory
assert _rm.note_row_changes() == -1, (
    "the FIRST note_row_changes must report -1, not 0: with no previous checksum there is nothing "
    "to compare, and a 0 there is the same number a permanently frozen table reports")
assert _rm.note_row_changes() == 0, "an untouched table reported a changed row"
with torch.no_grad():
    _rm.values.weight[7] += 1.0
assert _rm.note_row_changes() == 1, (
    "one row was changed by 1.0 and the checksum probe did not see it")
# THE BASELINE MUST ADVANCE. Without this call a mutant that counts against the ORIGINAL checksums
# forever still passes the three above: -1, 0, then 1 for the row it just changed. It is this
# fourth call -- nothing changed since the third -- that separates "changed since the previous diag
# step" from "changed since the run began", and the second is the wrong quantity: it converges to
# every touched row and stops being able to see the table freeze, exactly as a cumulative
# `touched` would.
assert _rm.note_row_changes() == 0, (
    "the row changed in the PREVIOUS window is still counted as changed, so the baseline is not "
    "advancing and the field reports change-since-start rather than change-since-previous")
# THE fp32 ACCUMULATION IS ITSELF A CLAIM, and neither a +1.0 change nor a whole-row nudge tests
# it -- both are visible in any precision. The case has to be the SMALLEST change the table can
# hold: one ULP in ONE element. That means the table must be bf16 HERE, as it is on the arms
# (train.py's fp8-branch bf16 cast), because the models built above are fp32 and one fp32 ULP is below the
# resolution of any dot product over 64 terms -- a case written on the fp32 model is red for both
# the right and the wrong reason. Measured at this d=64: with a bf16 table, a one-element bump moves
# the fp32 projection and leaves a bf16 one bit-identical, each bf16 partial sum being ~64x the
# change; at four bumped elements both see it, so the separation is about the accumulator and not
# the magnitude. THROUGH THE REAL row_checksums: a mutant reducing the probe to bf16 survived every
# assertion above, and an arithmetic case written beside the method would not have caught it either.
# THE fp32 ACCUMULATION IS ITSELF A CLAIM, and it can only be tested at the REAL row width. Whether
# a one-ULP change in one element survives a bf16 accumulator is decided by probe[0] / |sum(probe)|:
# at d=64 that ratio is order 1 and the answer flips with the seed -- measured, a bf16-probe mutant
# survived on one row and died on another. At d=1024 the partial sums grow as sqrt(d) while the
# change does not, so the ratio is ~1/32 and a bf16 accumulator is reliably blind. So this case
# builds a 64-value pool at d=1024 (tiny: 64 rows) rather than reusing the d=64 test model, and
# drives ProductKeyMemory's own method.
_rmb = model.ProductKeyMemory(64, 1024, top_k=8, sparse=False).to(DEV).to(torch.bfloat16)
assert _rmb.values.weight.dtype is torch.bfloat16, "the cast did not reach the value table"
# The row is set to a known value: at its init the bump's visibility depends on that row's
# magnitude, which is the seed dependence this case exists to avoid.
with torch.no_grad():
    _rmb.values.weight[13].fill_(1.0)
assert _rmb.note_row_changes() == -1, "baseline call on the bf16 module"
with torch.no_grad():
    _e = _rmb.values.weight[13, :1]
    _e.copy_(torch.nextafter(_e, torch.full_like(_e, 1e4)))
assert float(_e) != 1.0, "nextafter did not move the element by one bf16 ULP"
assert _rmb.note_row_changes() == 1, (
    "a ONE-ULP change in ONE element of a bf16 row at d=1024 was not resolved by row_checksums, so "
    "readout 6 under-reports precisely the small updates it exists to detect -- the projection must "
    "accumulate in fp32 over the bf16 table, not in bf16")
# THE BLOCK LOOP MUST BE EXERCISED, and no table above reaches one block: at 65,536 rows the pools
# used here (64, 256) all fit in the first iteration, so a regression to a single whole-table
# `weight.float()` would pass every assertion so far. This case makes the block smaller than the
# table for the duration, so the loop runs more than once and its boundary arithmetic is tested --
# a row in the LAST partial block is the one an off-by-one drops.
_saved_blk = model._ROW_CHECKSUM_BLOCK
try:
    model._ROW_CHECKSUM_BLOCK = 7          # 256 rows -> 37 blocks, the last one partial
    _rb = model.ProductKeyMemory(256, 64, top_k=8, sparse=False).to(DEV)
    assert _rb.note_row_changes() == -1, "baseline call on the chunked module"
    with torch.no_grad():                  # row 255 is in the final, partial block
        _rb.values.weight[255] += 1.0
    assert _rb.note_row_changes() == 1, (
        "a change in the LAST row was not seen: the block loop drops its final partial block, so "
        "every row past the last whole block is invisible to readout 6")
    with torch.no_grad():                  # row 0 is in the first block
        _rb.values.weight[0] += 1.0
    assert _rb.note_row_changes() == 1, "a change in the FIRST row was not seen"
    # DETERMINISM ACROSS CALLS is the property the `!=` comparison downstream needs, and it is NOT
    # the same as agreeing with an unchunked projection. Measured here: the chunked result differs
    # from `weight.float() @ probe` by up to 2.98e-07 on values of order 0.81 -- one fp32 ULP, from
    # a different summation order in the BLAS call, on 182 of 256 rows. That difference is
    # harmless because both sides of every real comparison are chunked; what would NOT be harmless
    # is the same table producing two different checksums on two calls, which would make every row
    # read as changed at every diag step and report rows_changed_since_prev as 1.0 forever.
    assert torch.equal(_rb.row_checksums(), _rb.row_checksums()), (
        "row_checksums is not deterministic for an unchanged table, so consecutive diag steps "
        "would compare two different projections and every row would read as changed")
    # THE BLOCK CONSTANT MUST GOVERN THE COMPUTATION, and no correctness assertion can establish
    # that: an unchunked `weight.float() @ probe` returns the RIGHT answer while allocating a
    # full-size fp32 temporary -- 5.45 GiB at side 1195 -- which is a peak-memory regression that
    # every check above passes. Measured, a mutant reverting the loop survived all of them. What
    # separates the two is that chunking changes the SUMMATION ORDER: at block 7 the result differs
    # from the whole-table product by up to 2.98e-07 on 182 of 256 rows. So a block size that
    # changes nothing in the output means the constant is not being read.
    _small = _rb.row_checksums().clone()
    model._ROW_CHECKSUM_BLOCK = 1 << 30          # one block: the whole table at once
    _whole = _rb.row_checksums()
    assert not torch.equal(_small, _whole), (
        "row_checksums returns bit-identical results at block 7 and block 2^30, so the block loop "
        "is not running and the fp32 temporary is the whole table -- 5.45 GiB at side 1195, "
        "charged to the peak at every diag step")
finally:
    model._ROW_CHECKSUM_BLOCK = _saved_blk
# NON-PERSISTENT, all four buffers: a window counter in the checkpoint would make two saves of the
# same weights differ, and row_sum_prev is 4 bytes per row -- 4 MiB at M1 -- of pure scratch.
_sd = HybridLM(Cfg).state_dict()
_leaked = [k for k in _sd if k.split(".")[-1] in
           ("row_probe", "row_sum_prev", "rows_changed", "row_probe_armed")]
assert not _leaked, f"readout 6 buffers leaked into the checkpoint: {_leaked}"

# TWO MASTERS, DISJOINT. build_optimizers takes ONE merged map, so an overlap would mean a
# parameter reachable as both the model tensor and a master -- the shape that lets an optimizer
# step a tensor the model no longer uses.
_mm = HybridLM(Cfg).to(DEV)
_mw = train.MasterWeights(_mm)
_tm = train.TableMaster(_mm)
assert _tm.pairs, "TableMaster matched no value table -- this case would be vacuous"
assert len(_tm.pairs) == 1, f"TableMaster took {len(_tm.pairs)} tensors; only the value table"
_tw = _mm.memory.values.weight
assert _tm.pairs[0][0] is _tw, "TableMaster's master is not of the value table"
assert not (set(map(id, _mw.map)) & set(map(id, _tm.map))), (
    "MasterWeights and TableMaster claim a parameter in common, so the merged map that "
    "build_optimizers reads depends on insertion order")
assert _tw not in _mw.map and any(p is _tw for p in _mw.unmastered), (
    "the value table is in MasterWeights' map: it must be excluded there and mastered by "
    "TableMaster, or --fp32_master would give it two masters")
# pull_grads must CLEAR the model's grad, which is the half that was deliberately absent while the
# table was unmastered. Leaving it would make the next backward accumulate into a running sum.
_tw.grad = torch.ones_like(_tw)
_tm.pull_grads()
assert _tw.grad is None, (
    "TableMaster.pull_grads left p.grad in place. The optimizer holds the master, so its "
    "zero_grad clears the master's grad only, and the next backward would accumulate on top of "
    "this one -- the running-sum bug MasterWeights.pull_grads records")
assert _tm.pairs[0][1].grad is not None and _tm.pairs[0][1].grad.dtype is torch.float32, (
    "the master did not receive an fp32 copy of the gradient")
Cfg.mem_values, Cfg.mem_layers = 0, "3,6,9"
print("readout 6 + table master: bf16 at lr 2^-11 freezes all 4 rows over 20 steps, an fp32 master "
      "moves all 4; note_row_changes -1/0/1; buffers non-persistent; the two masters are disjoint OK")
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

# de-106: _doc_id_per_pos replaces torch.bucketize with a broadcast comparison so torch.compile
# on CUDA does not meet the bucketize lowering's SliceView.get_stride (cu[1:] slice). The math
# must stay right-bucketize exactly over random packed layouts (varying batch and doc lengths).
# Compile-cleanliness itself is a CUDA property -- proven single-card on the pod, since the CPU
# bucketize lowering falls back to eager and never calls _boundaries_helper.
_torch.manual_seed(7)
for _ in range(200):
    _B = int(_torch.randint(1, 9, (1,)).item())
    _ends = _torch.sort(_torch.randint(1, 5000, (_B,))).values
    _cu = _torch.cat([_torch.zeros(1, dtype=_torch.long), _ends.cumsum(0)]).to(_torch.int32)
    _pos = _torch.arange(int(_ends[-1].item()))
    _ref = _torch.bucketize(_pos, _cu[1:], right=True)
    _got = model._doc_id_per_pos(_pos, _cu[1:].to(_torch.long))
    assert _torch.equal(_ref, _got), "_doc_id_per_pos disagrees with right-bucketize"
print("de-106: _doc_id_per_pos == right-bucketize over 200 random packed layouts OK")

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
    # train.py's `from model import` block re-exports 14 names -- flash_attn_varlen_func is not one of them.
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
# ON _model, NOT _train, and this site was broken the same way as _gpu_check's: train.py's `from model import`
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

# _n_active_params: MFU's denominator counts what a TOKEN multiplies, and the dense arm is
# bit-identical (de-71, 62's filing 2026-09-08).
#
# Cfg SUBCLASSES, not kwargs: Cfg is a plain class the model reads attributes off, so
# `Cfg(dim=64)` raises TypeError -- the same shape the file's other variant worlds use.


class _CfgPaDense(Cfg):
    # `layers`, NOT `n_layer` (train.Cfg.layers). Cfg has no n_layer, nothing reads it, and setting
    # left every world at the default depth 12 -- caught only because the mixed world below
    # asserts that its two predicates DISAGREE, and at depth 12 with moe_layers "0-1" it was
    # already mixed, so the all-MoE world was never all-MoE either.
    d = 64
    layers = 2
    vocab = 256
    heads = 4
    ffn_hidden = 128
    moe_experts = 0


class _CfgPaMoE(_CfgPaDense):
    # EQUAL-ACTIVE PARITY IS ENFORCED BY MoEFFN: (moe_top_k + moe_shared) * moe_expert_ffn must
    # equal ffn_hidden exactly, or construction raises -- so the world's numbers are not free.
    # ffn_hidden 128 with moe_expert_ffn 32 admits top_k 3 (3+1)*32 == 128.
    moe_experts = 8
    moe_top_k = 3
    moe_shared = 1
    moe_expert_ffn = 32
    moe_layers = "0-1"   # == every block, since layers is 2 here


# THE DENSE CASE IS THE LOAD-BEARING ONE. 62's acceptance criterion is that the dense arm's MFU
# come out unchanged, because a fix that lowers the MoE number by moving the shared denominator is
# indistinguishable from the right change by reading only the MoE line. Asserted as exact equality
# against the OLD expression, recomputed from the same primitives -- not "within rounding", since
# with moe_experts 0 nothing should match at all.
_pa_dense = HybridLM(_CfgPaDense)
_pa_old = (sum(p.numel() for p in _pa_dense.parameters())
           - sum(p.numel() for n, p in _pa_dense.named_parameters() if _train._is_mem_fqn(n)))
assert _train._n_active_params(_pa_dense, _CfgPaDense) == _pa_old, (
    f"dense arm's MFU denominator moved: {_train._n_active_params(_pa_dense, _CfgPaDense)} vs "
    f"{_pa_old}. Every arm shares this line, so a changed dense denominator is a changed "
    f"baseline, not a fix")

# THE MoE CASE, on a real model rather than arithmetic over config.
_pa_moe = HybridLM(_CfgPaMoE)
_pa_total = sum(p.numel() for p in _pa_moe.parameters())
_pa_routed = sum(p.numel() for p in _pa_moe.parameters() if p.dim() == 3 and p.shape[0] == _CfgPaMoE.moe_experts)
assert _pa_routed > 0, "the MoE world has no 3-D routed stack; the shape predicate has no subject"
_pa_act = _train._n_active_params(_pa_moe, _CfgPaMoE)
assert _pa_act == _pa_total - _pa_routed + _pa_routed * _CfgPaMoE.moe_top_k // _CfgPaMoE.moe_experts, (
    f"active {_pa_act} != total {_pa_total} - routed {_pa_routed} + its top_k share")
assert _pa_act < _pa_total, "active must be below total when experts are routed"


# top_k >= experts: every expert is reached, so the denominator is the full total and the
# subtraction must be zero rather than negative. Parity forbids constructing such a MODEL
# ((8+1)*32 != 128), so the CONFIG is varied against the model already built -- which is exactly
# what the function reads: it takes cfg and model separately.
#
# BOTH == AND >, because == alone leaves the clamp untested: at k == e the expression
# `n_routed * k // e` already equals n_routed and min() changes nothing, so dropping min()
# SURVIVED that case (measured). Only k > e distinguishes them, and there the unclamped form
# subtracts a negative -- inventing parameters the model does not have.
class _CfgPaAll(_CfgPaMoE):
    moe_top_k = 8


class _CfgPaOver(_CfgPaMoE):
    moe_top_k = 12   # nonsensical as a recipe, reachable as a typo, and the clamp's only witness


assert _train._n_active_params(_pa_moe, _CfgPaAll) == _pa_total, \
    "with top_k == experts the denominator must be the full total, never more"
assert _train._n_active_params(_pa_moe, _CfgPaOver) == _pa_total, (
    f"with top_k {_CfgPaOver.moe_top_k} > experts {_CfgPaOver.moe_experts} the denominator is "
    f"{_train._n_active_params(_pa_moe, _CfgPaOver)}, above the total {_pa_total}: an unclamped "
    f"k/e subtracts a negative and prices parameters the model does not contain")
# A NAME TEST MUST FAIL HERE, and only a MIXED model can show it. model.py:337-338 names the
# dense FFN's weights w13/w2 and MoEFFN (model.py:880-881) names its stacks the same -- but every
# world above is all-MoE or all-dense, where the two predicates happen to agree, so a name test
# SURVIVED them (measured). `moe_layers` is what makes a real model mixed: with "0-1" and 12
# layers, blocks 0-1 hold (E, 2w, d) stacks called `ffn.w13` while blocks 2-11 hold nn.Linear
# weights called `ffn.w13.weight`. Both are FLOPs a token pays; only the first is routed.
class _CfgPaMixed(_CfgPaMoE):
    layers = 12           # 0-1 routed, 2-11 dense FFN
    moe_layers = "0-1"


_pa_mixed = HybridLM(_CfgPaMixed)
_pa_mx_total = sum(p.numel() for p in _pa_mixed.parameters())
_pa_mx_shape = sum(p.numel() for p in _pa_mixed.parameters()
                   if p.dim() == 3 and p.shape[0] == _CfgPaMixed.moe_experts)
_pa_mx_name = sum(p.numel() for n, p in _pa_mixed.named_parameters()
                  if "w13" in n or "w2" in n)
assert _pa_mx_shape > 0, "the mixed world has no routed stack"
# THE SUBSTRING SPELLING is what the mixed world separates, and it is worth being precise about
# which name test is unsafe, because they are not equivalent. `"w13" in n` charges all 12 blocks
# (344064 here) against the routed 98304 -- a 3.5x error. The last-component spelling
# (`n.rsplit(".", 1)[-1] in ("w13", "w2")`) gives exactly 98304 and is CORRECT today, for a reason
# that has nothing to do with routing: nn.Linear registers `weight` under the module, so its
# leaf is "weight", while MoEFFN's bare nn.Parameter's leaf is "w13". Mutating the shape test into
# that spelling therefore SURVIVES this file, correctly -- it is not a defect, it is a coincidence,
# and it stops holding the day an expert stack becomes a module. The shape test does not depend on
# that coincidence, which is the reason it is the one in train.py.
assert _pa_mx_name > _pa_mx_shape, (
    f"the mixed world does not separate substring from shape: name {_pa_mx_name} vs shape "
    f"{_pa_mx_shape}. Without dense FFN blocks beside the routed ones a substring test gives the "
    f"same answer, and the 3.5x understatement it causes would go unmeasured")
assert _train._n_active_params(_pa_mixed, _CfgPaMixed) == (
    _pa_mx_total - _pa_mx_shape + _pa_mx_shape * _CfgPaMixed.moe_top_k // _CfgPaMixed.moe_experts
), ("the mixed model's active count subtracts something other than the routed stacks -- a name "
    "test would also charge the 10 DENSE FFN blocks as inactive, understating the denominator")

# THE MEMORY-TABLE HALF, which predates de-71 and must keep working: without a world holding a
# table, deleting the `- n_mem` term SURVIVES (measured). mem_values > 0 builds the sparse pool.
class _CfgPaMem(_CfgPaDense):
    mem_values = 256
    mem_layers = "0"
    layers = 2


try:
    _pa_mem = HybridLM(_CfgPaMem)
except Exception as _e:                                     # noqa: BLE001
    print(f"_n_active_params: memory world unavailable ({type(_e).__name__}), "
          f"the n_mem term is UNGUARDED here")
else:
    _pa_mm_total = sum(p.numel() for p in _pa_mem.parameters())
    _pa_mm_mem = sum(p.numel() for n, p in _pa_mem.named_parameters() if _train._is_mem_fqn(n))
    assert _pa_mm_mem > 0, "the memory world built no memory params; mem_values did not take"
    assert _train._n_active_params(_pa_mem, _CfgPaMem) == _pa_mm_total - _pa_mm_mem, (
        f"the memory table is back in the denominator: "
        f"{_train._n_active_params(_pa_mem, _CfgPaMem)} vs {_pa_mm_total - _pa_mm_mem}. This is "
        f"the M1 168%-MFU defect train._n_active_params's docstring records")
print(f"_n_active_params: dense denominator identical ({_pa_old}); MoE {_CfgPaMoE.moe_experts} "
      f"experts top_k {_CfgPaMoE.moe_top_k} counts "
      f"{_pa_act} of {_pa_total} ({_pa_routed} routed); top_k == experts is the full total OK")


# ---------------------------------------------------------------------------------------------
# b0-35 CSA: compressed coarse attention + top-k block selection + sliding window.
#
# FOUR CASES, and the first is the one that protects every existing run: with the flag off the
# forward must be BIT-IDENTICAL to model.py as it stood before CSA existed. Not "close", and not
# compared against a reference I typed -- compared against the real previous file, read out of
# git, so the assertion cannot drift into agreeing with my own new code.
#
# THE GIT CALL BELOW RUNS WITH GIT_* STRIPPED, and that is not hygiene. This file is run by the
# pre-commit hook, which exports GIT_DIR and GIT_INDEX_FILE for the commit it is checking. An
# inherited GIT_DIR makes `git show HEAD:model.py` resolve HEAD in whatever repository the
# caller is holding: the read still succeeds, so the case still runs, and it compares my new
# model.py against SOME OTHER TREE's file. That failure has no skip and no error -- it is a
# green bit-identity assertion made against the wrong reference. Measured: with GIT_DIR pointed
# at a scratch repo holding a 50-byte model.py, the old `cd repo && git show` form returned that
# stub with rc=0 and the case would have compared against it; the form below returns the real
# 118142 bytes. -C pins the repository and the scrubbed env stops the caller from redirecting
# it (same rule as pod_drift.py:61-66).
_csa_pre = os.path.join("/tmp", f"_csa_pre_model_{os.getpid()}.py")
_csa_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_csa_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
try:
    _csa_blob = subprocess.run(
        ["git", "-C", _csa_repo, "show", "HEAD:model.py"],
        env=_csa_env, capture_output=True, timeout=60)
    _csa_have_pre = _csa_blob.returncode == 0 and len(_csa_blob.stdout) > 0
    if _csa_have_pre:
        with open(_csa_pre, "wb") as _fh:
            _fh.write(_csa_blob.stdout)
except (OSError, subprocess.SubprocessError):
    _csa_have_pre = False


class _CfgCsaOff:
    d, heads, value_embed, csa = 64, 4, False, False


class _CfgCsaOn(_CfgCsaOff):
    csa, csa_compress, csa_topk, csa_window = True, 4, 2, 8


# 1. FLAG OFF IS THE OLD CODE, BIT FOR BIT.
if not _csa_have_pre:
    print("CSA parity: HEAD:model.py unavailable, the bit-identity case did NOT run")
else:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_csa_pre_model", _csa_pre)
    _pre = _ilu.module_from_spec(_spec)
    sys.modules["_csa_pre_model"] = _pre
    _spec.loader.exec_module(_pre)
    torch.manual_seed(7); _a = _pre.GatedMLA(_CfgCsaOff)
    torch.manual_seed(7); _b = model.GatedMLA(_CfgCsaOff)
    assert set(_a.state_dict()) == set(_b.state_dict()), (
        f"flag-off changed the state_dict: {set(_a.state_dict()) ^ set(_b.state_dict())}. Every "
        f"existing checkpoint loads through these keys")
    _x = torch.randn(3, 17, 64)
    with torch.no_grad():
        _ya, _yb = _a(_x), _b(_x)
    assert torch.equal(_ya, _yb), (
        f"flag-off is NOT bit-identical to the pre-CSA forward: max delta "
        f"{(_ya - _yb).abs().max().item():.3e}. The one `if` was supposed to be the only change "
        f"on this path; something above it moved")
    print(f"CSA parity: flag off is bit-identical to HEAD:model.py "
          f"({sum(p.numel() for p in _b.parameters())} params, keys unchanged)")

# 2. OFF CONSTRUCTS NOTHING. A skipped-in-forward module would still put tensors in every
#    checkpoint of every run that does not use CSA.
_off, _on = model.GatedMLA(_CfgCsaOff), model.GatedMLA(_CfgCsaOn)
assert _off.csa is None, "flag off still built the CSA module; its parameters would be checkpointed"
assert not [k for k in _off.state_dict() if "csa" in k], "flag off leaked csa keys into state_dict"
_added = sorted(set(_on.state_dict()) - set(_off.state_dict()))
assert _added == ["csa.branch_gate.bias", "csa.branch_gate.weight"], (
    f"CSA on adds unexpected parameters: {_added}")

# 3. CAUSALITY, BY PERTURBATION AT EVERY POSITION. This is the case that caught the real defect:
#    the first version made a compressed block visible from its FIRST position, so with m=4 a
#    query at t=5 read mean(k[4..7]) and saw tokens 6 and 7. Measured k[5] += 7 moving positions
#    0..4 by 1.44 while the window branch moved 0.0, which is what localised it to the compress
#    branch. A causal leak makes training loss BETTER and only surfaces as generation collapse,
#    so it must be asserted here rather than watched for.
_csa = model.CompressedSparseAttention(_CfgCsaOn, 4, 16).double()
_T = 20
torch.manual_seed(3)
_q, _k, _v = (torch.randn(1, _T, 4, 16, dtype=torch.double) for _ in range(3))
_y0 = _csa(_q, _k, _v)
assert torch.isfinite(_y0).all(), "CSA produced non-finite output on a clean input"
_leaks = []
for _t in range(1, _T):
    _k2, _v2 = _k.clone(), _v.clone()
    _k2[:, _t] += 7.0
    _v2[:, _t] += 7.0
    _d = (_csa(_q, _k2, _v2)[:, :_t] - _y0[:, :_t]).abs().max().item()
    if _d > 1e-12:
        _leaks.append((_t, _d))
assert not _leaks, (
    f"CSA leaks the future: perturbing position t moved outputs BEFORE t at {_leaks[:4]}. "
    f"Check the compressed-block visibility mask -- a block is attendable only when its LAST "
    f"member is at or before the query")

# 3b. THE LEAK IS REACHABLE, i.e. case 3 can fail. Restoring the original first-position mask
#     must reproduce it, or case 3 is asserting over a world where no leak was possible.
_m, _nb = _CfgCsaOn.csa_compress, (_T + _CfgCsaOn.csa_compress - 1) // _CfgCsaOn.csa_compress
_ar_b, _ar_t = torch.arange(_nb), torch.arange(_T)[:, None]
_first_vis = (_ar_b * _m)[None, :] <= _ar_t          # the mask that leaked
_last_vis = (_ar_b * _m + _m - 1)[None, :] <= _ar_t  # the mask now in model.py
assert bool((_first_vis & ~_last_vis).any()), (
    f"at T={_T} m={_m} the first-position and last-position visibility masks agree, so case 3 "
    f"is asserting over a world where the leak it was written for cannot occur -- pick a T and "
    f"m where an incomplete block is visible to some query")

# 4. DOC-PACKED INPUT IS ISOLATED, NOT REFUSED. cu=[0,10,20] splits the 20 positions into
#    two documents; blocks are built per document, so a perturbation in document 0 must not
#    move any output in document 1, in any branch. The old refusal is gone -- CSA now runs on
#    the packed training path, which is the whole point.
_cu = torch.tensor([0, 10, 20], dtype=torch.int32)
_yc = _csa(_q, _k, _v, cu=_cu)
assert torch.isfinite(_yc).all(), "CSA produced non-finite output on doc-packed input"
_cross = []
for _t in range(10):
    _k2, _v2 = _k.clone(), _v.clone()
    _k2[:, _t] += 7.0
    _v2[:, _t] += 7.0
    _d = (_csa(_q, _k2, _v2, cu=_cu)[:, 10:] - _yc[:, 10:]).abs().max().item()
    if _d > 1e-12:
        _cross.append((_t, _d))
assert not _cross, (
    f"CSA leaks across documents: perturbing doc 0 moved doc 1 outputs at {_cross[:4]}. "
    f"Blocks must be built per document and every branch masked to the same document")
# and the perturbation IS visible inside its own document, so the isolation is not a blanket
# zero -- a dead path passes the cross-document assertion for free.
_k2, _v2 = _k.clone(), _v.clone()
_k2[:, 5] += 7.0
_v2[:, 5] += 7.0
_within = (_csa(_q, _k2, _v2, cu=_cu)[:, :10] - _yc[:, :10]).abs().max().item()
assert _within > 1e-6, (
    f"a perturbation is invisible inside its own document (delta {_within:.2e}); the "
    f"cross-document assertion above is then asserting over a dead path")

# 4b. DOCUMENTS SHORTER THAN A BLOCK still isolate and stay visible within themselves
#     (de's edge on this PR): blocks are per-document, so a 1- and a 3-token doc each get
#     their own partial block, visible to themselves and to nothing across the boundary.
_csa16 = model.CompressedSparseAttention(_CfgCsaOn, 4, 16).double()
_q16, _k16, _v16 = (torch.randn(1, 20, 4, 16, dtype=torch.double) for _ in range(3))
_cu16 = torch.tensor([0, 1, 4, 20], dtype=torch.int32)   # docs of len 1, 3, 16
_y16 = _csa16(_q16, _k16, _v16, cu=_cu16)
assert torch.isfinite(_y16).all(), "CSA produced non-finite output on sub-block-length docs"
_k16b, _v16b = _k16.clone(), _v16.clone()
_k16b[:, 0] += 7.0
_v16b[:, 0] += 7.0
assert (_csa16(_q16, _k16b, _v16b, cu=_cu16)[:, 1:] - _y16[:, 1:]).abs().max().item() == 0.0, (
    "a 1-token document leaked across its boundary")
_k16c, _v16c = _k16.clone(), _v16.clone()
_k16c[:, 2] += 7.0
_v16c[:, 2] += 7.0
_y16c = _csa16(_q16, _k16c, _v16c, cu=_cu16)
assert (_y16c[:, 4:] - _y16[:, 4:]).abs().max().item() == 0.0, (
    "the 3-token document leaked into the next document")
assert (_y16c[:, 1:4] - _y16[:, 1:4]).abs().max().item() > 1e-6, (
    "a 3-token document is invisible inside itself -- its partial block and window are dead")
print(f"CSA: off constructs nothing, on adds {len(_added)} params; causal at all {_T - 1} "
      f"perturbed positions; doc-packed input isolated (cross-doc delta 0.0, within-doc "
      f"{_within:.2e}); sub-block-length docs isolated and self-visible")

# ---------------------------------------------------------------------------
# CSA2 (de-103): learned entries + dedicated indexer + one softmax over entries + SWA.
# Same property suite as CSA, plus the indexer's own leak path and a gradient for the
# non-differentiable top-k's parameters.
# ---------------------------------------------------------------------------
class _CfgCsa2On(_CfgCsaOff):
    csa, csa2 = True, True
    csa2_m, csa2_top_k, csa2_n_win = 4, 2, 8
    csa2_indexer_heads, csa2_indexer_dim = 2, 8   # 2 divides heads=4; each indexer head scores its group's mean


class _CfgCsa2M1(_CfgCsa2On):
    csa2_m = 1                             # the paper's uncompressed-main-KV special case


# 1. FINITE ON A CLEAN INPUT, packed and unpacked.
_csa2 = model.CompressedSparseAttention(_CfgCsa2On, 4, 16).double()
_T2 = 20
torch.manual_seed(11)
_q2, _k2, _v2 = (torch.randn(1, _T2, 4, 16, dtype=torch.double) for _ in range(3))
_x2 = torch.randn(1, _T2, 64, dtype=torch.double)
_y2 = _csa2(_q2, _k2, _v2, x=_x2)
assert torch.isfinite(_y2).all(), "CSA2 produced non-finite output on a clean input"
_cu2 = torch.tensor([0, 10, 20], dtype=torch.int32)
_y2c = _csa2(_q2, _k2, _v2, cu=_cu2, x=_x2)
assert torch.isfinite(_y2c).all(), "CSA2 produced non-finite output on a doc-packed input"

# 2. CAUSALITY BY PERTURBATION, now covering THREE leak paths: the learned entries, the
#    SWA window, and the indexer selection (a future block ranked top-k).
_leaks = []
for _t in range(1, _T2):
    _k3, _v3 = _k2.clone(), _v2.clone()
    _k3[:, _t] += 7.0
    _v3[:, _t] += 7.0
    _d = (_csa2(_q2, _k3, _v3, x=_x2)[:, :_t] - _y2[:, :_t]).abs().max().item()
    if _d > 1e-12:
        _leaks.append((_t, _d))
assert not _leaks, (
    f"CSA2 leaks the future: perturbing position t moved outputs BEFORE t at {_leaks[:4]}. "
    f"Check the entry visibility mask AND the indexer mask -- isc must be masked before topk")

# 3. THE INDEXER MASK IS LOAD-BEARING, NOT JUST PRESENT. At t=1 no block is complete
#    (m=4), so vis is empty there: an unmasked topk would still pick a block, the masked
#    one picks nothing, and the output equals PLAIN WINDOWED ATTENTION exactly -- the
#    entry branch contributes zero rather than a silent fallback.
_m2 = _CfgCsa2On.csa2_m
with torch.no_grad():
    _kc3, _, _vis3, _ = model.entries_per_doc(
        _q2, _k2.transpose(1, 2), _v2.transpose(1, 2),
        torch.arange(0, _T2 + 1, _T2, dtype=torch.int32), _m2,
        _csa2.compress_k, _csa2.compress_v)
    _kc3g = _kc3.view(1, _csa2.ih, 4 // _csa2.ih, _kc3.shape[-2], 16).mean(2)
    _iq3 = _csa2.indexer_q(_x2).view(1, _T2, _csa2.ih, _csa2.di)
    _ik3 = torch.einsum("bhnd,hde->bhne", _kc3g, _csa2.ik_weight)
    _isc3 = (_iq3.transpose(1, 2) @ _ik3.transpose(-1, -2))
assert _vis3[0, 1].sum() == 0, "t=1 has a visible block at m=4; the test premise is wrong"
assert torch.isfinite(_isc3[0, :, 1]).any(), (
    "the unmasked indexer has no finite scores at t=1 -- nothing for the mask to block, "
    "so this assertion would pass over a world where the leak cannot occur")
with torch.no_grad():
    _qh, _kh, _vh = _q2.transpose(1, 2), _k2.transpose(1, 2), _v2.transpose(1, 2)
    _full = (_qh @ _kh.transpose(-1, -2)) * (16 ** -0.5)
    _ar = torch.arange(_T2)
    _mw = (torch.ones(_T2, _T2, dtype=torch.bool).tril()
           & ((_ar[:, None] - _ar[None, :]) < _CfgCsa2On.csa2_n_win))
    _ref = torch.softmax(_full.masked_fill(~_mw[None, None], float("-inf")), -1) @ _vh
_dwin = (_y2[:, 1] - _ref[:, :, 1]).abs().max().item()
assert _dwin < 1e-12, (
    f"CSA2 at t=1 (no visible block) differs from plain windowed attention by {_dwin:.2e} "
    f"-- the entry branch must contribute exactly zero, not a silent fallback")

# 4. CROSS-DOCUMENT ISOLATION, and the perturbation is visible inside its own document.
_cross = []
for _t in range(10):
    _k3, _v3 = _k2.clone(), _v2.clone()
    _k3[:, _t] += 7.0
    _v3[:, _t] += 7.0
    _d = (_csa2(_q2, _k3, _v3, cu=_cu2, x=_x2)[:, 10:] - _y2c[:, 10:]).abs().max().item()
    if _d > 1e-12:
        _cross.append((_t, _d))
assert not _cross, f"CSA2 leaks across documents at {_cross[:4]}"
_k3, _v3 = _k2.clone(), _v2.clone()
_k3[:, 5] += 7.0
_v3[:, 5] += 7.0
_within2 = (_csa2(_q2, _k3, _v3, cu=_cu2, x=_x2)[:, :10] - _y2c[:, :10]).abs().max().item()
assert _within2 > 1e-6, (
    f"a perturbation is invisible inside its own document (delta {_within2:.2e}); the "
    f"cross-document assertion is then asserting over a dead path")

# 5. BACKWARD IS FINITE THROUGH EVERY PARAMETER, and the indexer gets a gradient at all.
#    The hard top-k is non-differentiable; without the straight-through softmax the
#    indexer parameters would have NO grad and could never learn.
_csa2.zero_grad()
_y2.sum().backward()
_bad = [n for n, p in _csa2.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all()]
assert not _bad, f"CSA2 backward missing or non-finite grads at: {_bad}"
assert _csa2.ik_weight.grad is not None and _csa2.indexer_q.weight.grad is not None, (
    "the indexer got no gradient -- the straight-through path is gone and the top-k is "
    "a dead selector")

# 6. m=1 IS THE UNCOMPRESSED SPECIAL CASE: no compressor is built, and the path still runs.
_csa2m1 = model.CompressedSparseAttention(_CfgCsa2M1, 4, 16).double()
assert _csa2m1.compress_k is None and _csa2m1.compress_v is None, (
    "m=1 built a compressor; the paper's uncompressed-main-KV special case is identity")
assert torch.isfinite(_csa2m1(_q2, _k2, _v2, x=_x2)).all(), "CSA2 m=1 produced non-finite output"

# 7. csa2 WITHOUT csa IS A CONSTRUCTION ERROR, not a silent no-op -- at both construction
#    sites: the module directly, and GatedMLA (which builds the arm only when csa is on,
#    so the refusal must live there too, not only inside the module).
class _CfgCsa2Alone(_CfgCsa2On):
    csa = False
try:
    model.CompressedSparseAttention(_CfgCsa2Alone, 4, 16)
except ValueError:
    pass
else:
    raise AssertionError("csa2=True without csa=True constructed silently; the arm/variant "
                         "convention must refuse")
try:
    model.GatedMLA(_CfgCsa2Alone)
except ValueError:
    pass
else:
    raise AssertionError("csa2=True without csa=True constructed silently through GatedMLA "
                         "-- the refusal must fire at the layer level, where the arm is built")

# 8. STATE_DICT: csa2 on adds the new parameters and drops branch_gate.
_on2 = model.GatedMLA(_CfgCsa2On)
_added2 = sorted(set(_on2.state_dict()) - set(_off.state_dict()))
assert _added2 == ["csa.compress_k.bias", "csa.compress_k.weight",
                   "csa.compress_v.bias", "csa.compress_v.weight",
                   "csa.ik_weight", "csa.indexer_q.weight"], (
    f"CSA2 on adds unexpected parameters: {_added2}")
assert not [k for k in _on2.state_dict() if "branch_gate" in k], (
    "csa2 still carries the deleted branch gate")
print(f"CSA2: finite packed+unpacked; causal at all {_T2 - 1} perturbed positions; indexer "
      f"mask load-bearing (t=1 output == window-only, {_dwin:.1e}); cross-doc delta 0.0, "
      f"within-doc {_within2:.2e}; backward finite on {len(list(_csa2.parameters()))} params "
      f"with indexer grad; m=1 uncompressed; csa2-alone refuses; {len(_added2)} new state_dict keys")

# ── CSA2 Reuse (3b-20): Full emits a KV package, Reuse consumes it ────────────
# The acceptance contract: a Reuse layer's output CHANGES when its source package
# changes, does NOT change when a neighbouring Reuse layer's parameters change,
# and Reuse adds zero global-branch parameters of its own.
_mla_f = model.GatedMLA(_CfgCsa2On).double()
_mla_r = model.GatedMLA(_CfgCsa2On, csa2_mode="R").double()
assert isinstance(_mla_r.csa, model.CSA2Reuse), "csa2_mode=R must build CSA2Reuse"
torch.manual_seed(31)
_xr = torch.randn(1, 20, 64, dtype=torch.double)
_cur = torch.tensor([0, 12, 20], dtype=torch.int32)
_yf = _mla_f(_xr, cu=_cur)
_pkg = _mla_f._pkg
assert _pkg is not None and _pkg.nb == 5, (
    f"Full must emit a package (20 tokens, m=4, 2 docs -> 5 blocks), got nb={_pkg.nb if _pkg else None}")
_mla_r._pkg = _pkg
_yr = _mla_r(_xr, cu=_cur)
assert torch.isfinite(_yr).all(), "Reuse produced non-finite output"

# (a) SOURCE-CHANGE PROPAGATES. Block 0 is the only complete block at t=3..6, so it
# is selected there for every head -- mutating it must move the output.
# Indexing note: kc is [B,H,NB,D], so block 0 of every head is [:, :, 0, :];
# [..., 0, :, :] would select HEAD 0 of every block instead.
for _field in ("kc", "vc"):
    _alt = _pkg._replace(**{_field: getattr(_pkg, _field).clone()})
    with torch.no_grad():
        getattr(_alt, _field)[:, :, 0, :] += 7.0
    _mla_r._pkg = _alt
    _d = (_mla_r(_xr, cu=_cur) - _yr).abs().max().item()
    assert _d > 1e-6, f"Reuse output unchanged by a {_field} mutation ({_d:.2e})"
_alt = _pkg._replace(topk_idx=_pkg.topk_idx.clone())
with torch.no_grad():
    _alt.topk_idx[...] = 0          # every query selects block 0 only
_mla_r._pkg = _alt
assert (_mla_r(_xr, cu=_cur) - _yr).abs().max().item() > 1e-6, (
    "Reuse output unchanged by a topk_idx mutation")

# (b) NEIGHBOUR INVARIANCE: a second Reuse layer's parameters are not read here.
_mla_r2 = model.GatedMLA(_CfgCsa2On, csa2_mode="R").double()
_mla_r2.load_state_dict(_mla_r.state_dict())
_mla_r2._pkg = _pkg
assert (_mla_r2(_xr, cu=_cur) - _yr).abs().max().item() == 0.0, (
    "identical Reuse weights+package gave different outputs -- shared state slipped in")
with torch.no_grad():
    _mla_r2.qg.weight += 0.5
    _mla_r2.o.weight += 0.5
_mla_r2._pkg = _pkg
assert (_mla_r2(_xr, cu=_cur) - _yr).abs().max().item() > 1e-6, (
    "a Reuse layer is blind to its own parameters")
_mla_r._pkg = _pkg
assert (_mla_r(_xr, cu=_cur) - _yr).abs().max().item() == 0.0, (
    "Reuse output moved when a NEIGHBOURING Reuse layer's parameters changed")

# (c) ZERO GLOBAL-BRANCH PARAMETERS: the Reuse module itself is parameter-free,
# and the layer's params are exactly Full's minus the global-branch set.
assert len(list(model.CSA2Reuse(_CfgCsa2On, 4, 16).parameters())) == 0, (
    "CSA2Reuse must hold no parameters of its own")
_pf, _pr = set(_mla_f.state_dict()), set(_mla_r.state_dict())
assert _pr <= _pf, f"Reuse carries params Full lacks: {sorted(_pr - _pf)}"
assert sorted(_pf - _pr) == ["csa.compress_k.bias", "csa.compress_k.weight",
                             "csa.compress_v.bias", "csa.compress_v.weight",
                             "csa.ik_weight", "csa.indexer_q.weight"], (
    f"Reuse must drop exactly the global-branch params, got {sorted(_pf - _pr)}")

# (d) MASKING AT THE CONSUMER, by perturbation: block 1 (positions 4..7) is
# invisible before t=4; the last block belongs to doc 2 and is invisible in doc 1.
_pkg_k = _pkg._replace(kc=_pkg.kc.clone())
with torch.no_grad():
    _pkg_k.kc[:, :, 1, :] += 7.0
_mla_r._pkg = _pkg_k
assert (_mla_r(_xr, cu=_cur)[:, :4] - _yr[:, :4]).abs().max().item() == 0.0, (
    "a package entry for a future block moved a past output -- causal leak")
_pkg_v = _pkg._replace(vc=_pkg.vc.clone())
with torch.no_grad():
    _pkg_v.vc[:, :, -1, :] += 7.0
_mla_r._pkg = _pkg_v
assert (_mla_r(_xr, cu=_cur)[:, :12] - _yr[:, :12]).abs().max().item() == 0.0, (
    "a package entry from another document moved this doc's output -- doc leak")

# (e) A PACKAGE CANNOT CROSS STREAMS: built for one cu, refused against another.
_mla_r._pkg = _pkg
try:
    _mla_r(_xr, cu=torch.tensor([0, 20], dtype=torch.int32))
except ValueError:
    pass
else:
    raise AssertionError("Reuse read a package against a different packed stream")
# and no package at all is a forward error, not a silent zero-branch
_mla_r0 = model.GatedMLA(_CfgCsa2On, csa2_mode="R").double()
try:
    _mla_r0(_xr, cu=_cur)
except ValueError:
    pass
else:
    raise AssertionError("Reuse ran with no package (a Reuse-first stack must refuse)")

# (f) END-TO-END THREADING through HybridLM._body: the slot is set, the Reuse
# layer receives that exact package, a Full-weight perturbation reaches it, and
# the slot is cleared after the forward.
class _CfgReuseStack(_CfgPaDense):
    layers, attn_every, attn_hybrid = 2, 1, True
    csa, csa2 = True, True
    csa2_m, csa2_top_k, csa2_n_win = 4, 2, 8
    csa2_indexer_heads, csa2_indexer_dim = 2, 8
    csa2_modes, n_swa_only_layers = "F,R", 0
    rope_dims = 8  # all-attention stack: lifts the zero-KDA refusal


_stack = HybridLM(_CfgReuseStack).double()
assert isinstance(_stack.blocks[0].mixer.csa, model.CompressedSparseAttention)
assert isinstance(_stack.blocks[1].mixer.csa, model.CSA2Reuse)
assert _stack.csa2_modes == {0: "F", 1: "R"}
_seen = []
_stack.blocks[1].mixer.csa.register_forward_pre_hook(
    lambda _m, _a: _seen.append(_a[-1]))
torch.manual_seed(32)
_idx = torch.randint(0, 256, (1, 20))
_y1 = _stack(_idx)[0]
assert len(_seen) == 1 and _seen[0] is not None and _seen[0].nb == 5, (
    "the Reuse layer did not receive the Full layer's package through _body")
with torch.no_grad():
    _stack.blocks[0].mixer.csa.compress_k.weight += 0.3
assert (_stack(_idx)[0] - _y1).abs().max().item() > 1e-6, (
    "the Reuse layer's output did not move when its source package changed")
assert _stack.blocks[0].mixer._pkg is None and _stack.blocks[1].mixer._pkg is None, (
    "the package slot outlived the forward -- a stale package would reach the next caller")


class _CfgReuseFirst(_CfgReuseStack):
    csa2_modes = "R,F"


try:
    HybridLM(_CfgReuseFirst)
except ValueError:
    pass
else:
    raise AssertionError("a Reuse-first mode map constructed -- the package source is missing")


class _CfgReuseCkpt(_CfgReuseStack):
    grad_ckpt = True


try:
    HybridLM(_CfgReuseCkpt)
except ValueError:
    pass
else:
    raise AssertionError("grad_ckpt + Reuse constructed -- the stash is not a checkpoint input")
print("CSA2Reuse: source-change propagates (kc/vc/topk_idx); neighbour-change invariant; "
      "zero global-branch params (6 keys dropped); causal+doc masking exact; stream mismatch, "
      "no-package, Reuse-first and grad_ckpt refuse; F,R body threads the package and clears the slot")

# ── PureSWA (V4.1 Step 3, task 0e-2): CSA's window branch, alone ──────────────
# Same perturbation discipline as the CSA cases above: a masked position must be EXACTLY
# invisible (its softmax weight is exactly 0), and the perturbation must be visible where
# the mask admits it, or the invariance is asserting over a dead path.
class _CfgSwa:
    csa_window = 8


_swa = model.PureSWA(_CfgSwa, 4, 16).double()
torch.manual_seed(4)
_sq, _sk, _sv = (torch.randn(1, 20, 4, 16, dtype=torch.double) for _ in range(3))
_sy0 = _swa(_sq, _sk, _sv)
assert torch.isfinite(_sy0).all(), "SWA produced non-finite output on a clean input"

# 1. OUTSIDE-WINDOW INVISIBILITY. Position 0 is within distance 8 only of queries 0..7;
#    queries 8..19 must not move AT ALL.
_sk2, _sv2 = _sk.clone(), _sv.clone()
_sk2[:, 0] += 7.0
_sv2[:, 0] += 7.0
assert (_swa(_sq, _sk2, _sv2)[:, 8:] - _sy0[:, 8:]).abs().max().item() == 0.0, (
    "SWA leaked outside its window: perturbing position 0 moved queries at distance >= 8"
)
assert (_swa(_sq, _sk2, _sv2)[:, :8] - _sy0[:, :8]).abs().max().item() > 1e-6, (
    "the perturbation is invisible INSIDE the window too -- the window branch is dead"
)

# 1b. csa2_n_win TAKES PRECEDENCE over csa_window, so the pure-SWA layers and de-103's CSA2
#     window branch share one width (ae's divergence, de's ruling 2026-09-10). With width 4,
#     position 0 is visible only to queries 0..3.
class _CfgSwaN(_CfgSwa):
    csa2_n_win = 4


_sw4 = model.PureSWA(_CfgSwaN, 4, 16).double()
assert (_sw4(_sq, _sk2, _sv2)[:, 4:] - _sw4(_sq, _sk, _sv)[:, 4:]).abs().max().item() == 0.0, (
    "csa2_n_win did not narrow the window: position 0 moved queries at distance >= 4"
)
assert (_sw4(_sq, _sk2, _sv2)[:, :4] - _sw4(_sq, _sk, _sv)[:, :4]).abs().max().item() > 1e-6, (
    "csa2_n_win=4 made the window dead inside its own range"
)

# 2. CAUSALITY. Perturbing position t must not move outputs before t.
_sleaks = []
for _t in range(1, 20):
    _sk2, _sv2 = _sk.clone(), _sv.clone()
    _sk2[:, _t] += 7.0
    _sv2[:, _t] += 7.0
    _d = (_swa(_sq, _sk2, _sv2)[:, :_t] - _sy0[:, :_t]).abs().max().item()
    if _d > 1e-12:
        _sleaks.append((_t, _d))
assert not _sleaks, f"SWA leaks the future: {_sleaks[:4]}"

# 3. PACKED PATH: cross-document invisibility, and self-visibility inside the document.
_scu = torch.tensor([0, 10, 20], dtype=torch.int32)
_syc = _swa(_sq, _sk, _sv, cu=_scu)
assert torch.isfinite(_syc).all(), "SWA produced non-finite output on doc-packed input"
for _t in range(10):
    _sk2, _sv2 = _sk.clone(), _sv.clone()
    _sk2[:, _t] += 7.0
    _sv2[:, _t] += 7.0
    _d = (_swa(_sq, _sk2, _sv2, cu=_scu)[:, 10:] - _syc[:, 10:]).abs().max().item()
    assert _d == 0.0, f"SWA leaked across documents: doc 0 pos {_t} moved doc 1 (delta {_d:.2e})"
_sk2, _sv2 = _sk.clone(), _sv.clone()
_sk2[:, 5] += 7.0
_sv2[:, 5] += 7.0
assert (_swa(_sq, _sk2, _sv2, cu=_scu)[:, :10] - _syc[:, :10]).abs().max().item() > 1e-6, (
    "packed path: a perturbation is invisible inside its own document -- dead path"
)

# 4. THE MOST-MASKED ROW THE MASK ADMITS. Self-attention is always visible, so a row cannot be
#    all-masked; the edge is a length-1 document: one visible entry, itself. It must stay finite
#    and equal its own value (softmax over one logit is weight 1).
_sy1 = _swa(_sq, _sk, _sv, cu=torch.tensor([0, 1, 20], dtype=torch.int32))
assert torch.isfinite(_sy1).all(), "SWA non-finite on a length-1 document (single visible entry)"
assert (_sy1[:, 0] - _sv[:, 0]).abs().max().item() < 1e-12, (
    "length-1 document output != its own value -- the single visible entry got weight != 1"
)


# 5. PLACEMENT. n_swa_only_layers=2 puts PureSWA in the first two ATTENTION layers and leaves
#    the rest of the interleave (and the n=0,1 CSA slots it replaces) exactly as before. The
#    mode map composes: kind[i] is still one string per attention layer, selected here.
class _CfgSwaPlace(_CfgPaDense):
    layers = 6
    attn_hybrid = True
    attn_every = 1
    rope_dims = 8  # lifts the zero-KDA refusal; position for an all-attention stack
    n_swa_only_layers = 2


_sp = HybridLM(_CfgSwaPlace)
_sp_kinds = {}
for _i, _b in enumerate(_sp.blocks):
    _m = _b.mixer
    _sp_kinds[_i] = "swa" if _m.swa is not None else "csa" if _m.csa is not None else "hca"
assert _sp_kinds == {0: "swa", 1: "swa", 2: "csa", 3: "hca", 4: "csa", 5: "hca"}, (
    f"n_swa_only_layers placement wrong: {_sp_kinds}"
)
# and n_swa_only_layers=0 reproduces the pre-V4.1 interleave byte for byte
_CfgSwaPlace.n_swa_only_layers = 0
_sp0 = HybridLM(_CfgSwaPlace)
_sp0_kinds = {
    _i: ("swa" if _b.mixer.swa is not None else "csa" if _b.mixer.csa is not None else "hca")
    for _i, _b in enumerate(_sp0.blocks)
}
assert _sp0_kinds == {0: "csa", 1: "csa", 2: "csa", 3: "hca", 4: "csa", 5: "hca"}, (
    f"n_swa_only_layers=0 changed the legacy interleave: {_sp0_kinds}"
)
print(
    "PureSWA: outside-window delta 0.0 (within-window visible), causal at all 19 perturbed "
    "positions; packed path cross-doc delta 0.0 and self-visible; length-1 doc finite and "
    "equal to its own value; placement swa,swa,csa,hca,csa,hca with n=2 and the legacy "
    "interleave unchanged with n=0"
)
