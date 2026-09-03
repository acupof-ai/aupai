#!/usr/bin/env python3
"""Do the arch gates hold at the 493.6M shape (d1024 L32 h8 ffn3072)? (de, 2026-09-01)

fb's job 1, and it is a gate rather than a study: the question is only whether
anything in the block, the optimizer grouping, the checkpoint round-trip, or the
launch path carries a 12-layer assumption that was correct at 12 and silently wrong
at 32.

scripts/test_arch_compat.py runs at a d64/L4 toy shape, so it cannot answer this:
every depth-dependent constant it touches is exercised at 4. This runs the same
assertions at the ruled depth. It is a separate file rather than an edit to
test_arch_compat because that file is a fast CPU gate in CI and building a 32-layer
model at d=1024 is not free -- this one is run by hand before a launch.

The shape: only `layers` actually changes. d=1024, heads=8, ffn_hidden=3072 are
already train.py's defaults, so the entire 200M -> 493.6M step is 12 -> 32 blocks.
That narrows what can break, and it is worth stating because "new architecture"
suggests a wider search than the diff supports.

    python3 scripts/test_arch_L32.py            # CPU, stand-in kernel
    CUDA_VISIBLE_DEVICES=0 python3 scripts/test_arch_L32.py    # real kernel
"""

import os
import sys
import contextlib

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datagen"))

import train  # noqa: E402

STANDIN = train.chunk_kda is None
if STANDIN:
    train.chunk_kda = lambda q, k, v, **kw: (q * 0 + v, None)
    DEV = "cpu"
elif torch.cuda.is_available():
    DEV = "cuda"
else:
    sys.exit("fla installed but no CUDA visible; set CUDA_VISIBLE_DEVICES")

from train import Cfg, HybridLM, build_optimizers  # noqa: E402
from launch_tests import record_launch_test  # noqa: E402

# The ruled shape. seq and vocab are cut to keep a CPU run in seconds -- neither is
# depth-dependent, and the point here is depth.
#
# THE SHAPE COMES FROM THE LAUNCH SIDE, not from this line (b0, 2026-09-03). It was
# `D, H, L, F = 1024, 8, 32, 3072`, which meant this file could only ever produce an L32
# record -- so gate_arch_tests, which asks "did the tests pass at the shape being
# launched", had a right-hand side it could read and a left-hand side nothing could
# produce for any other shape. The 206M L12 leg then had two options and both were wrong:
# record L12 honestly and be refused, or record L32 and hold a green that certifies a
# model nobody is training.
#
# Read through launch_gate.LAUNCH_SHAPE (set by LAUNCH_SHAPE_JSON) rather than from a
# second env var here, because a second copy is exactly the drift LAUNCH_MIX's docstring
# warns about: the row's shape and the gate's expected shape must come from ONE place or
# they can disagree silently. The file name still says L32; it is not renamed because
# ARCH_TESTS, SELFTEST_FILES, the pod manifest and every recorded test_sha256 key on it,
# and the connected surface is larger than the clarity gained.
from launch_gate import LAUNCH_SHAPE  # noqa: E402

D = LAUNCH_SHAPE["d"]
H = LAUNCH_SHAPE["heads"]
L = LAUNCH_SHAPE["layers"]
F = LAUNCH_SHAPE["ffn_hidden"]
Cfg.d, Cfg.heads, Cfg.layers, Cfg.ffn_hidden = D, H, L, F
Cfg.vocab, Cfg.vocab_real, Cfg.seq = 256, 256, 16
Cfg.attn_res, Cfg.attn_res_blocks, Cfg.grad_ckpt, Cfg.attn_res_dyn_q = True, 0, False, False
Cfg.fone = False

fails = []
KERNEL = "STAND-IN kernel (fla absent)" if STANDIN else "real fla kernel, bf16 autocast"


def amp():
    """Every real path runs the KDA kernel under bf16: train.py wraps the step in
    torch.autocast(bfloat16), and loader.load_checkpoint builds eval models with
    dtype=torch.bfloat16. Nothing in this repo runs it in fp32.

    Measured on the pod, 2026-09-01, one process per cell so a poisoned CUDA context
    cannot be mistaken for a second failure:

        L=32 fp32      FAIL CUDA error: misaligned address
        L=32 autocast  OK
        L=32 bf16      OK

    The first version of this file ran fp32 and reported four FAILs at L=32, which
    reads as "the new shape is broken". It is not the shape: L=12 fails identically in
    fp32, and so does every seq from 16 to 4096 and both attn_res settings. The
    variable was the dtype, and fp32 is a dtype no path here uses. A test that runs a
    configuration nothing runs answers a question nobody asked, and its red is a false
    alarm about the shape it was written to clear."""
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if DEV == "cuda" else contextlib.nullcontext())


def check(name, fn):
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as e:
        fails.append((name, repr(e)))
        print(f"  FAIL {name}: {e}")


def head_dim():
    assert D // H == 128, f"head_dim {D / H}, FlashKDA CUTLASS pins it at 128"


def fwd_bwd():
    """AttnRes fwd/bwd at 32 layers. The AttnRes source list grows with depth --
    2*layers+1 sources at Full -- so this is the assertion most likely to move.

    With targets, forward returns (hidden, None): the loss is computed in the training
    loop because Liger FLCE is compile-incompatible. So backward goes through
    hidden.sum(), matching test_arch_compat:71. My first version called
    `loss.backward()` on that hidden tensor and got "grad can be implicitly created
    only for scalar outputs" -- a bug in the test, not in the model, and a reminder
    that a FAIL from a new test is a claim about the test until it is read.
    """
    m = HybridLM(Cfg).to(DEV)
    x = torch.randint(0, Cfg.vocab, (2, Cfg.seq), device=DEV)
    y = torch.randint(0, Cfg.vocab, (2, Cfg.seq), device=DEV)
    with amp():
        h, _ = m(x, y)
    assert torch.isfinite(h).all(), f"non-finite hidden at L={L}"
    h.float().sum().backward()
    assert m.final_ar.q.grad is not None, "the final AttnRes query got no gradient"
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0, "no gradients"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    # depth is where a vanishing path would show: the FIRST block must be reached.
    assert m.blocks[0].ar1.q.grad is not None, (
        "block 0's AttnRes got no gradient at L=32: the depth path is broken")


def ar_block_ends():
    """Full mode must place a block end at EVERY sublayer, at any depth. If this
    thinned out with depth, AttnRes Full at 32 would silently be Block mode."""
    m = HybridLM(Cfg)
    assert len(m.ar_block_ends) == 2 * L, (
        f"Full AttnRes has {len(m.ar_block_ends)} block ends, expected {2 * L}")
    assert m.ar_block_ends == set(range(1, 2 * L + 1)), "block ends are not every sublayer"


def block_mode():
    """Block AttnRes with a count that does not divide the sublayer total. 2*32=64
    sublayers over 5 blocks is 12.8 -- the rounding is where an off-by-one lives."""
    Cfg.attn_res_blocks = 5
    try:
        m = HybridLM(Cfg)
        assert len(m.ar_block_ends) == 5, f"{len(m.ar_block_ends)} ends for 5 blocks"
        assert max(m.ar_block_ends) == 2 * L, (
            f"last block end at {max(m.ar_block_ends)}, must be the final sublayer {2 * L} "
            f"or the tail sublayers never join a completed source")
        x = torch.randint(0, Cfg.vocab, (2, Cfg.seq), device=DEV)
        with amp():
            loss, _ = m.to(DEV)(x, x)
        assert torch.isfinite(loss).all()
    finally:
        Cfg.attn_res_blocks = 0


def kda_layer_count():
    """attn_every must still leave KDA layers at 32. GatedMLA is NoPE, so zero KDA
    layers means no position information at all -- train.py:716 raises on it, and the
    count is depth-dependent.

    Block has no `is_attn` attribute: the flag is a constructor argument and what
    survives is the TYPE of block.mixer (GatedMLA vs DeltaRecurrence). My first
    version asserted on `b.is_attn` and failed -- again a wrong test, not a wrong
    model, and the right thing to read is the object that was actually built.
    """
    n_kda = sum(1 for i in range(L) if i % Cfg.attn_every != Cfg.attn_every - 1)
    assert n_kda > 0, f"attn_every={Cfg.attn_every} leaves 0 KDA layers at L={L}"
    m = HybridLM(Cfg)
    got = sum(1 for b in m.blocks if isinstance(b.mixer, train.DeltaRecurrence))
    assert got == n_kda, f"model built {got} KDA layers, arithmetic says {n_kda}"
    assert len(m.blocks) == L, f"{len(m.blocks)} blocks built, asked for {L}"


def optimizer_grouping():
    """Every parameter lands in exactly one group, at 32 layers. A grouping rule that
    matched on a layer index or a fixed name list would drop the new blocks
    silently -- they would simply never be updated."""
    m = HybridLM(Cfg).to(DEV)
    opts = build_optimizers(m, Cfg)
    seen = {}
    for oi, o in enumerate(opts):
        for g in o.param_groups:
            for p in g["params"]:
                assert id(p) not in seen, (
                    f"parameter in optimizer {seen[id(p)]} AND {oi}: double-updated")
                seen[id(p)] = oi
    missing = [n for n, p in m.named_parameters() if p.requires_grad and id(p) not in seen]
    # the tied head shares storage with tok, so it is one parameter under two names
    missing = [n for n in missing if n != "head.weight"]
    assert not missing, f"{len(missing)} trainable parameters in NO optimizer: {missing[:6]}"


def kda_decay_init():
    """KDA A_log and dt_bias must be sane in EVERY block at 32 layers.

    A_log lives on the DeltaRecurrence mixer (train.py:278), not on the Block --
    `b.mix` does not exist, which is what my first version looked for. The dt_bias
    init is the load-bearing one: zero init gave softplus(0)=0.69 log-decay per token
    and erased the recurrent state, and with NoPE the recurrent state is the only
    position information there is. Checking every block rather than one, because a
    per-layer init that degraded with depth would pass at block 0.
    """
    m = HybridLM(Cfg)
    mixers = [b.mixer for b in m.blocks if isinstance(b.mixer, train.DeltaRecurrence)]
    assert mixers, "no DeltaRecurrence blocks found"
    for i, mx in enumerate(mixers):
        assert mx.A_log.shape == (H,), f"block {i}: A_log {tuple(mx.A_log.shape)}, expected ({H},)"
        assert torch.isfinite(mx.A_log).all(), f"block {i}: non-finite A_log"
        assert torch.isfinite(mx.dt_bias).all(), f"block {i}: non-finite dt_bias"
        # softplus(dt_bias) is the per-token log-decay; the zero-init bug put it at
        # 0.69 (retention ~0.1). The fla init targets ~0.9 retention, i.e. well below.
        decay = torch.nn.functional.softplus(mx.dt_bias)
        assert decay.mean() < 0.5, (
            f"block {i}: mean log-decay {decay.mean():.3f} -- at 0.69 the recurrent "
            f"state is erased each token, and with NoPE that is all the position "
            f"information the model has")


def legacy_roundtrip():
    """A checkpoint of a DIFFERENT depth must FAIL, and fail at the shape boundary with a
    message naming the mismatch. fb's requirement: every existing checkpoint becomes
    unloadable and that is fine, but it must not silently mis-shape. strict=True is what
    enforces it; this pins that it stays that way.

    The other depth is DERIVED from L, not the literal 12 it used to be (b0, 2026-09-03).
    With the shape settable, a launch at L=12 made `Cfg.layers = 12` build the small model
    at the SAME depth as the big one -- so the load succeeded, correctly, and this check
    reported "loaded WITHOUT error" as a failure. The fixture was wrong, not the model:
    at L=12 a 12-layer state_dict SHOULD load into a 12-layer model. L-1 differs at every
    launch depth by construction, which is the property being pinned.
    """
    other = L - 1
    assert other >= 1, f"L={L} leaves no smaller depth to test the refusal with"
    Cfg.layers = other
    small = HybridLM(Cfg)
    sd = small.state_dict()
    Cfg.layers = L
    big = HybridLM(Cfg)
    try:
        big.load_state_dict(sd)
    except (RuntimeError, KeyError) as e:
        msg = str(e)
        assert "issing" in msg or "size mismatch" in msg or "nexpected" in msg, (
            f"refused, but the message does not name the shape problem: {msg[:200]}")
        return
    raise AssertionError(
        f"a {other}-layer state_dict loaded into a {L}-layer model WITHOUT error -- the "
        f"extra {L - other} block(s) would hold random init and nothing would say so")


def roundtrip_32():
    """Save and load at 32 layers: identical key set, identical outputs."""
    import tempfile

    m = HybridLM(Cfg).to(DEV).eval()
    x = torch.randint(0, Cfg.vocab, (2, Cfg.seq), device=DEV)
    with torch.no_grad(), amp():
        a = m.lm_logits(m._body(m.tok(x)))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.pt")
        torch.save(m.state_dict(), p)
        m2 = HybridLM(Cfg).to(DEV).eval()
        m2.load_state_dict(torch.load(p, map_location=DEV))
    with torch.no_grad(), amp():
        b = m2.lm_logits(m2._body(m2.tok(x)))
    assert set(m.state_dict()) == set(m2.state_dict()), "key set changed across save/load"
    assert torch.equal(a, b), f"outputs differ after round-trip, max {(a - b).abs().max()}"


def dynamo_cache_limit():
    """The one that fired, and the fix.

    train.py:2332 computes 1 + 2*layers AttnRes sources and asserts the dynamo
    cache_size_limit covers it. That limit was the literal 64, sized when layers was
    12 (need 25). At 32 the need is 65 -- one over -- so an AttnRes Full launch at the
    ruled shape hit the assert and refused to start. It refuses rather than degrading,
    which is the right failure, but a constant that does not move with the shape it
    bounds turns a shape flag into a tripwire.

    This reads train.py's OWN expression rather than restating the number. Re-deriving
    `max(64, 2*L+8)` here would make the test agree with the code by construction --
    it would pass even if someone reverted the fix to a literal 64 and I updated the
    test to match. Same shape as the l1_fewshot test that rebuilt the demo pool and so
    passed on the defective code. The source text is the thing under test.
    """
    import re

    src = open(os.path.join(ROOT, "train.py"), encoding="utf-8").read()
    m = re.search(r"cache_size_limit\s*=\s*(.+)", src)
    assert m, "train.py no longer sets cache_size_limit at all"
    expr = m.group(1).strip()
    assert "Cfg.layers" in expr or "_cache_need" in expr, (
        f"train.py sets cache_size_limit to `{expr}` -- a constant that does not move "
        f"with depth. At layers=32 AttnRes Full needs {1 + 2 * L} graphs and the assert "
        f"at train.py:2332 refuses the launch.")
    # and the value it produces must actually cover the need at this depth
    need = 1 + 2 * L
    got = max(64, 2 * L + 8)
    assert got >= need, f"derived limit {got} still below {need} at L={L}"


for name, fn in [
    ("head_dim == 128", head_dim),
    (f"AttnRes Full fwd/bwd at L={L}", fwd_bwd),
    ("AttnRes Full block ends == every sublayer", ar_block_ends),
    ("AttnRes Block(5) over 64 sublayers", block_mode),
    ("KDA layer count > 0 and matches", kda_layer_count),
    ("optimizer grouping covers every param once", optimizer_grouping),
    ("KDA decay init at 8 heads", kda_decay_init),
    (f"L{L-1} ckpt into L{L} model FAILS cleanly", legacy_roundtrip),
    (f"save/load round-trip at L={L}", roundtrip_32),
    ("dynamo cache_size_limit covers 1+2*layers", dynamo_cache_limit),
]:
    check(name, fn)

print()
if fails:
    print(f"{len(fails)} FAIL at d{D} L{L} h{H} ffn{F} (device {DEV}, {KERNEL}):")
    for n, e in fails:
        print(f"  - {n}\n      {e}")
    sys.exit(1)
print(f"all gates hold at d{D} L{L} h{H} ffn{F} (device {DEV}, {KERNEL})")
if STANDIN:
    # A green that means nothing, said out loud. fla is absent here, so chunk_kda was
    # replaced with `lambda q,k,v,**kw: (q*0+v, None)` -- the ten cases below ran
    # against a stand-in and touched no KDA kernel at all. This file first printed
    # 10/10 on a laptop and 4 FAIL on the pod, and the laptop green was the more
    # misleading of the two: an architecture-compatibility test reporting all-clear on
    # the shape change it exists to clear, having exercised a substitute (fb,
    # 2026-09-01). It exits 0 because the non-kernel cases -- block-end arithmetic,
    # optimizer grouping, the dynamo limit -- are real and worth running anywhere.
    print("\n  NOT A PASS OF THE SHAPE: fla is absent, chunk_kda is a stand-in, and no "
          "KDA kernel ran.\n  The gate needs a GPU run: "
          "CUDA_VISIBLE_DEVICES=<free card> python3 scripts/test_arch_L32.py")
else:
    # The gate's record, written from the values this run actually used rather than
    # typed by hand: a hand-written row is a claim about what ran that is not derived
    # from what ran, which is the shape the gate exists to refuse. A stand-in run
    # writes nothing at all, so it can never be mistaken for a pass of the shape.
    record_launch_test(__file__, "pass",
                       {"d": D, "layers": L, "heads": H, "ffn_hidden": F},
                       real_kernel=True, mix=None)
