"""P1 gates for step D: fp32-master training checkpoint (v41f/master.py).

Design: docs/standards/v41f_train_checkpoint_design.md (prereview #487). Direct-runner
compatible, process-private temp dirs. The 180M v41f_small means two live models + two
AdamW states in one process OOM on a laptop, so resume tests build model B only after model
A is gc'd, and the prod census is structural (no optimizer allocation); the full prod
backward is GPU-deferred (#497).
"""

import atexit
import gc
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # tests/v41f: ref_oracle sibling
sys.path.insert(0, str(_HERE.parents[1]))  # repo root: v41f package

import torch  # noqa: E402
from ref_oracle import synthetic_tokenizer  # noqa: E402

from v41f.config import v41f_small  # noqa: E402
from v41f.master import (  # noqa: E402
    OptimStateError,
    TrainState,
    load_train_checkpoint,
    save_train_checkpoint,
)
from v41f.model import V41FModel  # noqa: E402
from v41f.vocab import fingerprint  # noqa: E402

VOCAB = 12800
_SMALL = dict()

# EVERY scratch dir this file makes goes through _scratch(). Nine call sites used bare
# `tempfile.mkdtemp(prefix="td_gN_")` and never removed them; `_selftest` papers over that
# with a process-private TMPDIR root it rmtree's at the end, so the leak only shows on the
# DIRECT-RUN path this module's docstring advertises (`--gate <name>`, `--run <kind>`),
# which inherits the ambient TMPDIR -- /tmp on a normal box. Measured 2026-09-18 on digest:
# 31 `td_eq_*/c.pt` blobs of ~2.49 GB each, 77 GB, had filled the ROOT filesystem to 100%,
# which took out unrelated selftests with ENOSPC (two `launch_gate` reds appeared only on
# the second red-list run, on an unchanged commit). The `c.pt` files are model checkpoints,
# so one leaked dir is not a rounding error.
# The paths must outlive individual functions (a later gate in the same process reads what
# an earlier one wrote), so this is atexit + rmtree rather than TemporaryDirectory, and
# atexit also covers the `sys.exit(2)` refusal branches.
_SCRATCH = []


def _scratch(prefix):
    d = tempfile.mkdtemp(prefix=prefix)
    _SCRATCH.append(d)
    return d


def _clean_scratch():
    import shutil

    for d in _SCRATCH:
        shutil.rmtree(d, ignore_errors=True)
    _SCRATCH.clear()


atexit.register(_clean_scratch)


def _cfg(mode):
    return v41f_small(indexer_train_mode=mode)


def _build(cfg):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        m = V41FModel(cfg, max_batch_size=2)
    finally:
        torch.set_default_dtype(prev)
    return m


def _ids(cfg, seed, seq=16):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (2, seq), generator=g)


# G1/G5: membership census, two configs of the SAME small network --------------------------


def _six(cfg):
    return {
        f"layers.{l}.attn.indexer.{w}.weight"
        for l in cfg.index_source_layers
        for w in ("wq_b", "weights_proj")
    }


def _f(cfg):
    return {f"layers.{l}.attn.index_key.{w}.weight" for l in cfg.kv_source_layers for w in ("wk", "k_norm")}


def gate_census_membership():
    """off freezes F+SIX, ste freezes F only; counts name-derived."""
    for mode, expect_rg in (("off", 228), ("ste", 232)):
        cfg = _cfg(mode)
        m = _build(cfg)
        rg = {n for n, p in m.named_parameters() if p.requires_grad}
        assert len(rg) == expect_rg, f"{mode}: rg {len(rg)} != {expect_rg}"
        assert not (rg & _f(cfg)), f"{mode}: index_key F not frozen"
        if mode == "off":
            assert not (rg & _six(cfg)), "off: indexer projections not dormant"
        else:
            assert _six(cfg) <= rg, "ste: indexer projections not in group"
        del m
    print(f"  census: off 228 / ste 232; F={sorted(_f(_cfg('off')))} frozen both")


# G6: buffers ride in model, never in master/optim ----------------------------------------


def gate_buffers_in_model_not_master():
    cfg = _cfg("off")
    m = _build(cfg)
    st = TrainState(m, lr=1e-3)
    d = _scratch(prefix="td_g6_")
    f = os.path.join(d, "c.pt")
    save_train_checkpoint(f, model=m, cfg=cfg, state=st, tokenizer=synthetic_tokenizer())
    blob = torch.load(f, map_location="cpu", weights_only=False)
    biases = [k for k in blob["model"] if k.endswith("ffn.gate.bias")]
    assert len(biases) == 5, f"expected 5 small gate.bias buffers, got {len(biases)}"
    assert all(blob["model"][k].dtype == torch.float32 for k in biases)
    assert not any(".gate.bias" in n for n in blob["master_fp32"])
    assert not any(".gate.bias" in n for n in blob["optim_named"]["param_names"])
    print(f"  G6: {len(biases)} persistent gate.bias in model only, none in master/optim")


# G7: fp32-native alias survives a real save/load; bf16 master is distinct ------------------


def gate_alias_survives_save_load():
    cfg = _cfg("off")
    m = _build(cfg)
    st = TrainState(m, lr=1e-3)
    named = dict(m.named_parameters())
    # before save: alias one storage
    assert st.master["head.weight"].data_ptr() == named["head.weight"].data_ptr()
    bfname = "layers.0.attn.qproj.wq_b.weight"
    assert st.master[bfname].data_ptr() != named[bfname].data_ptr()
    d = _scratch(prefix="td_g7_")
    f = os.path.join(d, "c.pt")
    save_train_checkpoint(f, model=m, cfg=cfg, state=st, tokenizer=synthetic_tokenizer())
    del m, st
    gc.collect()
    m2, st2, _, _ = load_train_checkpoint(f, tokenizer=synthetic_tokenizer(), max_batch_size=2)
    n2 = dict(m2.named_parameters())
    assert st2.master["head.weight"].data_ptr() == n2["head.weight"].data_ptr(), (
        "fp32-native master not aliased to model after load"
    )
    assert st2.master[bfname].data_ptr() != n2[bfname].data_ptr(), (
        "bf16-native master must be distinct storage"
    )
    print("  G7: head fp32 master aliases model storage after load; bf16 master distinct")


# G3: tokenizer + vocab_id refusal ----------------------------------------------------------


def gate_tokenizer_and_vocab_id():
    cfg = _cfg("off")
    m = _build(cfg)
    st = TrainState(m, lr=1e-3)
    d = _scratch(prefix="td_g3_")
    f = os.path.join(d, "c.pt")
    tok = synthetic_tokenizer()
    save_train_checkpoint(f, model=m, cfg=cfg, state=st, tokenizer=tok)
    del m, st
    gc.collect()
    # wrong tokenizer (different pieces) -> refuse
    bad = synthetic_tokenizer(pieces=["x", "y", "z", "w"])
    try:
        load_train_checkpoint(f, tokenizer=bad, max_batch_size=2)
        raise AssertionError("mismatched tokenizer loaded")
    except ValueError as e:
        assert "vocab_id" in str(e)
    # tampered blob vocab_id -> refuse
    blob = torch.load(f, map_location="cpu", weights_only=False)
    blob["vocab_id"] = "0" * 16
    torch.save(blob, f)
    try:
        load_train_checkpoint(f, tokenizer=tok, max_batch_size=2)
        raise AssertionError("tampered vocab_id loaded")
    except ValueError as e:
        assert "vocab_id" in str(e)
    # correct fingerprint is deterministic for the same tokenizer
    assert fingerprint(tok) == fingerprint(synthetic_tokenizer())
    print("  G3: mismatched and tampered vocab_id refused; fingerprint deterministic")


# G2: name-keyed optimizer state, including a cross-construction reorder --------------------


def _g2_make_blobs(d):
    """Worker: train+save once, write the good blob and three refusal blobs into d."""
    from v41f.train import train_step

    cfg = _cfg("off")
    m = _build(cfg)
    st = TrainState(m, lr=1e-2)
    train_step(m, _ids(cfg, 0), None, state=st)
    f = os.path.join(d, "c.pt")
    save_train_checkpoint(f, model=m, cfg=cfg, state=st, tokenizer=synthetic_tokenizer(), step=1)
    blob = torch.load(f, map_location="cpu", weights_only=False)
    rev = dict(blob)
    rev["optim_named"] = dict(blob["optim_named"])
    rev["optim_named"]["param_names"] = list(reversed(blob["optim_named"]["param_names"]))
    torch.save(rev, os.path.join(d, "rev.pt"))
    extra = dict(blob)
    extra["optim_named"] = {
        k: (dict(v) if k == "state_by_name" else v) for k, v in blob["optim_named"].items()
    }
    k0 = next(iter(extra["optim_named"]["state_by_name"]))
    extra["optim_named"]["state_by_name"][k0 + ".ghost"] = extra["optim_named"]["state_by_name"][k0]
    torch.save(extra, os.path.join(d, "extra.pt"))
    shape = dict(blob)
    shape["optim_named"] = {
        k: (dict(v) if k == "state_by_name" else v) for k, v in blob["optim_named"].items()
    }
    n0 = next(iter(shape["optim_named"]["state_by_name"]))
    shape["optim_named"]["state_by_name"][n0]["shape"] = [1, 1]
    torch.save(shape, os.path.join(d, "shape.pt"))


def _g2_try_load(path, fragment):
    """Worker: load one blob; exit 0 iff it REFUSES with the expected error fragment."""
    try:
        mm, ss, _, _ = load_train_checkpoint(path, tokenizer=synthetic_tokenizer(), max_batch_size=2)
        del mm, ss
        print(f"FAIL: {fragment} case loaded")
        sys.exit(2)
    except OptimStateError as e:
        if fragment not in str(e):
            print(f"FAIL: wrong refusal {e}")
            sys.exit(3)
        print(f"refused: {fragment}")
        sys.exit(0)


def _g2_good_load(path):
    """Worker: the good blob must load and report step 1 with name-keyed state attached."""
    _, st2, _, step = load_train_checkpoint(path, tokenizer=synthetic_tokenizer(), max_batch_size=2)
    assert step == 1
    n = st2.in_group_names[5]
    assert n in st2.optimizer.state.get(st2.master[n], {}) or st2.master[n] in st2.optimizer.state
    print("good load step=1")


def gate_optim_named_roundtrip_and_reorder():
    """Name-keyed optim round-trips; non-canonical order, extra state name, and shape tamper
    each refuse. Every load is its own process (an 180M model + AdamW per load OOMs in one)."""
    import subprocess

    d = _scratch(prefix="td_g2_")
    env = dict(os.environ, OMP_NUM_THREADS="2")

    def sp(*args, check_rc=0):
        r = subprocess.run([sys.executable, __file__, *args], capture_output=True, text=True, env=env)
        if r.returncode != check_rc:
            print(r.stdout)
            print(r.stderr)
            raise AssertionError(f"worker {' '.join(args)} rc={r.returncode}")
        return r

    sp("--mkblobs", d)
    sp("--goodload", os.path.join(d, "c.pt"))
    sp("--loadblob", os.path.join(d, "rev.pt"), "order")
    sp("--loadblob", os.path.join(d, "extra.pt"), "not in param_names")
    sp("--loadblob", os.path.join(d, "shape.pt"), "shape")
    print("  G2: name-keyed optim round-trips; bad order / extra name / shape tamper refused")


# M11: inference loader refuses a train blob ------------------------------------------------


def gate_inference_refuses_train_blob():
    from v41f.ckpt import load_checkpoint

    cfg = _cfg("off")
    m = _build(cfg)
    st = TrainState(m, lr=1e-3)
    d = _scratch(prefix="td_m11_")
    f = os.path.join(d, "c.pt")
    save_train_checkpoint(f, model=m, cfg=cfg, state=st, tokenizer=synthetic_tokenizer())
    try:
        load_checkpoint(f)
        raise AssertionError("inference loader opened a train blob")
    except ValueError as e:
        assert "TRAINING" in str(e)
    print("  M11: inference load_checkpoint refuses a v41f_train_ckpt blob")


# M13: refreshing an alias truncates; refresh must skip it (structural dispatch) -------------


def gate_refresh_does_not_touch_alias():
    cfg = _cfg("off")
    m = _build(cfg)
    st = TrainState(m, lr=1e-3)
    h = st.master["head.weight"]
    # inject the distinctive value and refresh; the fp32 alias must keep the un-rounded value.
    # 1.0000305 is the fp32 value that a bf16 round-trip collapses to exactly 1.0 (M13).
    probe = torch.tensor(1.0000305, dtype=torch.float32)
    with torch.no_grad():
        h.fill_(probe)
    st.refresh_bf16()
    got = h.flatten()[0].detach().item()
    assert got != 1.0 and abs(got - probe.item()) < 1e-7, f"alias was rounded by refresh -> {got}"
    print(f"  M13: bf16 refresh leaves fp32 alias {got:.7f} (not collapsed to 1.0)")


# Resume equivalence (design section 3): save/load mid-run is bit-exact with running on ----


def _apply_diag_thread_env():
    """DIAGNOSTIC-ONLY thread/engine knob for the runner bimodal investigation (#549). It is a
    strict no-op unless the non-default env is set, so the required gate is unchanged in normal
    CI. GATE_OMP pins the intra-op pool AND OMP_NUM_THREADS (the subprocess env below reads it,
    default "2"); GATE_ONEDNN=0 disables the oneDNN/MKLDNN engine before any tensor op. Used to
    tell a oneDNN code-path selection apart from plain multithreaded reduction under load.
    Never a production setting: threads=1 / oneDNN-off are diagnostic arms, not a fix."""
    omp = os.environ.get("GATE_OMP")
    if omp:
        os.environ["OMP_NUM_THREADS"] = omp
        os.environ.setdefault("MKL_NUM_THREADS", omp)
        torch.set_num_threads(int(omp))
    if os.environ.get("GATE_ONEDNN") == "0":
        torch.backends.mkldnn.enabled = False


def _one_run(kind):
    """Subprocess worker: run one trajectory, write probe master/run tensor bytes to stdout.

    When the parent set GATE_DUMP_DIR, also write the resume-asymmetry instrumentation there
    (spec docs/standards/resume_gate_divergence_instrumentation.md): a manifest with the leaf
    names, the AdamW triple at K (control) / as-restored (restart), the checkpoint sha and key
    sets, RNG across the boundary, and a NAMED FAIL on a populated leaf whose state is absent
    from state_by_name. Unset -> no dump, so local/laptop runs of the gate stay cheap."""
    from v41f.train import train_step

    _apply_diag_thread_env()
    dump = os.environ.get("GATE_DUMP_DIR")
    diag = None
    leaves = []
    batches = [_ids(_cfg("off"), s, seq=16) for s in range(4)]
    k = 2
    torch.manual_seed(123)
    cfg = _cfg("off")
    m = _build(cfg)
    st = TrainState(m, lr=1e-2)
    f = os.path.join(_scratch(prefix="td_eq_"), "c.pt")

    if dump:
        import json as _json
        dk = os.path.join(dump, kind)
        os.makedirs(dk, exist_ok=True)
        # lazy import keeps the non-dump gate free of the diagnostic's engine-env module effects
        sys.path.insert(0, str(_HERE))
        import diag_resume_bimodal as diag
        leaves = diag._pick_leaves(st)
        _json.dump(leaves, open(os.path.join(dk, "leaves.json"), "w"))

    ctrl_present = []
    for i, ids in enumerate(batches):
        if dump and i == k:
            # control records its state AT K too (no save/load), so the restart's optK/loadK have
            # a same-point reference. Recorded before the step at index k (i.e. after k steps).
            dk = os.path.join(dump, kind)
            ctrl_present = diag._opt_present(st, leaves)
            diag._dump_opt_triple(dk, "optK", st, leaves)
            torch.save(torch.get_rng_state(), os.path.join(dk, "rng_atK.pt"))
        if kind in ("restart", "fresh") and i == k:
            if dump and kind == "restart":
                dk = os.path.join(dump, kind)
                ctrl_present = diag._opt_present(st, leaves)
                torch.save(torch.get_rng_state(), os.path.join(dk, "rng_preSave.pt"))
            save_train_checkpoint(f, model=m, cfg=cfg, state=st, tokenizer=synthetic_tokenizer(), step=i)
            del m, st
            gc.collect()
            m, st, _, _ = load_train_checkpoint(f, tokenizer=synthetic_tokenizer(), max_batch_size=2)
            if kind == "fresh":
                st.optimizer = torch.optim.AdamW([st.master[n] for n in st.in_group_names], lr=1e-2)
            if dump and kind == "restart":
                dk = os.path.join(dump, kind)
                import json as _json
                diag._dump_opt_triple(dk, "loadK", st, leaves)
                torch.save(torch.get_rng_state(), os.path.join(dk, "rng_postLoad.pt"))
                ident = diag._ckpt_identity(f, live_model_keys=set(m.state_dict().keys()))
                dropped = diag.missing_populated_states(ctrl_present, ident["state_by_name_keys"])
                ident["populated_but_dropped"] = dropped
                with open(os.path.join(dk, "ckpt_identity.json"), "w") as fh:
                    _json.dump(ident, fh, indent=2)
                # ASSERT, do not merely record: a leaf populated at K that the blob omitted is a
                # silent fresh-optimizer resume (the master.py `if not st: continue` branch).
                assert not dropped, (
                    f"{len(dropped)} leaf/leaves had optimizer state at step {k} but are absent "
                    f"from state_by_name (silently resumed with a fresh optimizer): {dropped}")
        train_step(m, ids, None, state=st)
    st.refresh_bf16()
    named = dict(m.named_parameters())
    probe = "layers.0.attn.qproj.wq_b.weight"
    sys.stdout.buffer.write(st.master[probe].detach().cpu().float().numpy().tobytes())
    sys.stdout.buffer.write(b"\x00SEP\x00")
    sys.stdout.buffer.write(named[probe].detach().cpu().float().numpy().tobytes())


def gate_resume_equivalent_to_uninterrupted():
    """Save/load mid-run is bit-exact with the uninterrupted run; a fresh optimizer diverges.
    Each trajectory runs in its own process (each builds+updates an ~2.5 GB model)."""
    import subprocess

    # default "2" keeps the required gate byte-identical; GATE_OMP lets the #549 diagnostic
    # arms vary the worker thread count, and GATE_* passes through dict(os.environ).
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("GATE_OMP", "2"))
    # GATE_DUMP_DIR (set by the CI upload step / a debugger) points the workers at one root and
    # switches on the save/load instrumentation; unset leaves the gate unchanged and cheap.
    dump_root = os.environ.get("GATE_DUMP_DIR")
    dump_root_env = {}
    if dump_root:
        os.makedirs(dump_root, exist_ok=True)
        dump_root_env = {"GATE_DUMP_DIR": dump_root}

    def run(kind):
        r = subprocess.run(
            [sys.executable, __file__, "--run", kind], capture_output=True,
            env={**env, **dump_root_env}, check=True
        )
        a, b = r.stdout.split(b"\x00SEP\x00")
        return torch.frombuffer(bytearray(a), dtype=torch.float32).clone(), torch.frombuffer(
            bytearray(b), dtype=torch.float32
        ).clone()

    cm, cb = run("control")
    rm, rb = run("restart")
    fm, _ = run("fresh")

    # DIAGNOSE, do not guess the cause: on a bit-exact failure print the magnitude, the FIRST
    # offending flat index, and how many elements differ. NOTE (2026-09-19, measured): the old
    # "1e-8..1e-3 sparse = cross-process thread/BLAS reduction noise" guide was WRONG and led
    # the investigation to a thread hypothesis that does not hold. Both workers are seeded
    # (manual_seed(123) before build; init is the only global-RNG consumer), and seeded builds
    # at threads=1 AND threads=2 are bit-identical across separate processes (0/524288). The
    # ~1.3e-2 / near-total-element CI red is therefore NOT bf16 reduction order: it is a real
    # asymmetry between the control and save/load-restart trajectories (name-keyed optim
    # rebind / exp_avg / exp_avg_sq / step / serialization under investigation in #549).
    # An earlier "reproduced" 99.9%-different gradient was an UNSEEDED-model-init artifact.
    # Read the fields as: n_nan>0 NaN corruption; near-total elements + ~1e-2 with n_nan=0 =
    # the save/load asymmetry to localize by microstage, NOT reduction noise; O(1)/index shift
    # = name-bind/copy/dtype regression.
    def _eq(tag, want, got):
        wf, gf = want.float(), got.float()
        if not torch.equal(wf, gf):
            # mismatch predicate MUST match torch.equal: ~eq is True for NaN too, whereas
            # abs()>0 is False for NaN and would report n_diff=0 next to a failing assert --
            # a NaN (the failure this gate exists to catch) reading as "zero diff".
            neq = ~torch.eq(wf, gf)
            n_diff = int(neq.sum().item())
            n_nan = int(torch.isnan(wf).sum().item() + torch.isnan(gf).sum().item())
            first = int(torch.nonzero(neq, as_tuple=False)[0].item()) if n_diff else -1
            d = (wf - gf).abs()
            raise AssertionError(
                f"{tag} differs after save/load resume: max|delta|={d.max().item():.3e} "
                f"n_diff={n_diff}/{d.numel()} first_flat_idx={first} n_nan={n_nan} "
                f"(n_nan>0 = NaN corruption; near-total ~1e-2 n_nan=0 = save/load asymmetry, "
                f"see #549; O(1) or wholesale shift = real regression)")

    _eq("fp32 master", cm, rm)
    _eq("bf16 run weight", cb, rb)
    assert not torch.equal(cm, fm), "a fresh optimizer must diverge (anti-tautology failed)"
    if dump_root:
        _report_gate_dumps(dump_root)
    print("  resume: save/load mid-run bit-identical to control; fresh optim diverges")


def _report_gate_dumps(root):
    """Green-run confirmation that every sampled leaf restored its AdamW triple bit-exactly and
    the dump set is complete (the red-run evidence is written by the workers regardless). On a
    red this is never reached, but the worker dumps + ckpt_identity are already on disk for the
    upload-artifact step."""
    import json
    sys.path.insert(0, str(_HERE))
    import diag_resume_bimodal as diag

    cdir, rdir = os.path.join(root, "control"), os.path.join(root, "restart")
    leaves = json.load(open(os.path.join(rdir, "leaves.json")))
    ident = json.load(open(os.path.join(rdir, "ckpt_identity.json")))
    assert not ident["populated_but_dropped"], ident["populated_but_dropped"]
    assert ident["model_missing_keys"] == [] and ident["model_unexpected_keys"] == []
    n_checked = 0
    for j, n in enumerate(leaves):
        for part in ("exp_avg", "exp_avg_sq", "step"):
            a = diag._load(cdir, f"optK.l{j}.{part}")
            b = diag._load(rdir, f"loadK.l{j}.{part}")
            if a is None and b is None:
                continue
            assert a is not None and b is not None, f"{n}.{part} present on only one side"
            assert torch.equal(a, b), f"{n}.{part} restored triple differs from control"
            n_checked += 1
    rng_c = diag._load(cdir, "rng_atK")
    rng_pre = diag._load(rdir, "rng_preSave")
    rng_post = diag._load(rdir, "rng_postLoad")
    # Saving must not move RNG (control-atK == restart-preSave). postLoad can differ: rebuilding
    # the model on load consumes init RNG the checkpoint does not persist; train_step consumes no
    # RNG, so this is recorded, not failed (a future rand/dropout would make it load-bearing).
    assert rng_c is not None and rng_pre is not None and torch.equal(rng_c, rng_pre), \
        "RNG changed during save (control-atK != restart-preSave)"
    assert rng_post is not None
    rng_note = "postLoad identical" if torch.equal(rng_c, rng_post) else \
        "postLoad differs (model-rebuild init RNG; train_step consumes none)"
    print(f"  resume instrumentation: {n_checked} leaf-triple comparisons bit-exact, "
          f"sha256={ident['sha256'][:12]}, dropped=[], keys complete, RNG {rng_note}")


# Prod census (structural, no optimizer allocation -- the 0.9B build OOMs a laptop; the full
# prod backward is GPU-deferred per #497). Pins the requires_grad name SETS on the default.
def gate_prod_census_structural():
    import dataclasses

    from v41f.config import v41f_s

    cfg = dataclasses.replace(
        v41f_s(tokenizer=synthetic_tokenizer()),
        engram_n_heads=2,
        engram_head_dim=8,
        engram_vocab_size=20,
        engram_pad_id=2,
    )
    cfg = cfg.with_derived_engram(tokenizer=synthetic_tokenizer())
    six = {
        f"layers.{l}.attn.indexer.{w}.weight"
        for l in cfg.index_source_layers
        for w in ("wq_b", "weights_proj")
    }
    expect = {"off": 2145, "ste": 2151}
    for mode, n_rg in expect.items():
        c2 = dataclasses.replace(cfg, indexer_train_mode=mode)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            m = V41FModel(c2, max_batch_size=1, max_seq_len=64, tokenizer=synthetic_tokenizer())
        finally:
            torch.set_default_dtype(prev)
        total = sum(1 for _ in m.named_parameters())
        rg = {n for n, p in m.named_parameters() if p.requires_grad}
        assert total == 2153, f"prod total {total} != 2153"
        assert len(rg) == n_rg, f"prod {mode} rg {len(rg)} != {n_rg}"
        # SIX derived from index_source_layers=(2,4,8); the two index_key leaves on kv layer 2
        idxk = {f"layers.2.attn.index_key.{w}.weight" for w in ("wk", "k_norm")}
        assert not (rg & idxk), "prod index_key leaves not frozen"
        if mode == "off":
            assert not (rg & six)
        else:
            assert six <= rg
        del m
    assert len(six) == 6
    print("  prod census (structural): total 2153; off 2145 / ste 2151; SIX 6, index_key 2 frozen")


# M2: a master saved as bf16 must be refused (the exact #441 truncation regression) ----------


def _m2_worker(path):
    try:
        mm, ss, _, _ = load_train_checkpoint(path, tokenizer=synthetic_tokenizer(), max_batch_size=2)
        del mm, ss
        print("FAIL: a bf16 master loaded")
        sys.exit(2)
    except (AssertionError, ValueError, RuntimeError) as e:
        print(f"refused bf16 master: {str(e)[:70]}")
        sys.exit(0)


def gate_legacy_engram_config_refused_on_load():
    """A blob whose engram-ON config lacks the derived fields is refused AT LOAD, by name.

    MEASURED 2026-09-18 (genA, second read of #529): load_train_checkpoint rebuilt the
    config and went straight to V41FModel, so this blob died inside NgramHashState with
    `AssertionError (6, 0)` -- the deep failure the config gate exists to move up, only
    moved from a fresh run to a resume, where the caller has a checkpoint in hand and no
    reason to suspect its config. The guard is on the rebuilt config, so the mutant that
    kills it is the guard line itself.
    """

    from v41f.config import v41f_small

    tok = synthetic_tokenizer()
    cfg = v41f_small(
        vocab_size=len(tok),
        tokenizer=tok,
        engram_layer_ids=(1,),
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=8,
        engram_vocab_size=20,
        engram_pad_id=2,
    )
    m = None  # engram-ON build needs the tokenizer; done below
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        m = V41FModel(cfg, max_batch_size=2, max_seq_len=64, tokenizer=tok)
    finally:
        torch.set_default_dtype(prev)
    st = TrainState(m, lr=1e-3)
    d = _scratch(prefix="td_legacy_")
    f = os.path.join(d, "c.pt")
    save_train_checkpoint(f, model=m, cfg=cfg, state=st, tokenizer=tok)
    del m, st
    gc.collect()

    # the legacy shape: engram ON, both derived fields back at their unusable defaults
    blob = torch.load(f, map_location="cpu", weights_only=False)
    assert blob["config"]["engram_layer_ids"], "fixture drift: needs an engram-ON blob"
    blob["config"] = {
        **blob["config"],
        "engram_num_embeddings": [],
        "engram_compressed_vocab_size": 0,
    }
    legacy = os.path.join(d, "legacy.pt")
    torch.save(blob, legacy)

    try:
        load_train_checkpoint(legacy, tokenizer=tok, max_batch_size=2)
    except ValueError as e:
        msg = str(e)
        assert "engram" in msg and "with_derived_engram" in msg, msg
        print("  legacy engram config refused at load by name (not in NgramHashState)")
        return
    except AssertionError as e:
        raise AssertionError(
            "the legacy blob still died deep in model construction instead of being refused "
            f"at load: {e!r} -- move the guard ahead of the V41FModel build"
        ) from e
    raise AssertionError("a legacy engram-ON config with unset derived fields loaded clean")


def gate_master_saved_bf16_refused():
    import subprocess

    cfg = _cfg("off")
    m = _build(cfg)
    st = TrainState(m, lr=1e-3)
    d = _scratch(prefix="td_m2_")
    f = os.path.join(d, "c.pt")
    save_train_checkpoint(f, model=m, cfg=cfg, state=st, tokenizer=synthetic_tokenizer())
    blob = torch.load(f, map_location="cpu", weights_only=False)
    n0 = next(iter(blob["master_fp32"]))
    blob["master_fp32"][n0] = blob["master_fp32"][n0].to(torch.bfloat16)
    torch.save(blob, os.path.join(d, "bf16.pt"))
    env = dict(os.environ, OMP_NUM_THREADS="2")
    r = subprocess.run(
        [sys.executable, __file__, "--m2", os.path.join(d, "bf16.pt")],
        capture_output=True,
        text=True,
        env=env,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError("M2 did not refuse a bf16 master")
    print("  M2: a master tensor saved as bf16 is refused on load")


def _stray_td(root):
    """The leak predicate. One implementation, shared by the live guard and its selftest.

    Only `td_` entries count: torch's compile cache lands under TMPDIR by design and is not
    this file's to remove. Asserting the root was EMPTY fired on `torchinductor_chenkailun.c`
    and named `_scratch()` as the fix for a dir this test never created -- a guard with a
    known false positive gets switched off.
    """
    return [x for x in sorted(os.listdir(root)) if x.startswith("td_")]


def _selftest():
    _leakguard_selftest()
    # Each gate builds a 180M model + fp32 master + AdamW (~2.5 GB) and several do save/load,
    # so running all seven in one process accumulates enough to OOM a laptop. Run each gate in
    # its OWN process (process-private memory, returned to the OS on exit); a failure in any
    # one fails the whole run by name.
    import shutil
    import subprocess

    # one process-private scratch root handed to every gate as TMPDIR (mkdtemp honours it);
    # a failed run leaves multi-GB blobs, so remove only our own root, never a shared prefix.
    root = _scratch(prefix="td_p1_root_")
    gates = [
        "gate_census_membership",
        "gate_buffers_in_model_not_master",
        "gate_alias_survives_save_load",
        "gate_tokenizer_and_vocab_id",
        "gate_optim_named_roundtrip_and_reorder",
        "gate_inference_refuses_train_blob",
        "gate_refresh_does_not_touch_alias",
        "gate_resume_equivalent_to_uninterrupted",
        "gate_prod_census_structural",
        "gate_legacy_engram_config_refused_on_load",
        "gate_master_saved_bf16_refused",
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "2"
    env["TMPDIR"] = root  # grandchildren mkdtemp under our private root too
    try:
        for name in gates:
            r = subprocess.run(
                [sys.executable, __file__, "--gate", name], capture_output=True, text=True, env=env
            )
            if r.returncode != 0:
                print(r.stdout)
                print(r.stderr)
                raise AssertionError(f"{name} failed (rc={r.returncode})")
            print(r.stdout.strip().splitlines()[-1])
            # THE LEAK GUARD. A child that exits leaves no scratch dir of OURS behind: if it
            # did, the direct-run path would keep filling the ambient /tmp exactly as it did
            # on 2026-09-18 (77 GB, root filesystem to 100%). Checked per gate, right after
            # the child exits, while a leak is still attributable to the gate that caused it
            # -- one check at the end cannot say which gate leaked, and naming the producer
            # is the whole point. The parent's rmtree below would otherwise erase the evidence.
            stray = _stray_td(root)
            if stray:
                raise AssertionError(
                    f"{name} left {len(stray)} scratch entr(ies) under its TMPDIR root: "
                    f"{stray[:4]} -- a dir this file created outlived the child. Route it "
                    f"through _scratch()."
                )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("p1 train ckpt OK")


def _leakguard_selftest():
    """Prove the per-gate leak guard DISCRIMINATES, on worlds built from the real predicate.

    A guard that never fires and a guard that always fires both pass a "the suite is green"
    reading, so the predicate is driven against four child processes that differ only in
    what they leave under TMPDIR. Cheap and model-free: no gate is run, so this can live in
    CI where the 180M gates cannot.

    It calls `_stray_td`, the SAME function the live guard calls, rather than restating the
    `td_` filter. A selftest that re-implements its subject certifies the re-implementation:
    the two agree today and the guard can change alone tomorrow, which is the drift this
    whole case exists to prevent.

    CALLED FROM `_selftest()` (genB 2026-09-18). It was reachable only by typing
    `--leakguard`, which no CI job and no hook entry names -- 76 lines of four-world
    discrimination that would never have run. `_selftest()` is what `ci.yml` invokes for
    this module and what the hook's SELFTEST_FILES entry runs, so the wiring is the fix.

    The two `torchinductor_*` worlds are the ones a naive `assert not os.listdir(root)` gets
    wrong, and it got them wrong in practice on 2026-09-18 -- which is why the predicate
    filters on `td_` rather than asserting emptiness.
    """
    import subprocess
    import tempfile

    guard = _stray_td

    worlds = [
        (
            "a leaking child (mkdtemp, no cleanup)",
            "import tempfile; tempfile.mkdtemp(prefix='td_x_')",
            True,
        ),
        (
            "a clean child",
            "import tempfile, shutil; shutil.rmtree(tempfile.mkdtemp(prefix='td_x_'))",
            False,
        ),
        (
            "a child that only warms the torch cache",
            "import os, tempfile; os.makedirs(os.path.join(tempfile.gettempdir(), 'torchinductor_x'))",
            False,
        ),
        (
            "a leaking child that also warms the cache",
            "import os, tempfile; tempfile.mkdtemp(prefix='td_x_'); "
            "os.makedirs(os.path.join(tempfile.gettempdir(), 'torchinductor_x'))",
            True,
        ),
    ]
    for why, code, want_fire in worlds:
        root = tempfile.mkdtemp(prefix="td_lg_root_")
        try:
            subprocess.run([sys.executable, "-c", code], env=dict(os.environ, TMPDIR=root), check=True)
            fired = bool(guard(root))
            assert fired == want_fire, f"leak guard wrong on {why!r}: fired={fired} want={want_fire}"
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)
    print(
        "leak guard OK: fires on a leaked td_ dir, silent on a clean child AND on torch's "
        "own cache dir (4/4 worlds)"
    )


if __name__ == "__main__":
    a = sys.argv
    if "--gate" in a:
        globals()[a[a.index("--gate") + 1]]()
    elif "--run" in a:
        _one_run(a[a.index("--run") + 1])
    elif "--mkblobs" in a:
        _g2_make_blobs(a[a.index("--mkblobs") + 1])
    elif "--goodload" in a:
        _g2_good_load(a[a.index("--goodload") + 1])
    elif "--loadblob" in a:
        _g2_try_load(a[a.index("--loadblob") + 1], a[a.index("--loadblob") + 2])
    elif "--m2" in a:
        _m2_worker(a[a.index("--m2") + 1])
    elif "--leakguard" in a:
        _leakguard_selftest()
    else:
        _selftest()
