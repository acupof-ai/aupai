"""P1 gates for step D: fp32-master training checkpoint (v41f/master.py).

Design: docs/standards/v41f_train_checkpoint_design.md (prereview #487). Direct-runner
compatible, process-private temp dirs. The 180M v41f_small means two live models + two
AdamW states in one process OOM on a laptop, so resume tests build model B only after model
A is gc'd, and the prod census is structural (no optimizer allocation); the full prod
backward is GPU-deferred (#497).
"""

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
from v41f.master import TrainState, save_train_checkpoint, load_train_checkpoint, OptimStateError  # noqa: E402
from v41f.model import V41FModel  # noqa: E402
from v41f.vocab import fingerprint  # noqa: E402

VOCAB = 12800
_SMALL = dict()


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
    d = tempfile.mkdtemp(prefix="td_g6_")
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
    d = tempfile.mkdtemp(prefix="td_g7_")
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
    d = tempfile.mkdtemp(prefix="td_g3_")
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

    d = tempfile.mkdtemp(prefix="td_g2_")
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
    d = tempfile.mkdtemp(prefix="td_m11_")
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


def _one_run(kind):
    """Subprocess worker: run one trajectory, write probe master/run tensor bytes to stdout."""
    from v41f.train import train_step

    batches = [_ids(_cfg("off"), s, seq=16) for s in range(4)]
    k = 2
    torch.manual_seed(123)
    cfg = _cfg("off")
    m = _build(cfg)
    st = TrainState(m, lr=1e-2)
    f = os.path.join(tempfile.mkdtemp(prefix="td_eq_"), "c.pt")
    for i, ids in enumerate(batches):
        if kind in ("restart", "fresh") and i == k:
            save_train_checkpoint(f, model=m, cfg=cfg, state=st, tokenizer=synthetic_tokenizer(), step=i)
            del m, st
            gc.collect()
            m, st, _, _ = load_train_checkpoint(f, tokenizer=synthetic_tokenizer(), max_batch_size=2)
            if kind == "fresh":
                st.optimizer = torch.optim.AdamW([st.master[n] for n in st.in_group_names], lr=1e-2)
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

    env = dict(os.environ, OMP_NUM_THREADS="2")

    def run(kind):
        r = subprocess.run(
            [sys.executable, __file__, "--run", kind], capture_output=True, env=env, check=True
        )
        a, b = r.stdout.split(b"\x00SEP\x00")
        return torch.frombuffer(bytearray(a), dtype=torch.float32).clone(), torch.frombuffer(
            bytearray(b), dtype=torch.float32
        ).clone()

    cm, cb = run("control")
    rm, rb = run("restart")
    fm, _ = run("fresh")
    assert torch.equal(cm, rm), "fp32 master differs after save/load resume"
    assert torch.equal(cb, rb), "bf16 run weight differs after save/load resume"
    assert not torch.equal(cm, fm), "a fresh optimizer must diverge (anti-tautology failed)"
    print("  resume: save/load mid-run bit-identical to control; fresh optim diverges")


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


def gate_master_saved_bf16_refused():
    import subprocess

    cfg = _cfg("off")
    m = _build(cfg)
    st = TrainState(m, lr=1e-3)
    d = tempfile.mkdtemp(prefix="td_m2_")
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


def _selftest():
    # Each gate builds a 180M model + fp32 master + AdamW (~2.5 GB) and several do save/load,
    # so running all seven in one process accumulates enough to OOM a laptop. Run each gate in
    # its OWN process (process-private memory, returned to the OS on exit); a failure in any
    # one fails the whole run by name.
    import shutil
    import subprocess

    # one process-private scratch root handed to every gate as TMPDIR (mkdtemp honours it);
    # a failed run leaves multi-GB blobs, so remove only our own root, never a shared prefix.
    root = tempfile.mkdtemp(prefix="td_p1_root_")
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
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("p1 train ckpt OK")


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
    else:
        _selftest()
