"""P1: model checkpoint save/load round-trip.

The default v41f_small config leaves engram/MTP off, so the checkpoint exercises the
exact production forward stack: the persistent gate.bias buffer must travel, and the
engram ModuleList of None must not emit empty/None tensors or break strict load.

Storage is exercised in the NATIVE dtype of a production-constructed model (bf16 weights,
fp32 head/HC/sink). fp32-master weights plus optimizer state belong to a later trainer
checkpoint PR; an fp32 model saved into a bf16-default reload would truncate weights, so
that mixed path is deliberately not asserted here.

Disk hygiene: each case does its model work in an inner function. That function's frame
(which owns the torch.load-backed tensors) is destroyed when it returns, and _tmpdir
gc-collects before removing the directory WITHOUT swallowing errors -- on macOS a still-open
storage handle can make an unlink fail intermittently, and ignoring that leaves ~965MB
behind. The last test asserts every path THIS process registered is gone (it does not scan
the shared temp root, which other processes legitimately use).
"""

import contextlib
import gc
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.ckpt import load_checkpoint, save_checkpoint
from v41f.config import v41f_small
from v41f.model import V41FModel

_PREFIX = "v41f_ckpt_"

# paths THIS process created through _tmpdir; the leak test asserts only these are gone,
# never a global gettempdir() glob (other processes/tests legitimately own other temp dirs).
_CREATED = set()


def _rmtree_loud(path, attempts=4, delay=0.05):
    """Remove a dir with a bounded retry for macOS' "handle just closed, unlink not yet
    released" window. Never swallows: if every attempt fails the last OSError propagates."""
    last = None
    for _ in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError as exc:  # pragma: no cover - exercised only under handle delay
            last = exc
            time.sleep(delay)
    raise last


@contextlib.contextmanager
def _tmpdir():
    """Create a temp dir registered in _CREATED and guarantee THIS path is removed on exit,
    loudly. On exit the body frame is destroyed; gc twice drops torch.load-backed file
    handles, then the tracked path is removed with a bounded retry (no ignore_errors)."""
    d = Path(tempfile.mkdtemp(prefix=_PREFIX))
    _CREATED.add(d)
    try:
        yield d
    finally:
        gc.collect()
        gc.collect()
        _rmtree_loud(d)
        _CREATED.discard(d)
        assert not d.exists(), f"rmtree returned but the tracked temp dir still exists: {d}"


def _bf16_model(seed=0):
    cfg = v41f_small()
    torch.manual_seed(seed)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = V41FModel(cfg, max_batch_size=2).eval()
    torch.set_default_dtype(prev)
    return model, cfg


def test_save_load_forward_bit_exact():
    def body(d):
        model, cfg = _bf16_model()
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        with torch.no_grad():
            before, mb = model(ids)
        save_checkpoint(d / "m.pt", model, cfg)
        loaded, lcfg = load_checkpoint(d / "m.pt")
        assert lcfg == cfg, "config did not round-trip"
        with torch.no_grad():
            after, ma = loaded(ids)
        assert mb is None and ma is None
        assert torch.equal(after, before), (after - before).abs().max().item()

    with _tmpdir() as d:
        body(d)


def test_gate_bias_buffer_present_and_restored():
    def body(d):
        model, _ = _bf16_model()
        save_checkpoint(d / "m.pt", model, v41f_small())
        sd = torch.load(d / "m.pt", map_location="cpu", weights_only=False)["state_dict"]
        assert [k for k in sd if k.endswith("ffn.gate.bias")], "persistent gate.bias missing"
        with torch.no_grad():
            model.layers[0].ffn.gate.bias.fill_(0.375)
        save_checkpoint(d / "m.pt", model, v41f_small())
        loaded, _ = load_checkpoint(d / "m.pt")
        assert loaded.layers[0].ffn.gate.bias.dtype == torch.float32
        assert torch.allclose(
            loaded.layers[0].ffn.gate.bias,
            torch.full_like(loaded.layers[0].ffn.gate.bias, 0.375),
            atol=1e-6,
        )

    with _tmpdir() as d:
        body(d)


def test_engram_none_slots_emit_no_keys_and_strict_load():
    def body(d):
        model, cfg = _bf16_model()
        assert all(slot is None for slot in model.engrams)
        save_checkpoint(d / "m.pt", model, cfg)
        sd = torch.load(d / "m.pt", map_location="cpu", weights_only=False)["state_dict"]
        assert not [k for k in sd if "engram" in k], "ModuleList-of-None emits no keys"
        missing = dict(sd)
        del missing["head.weight"]
        torch.save({"config": asdict(cfg), "state_dict": missing}, d / "missing.pt")
        with pytest.raises(RuntimeError, match="Missing key"):
            load_checkpoint(d / "missing.pt")
        extra = dict(sd)
        extra["bogus.extra"] = torch.zeros(3)
        torch.save({"config": asdict(cfg), "state_dict": extra}, d / "extra.pt")
        with pytest.raises(RuntimeError, match="Unexpected key"):
            load_checkpoint(d / "extra.pt")

    with _tmpdir() as d:
        body(d)


def test_config_mismatch_is_loud():
    def body(d):
        model, cfg = _bf16_model()
        save_checkpoint(d / "m.pt", model, cfg)
        raw = torch.load(d / "m.pt", map_location="cpu", weights_only=False)
        raw["config"] = asdict(replace(cfg, dim=cfg.dim * 2))
        torch.save(raw, d / "badcfg.pt")
        with pytest.raises(RuntimeError, match="size mismatch"):
            load_checkpoint(d / "badcfg.pt")

    with _tmpdir() as d:
        body(d)


def test_bf16_weight_resume_bit_exact_by_name():
    """Save a production (bf16) model, reload into the same dtype, and prove EVERY parameter
    and persistent buffer survives bit-for-bit by name -- bf16 linears/experts/norms/
    compressor AND the fp32 head, HC tables and attn_sink. fp32-master + optimizer
    continuity is a separate trainer-checkpoint concern."""

    def body(d):
        model, cfg = _bf16_model()
        with torch.no_grad():  # post-training movement so init-equality cannot pass
            for p in model.parameters():
                p.add_(0.01)
        save_checkpoint(d / "m.pt", model, cfg)
        loaded, _ = load_checkpoint(d / "m.pt")
        src_p, dst_p = dict(model.named_parameters()), dict(loaded.named_parameters())
        assert set(src_p) == set(dst_p)
        for name in src_p:
            assert src_p[name].dtype == dst_p[name].dtype, f"{name} dtype changed"
            assert torch.equal(src_p[name], dst_p[name]), f"{name} not bit-exact after reload"
        persisted = set(model.state_dict())  # non-persistent runtime buffers are rebuilt, not stored
        src_b = {n: b for n, b in model.named_buffers() if n in persisted}
        dst_b = {n: b for n, b in loaded.named_buffers() if n in persisted}
        assert set(src_b) == set(dst_b)
        for name in src_b:
            assert torch.equal(src_b[name], dst_b[name]), f"persistent buffer {name} changed"

    with _tmpdir() as d:
        body(d)


def test_weight_mutation_changes_loaded_forward():
    def body(d):
        model, cfg = _bf16_model()
        save_checkpoint(d / "m.pt", model, cfg)
        loaded, _ = load_checkpoint(d / "m.pt")
        raw = torch.load(d / "m.pt", map_location="cpu", weights_only=False)
        raw["state_dict"]["head.weight"].add_(1.0)
        torch.save(raw, d / "mut.pt")
        mutated, _ = load_checkpoint(d / "mut.pt")
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        with torch.no_grad():
            a, _ = loaded(ids)
            b, _ = mutated(ids)
        assert not torch.equal(a, b), "a changed weight must change the loaded forward"

    with _tmpdir() as d:
        body(d)


def test_zzz_no_tracked_temp_dirs_left_behind():
    """Runs last (sorted). Assert only the dirs THIS process created via _tmpdir are gone --
    never scan the shared temp root, which other processes/tests legitimately use."""
    leaked = [p for p in _CREATED if Path(p).exists()]
    assert not leaked, f"this process leaked checkpoint temp dirs: {leaked}"


def test_zzz_rmtree_failure_is_loud():
    """A removal failure must propagate, never be silently ignored (that was the flake:
    ignore_errors hid macOS unlink failures). Manually stub shutil.rmtree (no pytest
    fixture, so the direct no-args selftest runner can call this) to always raise, assert
    the context manager surfaces it after its bounded retries, then restore and clean up."""

    def _boom(*_a, **_k):
        raise OSError("injected unlink failure")

    real_rmtree, real_mkdtemp, real_sleep = shutil.rmtree, tempfile.mkdtemp, time.sleep
    created = []
    shutil.rmtree = _boom
    time.sleep = lambda *_a, **_k: None
    tempfile.mkdtemp = lambda *a, **k: created.append(real_mkdtemp(*a, **k)) or created[-1]
    try:
        with pytest.raises(OSError, match="injected unlink failure"), _tmpdir():
            pass
    finally:
        shutil.rmtree, tempfile.mkdtemp, time.sleep = real_rmtree, real_mkdtemp, real_sleep
        for path in created:
            real_rmtree(path, ignore_errors=True)
            _CREATED.discard(Path(path))
