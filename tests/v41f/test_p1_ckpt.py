"""P1: model checkpoint save/load round-trip.

The default v41f_small config leaves engram/MTP off, so the checkpoint exercises the
exact production forward stack: the persistent gate.bias buffer must travel, and the
engram ModuleList of None must not emit empty/None tensors or break strict load.
"""

import gc
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.ckpt import load_checkpoint, save_checkpoint
from v41f.config import v41f_small
from v41f.loss import shifted_cross_entropy as shifted_ce
from v41f.model import V41FModel
from v41f.train import train_step

_TMP = Path(tempfile.mkdtemp(prefix="v41f_ckpt_"))


def _p(name):
    return _TMP / name


def _bf16_model(seed=0):
    cfg = v41f_small()
    torch.manual_seed(seed)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = V41FModel(cfg, max_batch_size=2).eval()
    torch.set_default_dtype(prev)
    return model, cfg


def test_save_load_forward_bit_exact():
    model, cfg = _bf16_model()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        before, mb = model(ids)
    path = _p("m.pt")
    save_checkpoint(path, model, cfg)
    loaded, lcfg = load_checkpoint(path)
    assert lcfg == cfg, "config did not round-trip"
    with torch.no_grad():
        after, ma = loaded(ids)
    assert mb is None and ma is None
    assert torch.equal(after, before), (after - before).abs().max().item()
    del model, loaded
    gc.collect()


gc.collect()


def test_gate_bias_buffer_present_and_restored():
    model, _ = _bf16_model()
    path = _p("m.pt")
    save_checkpoint(path, model, v41f_small())
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    bias_keys = [k for k in sd if k.endswith("ffn.gate.bias")]
    assert bias_keys, "persistent gate.bias buffer missing from checkpoint"
    # a non-default buffer value survives
    with torch.no_grad():
        model.layers[0].ffn.gate.bias.fill_(0.375)
    save_checkpoint(path, model, v41f_small())
    loaded, _ = load_checkpoint(path)
    assert torch.allclose(
        loaded.layers[0].ffn.gate.bias, torch.full_like(loaded.layers[0].ffn.gate.bias, 0.375), atol=1e-6
    )


gc.collect()


def test_engram_none_slots_emit_no_keys_and_strict_load():
    model, cfg = _bf16_model()
    assert all(slot is None for slot in model.engrams)
    save_checkpoint(_p("m.pt"), model, cfg)
    sd = torch.load(_p("m.pt"), map_location="cpu", weights_only=False)["state_dict"]
    assert not [k for k in sd if "engram" in k], "ModuleList-of-None must emit no checkpoint keys"
    # strict load is exercised inside load_checkpoint; assert it really is strict by feeding
    # a checkpoint with a missing key directly.
    del sd["head.weight"]
    from dataclasses import asdict

    torch.save({"config": asdict(cfg), "state_dict": sd}, _p("broken.pt"))
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(_p("broken.pt"))
    # an unexpected extra key is also a strict-load error, not a silent ignore
    extra = dict(model.state_dict())
    extra["bogus.extra"] = torch.zeros(3)
    torch.save({"config": asdict(cfg), "state_dict": extra}, _p("extra.pt"))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_checkpoint(_p("extra.pt"))


gc.collect()


def test_config_mismatch_is_loud():
    """A config that builds but shapes tensors differently (dim) must fail strict load, not
    silently reshape."""
    from dataclasses import asdict, replace

    model, cfg = _bf16_model()
    save_checkpoint(_p("m.pt"), model, cfg)
    raw = torch.load(_p("m.pt"), map_location="cpu", weights_only=False)
    raw["config"] = asdict(replace(cfg, dim=cfg.dim * 2))
    torch.save(raw, _p("badcfg.pt"))
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_checkpoint(_p("badcfg.pt"))


gc.collect()


def test_resume_continues_training():
    """Train one step, checkpoint, reload, and keep training: the restored weights must
    pick up where they left off (loss decreases from the resumed point). Optimizer state is
    not in this model checkpoint, so this proves weight resume, not optimizer continuity."""
    cfg = v41f_small()
    torch.manual_seed(3)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = V41FModel(cfg, max_batch_size=2).train()
    torch.set_default_dtype(prev)
    model = model.float()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    train_step(model, ids, opt)
    save_checkpoint(_p("r.pt"), model, cfg)
    # release the source model and its optimizer before constructing the loaded one, so the
    # direct (no-per-test GC) selftest runner never holds two fp32 models at once
    del model, opt
    gc.collect()
    loaded, _ = load_checkpoint(_p("r.pt"), eval_mode=False)
    loaded = loaded.float().train()
    opt2 = torch.optim.AdamW(loaded.parameters(), lr=3e-3)
    with torch.no_grad():
        pre = shifted_ce(loaded(ids)[0], ids)
    for _ in range(20):
        train_step(loaded, ids, opt2)
    with torch.no_grad():
        post = shifted_ce(loaded(ids)[0], ids)
    assert post.item() < pre.item() * 0.2, (pre.item(), post.item())


gc.collect()


def test_weight_mutation_changes_loaded_forward():
    model, cfg = _bf16_model()
    save_checkpoint(_p("m.pt"), model, cfg)
    loaded, _ = load_checkpoint(_p("m.pt"))
    raw = torch.load(_p("m.pt"), map_location="cpu", weights_only=False)
    raw["state_dict"]["head.weight"].add_(1.0)
    torch.save(raw, _p("mut.pt"))
    mutated, _ = load_checkpoint(_p("mut.pt"))
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        a, _ = loaded(ids)
        b, _ = mutated(ids)
    assert not torch.equal(a, b), "a changed weight must change the loaded forward"
    del model, loaded, mutated
    gc.collect()


gc.collect()
