#!/usr/bin/env python3
"""Step 5 (AttnRes removal): the load contract the removal must not break.

WHY THIS EXISTS. `CLAUDE.md:48` states the contract the AttnRes-era code is serving:
"Old checkpoints still load via `_cfg` (`scripts/loader.py`)". Step 5 removes AttnRes,
and the thing that can go wrong quietly is this contract: `scripts/loader.py:90` rebuilds
the model from `ck["cfg"]`, so a checkpoint whose cfg lacks `attn_res` gets the LIVE
default backfilled (`loader.py:92`), and `HybridLM.load_state_dict` is what turns that
back into a model that loads (`scripts/test_arch_compat.py:145` already asserts the
effect on a synthetic state_dict). Remove the mechanism with nothing in its place and
that line reds with a strict-load traceback instead of a named refusal.

WHAT THIS MEASURES, and it is not a restatement of :145. That assertion uses a state_dict
built in-process from a live model. This one drives the two REAL loader entry points a
consumer actually uses -- `loader.load_checkpoint(path)` and the plain `_cfg` rebuild --
against a checkpoint written to disk, so a change that fixes the in-process path while
breaking the on-disk one is caught.

THE POD EVIDENCE THIS ENCODES (read 2026-09-23, read-only): of the 13 checkpoints
reachable on the pod, 12 record `attn_res=false` and the 13th
(`/data00/ckpt_k3-mla_2b_step2000.pt`) has NO `attn_res` key at all and zero
`final_ar.`/`ar1.`/`ar2.` keys in its state_dict -- so it is a KDA-line checkpoint that
predates AttnRes, and the auto-disable is what loads it. Declared here as data so the
test states which real file it stands for, rather than a synthetic shape with no
provenance.

RUN: python3 scripts/test_attnres_load_contract.py --selftest
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))


def _write_legacy_ckpt(path):
    """A checkpoint in the shape the K3 file has: a legacy state_dict, no attn_res key."""
    from model import HybridLM
    from train import Cfg

    Cfg.d, Cfg.heads, Cfg.layers, Cfg.ffn_hidden, Cfg.vocab, Cfg.seq = 64, 2, 4, 128, 100, 16
    Cfg.attn_res, Cfg.grad_ckpt = False, False
    old = HybridLM(Cfg)
    sd, cfg = {}, {}
    for k, v in old.state_dict().items():
        if k.endswith(".gb.weight"):
            sd[k.replace("gb", "gate_proj")] = v[: Cfg.d]
            sd[k.replace("gb", "beta_proj")] = v[Cfg.d : Cfg.d + Cfg.heads]
            continue
        for fused, (a, b) in {"w13": ("w1", "w3"), "kv_up": ("k_up", "v_up"),
                              "qg": ("q", "gate")}.items():
            if k.endswith(f".{fused}.weight"):
                va, vb = v.chunk(2)
                sd[k.replace(fused, a)] = va
                sd[k.replace(fused, b)] = vb
                break
        else:
            sd[k] = v
    for f in ("d", "heads", "layers", "ffn_hidden", "vocab", "seq", "attn_every"):
        cfg[f] = getattr(Cfg, f)
    # THE K3 SHAPE: no attn_res key, exactly as the pod file records it.
    assert "attn_res" not in cfg, "this fixture must omit attn_res to stand for the K3 file"
    torch.save({"model": sd, "cfg": cfg}, path)
    return path


def check_contract(tmpdir):
    """The contract: a ckpt with no attn_res key still LOADS, and the model says so.

    THE LIVE DEFAULT MUST BE True HERE, and that is the whole precondition. `loader.py:92`
    backfills a missing cfg key from the live `Cfg`, and the `Cfg.attn_res` field defaults
    `True`. A first version of this file left `Cfg.attn_res` at the False its
    own fixture had set, so the backfill produced False, no AttnRes was built, and the load
    succeeded with or without the auto-disable -- the mutant came back GREEN and said so.
    The fixture had removed the precondition the contract exists for.
    """
    from scripts.loader import load_checkpoint  # noqa: E402
    from train import Cfg  # noqa: E402

    path = os.path.join(tmpdir, "legacy_no_attn_res.pt")
    _write_legacy_ckpt(path)
    Cfg.attn_res = True  # the live default Cfg.attn_res ships; the backfill reads THIS

    model, cfg = load_checkpoint(path)
    # (a) it loaded at all -- a strict-load failure here is the regression
    assert model is not None, "load_checkpoint returned no model for a legacy ckpt"
    # (b) the model is honest about what it is: AttnRes must be OFF, not silently constructed
    assert getattr(cfg, "attn_res", None) is False or getattr(model, "attn_res", None) is False, (
        "a checkpoint with no attn_res key produced a model with AttnRes ON: the backfill at "
        "loader.py:92 set it from the live Cfg and nothing turned it back off. This is the "
        "contract CLAUDE.md:48 states and the K3 file depends on")
    # (c) NO AttnRes modules are left dangling on the model
    ar_params = [k for k in model.state_dict() if "final_ar" in k or ".ar1." in k or ".ar2." in k]
    assert not ar_params, (
        f"the loaded model carries {len(ar_params)} AttnRes parameter(s) ({ar_params[:3]}) for a "
        f"checkpoint that has none: a partially-disabled AttnRes is worse than either state")
    return True


def _selftest():
    """One positive, plus the mutant that must red."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        check_contract(td)
    print("  positive: a ckpt with no attn_res key loads, AttnRes off, no dangling params")

    # THE MUTANT: make the auto-disable inert, which is what "remove AttnRes" does if the
    # load path is deleted rather than replaced by a named refusal.
    from model import HybridLM
    orig = HybridLM.load_state_dict
    seen = {"n": 0}

    def inert(self, sd, strict=True):
        # TARGETED: keep the remap (so the failure is about AttnRes, not fused keys) and
        # skip ONLY the auto-disable branch.
        from model import remap_legacy_state_dict
        seen["n"] += 1
        return torch.nn.Module.load_state_dict(self, remap_legacy_state_dict(sd), strict=strict)

    HybridLM.load_state_dict = inert
    try:
        with tempfile.TemporaryDirectory() as td:
            try:
                check_contract(td)
            except AssertionError as e:
                print(f"  mutant: red as required (assert: {str(e)[:60]}...)")
                return 0
            except RuntimeError as e:
                # The real post-removal failure is torch's strict-load RuntimeError naming the
                # missing AttnRes keys, not an assertion -- so THAT is the signature to accept.
                msg = str(e)
                assert "ar1" in msg or "final_ar" in msg, (
                    f"the mutant red on an unrelated RuntimeError: {msg[:120]}")
                print("  mutant: red as required (strict-load RuntimeError naming the AttnRes keys)")
                return 0
            else:
                print("  mutant: GREEN -- this test does not discriminate; it would pass "
                      "after AttnRes's load path is deleted")
                return 1
    finally:
        HybridLM.load_state_dict = orig


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1:] != ["--selftest"]:
        sys.exit(f"usage: {os.path.basename(__file__)} [--selftest]")
    rc = _selftest()
    print("attnres load contract: OK" if rc == 0 else "attnres load contract: FAIL")
    sys.exit(rc)
