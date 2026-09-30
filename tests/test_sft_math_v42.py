"""sft_math.py must build, optimise and constrain the architecture the checkpoint names.

Four defects, all silent under --arch v42, all measured on the tree at d6678137 (de, 2026-09-30):

  a  sft_math.py:356 built HybridLM(Cfg) unconditionally. A v42 checkpoint's state dict has
     zero key overlap with it, so the SFT path could not touch the v42 line at all. Loud, but
     only at load_state_dict, and only after the pack had been read.
  b  sft_math.py:405 called train.build_optimizers, which routes parameters by NAME pattern
     against HybridLM's names. Not one V42LM parameter matches, so every v42 weight would have
     landed in the fallback group: no Muon on the backbone matrices, no Sinkhorn on embed/head,
     no per-group weight decay. Silent -- the run steps and converges to something else.
  c  sft_math.py:34 imported the SOFTCAP constant and applied it at the CE. train._softcap()
     returns None for v42 (V4.1 has no logit softcap), so SFT squashed logits through a tanh
     the pretrain never applied. Silent.
  d  sft_math.py:75 read `getattr(model, "blocks", [])` filtered on `ffn.w13`, against
     `ck_cfg["moe_experts"]`. v41f.moe.MoE has no w13 and the v42 launch sets no --moe_experts,
     so the assertion compared 0 to 0 and passed while the real stack is 64 routed / top-8.

Each case below fails if its defect comes back. (d) is the one that needs a negative control:
an assertion that cannot fail is byte-identical to a passing one from outside, so the test
asserts BOTH that the true shape passes AND that a wrong n_routed_experts is refused.

    python3 tests/test_sft_math_v42.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VOCAB = 97
N_ROUTED = 6
N_ACT = 2
N_LAYERS = 5


def _v42_cfg_dict():
    from dataclasses import asdict, replace

    from v41f.config import v42_s24

    vc = replace(
        v42_s24(vocab_size=VOCAB),
        dim=64, n_layers=N_LAYERS, n_heads=4, head_dim=32, rope_head_dim=16, q_lora_rank=32,
        o_groups=2, o_lora_rank=16, window_size=4, compress_ratios=(0, 2, 2, 1, 1),
        kv_source_layers=(1, 3), index_source_layers=(1, 3), index_n_heads=4, index_head_dim=32,
        index_topk=3, n_routed_experts=N_ROUTED, n_activated_experts=N_ACT, moe_inter_dim=32,
        hc_mult=2,
    )
    vc.validate()
    return asdict(vc)


def _cfg(**over):
    """A live-Cfg-shaped namespace, as save_checkpoint and build_model take one."""
    from train import Cfg

    d = {k: v for k, v in vars(Cfg).items() if not k.startswith("__") and not callable(v)}
    d.update(over)
    return SimpleNamespace(**d)


def _as_v42(sft_math):
    """Put sft_math's LIVE Cfg into the state main() leaves it in after the ck['cfg'] copy for a
    v42 checkpoint, and hand back the restore. The subject of this test is sft_math's own
    build_sft_model/build_sft_optimizers, which read that Cfg -- not train.build_model, which
    has dispatched correctly all along."""
    prev = {k: getattr(sft_math.Cfg, k, None) for k in ("arch", "vocab", "v42_cfg")}
    sft_math.Cfg.arch = "v42"
    sft_math.Cfg.vocab = VOCAB
    sft_math.Cfg.v42_cfg = _v42_cfg_dict()
    return prev


def _restore(sft_math, prev):
    for k, v in prev.items():
        setattr(sft_math.Cfg, k, v)


def _v42_model_and_cfg():
    import sft_math

    prev = _as_v42(sft_math)
    try:
        return sft_math.build_sft_model("cpu"), _cfg(
            arch="v42", vocab=VOCAB, v42_cfg=dict(sft_math.Cfg.v42_cfg))
    finally:
        _restore(sft_math, prev)


def test_a_build_model_returns_v42lm():
    """(a) sft_math.build_sft_model on a v42 Cfg must build V42LM, not HybridLM."""
    model, _ = _v42_model_and_cfg()
    assert type(model).__name__ == "V42LM", f"built {type(model).__name__}"
    # The property, not the class name: a HybridLM would carry `blocks.0.mixer.qkv.weight`.
    keys = list(model.state_dict())
    assert any(k.startswith("layers.0.attn.") for k in keys), keys[:5]
    # And the hybrid arm must not have moved.
    import sft_math

    h = sft_math.build_sft_model("cpu") if getattr(sft_math.Cfg, "arch", "hybrid") == "v42" else None
    assert h is None, "sft_math.Cfg was left on arch=v42"
    print(f"  a: build_sft_model -> V42LM, {len(keys)} state-dict keys")


def test_b_v42_optimizer_groups():
    """(b) sft_math.build_sft_optimizers must produce build_v42_optimizers' three rules over ALL
    parameters.

    The defect is silent, so the assertion is on coverage, not on the call: every trainable
    parameter must be claimed by exactly one group, and the Muon/Sinkhorn groups must be
    non-empty -- which is precisely what name-pattern routing against HybridLM names loses.
    """
    import sft_math

    prev = _as_v42(sft_math)
    try:
        model = sft_math.build_sft_model("cpu")
        opts = sft_math.build_sft_optimizers(model)
    finally:
        _restore(sft_math, prev)
    cfg = _cfg(arch="v42", vocab=VOCAB, v42_cfg=_v42_cfg_dict())
    names = [o.aupai_group for o in opts]
    assert names == ["muon", "sinkhorn", "adamw"], names
    claimed = [p for o in opts for g in o.param_groups for p in g["params"]]
    n_claimed = len(claimed)
    assert len({id(p) for p in claimed}) == n_claimed, "a parameter landed in two groups"
    total = [p for p in model.parameters() if p.requires_grad]
    assert n_claimed == len(total), f"{n_claimed} of {len(total)} parameters routed"
    sizes = {o.aupai_group: sum(len(g["params"]) for g in o.param_groups) for o in opts}
    assert sizes["muon"] > 0 and sizes["sinkhorn"] > 0, sizes
    for o in opts:
        for g in o.param_groups:
            assert "initial_lr" in g and "initial_wd" in g, (
                "set_schedule reads initial_lr/initial_wd; a group without them keeps a "
                "constant LR for the whole SFT")
    print(f"  b: {n_claimed}/{len(total)} params routed, groups {sizes}")

    # The negative control: train.build_optimizers is what the defect called. Assert it does
    # NOT reproduce these groups, so this test would have been red before the fix.
    from train import build_optimizers

    wrong = build_optimizers(model, cfg)
    n_wrong = len({id(p) for o in wrong for g in o.param_groups for p in g["params"]})
    assert n_wrong != n_claimed or [getattr(o, "aupai_group", None) for o in wrong] != names, (
        "build_optimizers is indistinguishable from the v42 rules here -- defect (b) would be "
        "invisible to this test")
    print(f"  b: control -- build_optimizers claims {n_wrong} params in "
          f"{[getattr(o, 'aupai_group', type(o).__name__) for o in wrong]}")


def test_c_softcap_is_none_for_v42():
    """(c) the CE softcap under v42 is None, not the SOFTCAP constant."""
    import train
    from train import SOFTCAP, _softcap

    prev = train.Cfg.arch
    try:
        train.Cfg.arch = "v42"
        got = _softcap()
        assert got is None, f"_softcap() returned {got!r} under arch=v42"
        train.Cfg.arch = "hybrid"
        assert _softcap() == SOFTCAP, "the hybrid arm lost its softcap"
    finally:
        train.Cfg.arch = prev
    # sft_math must call the function; importing the constant is the defect.
    src = (ROOT / "sft_math.py").read_text()
    assert "SOFTCAP" not in src.replace("SOFTCAP constant", ""), (
        "sft_math.py names SOFTCAP again -- under v42 that adds a logit squash the pretrain "
        "never applied")
    assert "_softcap()" in src
    print(f"  c: _softcap() None under v42, {SOFTCAP} under hybrid; sft_math calls the function")


def test_d_moe_assertion_is_not_vacuous():
    """(d) the MoE assertion must read the v42 shape and actually refuse a wrong one."""
    import sft_math

    model, cfg = _v42_model_and_cfg()
    ck_cfg = {"arch": "v42", "v42_cfg": cfg.v42_cfg}
    sft_math.Cfg.v42_cfg = cfg.v42_cfg
    prev_arch = sft_math.Cfg.arch
    try:
        sft_math.Cfg.arch = "v42"
        got = sft_math.assert_moe_matches_ckpt(model, ck_cfg)
        assert got == N_ROUTED, f"reported {got} routed experts, not {N_ROUTED}"
        assert got != 0, "0 is the vacuous answer the old code returned for every v42 model"

        # NEGATIVE CONTROL, the point of this case: a checkpoint claiming a different expert
        # count must be refused. Without this the assertion could pass by not checking.
        for field, bad in (("n_routed_experts", N_ROUTED + 1), ("n_activated_experts", N_ACT + 1)):
            wrong = dict(cfg.v42_cfg)
            wrong[field] = bad
            sft_math.Cfg.v42_cfg = wrong
            try:
                sft_math.assert_moe_matches_ckpt(model, {"arch": "v42", "v42_cfg": wrong})
            except SystemExit as e:
                assert str(bad) in str(e), str(e)
            else:
                raise AssertionError(f"a checkpoint claiming {field}={bad} was accepted")
        # And the old, vacuous basis must no longer be what is read: moe_experts stays 0.
        assert int(getattr(sft_math.Cfg, "moe_experts", 0) or 0) == 0, (
            "Cfg.moe_experts is non-zero here, so this case no longer proves the v42 shape is "
            "read from v42_cfg rather than from the hybrid field")
    finally:
        sft_math.Cfg.arch = prev_arch
        sft_math.Cfg.v42_cfg = None
    print(f"  d: {N_ROUTED} routed / top-{N_ACT} asserted from v42_cfg; both wrong shapes refused "
          f"while Cfg.moe_experts is 0")


def test_e_aux_loss_is_real_and_added():
    """(e) the v42 auxiliary term exists and sft_math adds it.

    V42LM.aux_loss() carries the MoE balance loss AND indexer_loss, the KL that is the indexer's
    ONLY gradient (its inputs are detached). Dropping it trains nothing in the indexer while the
    CE falls normally, so the first assertion is that the term is non-None and non-zero -- a
    source grep alone would pass against a term that was always None.
    """

    model, _ = _v42_model_and_cfg()
    model = model.float()
    model.train()
    ids = torch.randint(2, VOCAB, (1, 16))
    model(ids, targets=ids)
    aux = model.aux_loss()
    assert aux is not None, "V42LM.aux_loss() is None after a training forward"
    aux_v = float(aux.detach().abs())
    assert torch.isfinite(aux.detach()) and aux_v > 0, f"aux_loss is {aux!r}"
    src = (ROOT / "sft_math.py").read_text()
    assert "raw_model.aux_loss()" in src, (
        "sft_math.py does not add raw_model.aux_loss(): under v42 the indexer receives no "
        "gradient at all for the whole SFT")
    print(f"  e: aux_loss {aux_v:.6g} (non-zero), sft_math adds it")


def test_f_dynamo_optimize_ddp_off_under_v42():
    """(f) the v42 DDPOptimizer workaround is migrated, with the same env name train.py uses."""
    src = (ROOT / "sft_math.py").read_text()
    assert "optimize_ddp" in src and "DYNAMO_OPTIMIZE_DDP" in src, (
        "sft_math.py does not set optimize_ddp: on v42 the split compiled graph returned a "
        "hidden with no grad_fn and the first backward raised")
    # The defaults must agree between the two files, or one launch trains and the other does not.
    for f in ("train.py", "sft_math.py"):
        t = (ROOT / f).read_text()
        assert '_ddp_opt_default = "0" if' in t and '"v42"' in t, f
    print("  f: optimize_ddp default 0 under v42 in both train.py and sft_math.py, "
          "same DYNAMO_OPTIMIZE_DDP override")


def test_g_refuses_inert_flags_under_v42():
    """(g) --fp32_master / --stochastic_round must REFUSE under v42, not run silently.

    build_v42_optimizers takes no master map, so master.push() would write unstepped fp32 copies
    back over the weights the optimizer just moved; V42Muon never reads stochastic_round. Both
    are "supported and silently off" without this, which is the same log line as supported.
    """
    import sft_math

    ns = SimpleNamespace(fp32_master=False, stochastic_round=False, loop=None, prefix=None)
    prev = sft_math.Cfg.arch
    try:
        sft_math.Cfg.arch = "v42"
        sft_math.refuse_v42_unsupported(ns)  # the clean case must NOT raise
        for field in ("fp32_master", "stochastic_round"):
            bad = SimpleNamespace(**{**vars(ns), field: True})
            try:
                sft_math.refuse_v42_unsupported(bad)
            except SystemExit as e:
                assert field in str(e), str(e)
            else:
                raise AssertionError(f"--{field} was accepted under v42")
        # And the hybrid arm must still allow both, or this refusal has broken the CED SFT.
        sft_math.Cfg.arch = "hybrid"
        sft_math.refuse_v42_unsupported(SimpleNamespace(
            fp32_master=True, stochastic_round=False, loop=None, prefix=None))
    finally:
        sft_math.Cfg.arch = prev
    print("  g: --fp32_master and --stochastic_round refused under v42, allowed under hybrid")


def test_h_expert_bias_actually_moves():
    """(h) the expert-bias balancer's policy is named AND has a caller.

    MoE.update_bias had no caller in sft_math.py, so expert_bias stayed identically zero for the
    whole SFT -- a different balancer than the pretrain ran. The assertion is that the bias MOVES
    under an unbalanced count and re-centres, which a call that ran against a zero count would
    not show.
    """

    model, _ = _v42_model_and_cfg()
    ffns = [b.ffn for b in model.layers]
    assert ffns, "no MoE layers resolved"
    f = ffns[0]
    assert float(f.gamma) > 0, (
        f"gamma is {f.gamma}: V42LM.__init__ did not set it, so update_bias would be a no-op "
        "even with a caller")
    before = f.gate.bias.detach().clone()
    counts = torch.zeros(N_ROUTED)
    counts[0] = 100.0  # one expert grossly overloaded
    f.update_bias(counts)
    after = f.gate.bias.detach()
    assert not torch.equal(before, after), "update_bias did not move the bias"
    assert after[0] < before[0], "the overloaded expert's bias did not go down"
    assert abs(float(after.mean())) < 1e-5, f"bias not re-centred, mean {float(after.mean())}"
    src = (ROOT / "sft_math.py").read_text()
    assert "update_bias(" in src, "sft_math.py has no update_bias caller"
    print(f"  h: gamma {float(f.gamma)}, bias[0] {float(before[0]):.4g} -> {float(after[0]):.4g}, "
          f"re-centred; sft_math calls update_bias")


TESTS = [
    test_a_build_model_returns_v42lm,
    test_b_v42_optimizer_groups,
    test_c_softcap_is_none_for_v42,
    test_d_moe_assertion_is_not_vacuous,
    test_e_aux_loss_is_real_and_added,
    test_f_dynamo_optimize_ddp_off_under_v42,
    test_g_refuses_inert_flags_under_v42,
    test_h_expert_bias_actually_moves,
]

if __name__ == "__main__":
    # --selftest is accepted and ignored: the four cases ARE the selftest. Taking the
    # conventional flag keeps this file out of the hook's SELFTEST_FLAG map, where a flag
    # argparse rejects reads as exit 2 and refuses every commit that stages the file.
    torch.manual_seed(0)
    for t in TESTS:
        t()
        print(f"ok   {t.__name__}")
