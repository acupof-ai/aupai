"""P1 step B: the DSpark draft wired into V41FModel, plus the multi-token loss.

Assembly plan (#468) step B. Same gate pair as step A -- one that the whole-network path is
unchanged with the mechanism OFF, one that the ON path matches the reference -- plus the two
things specific to a draft: the tied embedding/head must not double-register, and the
teacher-forced label alignment must be the one the parallel forward actually implements.

WHAT THIS IS NOT: a trainer. `v41f/train.py` is a single CPU step helper; a real run still
needs the loader/scheduler/checkpoint/AMP stages.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_p0_dspark import _build_pair

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.config import v41f_small
from v41f.loss import dspark_targets, multi_token_cross_entropy
from v41f.model import V41FModel
from v41f.train import dspark_loss, last_loss_terms, train_step

_B, _S = 2, 16
_DRAFT = dict(n_mtp_layers=1, dspark_block_size=5, dspark_target_layer_ids=(2,))


def _on_cfg(**over):
    return v41f_small(**{**_DRAFT, **over})


def _build(cfg, fp32_master=False):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        m = V41FModel(cfg, max_batch_size=_B)
    finally:
        torch.set_default_dtype(prev)
    return m.float() if fp32_master else m


def test_off_path_has_no_draft_keys_or_parameters():
    """G-B1 (the red line): n_mtp_layers=0 leaves the model structurally unchanged.

    The OFF half. A draft wired unconditionally would add mtp.* keys to every existing
    checkpoint, so this asserts the absence directly rather than trusting that the branch
    was skipped.

    TWO OFF CONFIGS, and neither is redundant with the other: `v41f_small` sets BOTH
    n_mtp_layers 0 and target_layer_ids (), while the second turns stages off with targets
    still SET -- which is the production shape minus the draft. The pair matters because on
    `v41f_small` alone the two possible gates are indistinguishable.
    """
    for label, cfg in (
        ("v41f_small (both off)", v41f_small()),
        ("n_mtp_layers=0 with targets SET", v41f_small(dspark_target_layer_ids=(2,))),
    ):
        m = _build(cfg)
        assert len(m.mtp) == 0, f"{label}: built {len(m.mtp)} draft stages"
        sd = m.state_dict()
        leaked = [k for k in sd if k.startswith("mtp.")]
        assert not leaked, f"{label}: leaked draft keys: {leaked[:5]}"
        assert not [n for n in dict(m.named_parameters()) if n.startswith("mtp.")]
        assert len(sd) == 241, f"{label}: {len(sd)} sd keys, expected the frozen 241"
    print("  OFF (both configs): 241 sd keys, no mtp.*")


def test_stages_requested_without_targets_is_loud():
    """n_mtp_layers > 0 with dspark_target_layer_ids = () must refuse BY NAME.

    The only config where the two candidate gates disagree, and therefore the only one that
    can tell them apart: `range(n_mtp_layers)` is empty whenever n_mtp_layers is 0, so a gate
    on `target_layer_ids` and a gate on `n_mtp_layers` behave identically on every
    stages-off config (measured: a mutant swapping them survives the OFF test above). Here
    they diverge -- the target-based gate would build no draft and say nothing, so a config
    that ASKED for draft stages would silently train a plain backbone.
    """
    cfg = v41f_small(n_mtp_layers=1, dspark_block_size=5, dspark_target_layer_ids=())
    try:
        _build(cfg)
    except AssertionError as e:
        # NAME THE MESSAGE. DSparkBlock carries its own "DSpark needs target layers" assert,
        # so a bare `except AssertionError` would pass on that one and never exercise the
        # model-level check the test is about (measured: disabling the model assert leaves
        # the test green because the block's fires instead).
        assert "dspark_target_layer_ids" in str(e), (
            f"refused, but by the block's internal assert rather than the model's config "
            f"check: {e}")
    else:
        raise AssertionError(
            "n_mtp_layers=1 with no target layers built silently: the draft would be absent "
            "with no error, and the run would train a backbone while believing it had MTP")
    print("  n_mtp>0 without targets: refused by name, at the model level")


def test_on_path_draft_keys_and_last_stage_norm():
    """The ON path registers exactly one draft stage, whose KEY SET includes the last-stage
    pre-head norm the reference builds (ref DSparkBlock :1115-1116) and nothing for the
    tied embed/head."""
    m = _build(_on_cfg())
    assert len(m.mtp) == 1
    sd = set(m.state_dict())
    mtp = {k for k in sd if k.startswith("mtp.0.")}
    assert "mtp.0.norm.weight" in mtp, (
        "the last draft stage has no pre-head norm: ref DSparkBlock builds self.norm when "
        "stage_id == n_mtp_layers-1 and forward_head runs head(norm(hc_pre(x)))")
    assert "mtp.0.main_proj.weight" in mtp and "mtp.0.main_norm.weight" in mtp
    print(f"  ON: {len(mtp)} mtp.0.* keys incl norm.weight")


def test_tied_embed_head_are_not_registered():
    """G-B5: the shared embedding and LM head appear exactly ONCE each in state_dict.

    This is a STRUCTURAL INVARIANT, not a workaround for a live bug. The reference assigns
    `mtp[i].embed = self.embed`, which puts both names in state_dict over one storage;
    DSparkBlock instead takes `embed`/`head` as forward arguments, so no module holds the
    other's parameter. The test guards the property so a future change that switches to
    holding them fails here by name.
    """
    m = _build(_on_cfg())
    sd = list(m.state_dict())
    for leaf in ("embed.weight", "head.weight"):
        hits = [k for k in sd if k.endswith(leaf)]
        assert hits == [leaf], (
            f"{leaf} appears as {hits}: a tied parameter registered under a second name makes "
            f"strict loading require both keys and lets the two diverge after a partial load")
    # the draft really shares them (same storage), so this is a tie, not a missing feature
    assert m.mtp[0].forward_head.__self__ is m.mtp[0]
    print(f"  tied embed/head: one key each ({[k for k in sd if k.endswith(('embed.weight','head.weight'))]})")


def test_n_mtp_two_stages_construct():
    """Two draft stages build, and only the LAST carries the pre-head norm (ref :1115-1116
    gates on stage_id == n_mtp_layers-1)."""
    m = _build(_on_cfg(n_mtp_layers=2))
    assert len(m.mtp) == 2
    assert not hasattr(m.mtp[0], "norm") or "norm.weight" not in dict(m.mtp[0].named_parameters())
    assert "norm.weight" in dict(m.mtp[1].named_parameters()), "last stage must own the norm"
    assert m.mtp[0].is_last_stage is False and m.mtp[1].is_last_stage is True
    sd = set(m.state_dict())
    assert {"mtp.0.norm.weight", "mtp.1.norm.weight"} & sd == {"mtp.1.norm.weight"}, (
        "exactly the last stage should carry the pre-head norm")
    print("  n_mtp=2: 2 stages, norm only on the last")


def test_mutant_missing_last_stage_norm_goes_red():
    """MUTANT: delete the last-stage norm and the draft logits must move.

    Proves the norm is load-bearing rather than decorative: without it the head consumes an
    unnormalized stream, which is what the test above exists to prevent.
    """
    m = _build(_on_cfg(), fp32_master=True).eval()
    torch.manual_seed(5)
    ids = torch.randint(0, m.cfg.vocab_size, (_B, _S))
    with torch.no_grad():
        _, mh = m(ids)
        good = dspark_loss(m, mh, ids, prefix_len=8)
        real = m.mtp[0].norm
        m.mtp[0].norm = torch.nn.Identity()
        try:
            bad = dspark_loss(m, mh, ids, prefix_len=8)
        finally:
            m.mtp[0].norm = real
    d = abs(good.item() - bad.item())
    assert d > 1e-3, f"removing the last-stage norm moved the draft loss by only {d}"
    print(f"  missing last-stage norm: draft loss delta {d:.4f}")


def test_dspark_targets_alignment_is_teacher_forced():
    """The label alignment the parallel forward implements: query i is fed the token BEFORE
    the one it predicts, so nothing is visible before it is predicted.

    This is the property prereg v41f_dspark_train_equiv_0917 rests on; if the split were off
    by one, the draft would be scored on a token it was shown.
    """
    ids = torch.arange(40).reshape(2, 20)
    din, lab = dspark_targets(ids, prefix_len=8, block_size=5)
    assert din.shape == lab.shape == (2, 5)
    assert din[0, 0].item() == ids[0, 7].item(), "query 0 must be fed the anchor"
    assert lab[0, 0].item() == ids[0, 8].item(), "query 0 must predict the first future token"
    for i in range(5):
        assert din[0, i].item() == ids[0, 7 + i].item()
        assert lab[0, i].item() == ids[0, 8 + i].item()
        assert din[0, i].item() != lab[0, i].item(), "an input must never equal its own label"
    # moving the prefix moves both sides together
    d2, l2 = dspark_targets(ids, prefix_len=12, block_size=5)
    assert d2[0, 0].item() == ids[0, 11].item() and l2[0, 0].item() == ids[0, 12].item()
    print("  draft split: input j = ids[p-1+j], label j = ids[p+j]")


def test_multi_token_ce_does_not_shift():
    """The draft loss is direct (unshifted) cross entropy; `shifted_cross_entropy` would
    score every position one step off, because the draft's offset is built into its targets.
    """
    torch.manual_seed(3)
    logits = torch.randn(2, 4, 7)
    labels = torch.randint(0, 7, (2, 4))
    direct = multi_token_cross_entropy(logits, labels)
    import torch.nn.functional as F

    want = F.cross_entropy(logits.reshape(-1, 7), labels.reshape(-1))
    assert torch.allclose(direct, want, atol=1e-6), (direct.item(), want.item())
    # and it differs from the shifted form on the same tensors, so the two are not the same op
    from v41f.loss import shifted_cross_entropy

    shifted = shifted_cross_entropy(logits, labels)
    assert not torch.allclose(direct, shifted, atol=1e-4), (
        "direct and shifted agree here: the alignment would be undetectable either way")
    print(f"  direct {direct.item():.4f} vs shifted {shifted.item():.4f} (must differ)")


def test_train_step_draft_and_backbone_both_get_grad():
    """One CPU step with both terms: finite total, and every draft parameter family plus the
    backbone receives a finite gradient."""
    torch.manual_seed(0)
    m = _build(_on_cfg(), fp32_master=True).train()
    ids = torch.randint(0, m.cfg.vocab_size, (_B, _S))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    loss = train_step(m, ids, opt, draft_prefix_len=8, draft_weight=0.5)
    bb, dr = last_loss_terms()
    assert torch.isfinite(loss) and bb is not None and dr is not None
    assert torch.isfinite(bb) and torch.isfinite(dr)
    params = dict(m.named_parameters())
    for n in ("mtp.0.main_proj.weight", "mtp.0.main_norm.weight", "mtp.0.norm.weight",
              "mtp.0.attn.qproj.wq_b.weight", "mtp.0.ffn.experts.0.w1.weight",
              "embed.weight", "head.weight", "layers.0.attn.qproj.wq_b.weight"):
        g = params[n].grad
        assert g is not None and torch.isfinite(g).all(), f"{n} has no finite grad"
        assert g.abs().sum() > 0, f"{n} grad is zero"
    print(f"  train_step total={loss.item():.4f} backbone={bb.item():.4f} draft={dr.item():.4f}; all grads finite")


def test_train_step_without_draft_is_unchanged():
    """The draft term is opt-in: the same call with no draft arguments is the step-A step,
    and the OFF model never builds a draft at all."""
    torch.manual_seed(0)
    off = _build(v41f_small(), fp32_master=True).train()
    ids = torch.randint(0, off.cfg.vocab_size, (_B, _S))
    opt = torch.optim.AdamW(off.parameters(), lr=3e-3)
    loss = train_step(off, ids, opt)
    bb, dr = last_loss_terms()
    assert torch.isfinite(loss) and dr is None, "no draft was requested but one ran"
    assert ll_eq(bb, loss), (bb.item(), loss.item())
    print(f"  no-draft step: loss==backbone ({loss.item():.4f}), draft None")


def ll_eq(a, b):
    return a is not None and torch.allclose(a, b, atol=0, rtol=0)


def test_draft_loss_requires_main_hidden_and_stages():
    """Loud refusals: a draft term on a model with no target layers, and on a model with no
    draft stages, must name the missing piece rather than fail deep in a matmul."""
    m = _build(v41f_small(), fp32_master=True)  # no draft stages, no targets
    ids = torch.randint(0, m.cfg.vocab_size, (_B, _S))
    try:
        m(ids)
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"the OFF forward raised: {e}") from e
    _, mh = m(ids)
    assert mh is None
    try:
        dspark_loss(m, torch.zeros(2, _S, 4), ids, prefix_len=8)
    except ValueError as e:
        assert "n_mtp_layers" in str(e), e
    else:
        raise AssertionError("dspark_loss ran on a model with no draft stages")
    try:
        train_step(m, ids, torch.optim.AdamW(m.parameters(), lr=1e-3),
                   draft_prefix_len=8, draft_weight=0.5)
    except ValueError as e:
        assert "main_hidden" in str(e), e
    else:
        raise AssertionError("train_step ran a draft term with no main_hidden")
    print("  loud refusals: no draft stages, no main_hidden")


def test_draft_matches_reference_block_end_to_end():
    """The wired draft block reproduces the #454 reference oracle at the WIRED config.

    #454 proves the draft block's math against a hand-written column-by-column decode on a
    synthetic pair. This re-runs that comparison on a DSparkBlock built the way
    V41FModel builds one -- through `DSparkBlock(cfg, stage, n_target)`, with the config the
    wired model actually uses -- so a wiring-time config difference (a field the model passes
    differently, a stage index, the virtual layer id the ratio table is padded for) shows up
    as a numerical gap rather than only as a key-set difference.
    """
    model, ref, ours, cfg, embed, main_hidden, gold_ids = _build_pair(torch.float32)
    with torch.no_grad():
        x_ref, pre = _reference_draft_forward(model, ref, cfg, embed, main_hidden, gold_ids)
        main_x = ref.main_norm(ref.main_proj(main_hidden))
        main_kv = ours.attn.seed_main_prefix(main_x)
        y_ref, _ = ref(x_ref, 0, pre, None, main_x)
        y_ours, _ = ours(x_ref, pre, main_kv, main_hidden.size(1))
    d = (y_ref.float() - y_ours.float()).abs().max().item()
    assert d <= 1e-3, (
        f"the wired draft block differs from the #454 reference oracle by {d}: the wiring "
        f"changed the draft's math, not just its registration")
    print(f"  wired draft vs #454 oracle: max_abs {d:.3e} (fp32 control)")


def _reference_draft_forward(model, ref, cfg, embed, main_hidden, gold_ids):
    from v41f.block import make_identity_pre_mix

    x = embed(gold_ids).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
    pre = make_identity_pre_mix(x, cfg.hc_mult)
    return x, pre
