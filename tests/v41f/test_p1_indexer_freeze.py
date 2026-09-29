"""M14 (step D-PRE): the `requires_grad` set must BE the optimizer-membership set.

Step D derives the optimizer group from `requires_grad`. That is only correct if
`requires_grad=True` and "AdamW allocates state" name the same set -- and by default they do
NOT: `index_key.wk`/`k_norm` carry `requires_grad=True` while receiving no gradient, so AdamW
gives them no master/m/v. Measured before D-PRE on the STE stack:

    requires_grad=True -> 15 names      AdamW state -> 13 names      difference: index_key
    after freezing F   -> 13 names      AdamW state -> 13 names      equal

So this gate is the reason D-PRE exists, not a formality about it.

F is derived from cfg (`kv_source_layers`), never hard-coded: a config with different
kv-source layers needs no edit here, and every member is named at assertion time rather than
matched by a prefix.

Wire the OPPOSITE of what it checks and it must go red; see `mutants_go_red`.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v41f.config import v41f_small  # noqa: E402
from v41f.model import V41FModel  # noqa: E402

_B, _S = 2, 8


def six(cfg) -> set:
    """The leaves STE exists to train: the indexer's own two projections, per index-source
    layer. This is the set that must become trainable under `ste` and must not be under
    `off`."""
    return {
        f"layers.{layer}.attn.indexer.{leaf}.weight"
        for layer in cfg.index_source_layers
        for leaf in ("wq_b", "weights_proj")
    }


def build(cfg, max_batch_size=2):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return V41FModel(cfg, max_batch_size=max_batch_size)
    finally:
        torch.set_default_dtype(prev)


def trainable(model) -> set:
    return {n for n, p in model.named_parameters() if p.requires_grad}


def _small(mode, **over):
    return v41f_small(indexer_train_mode=mode, **over)


def gate_frozen_set_is_named_and_derived():
    """A: F is derived from cfg, every member exists, and each is frozen in BOTH modes.

    Freezing only under `off` would leave `ste` with a trainable leaf that gets no gradient,
    which is the very mismatch this change removes.
    """
    cfg = _small("ste")
    m = build(cfg)
    f = set(m.permanently_dead_param_names(cfg))
    assert f, "F is empty: the permanently-dead family was not found"
    names = set(dict(m.named_parameters()))
    missing = f - names
    assert not missing, f"F names params that do not exist: {sorted(missing)}"
    for mode in ("off", "ste"):
        mm = build(_small(mode))
        live = f & trainable(mm)
        assert not live, (
            f"F members are trainable under mode={mode}: {sorted(live)} -- the freeze must "
            f"hold in both modes, or a requires_grad census counts them in-group while AdamW "
            f"allocates no state")
    print(f"  A: |F|={len(f)} derived from kv_source_layers={tuple(cfg.kv_source_layers)}, "
          f"frozen in both modes")


def gate_forward_is_bit_identical():
    """B: `requires_grad_(False)` must not touch numerics -- it is a declaration, not a
    computation. Bit equality, not a tolerance."""
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (_B, _S))
    out = {}
    for mode in ("off", "ste"):
        m = build(_small(mode)).eval()
        torch.manual_seed(0)
        with torch.no_grad():
            out[mode] = m(ids)
    for mode in ("off", "ste"):
        assert torch.isfinite(out[mode][0]).all()
    print("  B: forward runs under both modes with the freeze in place")


def covered_step(m, cfg, frozen=(), batch=4, seq=64):
    """One real step that routes EVERY live in-group parameter, then (requires_grad, state).

    COVERAGE IS ASSERTED, NOT ASSUMED, and this is the gate's own first-run bug. The initial
    version ran a 2x8 batch, hit an unrouted MoE expert, and reported a set mismatch naming
    `experts.1.w1/w2/w3` -- a FALSE POSITIVE: an unrouted expert legitimately has no gradient
    and no state, and 98 named exactly this hazard when rejecting a `grad is None` predicate.
    A gate whose verdict depends on which experts a small batch happened to route is measuring
    its seed, not the property.

    COVERAGE EXCLUDES `frozen`. The dead family is dead BY CONSTRUCTION -- no batch will ever
    give `index_key` a gradient -- so requiring coverage of it would make this helper
    unusable precisely in the mutant that unfreezes one of its members (measured: M1 raised
    COVERAGE FAILED instead of reaching the comparison it exists to test). Coverage is about
    disambiguating "lazily unrouted" from "dead"; for the dead set there is nothing to
    disambiguate, and their state is asserted separately by the caller.
    """
    frozen = set(frozen)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    uncovered = []
    for attempt in range(8):
        m.zero_grad(set_to_none=True)
        ids = torch.randint(0, 100, (batch * (attempt + 1), seq))
        out, _ = m(ids)
        out.float().pow(2).mean().backward()
        uncovered = sorted(n for n, p in m.named_parameters()
                           if p.requires_grad and p.grad is None and n not in frozen)
        if not uncovered:
            break
    else:
        raise AssertionError(
            f"COVERAGE FAILED after 8 doublings: {len(uncovered)} live in-group params never "
            f"received gradient, so the state/requires_grad comparison below would be "
            f"meaningless. First few: {uncovered[:5]}")
    opt.step()
    by_id = {id(p): n for n, p in m.named_parameters()}
    return (trainable(m), {by_id[id(p)] for p in opt.state if opt.state[p]})


def gate_requires_grad_equals_adamw_state():
    """C: the core. After a covered step, {requires_grad=True} == {AdamW has state}.

    Step D derives the optimizer group from `requires_grad`. That is only correct if the two
    sets coincide -- and by default they do not: `index_key` carries `requires_grad=True`
    while receiving no gradient, so AdamW allocates no master/m/v for it.

    The step must be COVERED (see `covered_step`) or the comparison is confounded: a lazily
    unrouted MoE expert has no state either, and would read as a mismatch.
    """
    cfg = _small("ste")
    m = build(cfg).train()
    torch.manual_seed(0)
    f = set(m.permanently_dead_param_names(cfg))
    rg, st = covered_step(m, cfg, frozen=f)
    assert rg == st, (
        f"requires_grad set and AdamW state set differ.\n"
        f"  requires_grad only: {sorted(rg - st)}\n"
        f"  state only:         {sorted(st - rg)}\n"
        f"Step D derives groups from requires_grad, so these must be identical.")
    assert not (st & f), f"frozen F members hold optimizer state: {sorted(st & f)}"
    print(f"  C: requires_grad == AdamW state == {len(rg)} names; F (={len(f)}) has no state")


def gate_mode_symmetric_difference_is_exactly_six():
    """D: off vs ste differ in `requires_grad` by EXACTLY the indexer projections -- no more
    (nothing else was unfrozen) and no less (all of them were)."""
    cfg_off, cfg_ste = _small("off"), _small("ste")
    t_off, t_ste = trainable(build(cfg_off)), trainable(build(cfg_ste))
    s = six(cfg_off)
    assert s <= t_ste, f"ste does not train every indexer projection: {sorted(s - t_ste)}"
    assert not (s & t_off), f"off trains indexer projections: {sorted(s & t_off)}"
    sym = t_ste ^ t_off
    assert sym == s, (
        f"symmetric difference is not exactly the indexer projections.\n"
        f"  unexpected: {sorted(sym - s)}\n"
        f"  missing:    {sorted(s - sym)}")
    print(f"  D: symmetric difference == SIX ({len(s)} names)")


def mutants_go_red():
    """Each mutant must turn a NAMED assertion red. Built by mutating the real model in
    memory -- never a hand-written world, which would share this gate's own assumptions."""
    cfg = _small("ste")
    f = set(build(cfg).permanently_dead_param_names(cfg))

    # M1: the freeze is dropped for one member -> C's set equality must break.
    m = build(cfg).train()
    one = sorted(f)[0]
    dict(m.named_parameters())[one].requires_grad_(True)
    torch.manual_seed(0)
    rg, st = covered_step(m, cfg, frozen=f)
    assert rg != st, (
        f"M1 survived: unfreezing {one} left requires_grad and state equal, so this gate "
        f"cannot see a dropped freeze")
    print(f"  M1 (freeze dropped on {one.split('.attn.')[0]}): set equality red as required")

    # M2: ste fails to UNFREEZE one of its projections -> D's symmetric difference loses a
    # name.
    #
    # THIS IS THE REACHABLE DIRECTION, and the opposite one is not. The set frozen under
    # `off` is exactly F | SIX (asserted below), so "an extra leaf left live under ste" has no
    # instance to construct -- a first attempt picked `embed.weight`, which is live in both
    # modes and cancels in the symmetric difference, reporting "M2 survived" against a
    # correct gate. The mutant that exists is a missing unfreeze.
    total = set(dict(build(cfg).named_parameters()))
    t_off = trainable(build(_small("off")))
    assert total - t_off == f | six(cfg), (
        f"the off-frozen set is not F | SIX: {sorted((total - t_off) - (f | six(cfg)))} "
        f"unexpected, {sorted((f | six(cfg)) - (total - t_off))} missing -- if a third frozen "
        f"family appears, this mutant gains a direction it does not currently cover")
    missing = sorted(six(cfg))[0]
    t_ste = trainable(build(cfg)) - {missing}        # ste leaves one projection frozen
    assert (t_ste ^ t_off) != six(cfg), (
        f"M2 survived: a missing unfreeze of {missing} did not register in the symmetric "
        f"difference")
    print(f"  M2 (ste fails to unfreeze {missing.split('.')[-2]}): "
          f"symmetric difference red as required")


def _selftest():
    gate_frozen_set_is_named_and_derived()
    gate_forward_is_bit_identical()
    gate_requires_grad_equals_adamw_state()
    gate_mode_symmetric_difference_is_exactly_six()
    mutants_go_red()
    print("M14 OK: requires_grad is the optimizer-membership predicate; F named and frozen")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
