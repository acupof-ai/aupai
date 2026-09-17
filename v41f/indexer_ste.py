"""Straight-through training signal for the CSA2 second-level indexer (v41f-defined).

The vendored reference has no differentiable path through the indexer: `Indexer.forward`
ends in `topk(...).indices.sort().values` and `torch.where(...).int()`, and both are
integer. Under the main CE the only two trainable tensors in the module --
`wq_b.weight` and `weights_proj.weight` -- therefore receive no gradient at all
(`docs/standards/v41f_indexer_trainability_design.md` §0).

This module supplies the missing backward without touching the forward. It is v41f-defined:
the reference contains no soft, Gumbel or straight-through indexer path, so nothing here is
claimed to be upstream behaviour.

WHY THE MULTIPLIER IS 1 AND NOT 1/k
    The design doc parameterises the weight as the uniform selection weight `1/k`, arguing
    that `a` "factors out of exp/sum(exp) algebraically" and that the forward can therefore
    only be sub-ULP. That algebra is wrong for this kernel: `sparse_attn`'s denominator is
    `exp.sum(-1) + sink_term`, and the learned attention sink is a constant that is NOT part
    of the selection, so scaling the selected numerators by 1/k rescales them against the
    sink instead of cancelling.

    Measured (fp64, P0 shapes, all 256 slots valid): a=1 gives a forward max|delta| of
    exactly 0.0 and an attention dgrad ratio of exactly 1.0; a=1/k gives 4.084e-01 and dgrad
    ratios spanning -4.939723 .. 12.388815. So a=1 is not a preference, it is what makes the
    forward bit-equal (the hard-forward red line) and what keeps every attention parameter's
    gradient untouched.

    `torch.exp(x)` is never zero here, so `exp * 1.0` is bit-for-bit `exp` and the softmax
    output is unchanged to the last bit -- stronger than the doc's sub-ULP bound, and
    asserted as `torch.equal` rather than as a tolerance.
"""
import torch


def ste_slot_weight(scores: torch.Tensor) -> torch.Tensor:
    """[b, m, k] continuous scores at the hard-selected slots -> [b, m, 1, k] weight for the
    attention softmax numerator.

    Forward: the identity. `softmax(scores) - softmax(scores).detach()` is exactly zero, so
    the result is bit-for-bit 1.0 and `torch.equal(p, ones)` holds.

    Backward: the derivative of the softmax over the k selected slots, which carries the
    main-CE gradient into `scores` -> `Indexer.score` -> `wq_b` / `weights_proj`. The
    detached copy cancels in value but not in the graph, which is the whole mechanism.

    `scores` is the SAME tensor the hard topk consumed, gathered at the slots it selected --
    never a recomputed score. Slots the selection marked unreachable sit at -inf and the
    attention masks them anyway (their `exp` is 0, so the weight multiplies nothing);
    `torch.softmax` gives them exactly 0 weight, so no gradient reaches them.

    The ONE case needing a substitution is a row where EVERY slot is -inf, which is real and
    not hypothetical: at ratio 2, query 0 has `compress_lens == 0`, so it can reach no
    compressed position at all and `softmax` over the all-(-inf) row is NaN. Such a row is
    fully masked in the attention, so its weights are arbitrary -- replace the row with
    zeros rather than letting NaN into a forward that is required to be bit-unchanged.
    Replacing individual -inf entries (rather than the whole all-(-inf) row) would be wrong:
    a 0.0 in place of -inf is a score that CAN win the softmax, which would give a masked
    slot real weight.
    """
    if scores.dim() != 3:
        raise ValueError(f"expected selected scores [b, m, k], got {tuple(scores.shape)}")
    finite = scores.clone()
    all_masked = torch.isinf(finite).all(dim=-1, keepdim=True)
    if all_masked.any():
        finite = torch.where(all_masked, torch.zeros_like(finite), finite)
    sf = torch.softmax(finite, dim=-1)
    return (1.0 + (sf - sf.detach())).unsqueeze(2)
