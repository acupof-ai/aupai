"""One CPU training step for v41f.

Deliberately minimal: the whole forward/backward/update the smoke test needs, not a
trainer loop (no data loader, scheduler, checkpointing, AMP wrapper, DDP). Production runs
bf16 forward with fp32 master weights (the optimizer owns the fp32 copy); on CPU the smoke
test simply passes a model cast to fp32, so this helper is dtype-agnostic.

THIS IS NOT A 30B-RUN ENTRY POINT. Step B of the assembly plan wires the DSpark draft and
its multi-token loss here so both halves of the loss are exercisable on CPU; a real run
still needs the loader/scheduler/checkpoint/AMP stages, which are later work.
"""


from .loss import dspark_targets, multi_token_cross_entropy, shifted_cross_entropy


def dspark_loss(model, main_hidden, input_ids, prefix_len, pos_weights=None, ignore_index=-100):
    """Multi-token draft loss for one prefix, teacher-forced.

    main_hidden: [b, t, dim*n_target] the target layers' attention inputs, exactly what
        `V41FModel.forward` returns as its second value (it collects them BEFORE each target
        block runs, ref :1264-1267).
    input_ids: [b, t] the same sequence the backbone was scored on.
    prefix_len: the anchor position; the draft predicts the `block_size` tokens after it.

    Runs the LAST draft stage: stage 0 owns main_proj/main_norm and seeds the main-window KV,
    and every stage then runs the same parallel forward, carrying `pre_mix` between them
    (ref forward_spec :1276-1280). The draft shares the backbone's embedding and head, so
    both are passed in rather than held -- see DSparkBlock.forward_head.
    """
    if not len(model.mtp):
        raise ValueError("dspark_loss needs n_mtp_layers >= 1; this model has no draft stages")
    draft_inputs, labels = dspark_targets(input_ids, prefix_len, model.mtp[0].block_size)
    x, pre_mix, main_kv, main_len = model.mtp[0].forward_train_embed(
        main_hidden, draft_inputs, model.embed, _identity_pre_mix
    )
    for stage in model.mtp:
        x, pre_mix = stage(x, pre_mix, main_kv, main_len)
    logits = model.mtp[-1].forward_head(x, pre_mix, model.head)
    return multi_token_cross_entropy(logits, labels, ignore_index=ignore_index)


def _identity_pre_mix(x, hc_mult):
    from .block import make_identity_pre_mix

    return make_identity_pre_mix(x, hc_mult)


def train_step(
    model,
    input_ids,
    optimizer,
    ignore_index: int = -100,
    draft_prefix_len: int | None = None,
    draft_weight: float = 0.0,
    draft_pos_weights=None,
    state=None,
):
    """forward -> CE (+ optional DSpark multi-token CE) -> backward -> AdamW(step).

    With `state` (a v41f.master.TrainState) the optimizer arg is ignored and the step runs
    the bf16-forward / fp32-master path (refresh -> forward -> backward -> cast grads to fp32
    -> step the master optimizer). state=None keeps the existing optimizer-based call.

    The backbone term is always `shifted_cross_entropy` over the whole sequence. The draft
    term is added only when `draft_prefix_len` is given AND `draft_weight` is non-zero, so
    a model with no draft stages trains exactly as before -- the OFF path is one branch
    away, not a differently-shaped call.

    Returns the detached scalar total loss -- the same contract as before the draft term
    existed, so every existing caller keeps working. The two terms are available separately
    from `last_loss_terms` when a caller wants to watch them; a tuple return would have
    changed the contract for all of them to serve one new one.
    """
    global _last_loss_terms
    if state is not None:
        state.zero_model_grads()
        state.refresh_bf16()
        logits, main_hidden = model(input_ids)
        backbone = shifted_cross_entropy(logits, input_ids, ignore_index=ignore_index)
        total = backbone
        draft = None
        if draft_prefix_len is not None and draft_weight:
            if main_hidden is None:
                raise ValueError("draft_prefix_len was given but the model returned no main_hidden")
            draft = dspark_loss(
                model,
                main_hidden,
                input_ids,
                draft_prefix_len,
                pos_weights=draft_pos_weights,
                ignore_index=ignore_index,
            )
            total = backbone + draft_weight * draft
        total.backward()
        state.collect_master_grads()
        state.optimizer.step()
        _last_loss_terms = (backbone.detach(), draft.detach() if draft is not None else None)
        return total.detach()

    optimizer.zero_grad(set_to_none=True)
    logits, main_hidden = model(input_ids)
    backbone = shifted_cross_entropy(logits, input_ids, ignore_index=ignore_index)
    total = backbone
    draft = None
    if draft_prefix_len is not None and draft_weight:
        if main_hidden is None:
            raise ValueError(
                "draft_prefix_len was given but the model returned no main_hidden: "
                "dspark_target_layer_ids is empty, so there is nothing to condition the "
                "draft on")
        draft = dspark_loss(
            model, main_hidden, input_ids, draft_prefix_len,
            pos_weights=draft_pos_weights, ignore_index=ignore_index,
        )
        total = backbone + draft_weight * draft
    total.backward()
    optimizer.step()
    _last_loss_terms = (backbone.detach(), draft.detach() if draft is not None else None)
    return total.detach()


# (backbone, draft) from the most recent train_step; the draft entry is None when no draft
# term was requested. Read through last_loss_terms() rather than imported by value: a
# `from .train import last_loss_terms` binding would freeze the value at import time and
# silently read (None, None) forever.
_last_loss_terms = (None, None)


def last_loss_terms():
    """The (backbone, draft) losses from the most recent `train_step`."""
    return _last_loss_terms
