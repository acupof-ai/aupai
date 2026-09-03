#!/usr/bin/env python3
"""Prefix-LM attention in ONE post-loop layer, as a forward patch. N7 Stage C.

WHAT THE ARMS ARE. Both arms train the looped model (blocks 4-7 twice, eval/loop_wrapper.py).
They differ only in the attention mask at block index 11:
  causal arm  -- every query attends to itself and earlier tokens, within its document.
  prefix arm  -- PROMPT tokens attend bidirectionally among prompt tokens of their own row;
                 RESPONSE tokens attend causally over prompt+response. Response tokens are
                 never visible to a prompt token, so no label is ever readable from its own
                 position or from a later one.

WHY NOT A PLAIN NON-CAUSAL MASK. Position t would read token t+1, which IS its label, so the
loss collapses by leakage and generation is undefined -- the ShortConv padding=2 incident in
this repo. Prefix-LM is the version that is well defined, and the SFT pack's loss mask already
draws the prompt/response boundary, so the boundary is read from the data rather than invented.

THE INTERVENTION IS ONE LAYER DEEP AND FEEDS THE HEAD DIRECTLY. cfg says layers 12,
attn_every 4, and model.py:354 places attention where `i % attn_every == attn_every - 1`, so
the MLA layers are 3, 7 and 11 (0-based). In the post-loop range 8-11 there is exactly ONE,
and it is 11 -- the last block in the network. Nothing sits above it but final_ar and the head.
So a null result here is a null for ONE-LAYER prefix attention immediately below the head, and
says nothing about prefix-LM in the usual sense where every attention layer over the prompt is
bidirectional. Layer 7 would give a second one, and it is INSIDE the looped block, so using it
would confound the two interventions; ruled out deliberately (6e, 2026-09-04).

HOW THE MASK REACHES THE KERNEL, and this is the part that had to be read rather than assumed.
model.py:38's `from flash_attn import ...` FAILS on this pod; HAS_FA is True through the
`.cute` fallback at model.py:47, and flash_attn.cute's varlen entry point takes a `mask_mod`
callback. So prefix-LM runs on the SAME kernel the causal path already trains on -- not on the
SDPA fallback at model.py:196, whose own comment measures ~20x slower and which would also
materialise ~4 GiB of bf16 scores per layer at B=16 T=4096.

THREE CONSTRAINTS OF THAT PATH, each read from the installed package:
  1. interface.py:270 `_resolve_causal_local_window` returns causal=False whenever mask_mod is
     not None. mask_mod REPLACES causality; it does not intersect with it. So this file's
     mask_mod must encode the causal half ITSELF, or the prefix arm goes silently
     bidirectional everywhere -- which is exactly what the full-row leak test catches.
  2. interface.py:1286 asserts mask_mod is None for the BACKWARD on SM 12.0. Card 4 is an H20
     at compute capability 9.0, so training with a mask_mod is available here. Checked, because
     a forward-only mask would make the training arm impossible rather than slow.
  3. interface.py:655 asserts mask_mod is None on the `qv` path; model.py passes no qv.

THE CALLBACK SIGNATURE COMES FROM THE KERNEL, NOT THE DOCSTRING. flash_fwd.py:77 documents
five arguments (batch_idx, head_idx, q_idx, kv_idx, aux_tensors) and is STALE: mask.py calls it
at four sites (:232, :447, :539, :834) with SIX --
    mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors) -> Boolean
and compute_block_sparsity.py:227 passes six as well. True means KEEP the position. Following
the docstring would have dropped seqlen_info and broken the call.

BOTH ARMS GO THROUGH mask_mod, including the causal one. The alternative -- causal=True for one
arm and mask_mod for the other -- makes the arms differ in KERNEL PATH as well as in mask, so a
difference between them could be a kernel difference. The cost is one extra gate: the
causal-via-mask_mod arm must reproduce causal=True bitwise before either arm trains.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PREFIX_LAYER = 11  # confirmed from the checkpoint cfg, not assumed; see the module docstring


def build_mask_mods():
    """(causal_mod, prefix_mod). Imported lazily: cutlass is a pod-only dependency, and this
    module's selftest runs the pure-python mask logic off-pod without it."""
    import cutlass.cute as cute  # noqa: PLC0415

    @cute.jit
    def causal_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        # The causal half, written out because mask_mod REPLACES causal=True (constraint 1).
        return kv_idx <= q_idx

    @cute.jit
    def prefix_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        # EVERY ARGUMENT ARRIVES AS A SHAPE-(1,) SSA VECTOR, not a python int. mask.py:228-231
        # wraps each one with utils.scalar_to_ssa before the call, and utils.ssa_to_scalar is
        # simply `val[0]`, so a scalar is recovered by subscripting. My first version wrote
        # `aux_tensors[0][batch_idx]` and cutlass raised "Expected Coord, whose leaves are
        # integers or None, but got tensor_value<vector<1xi32> o 1>" -- a tensor cannot be
        # indexed by an SSA vector. The comparisons below work on SSA values directly; only the
        # tensor lookup needs the scalar.
        b = batch_idx[0]
        p = aux_tensors[0][b]
        #
        # aux_tensors[0][b] is this row's prompt length P, in tokens.
        #
        # A query INSIDE the prompt (q_idx < P) may see any prompt key (kv_idx < P), including
        # later ones -- that is the bidirectional half, and it is safe because prompt positions
        # carry no loss: the pack masks them to -100, so nothing there is a label to leak.
        # A query in the RESPONSE keeps kv_idx <= q_idx, so it never sees its own label or a
        # later token.
        #
        # NOTE the asymmetry: a prompt query must NOT be allowed to see response keys. Writing
        # this as `(q_idx < P) or (kv_idx <= q_idx)` would be wrong -- it would let a prompt
        # query at q_idx read every key in the row, response included, and the loss would then
        # drop by leakage through the residual stream feeding the head.
        in_prompt_q = q_idx < p
        in_prompt_k = kv_idx < p
        return (in_prompt_q and in_prompt_k) or ((not in_prompt_q) and kv_idx <= q_idx)

    return causal_mod, prefix_mod


def reference_mask(q_idx, kv_idx, prompt_len, prefix):
    """The same predicate in pure python, for the off-pod selftest and for the leak tests.

    The kernel version cannot run without a GPU, so the property tests below check THIS, and
    the bitwise gate on the pod checks that the kernel agrees with causal=True. Keeping the two
    in one file is what makes a divergence visible; the risk that they drift apart is real and
    is why the gate is a bitwise comparison against the shipped causal path rather than against
    this function."""
    if not prefix:
        return kv_idx <= q_idx
    in_q, in_k = q_idx < prompt_len, kv_idx < prompt_len
    return (in_q and in_k) or ((not in_q) and kv_idx <= q_idx)


def prompt_lengths(labels, ignore_index=-100):
    """Per-row prompt length from the SFT pack's own loss mask.

    The boundary is READ, not invented: format_agentic masks prompt and tool turns to -100 and
    supervises the assistant's text, so the first supervised position is where the response
    starts. A row with no supervised token at all gets its full length, which makes it fully
    causal rather than fully bidirectional -- the safe direction, since a row we cannot
    interpret must not be the one that leaks.
    """
    import torch  # noqa: PLC0415

    sup = labels != ignore_index
    any_sup = sup.any(dim=1)
    first = torch.argmax(sup.to(torch.int32), dim=1)
    return torch.where(any_sup, first, torch.full_like(first, labels.shape[1]))


def _selftest():
    """Property tests on the pure-python predicate, plus the two leak conditions as ASSERTIONS
    rather than as prose. No GPU, no cutlass, no pack."""
    import torch  # noqa: PLC0415

    fails = []

    def check(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}{'' if cond else '  <- ' + detail}")
        if not cond:
            fails.append(name)

    T, P = 8, 3

    # 1. NO POSITION EVER SEES A LATER TOKEN THAT CARRIES LOSS. This is the leak property
    #    stated over the mask itself: a key at kv_idx > q_idx is allowed only when BOTH ends are
    #    inside the prompt, and prompt positions are masked to -100 in the pack, so they are
    #    not labels.
    bad = [(q, k) for q in range(T) for k in range(T)
           if k > q and reference_mask(q, k, P, prefix=True) and not (q < P and k < P)]
    check("no future key is visible outside the prompt block", not bad, f"{bad[:4]}")

    # 2. A PROMPT QUERY MUST NOT SEE RESPONSE KEYS. The wrong formulation
    #    `(q_idx < P) or (kv_idx <= q_idx)` passes test 1 and fails this one, which is why both
    #    exist.
    leak = [(q, k) for q in range(P) for k in range(P, T) if reference_mask(q, k, P, prefix=True)]
    check("a prompt query cannot see response keys", not leak, f"{leak[:4]}")

    # 3. THE BIDIRECTIONAL HALF IS REAL. Without this the prefix arm could be a no-op that
    #    passes every safety test above by simply being causal -- a null result would then be
    #    meaningless. This is the positive control.
    fwd = [(q, k) for q in range(P) for k in range(P) if k > q
           and reference_mask(q, k, P, prefix=True)]
    check("prompt queries do see later prompt keys", len(fwd) == P * (P - 1) // 2,
          f"{len(fwd)} of {P * (P - 1) // 2}")

    # 4. LEAK TEST A, as an assertion: prompt_len = full row must make the mask NO WIDER than
    #    causal on any position that carries loss. With P = T every pair is inside the prompt,
    #    so the mask is fully bidirectional -- and that is precisely the configuration whose
    #    LOSS must not fall below causal on the pod, because with P = T the pack supervises
    #    nothing. Here we assert the shape of it: full-row prompt is bidirectional everywhere.
    full = all(reference_mask(q, k, T, prefix=True) for q in range(T) for k in range(T))
    check("full-row prompt length is bidirectional everywhere (leak test A's premise)", full)

    # 5. LEAK TEST B, as an assertion: prompt_len = 0 must be EXACTLY causal, every pair.
    same = [(q, k) for q in range(T) for k in range(T)
            if reference_mask(q, k, 0, prefix=True) != reference_mask(q, k, 0, prefix=False)]
    check("zero prompt length reproduces causal exactly", not same, f"{same[:4]}")

    # 6. THE BOUNDARY COMES FROM THE PACK'S MASK, including the row we cannot interpret.
    labels = torch.tensor([[-100, -100, 5, 6], [-100, 7, -100, 8], [-100, -100, -100, -100]])
    got = prompt_lengths(labels).tolist()
    check("prompt_lengths reads the first supervised position", got == [2, 1, 4], f"{got}")
    check("a fully masked row becomes fully causal, not fully bidirectional", got[2] == 4)

    # 7. THE SIGNATURE IS THE KERNEL'S, NOT THE DOCSTRING'S. Six parameters; following
    #    flash_fwd.py:77's five would drop seqlen_info and break the call at mask.py:232.
    import inspect  # noqa: PLC0415
    src = inspect.getsource(build_mask_mods)
    for name in ("causal_mod", "prefix_mod"):
        sig = src.split(f"def {name}(", 1)[1].split(")", 1)[0]
        check(f"{name} takes the kernel's six arguments",
              [a.strip() for a in sig.split(",")] ==
              ["batch_idx", "head_idx", "q_idx", "kv_idx", "seqlen_info", "aux_tensors"], sig)

    print(f"\n{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(_selftest())
    raise SystemExit(f"usage: {os.path.basename(__file__)} --selftest  "
                     "(the arms are launched through sft_math.py; this file is the mask)")
