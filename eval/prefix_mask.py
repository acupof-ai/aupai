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
    import cutlass  # noqa: PLC0415
    import cutlass.cute as cute  # noqa: PLC0415
    from flash_attn.cute import utils as fa_utils  # noqa: PLC0415

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
        # indexed by an SSA vector.
        #
        # batch_idx IS THE DOCUMENT INDEX, NOT THE ROW, and q_idx/kv_idx are DOCUMENT-LOCAL.
        # Both were measured on the pod (2026-09-04), not inferred, because getting either wrong
        # produces a mask that runs and is silently wrong:
        #   - aux of length 4 (one per row) made gate C read 0.848 against a causal 1.614; the
        #     same all-zero aux at length 61 (one per document) reproduces causal EXACTLY. So b
        #     ranges over documents -- 61 of them for a 4x4096 batch -- and a row-sized tensor was
        #     being read out of bounds for 57 of them. Out-of-bounds garbage reads large, every
        #     position lands "inside the prompt", and the mask goes bidirectional everywhere,
        #     which is exactly the collapse gates B and C reported.
        #   - the LAST document is flat 16355..16383. Giving it alone P=4096 moved the loss by
        #     -9.9e-04. Under flat-stream indices none of its positions are below 4096 and the
        #     loss could not have moved at all, so the indices are document-local.
        # A prompt length must therefore be supplied PER DOCUMENT; doc_prompt_lengths below maps
        # the pack's row-relative boundary onto that space.
        b = batch_idx[0]
        p = aux_tensors[0][b]
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
        #
        # THE LOGIC RUNS ON SCALARS, NOT ON SSA VECTORS, and this took three failed runs to
        # locate. A comparison of two SSA vectors yields an i1 VECTOR (vector<1xi1>), and
        # typing.py's _from_mlir_type has no entry for that type in its type_map (:1250) -- so
        # anything that has to resolve its dtype raises "Unsupported DSL type: vector<1xi1>".
        # `and`/`or`/`not` fail there, and so do `&`/`|`/`~`: the operator was never the issue,
        # the vector shape was. causal_mod above survives only because `kv_idx <= q_idx` merely
        # PRODUCES such a vector and the kernel immediately unwraps it with ssa_to_scalar.
        # The kernel's own mask code does the same thing -- block_sparsity.py:226 pulls the
        # callback's result through ssa_to_scalar and only THEN combines it with `&` on
        # scalars (:237-238), and mask.py:242 uses a plain `or` on scalars. So: subscript every
        # argument to a scalar, do the boolean algebra there, and hand back one i1 vector built
        # the way scalar_to_ssa builds it.
        q = q_idx[0]
        k = kv_idx[0]
        in_prompt_q = q < p
        in_prompt_k = k < p
        keep = (in_prompt_q and in_prompt_k) or ((not in_prompt_q) and k <= q)
        return fa_utils.scalar_to_ssa(cutlass.Boolean(keep), cutlass.Boolean)

    return causal_mod, prefix_mod


def doc_prompt_lengths(labels, cu, ignore_index=-100):
    """Per-DOCUMENT prompt length, in document-local coordinates, for aux_tensors[0].

    WHY THIS EXISTS. The pack's prompt/response boundary is per ROW, but the kernel's masking
    unit is the DOCUMENT: doc_cu_seqlens (train.py:669) opens a document at every row start and
    after every <eos> run, and mask_mod is called with a document index and document-local
    positions. On the four gate rows that is 61 documents over 4 rows -- 10 to 23 per row, with
    9 to 22 boundaries falling INSIDE the supervised span. So the boundary has to be projected
    from row coordinates into document coordinates, and a row-sized tensor is not merely
    mislabelled: it is read out of bounds for most documents.

    THE PROJECTION. Document d starts at flat position cu[d], which is row cu[d] // T at
    row-offset cu[d] % T. The row's prompt ends at row-offset `end`, so within d the prompt
    covers the first `end - (cu[d] % T)` positions, clamped to [0, len(d)]. Documents that start
    at or after the row's prompt end get 0 and are therefore fully causal, which is both correct
    and the safe direction.

    ON THE FOUR GATE ROWS THE PROMPT LIES ENTIRELY INSIDE THE FIRST DOCUMENT (66 < 70, 45 < 49,
    43 < 614, 215 < 1969), so the clamp is not exercised there. It is written anyway because
    nothing in the pack guarantees it -- an <eos> inside a prompt would split it, and then a
    formula that only handled the first document would silently drop the rest of the prompt to
    causal.

    A SEPARATE FACT THIS MADE VISIBLE, and it is not something this mask introduces: because
    varlen attention never crosses a document boundary, response tokens after an internal <eos>
    ALREADY cannot see the prompt at all, in both arms. That is what doc_mask training means
    here. It bounds how much of the sequence the prefix intervention can even reach, and belongs
    in the Stage C exp row beside the one-layer caveat.
    """
    import torch  # noqa: PLC0415

    T = labels.shape[1]
    row_end = prompt_lengths(labels, ignore_index).to(torch.int64)
    start = cu[:-1].to(torch.int64)
    doc_len = (cu[1:] - cu[:-1]).to(torch.int64)
    within = row_end[start // T] - (start % T)
    return torch.clamp(within, min=0).minimum(doc_len)


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

    # 7. THE PROJECTION FROM ROW COORDINATES TO DOCUMENT COORDINATES. This is the test that would
    #    have caught the aux-length defect off-pod: a row-sized tensor is not just mislabelled,
    #    it is read out of bounds for most documents, and the on-pod symptom (gate C reading
    #    0.848 against a causal 1.614) said "leak" rather than "wrong length". Two rows of T=8,
    #    documents at flat 0, 3, 8, 12 -- so row 0 holds two documents and row 1 holds two.
    #    Prompt ends at row-offset 5 in row 0 and 2 in row 1.
    #      doc 0: flat 0-2,   row 0 offset 0, len 3 -> prompt covers 5-0=5, clamped to len 3
    #      doc 1: flat 3-7,   row 0 offset 3, len 5 -> prompt covers 5-3=2
    #      doc 2: flat 8-11,  row 1 offset 0, len 4 -> prompt covers 2-0=2
    #      doc 3: flat 12-15, row 1 offset 4, len 4 -> prompt covers 2-4=-2, clamped to 0
    #    The clamps are the point: doc 0 exercises the upper clamp (a prompt longer than the
    #    document) and doc 3 the lower one (a document starting after the prompt ends, which must
    #    be fully causal rather than fully bidirectional).
    labels2 = torch.tensor([[-100, -100, -100, -100, -100, 9, 9, 9],
                            [-100, -100, 9, 9, 9, 9, 9, 9]])
    cu2 = torch.tensor([0, 3, 8, 12, 16], dtype=torch.int32)
    dp = doc_prompt_lengths(labels2, cu2).tolist()
    check("doc_prompt_lengths projects the row boundary onto documents", dp == [3, 2, 2, 0], f"{dp}")
    check("a prompt longer than its document is clamped to the document", dp[0] == 3, f"{dp[0]}")
    check("a document starting after the prompt is fully causal, not bidirectional",
          dp[3] == 0, f"{dp[3]}")
    # ONE ENTRY PER DOCUMENT, never per row: length 4 here, not 2. The kernel indexes this tensor
    # by document (measured on the pod), so a row-sized tensor reads past its end.
    check("doc_prompt_lengths returns one entry per document", len(dp) == cu2.numel() - 1,
          f"{len(dp)} for {cu2.numel() - 1} documents")

    # 8. THE SIGNATURE IS THE KERNEL'S, NOT THE DOCSTRING'S. Six parameters; following
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
