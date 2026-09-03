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

WHICH LAYERS, and the first answer was a null by construction. cfg says layers 12, attn_every 4,
and model.py:354 places attention where `i % attn_every == attn_every - 1`, so the MLA layers are
3, 7 and 11 (0-based). The original plan masked block 11 alone, the only MLA layer in the post-loop
range 8-11. That cannot work, and the reason is architectural rather than a bug: prefix and causal
differ ONLY for prompt queries, prompt positions are exactly the positions the pack masks to -100,
and block 11 is the last block -- above it sit final_ar and the per-position head, so a changed
prompt position has no path to any supervised position. Measured on the pod (2026-09-04): block 11
alone moves 3660 of 16384 logit positions with 0 supervised, and the loss is bitwise identical at
1.6138908863067627. Two arms would have trained to the same number.

So the arms are PREFIX_ARMS below: "p3" masks all three MLA layers (prefix-LM in the usual sense,
twin ckpt_n7c_unlooped.pt) and "p7" masks layer 7 alone inside the looped block (twin
ckpt_n7c_looped.pt, loop held constant across both arms so only the mask differs). Layer 7 works
where 11 does not because it has KDA layers above it to carry a prompt change into a supervised
position.

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

PREFIX_LAYER = 11  # kept for the gate's default; see PREFIX_ARMS for what the arms actually use

# THE TWO ARMS, and why layer 11 alone is NOT one of them. cfg is layers 12 / attn_every 4, so
# MLA sits at blocks 3, 7 and 11. A prefix mask changes attention ONLY for prompt queries -- a
# response query keeps kv <= q in both arms -- and prompt positions are exactly the positions the
# pack sets to -100. Block 11 is the last block, with only final_ar and the per-position head
# above it, so a changed prompt position has no path to a supervised position and the training
# loss cannot see the mask at all. Measured on the pod 2026-09-04: block 11 alone moves 3660 of
# 16384 positions and 0 of them are supervised (max|dlogit| 0.000e+00 on supervised, 8.15 on
# masked), and the loss is bitwise identical at 1.6138908863067627. Two arms at layer 11 would
# train to the same loss and differ only in wasted compute -- a null by construction.
#
#   "p3": all three MLA layers. Prefix-LM in the usual sense on the adopted architecture; its
#         causal twin is ckpt_n7c_unlooped.pt (same pack, seed and 500 steps), so nothing is
#         retrained for the baseline.
#   "p7": layer 7 alone, looped 4-7. The closest well-defined form of the original one-layer
#         proposal: layer 7 has KDA layers above it to carry a prompt change into a supervised
#         position. It sits INSIDE the looped block, so prefix and loop would be confounded if the
#         arms differed in the loop -- they do not. Both arms are looped, the loop is held
#         constant, and only the mask differs. Its twin is ckpt_n7c_looped.pt.
PREFIX_ARMS = {"p3": (3, 7, 11), "p7": (7,)}



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

    WHY THIS EXISTS. The pack's loss mask is per row, but the kernel's masking unit is the
    DOCUMENT: doc_cu_seqlens (train.py:669) opens a document at every row start and after every
    <eos> run, and mask_mod is called with a document index and document-local positions
    (measured 2026-09-04, see build_mask_mods). A row-sized tensor is not merely mislabelled --
    it is read out of bounds for most documents.

    EACH DOCUMENT IS EXACTLY ONE PROMPT/RESPONSE TURN. Measured over 32 rows / 359 documents of
    control_sft_ours.pt: 359 of 359 have exactly ONE masked->supervised transition, none begins
    supervised, none is entirely masked, and the per-document prompt length runs 12 to 1143
    (mean 71.7). documents-per-row and masked->supervised-transitions-per-row are the same
    distribution (min 2, max 23, mean 11.2). format_agentic emits one prompt/response pair per
    assistant turn and each pair ends with <eos>, so the document boundary and the turn boundary
    are the same boundary.

    SO THE PROMPT LENGTH IS LOCAL: the offset of the document's first supervised token within
    that document. An earlier version projected the ROW's single boundary onto documents, which
    gave a prompt to each row's first document only and left the other ~10 fully causal -- 32 of
    359 documents covered instead of 359 of 359. The gates passed on it, because a mask that is
    causal on most documents leaks nothing and still moves the loss; only counting which
    documents got a nonzero length showed it. The "4/61 documents" scope reported from the
    earlier gate run was that artifact, not a property of the pack.

    A document with no supervised token at all gets its full length, which makes it fully causal
    rather than fully bidirectional -- the safe direction, and the same rule prompt_lengths uses
    per row. It does not occur in this pack (0 of 359) but nothing guarantees it.
    """
    import torch  # noqa: PLC0415

    flat = labels.reshape(-1)
    start = cu[:-1].to(torch.int64)
    doc_len = (cu[1:] - cu[:-1]).to(torch.int64)
    assert int((start + doc_len).max()) <= flat.numel(), \
        "cu extends past the label tensor; it was built for a different batch"
    # Vectorised per-document argmax of the supervised mask, over a [ndoc, max_len] index grid
    # masked to each document's own length. A python loop is correct too, but this runs once per
    # training step and B=16 packs ~250 documents.
    idx = start.unsqueeze(1) + torch.arange(int(doc_len.max()), device=labels.device).unsqueeze(0)
    inside = idx < (start + doc_len).unsqueeze(1)
    sup = (flat[idx.clamp(max=flat.numel() - 1)] != ignore_index) & inside
    any_sup = sup.any(dim=1)
    first = torch.argmax(sup.to(torch.int32), dim=1)
    return torch.where(any_sup, first, doc_len)


def prefix_two_call(orig, q, k, v, cu, plens, **kw):
    """Prefix-LM attention as TWO calls the kernel already gets right, instead of one mask_mod.

    WHY: flash_attn_4-4.0.0b15's mask_mod has a correct FORWARD and a WRONG BACKWARD on SM 9.0.
    Measured (facts/efficiency.json#eff.flash_attn_cute_mask_mod_backward_wrong_sm90): a mask_mod
    bitwise-identical to causal=True in the forward disagrees with a same-mask SDPA reference on 160
    of 169 gradient tensors, norm ratio median 21.65. So training cannot go through mask_mod at all,
    whatever the predicate says. The SDPA fallback is the correct reference but materialises a dense
    (B*T)^2 mask -- 17 GiB at B=16, T=4096, before any scores.

    THE DECOMPOSITION (6e, 2026-09-04), and it is exact rather than an approximation:
      call 1  causal=True over the documents, exactly as the causal arm runs today.
      call 2  causal=False over the PROMPT SEGMENTS ONLY, cu built from the prompt lengths.
      output  call 2's rows at prompt positions, call 1's rows everywhere else.
    Check both rows against the predicate in reference_mask:
      a RESPONSE query (q >= P) takes call 1, which gives it every k <= q -- prompt keys and its
        own causal past. reference_mask's response branch is exactly `k <= q`.
      a PROMPT query (q < P) takes call 2, which gives it every prompt key k < P, in both
        directions. reference_mask's prompt branch is exactly `k < P`.
    Attention is a per-query softmax over that query's own allowed keys, so replacing a whole query
    ROW is valid -- no row's output depends on another row's. That is what makes this a rewrite of
    the same function and not an approximation of it.

    COST: one extra varlen call over the prompt tokens only. On the gate batch that is 3721 of
    16384 positions (22.7%), and the second call is quadratic in P per document rather than in T,
    so the added work is far below the 20x the SDPA fallback costs.

    Returns the [B*T, h, hd] output the single call would have returned.
    """
    import torch  # noqa: PLC0415

    y1 = orig(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, **kw)
    # THE PROMPT INDEX. A document's prompt is its first P positions, so the gathered rows are one
    # contiguous slice per document -- built here as an explicit index because the slices are not
    # contiguous with each other in the flat stream.
    starts = cu[:-1].to(torch.int64)
    pl = plens.to(torch.int64)
    if int(pl.sum()) == 0:
        return y1  # no prompt anywhere: prefix reduces to causal, and call 2 would be empty
    idx = torch.cat([torch.arange(int(s), int(s) + int(p), device=q.device)
                     for s, p in zip(starts.tolist(), pl.tolist(), strict=True) if p > 0])
    cu2 = torch.zeros(int((pl > 0).sum()) + 1, dtype=torch.int32, device=q.device)
    cu2[1:] = torch.cumsum(pl[pl > 0], 0).to(torch.int32)
    mx = int(pl.max())
    kw2 = {kk: vv for kk, vv in kw.items() if kk not in ("causal", "max_seqlen_q", "max_seqlen_k")}
    y2 = orig(q[idx], k[idx], v[idx], cu_seqlens_q=cu2, cu_seqlens_k=cu2,
              max_seqlen_q=mx, max_seqlen_k=mx, causal=False, **kw2)
    # OUT-OF-PLACE index_copy: the in-place form would mutate a tensor autograd still needs, and
    # this runs inside the training graph.
    return y1.index_copy(0, idx, y2.to(y1.dtype))


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

    # 4. FULL-ROW PROMPT IS BIDIRECTIONAL EVERYWHERE. This is the premise of the pod's gate B2,
    #    which is a POSITIVE CONTROL and not a leak test. An earlier version of this comment said
    #    "with P = T the pack supervises nothing", and that was false: the loss mask lives in
    #    `labels` and P does not touch it, so at P = T every supervised position is still
    #    supervised and now attends to its own label. The loss therefore MUST collapse, which is
    #    why the pod gate requires the collapse rather than forbidding it -- the leak test runs at
    #    the real prompt lengths instead, where the prompt ends before the first supervised token.
    full = all(reference_mask(q, k, T, prefix=True) for q in range(T) for k in range(T))
    check("full-row prompt length is bidirectional everywhere (gate B2's premise)", full)

    # 5. LEAK TEST B, as an assertion: prompt_len = 0 must be EXACTLY causal, every pair.
    same = [(q, k) for q in range(T) for k in range(T)
            if reference_mask(q, k, 0, prefix=True) != reference_mask(q, k, 0, prefix=False)]
    check("zero prompt length reproduces causal exactly", not same, f"{same[:4]}")

    # 6. THE BOUNDARY COMES FROM THE PACK'S MASK, including the row we cannot interpret.
    labels = torch.tensor([[-100, -100, 5, 6], [-100, 7, -100, 8], [-100, -100, -100, -100]])
    got = prompt_lengths(labels).tolist()
    check("prompt_lengths reads the first supervised position", got == [2, 1, 4], f"{got}")
    check("a fully masked row becomes fully causal, not fully bidirectional", got[2] == 4)

    # 7. THE PER-DOCUMENT PROMPT LENGTH IS LOCAL TO THE DOCUMENT. Two rows of T=8 with documents
    #    at flat 0, 3, 8, 12 -- so each row holds two documents, and EACH document carries its own
    #    masked prompt, which is what control_sft_ours.pt actually looks like (359 of 359
    #    documents have exactly one masked->supervised transition).
    #      doc 0: flat 0-2   labels [-100,-100,9]      -> local prompt 2
    #      doc 1: flat 3-7   labels [-100,9,9,9,9]     -> local prompt 1
    #      doc 2: flat 8-11  labels [-100,-100,-100,9] -> local prompt 3
    #      doc 3: flat 12-15 labels [-100,9,9,9]       -> local prompt 1
    #    THIS IS THE TEST THAT WOULD HAVE CAUGHT THE ROW-LEVEL PROJECTION, which gave doc 1 and
    #    doc 3 a length of 0 and left them fully causal. Every pod gate still passed on that,
    #    because a mask that is causal on most documents leaks nothing and still moves the loss --
    #    only the per-document coverage count showed it.
    labels2 = torch.tensor([[-100, -100, 9, -100, 9, 9, 9, 9],
                            [-100, -100, -100, 9, -100, 9, 9, 9]])
    cu2 = torch.tensor([0, 3, 8, 12, 16], dtype=torch.int32)
    dp = doc_prompt_lengths(labels2, cu2).tolist()
    check("doc_prompt_lengths reads each document's OWN prompt", dp == [2, 1, 3, 1], f"{dp}")
    check("a document after the row's first still gets a nonzero prompt",
          dp[1] > 0 and dp[3] > 0, f"{dp}")
    # ONE ENTRY PER DOCUMENT, never per row: length 4 here, not 2. The kernel indexes this tensor
    # by document (measured on the pod), so a row-sized tensor reads past its end.
    check("doc_prompt_lengths returns one entry per document", len(dp) == cu2.numel() - 1,
          f"{len(dp)} for {cu2.numel() - 1} documents")
    # A DOCUMENT WITH NO SUPERVISED TOKEN gets its full length -> fully causal, not fully
    # bidirectional. Does not occur in this pack (0 of 359) and is asserted anyway.
    labels3 = torch.tensor([[-100, -100, -100, -100, -100, 9, 9, 9]])
    cu3 = torch.tensor([0, 4, 8], dtype=torch.int32)
    dp3 = doc_prompt_lengths(labels3, cu3).tolist()
    check("a document with no supervised token becomes fully causal", dp3 == [4, 1], f"{dp3}")

    # 8. THE TWO-CALL DECOMPOSITION IS THE SAME MASK AS THE PREDICATE. This is the claim
    #    prefix_two_call rests on, checked as a mask identity in pure python rather than trusted
    #    from the argument: for every (q, k) in a document of length L with prompt length P, the row
    #    a query takes -- call 2 if q < P, call 1 otherwise -- must allow exactly the keys
    #    reference_mask allows. Exhaustive over several (L, P) pairs including the degenerate ends.
    #    Attention is a per-query softmax over that query's own keys, so replacing whole query ROWS
    #    is valid and this identity is the whole correctness argument.
    for L, P in ((8, 3), (8, 0), (8, 8), (5, 1), (12, 7)):
        wrong = []
        for q in range(L):
            for k in range(L):
                want = reference_mask(q, k, P, prefix=True)
                # call 2 (causal=False over the first P positions) for a prompt query; call 1
                # (causal=True over the whole document) for a response query.
                got = (k < P) if q < P else (k <= q)
                if want != got:
                    wrong.append((q, k, want, got))
        check(f"two-call decomposition == the predicate at L={L} P={P}", not wrong,
              f"{wrong[:4]}")

    # 9. THE SIGNATURE IS THE KERNEL'S, NOT THE DOCSTRING'S. Six parameters; following
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
