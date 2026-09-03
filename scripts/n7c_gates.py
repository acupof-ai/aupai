#!/usr/bin/env python3
"""The gates N7 Stage C must pass on the pod BEFORE either arm trains.

Order matters and is not cosmetic. Gate A comes first because B and C are uninterpretable if
the mask_mod is not a correct causal mask to begin with: a callback that silently dropped
causality would make B fail for the right reason and C pass for the wrong one, and neither
would tell me which.

  A. causal-via-mask_mod == causal=True, BITWISE, on one real batch through the real model.
     interface.py:270 sets causal=False whenever mask_mod is not None, so the callback must
     encode causality itself. If A fails, the callback is not a causal mask and nothing else
     said about it means anything.
  A2. a deliberately WRONG mask -- fully bidirectional -- must MOVE the loss. 6e's addition, and
     it is the gate A cannot be: bitwise equality is also what a mask that never reaches the
     kernel produces.
  B. NO newly visible key may carry loss. Decided on the MASK, exactly, by enumerating every pair
     prefix allows that causal forbids and requiring each to have an ignore_index key seen by a
     prompt query. Not decided on the loss -- see "GATE B HAS BEEN WRONG FOUR TIMES" below.
  B2. P = T MUST collapse the loss, as a positive control. With every position inside the prompt
     each supervised token attends to its own label, so the loss has to fall far below causal. If
     it does not, the bidirectional half never reached the kernel and B would pass by being
     causal everywhere -- which is how a no-op mask passes a leak test.
  C. prefix with prompt_len = 0 must reproduce the causal loss EXACTLY on response tokens.
     This separates mask CONSTRUCTION from mask APPLICATION: with no prompt the prefix
     predicate reduces to `kv <= q`, so any difference is in how the mask is wired in, not in
     what it says.
  D. a prompt length on ONE document alone must move the loss, by less than the all-document
     case. B2 and C both use a uniform aux tensor -- B2 all bidirectional, C all causal -- so
     neither shows the lookup varies with the document index.
  E. EVERY document must get a nonzero prompt length. A-D all passed while the mask reached 4 of
     61 documents: a mask that is causal on the other 57 leaks nothing, collapses under P=T,
     reproduces causal at P=0 and moves under a single-document write. Coverage is invisible to
     every loss comparison, so it is a gate of its own.
  F. the mask must change at least one SUPERVISED position. A-E all passed for a layer set whose
     effect on the loss was BITWISE ZERO -- so this is the gate that decides whether the two arms
     can differ at all. See "GATE F IS THE ONE THAT MATTERS" below.

EVERY GATE RUNS PER ARM, on that arm's own layer set (eval/prefix_mask.py's PREFIX_ARMS). A gate
certified on one layer set says nothing about another, which is not a hypothetical: layer 11 alone
passed A through E while being a null by construction.

GATE F IS THE ONE THAT MATTERS, and it exists because the original design was unfalsifiable.
Stage C first masked block 11 alone -- the only MLA layer above the looped range. Gates A-E passed
and gate B printed "+0.000000 vs causal", which B accepted because it tests `prefix >= causal` and
equality satisfies it. The loss was bitwise identical: 1.6138908863067627 both ways. The cause is
architectural. prefix and causal differ ONLY for prompt queries; prompt positions are exactly the
positions the pack sets to -100; and block 11 is the last block, with only final_ar and the
per-position head above it. So a changed prompt position has no path to a supervised position.
Measured: block 11 alone moves 3660 of 16384 positions, 0 supervised. All three MLA layers move
12564 supervised positions, because layers 3 and 7 have KDA layers above them. Two arms at layer
11 would have trained to the same number, and no loss comparison could have said so -- only
counting supervised positions whose logits moved.

WHY THE LOSS AND NOT THE MASK. eval/prefix_mask.py's selftest already checks the predicate as
pure python, off-pod, thirteen ways. What that cannot check is whether the kernel APPLIES what
the predicate says -- a mask_mod that never reaches the kernel, or reaches it with permuted
arguments, or indexes aux_tensors in the wrong space, passes every predicate test and trains a
different model. So these gates read the LOSS through the real forward pass on the real
checkpoint, which is the only observable that moves when the mask is actually in effect.

GATE B HAS BEEN WRONG FOUR TIMES, all four by inferring a mask property from a loss number.
  1. B and C read 0.792 and 0.848 against a causal 1.614 and both said "THE MASK LEAKS" -- while
     the mask was correct and the AUX TENSOR was row-sized (4) against 61 documents, so most
     lookups ran out of bounds and read large enough to put every position inside the prompt.
  2. With the shape fixed, B at P = T read 0.087. Also not a leak: P = T ASKS for bidirectional
     attention everywhere. The false premise was beside it -- "with P = T the pack supervises
     nothing" -- when the loss mask lives in `labels`, which P does not touch. Every supervised
     position was still supervised and now saw its own label, so the collapse was guaranteed and
     the gate could only ever fail. P = T became B2, a positive control that REQUIRES the collapse.
  3. At the real lengths on layer 11 alone, B read +0.000000 and PASSED, because `prefix >= causal`
     is satisfied by equality -- while the loss was bitwise identical and the arms could not have
     differed at all. That is gate F.
  4. Arm p7 (layer 7 alone) read 1.6136748791, LOWER than causal by 2.16e-04, and B called it a
     leak. It is not: a lower loss is also what an IMPROVEMENT looks like, and whether
     bidirectional prompt encoding helps is the hypothesis this experiment tests. A gate that
     rejects the effect it was built to measure is not a gate.
So B no longer reads the loss. Leakage is decidable from the MASK, exactly: enumerate every pair
prefix allows that causal forbids and require that each has an ignore_index key and a query inside
the prompt. At the real lengths over 61 documents that is 292919 newly visible pairs, 0 with a
supervised key, 0 with a query outside the prompt. A proof, not a comparison. The loss is still
printed because its size and direction are the experiment's first signal, but it gates nothing.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CKPT = "ckpt_data_leg_206m_8b.pt"
PACK = "data/sft/control_sft_ours.pt"
ROWS = 4  # one small batch; these gates are about correctness, not throughput


def main():
    import torch  # noqa: PLC0415

    import model as M  # noqa: PLC0415
    from eval.prefix_mask import (  # noqa: PLC0415
        PREFIX_ARMS,
        build_mask_mods,
        doc_prompt_lengths,
        reference_mask,
    )
    from scripts.loader import load_checkpoint  # noqa: PLC0415

    if not M.HAS_FA:
        raise SystemExit(
            "REFUSING: HAS_FA is False here, so GatedMLA takes the SDPA fallback at "
            "model.py:196 and no mask_mod is ever called. Every gate below would pass "
            "vacuously while testing nothing. Run this on the pod.")

    fails = []
    arm_now = [""]

    def check(name, cond, detail=""):
        # THE ARM IS PART OF THE FAILURE NAME. Both arms run every gate, so a bare "F" in the
        # summary would not say which layer set failed -- and the two arms are exactly what this
        # run is comparing.
        print(f"  {'ok  ' if cond else 'FAIL'} {name}{'' if cond else '  <- ' + detail}")
        if not cond:
            fails.append(f"{arm_now[0]}/{name.split(':')[0]}")

    # (model, cfg) -- two values. Read from scripts/loader.py:44 after I guessed three and the
    # gate crashed on the unpack; cfg carries vocab_id rather than it being returned separately.
    mdl, _cfg = load_checkpoint(CKPT, dtype=torch.bfloat16)
    mdl = mdl.cuda().eval()
    pack = torch.load(PACK, map_location="cpu", weights_only=False)
    # THE PACK STORES 4097 COLUMNS AND THE MODEL TAKES 4096: sft_math.py:156-157 feeds
    # input_ids[:, :-1] against labels[:, 1:], the next-token shift. My first version handed the
    # raw 4097 straight to the model and it died in chunk_kda with CUDA "misaligned address"
    # (model.py:125) on the PLAIN baseline forward, before any mask_mod was involved -- an odd
    # sequence length misaligns that kernel. The crash was the lucky half: without the shift the
    # labels were also off by one, so every gate number would have been computed against the
    # wrong targets and gates B and C could have "passed" on a comparison that meant nothing.
    ids = pack["input_ids"][:ROWS, :-1].long().contiguous().cuda()
    labels = pack["labels"][:ROWS, 1:].long().contiguous().cuda()
    assert ids.shape == labels.shape, (ids.shape, labels.shape)
    assert ids.shape[1] % 2 == 0, f"odd sequence length {ids.shape[1]} misaligns chunk_kda"
    causal_mod, prefix_mod = build_mask_mods()

    def loss_with(mod=None, plens=None, want_logits=False, layers=()):
        """Cross-entropy over supervised positions only, with the attention call of the blocks in
        `layers` -- and only those -- redirected through `mod`.

        THE LAYER SET IS AN ARGUMENT, not a constant, because the two arms use different sets and
        a gate that hardcodes one would certify the wrong experiment. An earlier version patched
        M.flash_attn_varlen_func, the MODULE GLOBAL, which every GatedMLA looks up at call time --
        so all three MLA layers got the mask while PREFIX_LAYER = 11 said one. Every gate number
        then described a three-layer intervention. The docstring said "patched on the INSTANCE" the
        whole time; the code did not, and the "(3 varlen calls)" in every green run was the tell.
        The patch is now scoped by identity: hooks mark the target modules, and the wrapper masks
        only a call made from inside one of them.

        RETURNS (loss, n_calls) or (loss, n_calls, logits) with want_logits. n_calls counts calls
        on the TARGET layers. The count is not diagnostics -- it is the only thing that
        distinguishes "my mask ran" from "my mask was never reached", and an earlier version could
        not tell those apart: it patched flash_attn_varlen_func while model.py:189 requires
        `cu is not None` to reach that function, so with cu=None every forward took
        flash_attn_func at :194 and all three gates compared causal against causal. Three green
        lines, zero information. No comparison of two losses can detect that; only counting can.
        """
        targets = []
        for li in layers:
            mixer = mdl.blocks[li].mixer
            if not isinstance(mixer, M.GatedMLA):
                raise SystemExit(
                    f"REFUSING: block {li}'s mixer is {type(mixer).__name__}, not GatedMLA. The "
                    "layer indices come from cfg (layers 12, attn_every 4 -> MLA at 3, 7, 11) and "
                    "this asserts them against the built model rather than trusting arithmetic.")
            targets.append(mixer)
        n_mla = sum(1 for b in mdl.blocks if isinstance(b.mixer, M.GatedMLA))
        orig = M.flash_attn_varlen_func
        aux = [plens.to(torch.int32)] if plens is not None else None
        calls = [0]
        others = [0]

        # PATCHING THE MODULE GLOBAL IS WHAT WORKS AT ALL, because model.py:191 looks the name up
        # at call time rather than binding it into the closure -- but it is global, so the wrapper
        # has to decide for itself which layer is calling. `_n7c_active` is set by hooks on the
        # target mixers below; a call with no flag set is another MLA layer and passes through
        # unmasked.
        # What `orig` holds is already wrapped in torch._dynamo.disable (model.py:60, for a
        # measured reason: flash's varlen wrapper validates shapes against a python int, so
        # dynamo's guard set never closes -- 70 recompiles in 110 steps). Calling through orig
        # preserves that wrapper; rebuilding the call from the raw flash import would silently
        # drop it and reintroduce the recompiles.
        def patched(q, k, v, **kw):
            if not any(getattr(t, "_n7c_active", False) for t in targets):
                others[0] += 1
                return orig(q, k, v, **kw)
            calls[0] += 1
            if mod is None:
                return orig(q, k, v, **kw)
            kw.pop("causal", None)  # mask_mod replaces it (interface.py:270)
            return orig(q, k, v, mask_mod=mod, aux_tensors=aux, **kw)

        # THE FLAGS ARE SET BY HOOKS ON THE TARGET MODULES, so a flag is true only while that
        # module's own forward is running. model.py is frozen, so nothing there can be changed to
        # identify the caller; a pre/post hook pair is the smallest thing that does it from
        # outside. Registered here and removed in the finally, so a raised exception cannot leave
        # the flags or the hooks behind for the next gate.
        hooks = []
        for t in targets:
            hooks.append(t.register_forward_pre_hook(
                lambda m, _i: setattr(m, "_n7c_active", True)))
            hooks.append(t.register_forward_hook(
                lambda m, _i, _o: setattr(m, "_n7c_active", False)))
        M.flash_attn_varlen_func = patched
        try:
            # bf16 UNDER AUTOCAST, matching eval/domain_loss.py:613/228 -- it loads the checkpoint
            # with dtype=torch.bfloat16 and wraps the forward in autocast, and that is the only
            # configuration these kernels are exercised in. My first version ran fp32 with no
            # autocast and chunk_kda died with CUDA "misaligned address" on the plain baseline
            # forward. I first blamed the pack's odd 4097 length, fixed that (it was a real second
            # bug -- the labels were unshifted), and the crash survived unchanged, which is what
            # said the length was not the cause.
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                # forward returns (logits, hidden) -- model.py:571. Taking [0] rather than
                # assuming a bare tensor; the tuple is why the first bf16 run failed on
                # .float().
                # cu BY KEYWORD. forward's signature is (idx, targets=None, cu=None, ...) at
                # model.py:533, so `mdl(ids, cu)` bound cu to TARGETS -- and with targets set the
                # method returns (hidden, None), so [0] was the hidden state. cross_entropy then
                # saw n_classes = 1024, the model dim, and CUDA asserted `t >= 0 && t < n_classes`
                # on every label above 1023. The labels were never out of range; the classes were.
                logits = mdl(ids, cu=cu)[0]
                lv = torch.nn.functional.cross_entropy(
                    logits.float().view(-1, logits.shape[-1]), labels.view(-1),
                    ignore_index=-100)
            # THE SPLIT IS ASSERTED, not assumed: one masked call per target layer and the rest
            # passed through. n_mla is read from the built model, so this fails if the model ever
            # has a different number of attention layers rather than silently masking more of them.
            if calls[0] != len(targets) or others[0] != n_mla - len(targets):
                raise SystemExit(
                    f"REFUSING: {calls[0]} masked varlen call(s) and {others[0]} passed through, "
                    f"expected {len(targets)} and {n_mla - len(targets)} for layers "
                    f"{list(layers)} of {n_mla} MLA layers. A different split means the patch is "
                    "scoped wrong and every gate number would describe a different experiment "
                    "than the arm runs.")
            return (lv.item(), calls[0], logits.float().clone()) if want_logits \
                else (lv.item(), calls[0])
        finally:
            M.flash_attn_varlen_func = orig
            for h in hooks:
                h.remove()
            for t in targets:
                t._n7c_active = False

    # cu IS REQUIRED, not optional. model.py:189 reads `if HAS_FA and cu is not None:` before
    # reaching flash_attn_varlen_func, so passing cu=None routes every forward to flash_attn_func
    # at :194 -- the function this file does NOT patch. That is how the first version of these
    # gates printed three passes while never invoking a mask. cu is built by train.py's own
    # doc_cu_seqlens so the gate exercises the same document-masked varlen path the SFT arms train
    # under, rather than a path chosen for the gate's convenience.
    from tokenizers import Tokenizer  # noqa: PLC0415

    from train import doc_cu_seqlens  # noqa: PLC0415

    # eos_id FROM THE TOKENIZER, as train.py:2102 does, not from cfg (it is not a cfg field --
    # my first version read getattr(cfg, "eos_id") and would have refused on every checkpoint)
    # and not from sft_math.py:44's hardcoded EOS_ID = 1: a literal and a lookup can drift, and
    # the wrong eos id silently changes every document boundary, which changes what the mask
    # means without changing anything visible.
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    eos = tok.token_to_id("<eos>")
    if eos is None:
        raise SystemExit("REFUSING: data/tokenizer.json has no <eos> token, so doc_cu_seqlens "
                         "cannot build document boundaries and the gate would run on a "
                         "different masking path than the arms.")
    cu = doc_cu_seqlens(ids, eos)
    print(f"  cu: {cu.numel()} document boundaries over {ids.shape[0]}x{ids.shape[1]} "
          f"(eos_id {eos})")

    for arm, layers in PREFIX_ARMS.items():
        # EVERY GATE RUNS PER ARM, on that arm's exact layer set. A gate certified on one layer set
        # says nothing about another: layer 11 alone passed A-E while being a null by construction,
        # and only gate F -- run on the arm's own layers -- can tell.
        print(f"\n== arm {arm}: prefix mask on MLA layer(s) {list(layers)}")
        arm_now[0] = arm

        def L(mod=None, plens=None, want_logits=False, _ls=layers):
            return loss_with(mod, plens, want_logits, _ls)

        base, n_base = L(None)
        # THE COUNTER IS CHECKED BEFORE ANY GATE IS READ. A zero here means the patched function was
        # never called, which makes every comparison below vacuous -- and vacuous comparisons come out
        # GREEN, which is why this is a refusal and not a warning.
        if n_base == 0:
            raise SystemExit(
                "REFUSING: the patched flash_attn_varlen_func was never called, so no gate below "
                "would be testing a mask. Every comparison would pass trivially by comparing the "
                "unmasked path against itself, exactly as it did before cu was passed.")
        print(f"  causal=True baseline loss {base:.10f}  ({n_base} masked varlen call(s) on "
              f"block(s) {list(layers)}; the other MLA layers pass through unmasked)")

        # GATE A -- bitwise. Not "close": the two paths compute the same mask, so any difference is
        # a real difference in what was computed, and a tolerance here would hide exactly the bug
        # this gate exists to find.
        a, n_a = L(causal_mod)
        check("A: causal-via-mask_mod reproduces causal=True bitwise", a == base and n_a == n_base,
              f"mask_mod {a:.10f} vs causal=True {base:.10f}, diff {a - base:.3e}, "
              f"calls {n_a} vs {n_base}")

        # GATE A2 -- THE POSITIVE CONTROL (6e's addition), and it is the gate A cannot be:
        # bitwise equality is ALSO what a mask_mod that never reaches the kernel produces, so gate A
        # alone cannot distinguish a correct causal mask from no mask at all. A deliberately WRONG
        # mask -- fully bidirectional, every position sees every position -- must move the loss. If it
        # does not, the hook is not on the path, and that conclusion holds independently of the
        # counter above.
        import cutlass.cute as cute  # noqa: PLC0415

        @cute.jit
        def all_visible(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
            return kv_idx >= 0  # deliberately wrong: no causality at all

        a2, _ = L(all_visible)
        check("A2: a fully bidirectional mask CHANGES the loss (the hook is on the path)",
              abs(a2 - base) > 1e-4,
              f"bidirectional {a2:.10f} vs causal {base:.10f}, diff {a2 - base:.3e} -- a mask that "
              "sees the future changed nothing, so the mask is not being applied")

        T = ids.shape[1]
        ndoc = cu.numel() - 1
        # ONE ENTRY PER DOCUMENT, NOT PER ROW, and the gates are written in document space for a
        # measured reason. mask_mod's batch_idx is the DOCUMENT index and q_idx/kv_idx are
        # DOCUMENT-LOCAL (both measured 2026-09-04, see eval/prefix_mask.py). My first version passed
        # a 4-element row-sized tensor against 61 documents, so 57 of them read out of bounds; the
        # garbage read large, every position landed "inside the prompt", and gates B and C reported
        # 0.792 and 0.848 against a causal 1.614 -- diagnosed as A LEAK when the mask was correct and
        # the TENSOR was the wrong shape. The two-point test that settled it: all-zero aux at length
        # 4 moved the loss by -7.65e-01, the same all-zero aux at length 61 reproduced causal
        # bitwise. A gate that reads "leak" for an out-of-bounds index is a gate that would have
        # blocked a correct mask, so the length is now derived from cu rather than from ids.shape[0].
        # GATE B -- THE LEAK TEST, AT THE REAL PROMPT LENGTHS, and this is the second time B has had
        # to be rewritten because its premise was wrong rather than its arithmetic.
        #
        # B WAS "P = T must not read below causal", and it read 0.0874 against a causal 1.6139. That
        # is not a leak, it is the DEFINITION of what P = T asks for: every position of every document
        # is inside the prompt, so attention is bidirectional everywhere, and prefix_mask's selftest
        # case 4 asserts exactly that. What made the old gate wrong was the sentence I wrote next to
        # it -- "with P = T the pack supervises nothing". The loss mask lives in `labels`, which P does
        # not touch; every supervised position was still supervised, and now attended to its own label.
        # A collapse to 0.087 was guaranteed by construction, so the gate could only ever fail.
        #
        # THE LEAK CONDITION THAT IS ACTUALLY LOAD-BEARING is at the REAL prompt lengths: the prompt
        # ends before the first supervised token by construction (doc_prompt_lengths reads the boundary
        # from the loss mask), so no supervised position may become visible to any query and the loss
        # must not fall below causal. That is the configuration the arms train in, so it is also the
        # only configuration whose leak matters.
        real = doc_prompt_lengths(labels, cu).to(ids.device)
        r, _ = L(prefix_mod, real)
        nz = int((real > 0).sum())
        # GATE B -- THE LEAK TEST, DECIDED ON THE MASK AND NOT ON THE LOSS, and this is the fourth
        # time B has had to be rewritten. The first three were premise errors; this one is a
        # category error.
        #
        # B USED TO ASSERT `prefix_loss >= causal_loss` at the real prompt lengths, reasoning that
        # leakage is free information so a leak shows up as a lower loss. Arm p7 (layer 7 alone)
        # read 1.6136748791 against a causal 1.6138908863 -- LOWER by 2.16e-04 -- and B called it a
        # leak. It is not. A loss drop is also what a genuine improvement looks like, and "does
        # bidirectional prompt encoding help" is the HYPOTHESIS THIS EXPERIMENT TESTS. A gate that
        # rejects the effect it was built to measure is not a gate.
        #
        # LEAKAGE IS DECIDABLE FROM THE MASK ALONE, exactly, with no reference to any loss: a leak
        # means some position can read a token that carries loss and is not already causally
        # visible. So enumerate the pairs prefix allows that causal forbids, and require that every
        # one of them has an ignore_index key -- a prompt token, which is not a label -- and a query
        # inside the prompt. Measured over all 61 documents at the real lengths: 292919 newly
        # visible pairs, 0 with a supervised key, 0 with a query outside the prompt, and prefix
        # never forbids a pair causal allows. That is a proof, not a comparison.
        #
        # The loss is still printed, because its SIZE and DIRECTION are the experiment's first
        # signal -- but it decides nothing here.
        leak_sup, leak_q, wider = 0, 0, 0
        _flat = labels.reshape(-1)
        _start = cu[:-1].to(torch.int64)
        _dlen = (cu[1:] - cu[:-1]).to(torch.int64)
        for _d in range(ndoc):
            _P, _L, _s = int(real[_d]), int(_dlen[_d]), int(_start[_d])
            _sup = (_flat[_s:_s + _L] != -100)
            for _q in range(_L):
                for _k in range(_L):
                    _pre = reference_mask(_q, _k, _P, prefix=True)
                    _cau = reference_mask(_q, _k, _P, prefix=False)
                    if _pre and not _cau:
                        wider += 1
                        leak_sup += int(bool(_sup[_k]))
                        leak_q += int(_q >= _P)
                    elif _cau and not _pre:
                        leak_sup += 1  # narrower than causal: the response half is broken
        check("B: no newly visible key carries loss (leak decided on the mask, not the loss)",
              leak_sup == 0 and leak_q == 0,
              f"{wider} pairs prefix allows that causal forbids; {leak_sup} have a SUPERVISED key "
              f"and {leak_q} have a query outside the prompt. Either is a position reading a "
              "label it must not see")
        print(f"       {wider} newly visible pairs over {ndoc} documents, all "
              f"ignore_index keys seen by prompt queries; loss {r:.10f} vs causal {base:.10f} "
              f"({r - base:+.3e}). The SIGN is the experiment's signal, not a gate: a drop is what "
              f"an improvement looks like, and rejecting it would reject the hypothesis.")

        # B2 -- P = T IS STILL RUN, as a POSITIVE CONTROL rather than as a leak test. With every
        # position inside the prompt each supervised token attends to its own label, so the loss MUST
        # collapse far below causal. If it does not, the bidirectional half of the predicate is not
        # reaching the kernel and B above would pass by being causal everywhere -- which is precisely
        # how a no-op mask passes a leak test.
        full = torch.full((ndoc,), T, device=ids.device)
        b, _ = L(prefix_mod, full)
        check("B2: P = T DOES collapse the loss (the bidirectional half is real)", b < base - 0.1,
              f"P=T {b:.10f} vs causal {base:.10f}, diff {b - base:.3e} -- with every position "
              "inside the prompt, every label is visible to its own query and the loss has to fall. "
              "It did not, so the bidirectional half of the mask is not in effect")

        zero = torch.zeros((ndoc,), device=ids.device)
        c, _ = L(prefix_mod, zero)
        check("C: zero prompt length reproduces causal exactly", c == base,
              f"prefix@P=0 {c:.10f} vs causal {base:.10f}, diff {c - base:.3e}")

        # GATE D -- THE PROJECTION IS LIVE PER DOCUMENT. B2 and C both hold for a mask that ignores
        # the per-document INDEX: B2 by making everything bidirectional, C by making everything
        # causal, and both use a uniform aux tensor. Neither shows the lookup varies with b. Giving
        # exactly one document a full prompt length and leaving every other at zero must move the
        # loss, and by less than the all-bidirectional case -- that is a change only a live
        # per-document read can produce.
        one = torch.zeros((ndoc,), device=ids.device)
        one[0] = T
        d, _ = L(prefix_mod, one)
        check("D: a prompt length on ONE document alone moves the loss (per-document read is live)",
              abs(d - base) > 1e-6 and abs(d - base) < abs(b - base),
              f"one-document {d:.10f} vs causal {base:.10f} (diff {d - base:.3e}); all-documents was "
              f"{b - base:+.3e}. Equal to causal means aux_tensors is not read; equal to the "
              "all-documents case means the index is ignored and every document gets the same value")

        # GATE E -- COVERAGE. Every gate above passed while the mask reached 4 of 61 documents,
        # because doc_prompt_lengths projected the ROW's single boundary and gave a prompt to each
        # row's first document only. A mask that is causal on 57 of 61 documents leaks nothing (B),
        # collapses under P=T (B2), reproduces causal at P=0 (C) and moves under a single-document
        # write (D). None of them can see the coverage, so it is a gate of its own.
        #
        # THE PACK'S STRUCTURE, measured over 32 rows / 359 documents of control_sft_ours.pt: every
        # document is exactly ONE masked-prompt/supervised-response turn -- 359 of 359 with a single
        # masked->supervised transition, none beginning supervised, none entirely masked, local prompt
        # lengths 12 to 1143 (mean 71.7). format_agentic emits one pair per assistant turn and each
        # pair ends with <eos>, so the turn boundary and the document boundary are the same boundary.
        # Every document therefore carries a prompt, and anything short of full coverage is a defect
        # in this file rather than a property of the pack.
        # GATE F -- THE INTERVENTION REACHES A SUPERVISED POSITION. This is the gate that catches what
        # B cannot, and B has now been wrong three times for the same reason: a SIGN test cannot tell
        # "no leak" from "no effect". At PREFIX_LAYER = 11 gate B printed "+0.000000 vs causal" and
        # PASSED, because `prefix >= causal` is satisfied by equality -- while the loss was bitwise
        # identical at full precision, 1.6138908863067627 both ways.
        #
        # AND THE CAUSE IS STRUCTURAL, not a wiring bug. prefix and causal differ ONLY for PROMPT
        # queries (a response query keeps kv <= q in both arms), and prompt positions are exactly the
        # positions `labels` sets to -100. Block 11 is the LAST block: above it sit final_ar and the
        # head, and the head is per-position. So a changed prompt position has no path to a supervised
        # position, and the loss cannot see the mask at all. Measured: block 11 alone changes 3660 of
        # 16384 positions with 0 supervised (max|dlogit| 0.000e+00 on supervised, 8.15 on masked);
        # all three MLA layers change 12564 supervised positions (max|dlogit| 14.67), because layers
        # 3 and 7 have KDA layers above them to carry prompt changes forward.
        #
        # So this gate counts SUPERVISED POSITIONS WHOSE LOGITS MOVED. Zero means the two arms would
        # train to identical losses -- a null by construction, decidable here rather than after two
        # 500-step runs.
        _, _, lg_causal = L(causal_mod, None, want_logits=True)
        _, _, lg_prefix = L(prefix_mod, real, want_logits=True)
        moved = (lg_prefix - lg_causal).abs().amax(dim=-1) > 0
        sup = labels != -100
        n_sup_moved = int((moved & sup).sum())
        n_msk_moved = int((moved & ~sup).sum())
        check("F: the mask changes at least one SUPERVISED position (the arms can differ)",
              n_sup_moved > 0,
              f"{n_sup_moved} supervised and {n_msk_moved} masked positions moved. Zero supervised "
              f"means the loss cannot see the mask: prefix and causal differ only for PROMPT queries, "
              f"prompt positions are ignore_index, and the last block has no layer above it to "
              f"carry the change into a supervised position. The two arms would train identically")
        print(f"       positions moved: {n_sup_moved} of {int(sup.sum())} supervised, "
              f"{n_msk_moved} masked, out of {moved.numel()}. THE COUNT IS COVERAGE, NOT "
              f"MAGNITUDE: both arms move ALL {int(sup.sum())} supervised positions, so this "
              f"number separates a null from a non-null and says nothing about size. The size is "
              f"max|dlogit| on supervised "
              f"{float((lg_prefix - lg_causal).abs().amax(dim=-1)[sup].max()):.3e}, which does "
              f"differ by arm (3.894 for layer 7 alone, 14.67 for all three).")

        check("E: every document carries a nonzero prompt length (full coverage)", nz == ndoc,
              f"{nz} of {ndoc} documents have a nonzero prompt. Each document is one "
              "prompt/response turn in this pack, so a document without one means the prompt length "
              "is being derived at the wrong granularity -- the row-level projection gave 4 of 61 "
              "and every other gate passed on it")
        print(f"       local prompt lengths: min {int(real.min())}, max {int(real.max())}, "
              f"mean {float(real.float().mean()):.1f} over {ndoc} documents")
        # WHAT THE EVAL SIDE SEES, stated beside the training number because they differ and the
        # comparison is what the exp row needs: HumanEval BPB scores one prompt+solution per task,
        # so every one of its 164 tasks has a real prompt and the intervention is in effect on
        # 164/164 there. Training sees it on ndoc/ndoc documents per batch.
        print(f"  informational: training coverage {nz}/{ndoc} documents per batch; eval coverage "
              f"164/164 HumanEval tasks (one prompt per task). Both sides see the intervention.")

    print(f"\n{'ALL GATES PASS' if not fails else 'GATES FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
