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
  B. prefix at the REAL per-document prompt lengths must not read BELOW the causal loss. The
     prompt ends before the first supervised token by construction, so no supervised position may
     become visible and the loss cannot fall. This is the configuration the arms train in, which
     is the only configuration whose leak matters.
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

WHY THE LOSS AND NOT THE MASK. eval/prefix_mask.py's selftest already checks the predicate as
pure python, off-pod, thirteen ways. What that cannot check is whether the kernel APPLIES what
the predicate says -- a mask_mod that never reaches the kernel, or reaches it with permuted
arguments, or indexes aux_tensors in the wrong space, passes every predicate test and trains a
different model. So these gates read the LOSS through the real forward pass on the real
checkpoint, which is the only observable that moves when the mask is actually in effect.

A LOW LOSS IS NOT BY ITSELF A LEAK, and B has now been wrong twice for that reason.
  First: B and C read 0.792 and 0.848 against a causal 1.614 and both said "THE MASK LEAKS" --
  while the mask was correct and the AUX TENSOR was the wrong shape, row-sized (4) against 61
  documents, so most lookups ran out of bounds and read large enough to put every position inside
  the prompt.
  Second: with the shape fixed, B at P = T read 0.087. Also not a leak -- P = T ASKS for
  bidirectional attention everywhere. The false premise was in the comment beside it, "with P = T
  the pack supervises nothing": the loss mask lives in `labels`, which P does not touch, so every
  supervised position was still supervised and now saw its own label. The collapse was guaranteed
  by construction and the gate could only ever fail.
So B now runs at the real lengths and P = T is kept as B2, a positive control that REQUIRES the
collapse. Gate D exists because the first failure looked exactly like the leak B was written to
catch.

GATE B'S DIRECTION IS THE SUBTLE ONE. A leaking mask makes the loss LOWER, not higher, because
leakage is free information. So the failure condition is `prefix_loss < causal_loss`, and a
prefix loss ABOVE causal is fine at this gate -- bidirectional prompt attention on a model
trained causally should hurt. Testing for "close to causal" would reject the healthy case and
accept nothing useful.
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
        PREFIX_LAYER,
        build_mask_mods,
        doc_prompt_lengths,
    )
    from scripts.loader import load_checkpoint  # noqa: PLC0415

    if not M.HAS_FA:
        raise SystemExit(
            "REFUSING: HAS_FA is False here, so GatedMLA takes the SDPA fallback at "
            "model.py:196 and no mask_mod is ever called. Every gate below would pass "
            "vacuously while testing nothing. Run this on the pod.")

    fails = []

    def check(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}{'' if cond else '  <- ' + detail}")
        if not cond:
            fails.append(name)

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

    def loss_with(mod=None, plens=None):
        """Cross-entropy over supervised positions only, with GatedMLA's attention call
        redirected through `mod`. Patched on the INSTANCE for the same reason patch_body is:
        a class patch would leak into any other model in the process.

        RETURNS (loss, n_calls). The call count is not diagnostics -- it is the only thing that
        distinguishes "my mask ran" from "my mask was never reached", and the first version of
        this file could not tell those apart: it patched flash_attn_varlen_func while
        model.py:189 requires `cu is not None` to reach that function, so with cu=None every
        forward took flash_attn_func at :194 and all three gates compared causal against causal.
        Three green lines, zero information. No comparison of two losses can detect that; only
        counting the calls can.
        """
        blk = mdl.blocks[PREFIX_LAYER]
        mixer = blk.mixer
        if not isinstance(mixer, M.GatedMLA):
            raise SystemExit(
                f"REFUSING: block {PREFIX_LAYER}'s mixer is {type(mixer).__name__}, not "
                "GatedMLA. The layer index came from cfg (layers 12, attn_every 4 -> MLA at "
                "3, 7, 11) and this asserts it against the built model rather than trusting "
                "the arithmetic.")
        orig = M.flash_attn_varlen_func
        aux = [plens.to(torch.int32)] if plens is not None else None
        calls = [0]

        # PATCHING THE MODULE GLOBAL WORKS because model.py:191 looks the name up at call time
        # rather than binding it into the closure. What `orig` holds is already wrapped in
        # torch._dynamo.disable (model.py:60, for a measured reason: flash's varlen wrapper
        # validates shapes against a python int, so dynamo's guard set never closes -- 70
        # recompiles in 110 steps). Calling through orig preserves that wrapper; rebuilding the
        # call from the raw flash import would silently drop it and reintroduce the recompiles.
        def patched(q, k, v, **kw):
            calls[0] += 1
            if mod is None:
                return orig(q, k, v, **kw)
            kw.pop("causal", None)  # mask_mod replaces it (interface.py:270)
            return orig(q, k, v, mask_mod=mod, aux_tensors=aux, **kw)

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
            return lv.item(), calls[0]
        finally:
            M.flash_attn_varlen_func = orig

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

    base, n_base = loss_with(None)
    # THE COUNTER IS CHECKED BEFORE ANY GATE IS READ. A zero here means the patched function was
    # never called, which makes every comparison below vacuous -- and vacuous comparisons come out
    # GREEN, which is why this is a refusal and not a warning.
    if n_base == 0:
        raise SystemExit(
            "REFUSING: the patched flash_attn_varlen_func was never called, so no gate below "
            "would be testing a mask. Every comparison would pass trivially by comparing the "
            "unmasked path against itself, exactly as it did before cu was passed.")
    print(f"  causal=True baseline loss {base:.10f}  ({n_base} varlen calls)")

    # GATE A -- bitwise. Not "close": the two paths compute the same mask, so any difference is
    # a real difference in what was computed, and a tolerance here would hide exactly the bug
    # this gate exists to find.
    a, n_a = loss_with(causal_mod)
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

    a2, _ = loss_with(all_visible)
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
    r, _ = loss_with(prefix_mod, real)
    nz = int((real > 0).sum())
    check("B: at the REAL prompt lengths the loss does not fall BELOW causal (no leak)",
          r >= base,
          f"prefix {r:.10f} < causal {base:.10f} by {base - r:.3e} -- a supervised position is "
          "visible to some query, which is a leak in the configuration the arms train in")
    print(f"       {ndoc} documents over {ids.shape[0]} rows, {nz} carry a prompt "
          f"(lengths {real[real > 0].tolist()}); loss {r:.6f} ({r - base:+.6f} vs causal). A "
          f"loss ABOVE causal is the expected cost of bidirectional prompt attention on a "
          f"causally trained model. A delta of EXACTLY zero would not be good news -- it is the "
          f"signature of a mask that is not applied.")

    # B2 -- P = T IS STILL RUN, as a POSITIVE CONTROL rather than as a leak test. With every
    # position inside the prompt each supervised token attends to its own label, so the loss MUST
    # collapse far below causal. If it does not, the bidirectional half of the predicate is not
    # reaching the kernel and B above would pass by being causal everywhere -- which is precisely
    # how a no-op mask passes a leak test.
    full = torch.full((ndoc,), T, device=ids.device)
    b, _ = loss_with(prefix_mod, full)
    check("B2: P = T DOES collapse the loss (the bidirectional half is real)", b < base - 0.1,
          f"P=T {b:.10f} vs causal {base:.10f}, diff {b - base:.3e} -- with every position "
          "inside the prompt, every label is visible to its own query and the loss has to fall. "
          "It did not, so the bidirectional half of the mask is not in effect")

    zero = torch.zeros((ndoc,), device=ids.device)
    c, _ = loss_with(prefix_mod, zero)
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
    d, _ = loss_with(prefix_mod, one)
    check("D: a prompt length on ONE document alone moves the loss (per-document read is live)",
          abs(d - base) > 1e-6 and abs(d - base) < abs(b - base),
          f"one-document {d:.10f} vs causal {base:.10f} (diff {d - base:.3e}); all-documents was "
          f"{b - base:+.3e}. Equal to causal means aux_tensors is not read; equal to the "
          "all-documents case means the index is ignored and every document gets the same value")

    # WHAT THE DOCUMENT COUNT MEANS FOR THE EXPERIMENT, printed because it bounds the result and
    # is not visible anywhere else: doc_cu_seqlens opens a document at every row start and after
    # every <eos> run, so a 4x4096 SFT batch holds 61 documents and only the FIRST document of
    # each row contains the prompt. Varlen attention never crosses a document boundary in either
    # arm, so response tokens after an internal <eos> cannot see the prompt at all, prefix or
    # causal. The intervention therefore reaches a minority of the sequence.
    print(f"  informational: prompt-carrying documents {nz}/{ndoc}; the other "
          f"{ndoc - nz} are fully causal in BOTH arms because varlen attention does not cross "
          f"document boundaries. This bounds how much of the batch Stage C can affect.")

    print(f"\n{'ALL GATES PASS' if not fails else 'GATES FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
