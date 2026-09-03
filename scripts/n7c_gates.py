#!/usr/bin/env python3
"""The three gates N7 Stage C must pass on the pod BEFORE either arm trains.

Order matters and is not cosmetic. Gate A comes first because B and C are uninterpretable if
the mask_mod is not a correct causal mask to begin with: a callback that silently dropped
causality would make B fail for the right reason and C pass for the wrong one, and neither
would tell me which.

  A. causal-via-mask_mod == causal=True, BITWISE, on one real batch through the real model.
     interface.py:270 sets causal=False whenever mask_mod is not None, so the callback must
     encode causality itself. If A fails, the callback is not a causal mask and nothing else
     said about it means anything.
  B. prefix with prompt_len = the FULL ROW must not read BELOW the causal loss. With P = T
     every position is inside the prompt, so attention is bidirectional everywhere -- if the
     mask leaks, this is where the loss collapses. A loss below causal here means the mask
     leaks and Stage C stops.
  C. prefix with prompt_len = 0 must reproduce the causal loss EXACTLY on response tokens.
     This separates mask CONSTRUCTION from mask APPLICATION: with no prompt the prefix
     predicate reduces to `kv <= q`, so any difference is in how the mask is wired in, not in
     what it says.

WHY THE LOSS AND NOT THE MASK. eval/prefix_mask.py's selftest already checks the predicate as
pure python, off-pod, nine ways. What that cannot check is whether the kernel APPLIES what the
predicate says -- a mask_mod that never reaches the kernel, or reaches it with permuted
arguments, passes every predicate test and trains a different model. So these gates read the
LOSS through the real forward pass on the real checkpoint, which is the only observable that
moves when the mask is actually in effect.

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
    from eval.prefix_mask import PREFIX_LAYER, build_mask_mods, prompt_lengths  # noqa: PLC0415
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
    mdl, _cfg = load_checkpoint(CKPT)
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
        a class patch would leak into any other model in the process."""
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

        # PATCHING THE MODULE GLOBAL WORKS because model.py:191 looks the name up at call time
        # rather than binding it into the closure. What `orig` holds is already wrapped in
        # torch._dynamo.disable (model.py:60, for a measured reason: flash's varlen wrapper
        # validates shapes against a python int, so dynamo's guard set never closes -- 70
        # recompiles in 110 steps). Calling through orig preserves that wrapper; rebuilding the
        # call from the raw flash import would silently drop it and reintroduce the recompiles.
        def patched(q, k, v, **kw):
            if mod is None:
                return orig(q, k, v, **kw)
            kw.pop("causal", None)  # mask_mod replaces it (interface.py:270)
            return orig(q, k, v, mask_mod=mod, aux_tensors=aux, **kw)

        M.flash_attn_varlen_func = patched
        try:
            with torch.no_grad():
                logits = mdl(ids)
                lv = torch.nn.functional.cross_entropy(
                    logits.float().view(-1, logits.shape[-1]), labels.view(-1),
                    ignore_index=-100)
            return lv.item()
        finally:
            M.flash_attn_varlen_func = orig

    base = loss_with(None)
    print(f"  causal=True baseline loss {base:.10f}")

    # GATE A -- bitwise. Not "close": the two paths compute the same mask, so any difference is
    # a real difference in what was computed, and a tolerance here would hide exactly the bug
    # this gate exists to find.
    a = loss_with(causal_mod)
    check("A: causal-via-mask_mod reproduces causal=True bitwise", a == base,
          f"mask_mod {a:.10f} vs causal=True {base:.10f}, diff {a - base:.3e}")

    T = ids.shape[1]
    full = torch.full((ids.shape[0],), T, device=ids.device)
    b = loss_with(prefix_mod, full)
    check("B: full-row prompt does not read BELOW causal (no leak)", b >= base,
          f"prefix {b:.10f} < causal {base:.10f} by {base - b:.3e} -- THE MASK LEAKS")

    zero = torch.zeros((ids.shape[0],), device=ids.device)
    c = loss_with(prefix_mod, zero)
    check("C: zero prompt length reproduces causal exactly", c == base,
          f"prefix@P=0 {c:.10f} vs causal {base:.10f}, diff {c - base:.3e}")

    real = prompt_lengths(labels)
    r = loss_with(prefix_mod, real)
    print(f"  informational: real prompt lengths {real.tolist()} -> loss {r:.6f} "
          f"({r - base:+.6f} vs causal). NOT a gate: a higher loss here is the expected "
          f"cost of bidirectional prompt attention on a causally trained model.")

    print(f"\n{'ALL GATES PASS' if not fails else 'GATES FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
