#!/usr/bin/env python3
"""Do the mask_mod kernel and an explicit SDPA mask produce the SAME GRADIENTS?

6e's diagnostic, and it is the right one: a loss that starts at the twin's value and climbs
monotonically is the shape of a WRONG GRADIENT, not of a distribution shift -- a shift starts high
and falls. Every gate in scripts/n7c_gates.py reads a frozen model's forward loss, so a mask whose
forward is exact and whose backward is wrong passes all six and then diverges in training, which is
what both arms did.

THE COMPARISON. Same weights, same batch, bf16, fp8 off:
  path A  the training path -- flash_attn.cute's varlen kernel with mask_mod=prefix_mod
  path B  the reference -- torch SDPA with the SAME mask built EXPLICITLY as a dense boolean
          matrix: block-diagonal by document, causal within a document, plus bidirectional inside
          each document's prompt
Forward losses must agree to bf16 tolerance. Then backward from each and compare parameter
gradients tensor by tensor: cosine ~1 everywhere. A tensor whose cosine is not ~1 NAMES the
defect -- and if every cosine is ~1 the gradient is fine and the divergence is a real optimisation
effect, not a bug.

WHY AN EXPLICIT DENSE MASK IS THE RIGHT REFERENCE. It is built from reference_mask, the pure-python
predicate the off-pod selftest already checks 13 ways, evaluated per (q, k) pair with no kernel
involved. So the two paths share only the PREDICATE, not any of the machinery that could be wrong:
the SSA calling convention, the aux_tensors indexing, the document-local coordinates, the backward's
recovery of aux from ctx.saved_tensors.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CKPT = "ckpt_data_leg_206m_8b.pt"
PACK = "data/sft/control_sft_ours.pt"
ROWS = 2  # two rows: enough for a per-tensor cosine, small enough for a dense TxT mask per doc
ARM = "p3"


def main():
    import torch  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415

    import model as M  # noqa: PLC0415
    from eval.prefix_mask import (  # noqa: PLC0415
        PREFIX_ARMS,
        build_mask_mods,
        doc_prompt_lengths,
        reference_mask,
    )
    from scripts.loader import load_checkpoint  # noqa: PLC0415
    from train import doc_cu_seqlens  # noqa: PLC0415

    if not M.HAS_FA:
        raise SystemExit("REFUSING: HAS_FA is False, so path A would not run the kernel at all.")

    mdl, _cfg = load_checkpoint(CKPT, dtype=torch.bfloat16)
    mdl = mdl.cuda().train()  # train(), not eval(): the gradient is what is being compared
    pack = torch.load(PACK, map_location="cpu", weights_only=False)
    ids = pack["input_ids"][:ROWS, :-1].long().contiguous().cuda()
    labels = pack["labels"][:ROWS, 1:].long().contiguous().cuda()
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    cu = doc_cu_seqlens(ids, tok.token_to_id("<eos>"))
    plens = doc_prompt_lengths(labels, cu).to(ids.device)
    layers = PREFIX_ARMS[ARM]
    B, T = ids.shape
    ndoc = cu.numel() - 1
    print(f"{B}x{T}, {ndoc} documents, arm {ARM} layers {list(layers)}")

    # THE DENSE REFERENCE MASK, built from reference_mask per (q, k) inside each document and False
    # across documents. Shape [T, T] over the FLAT stream, which is what SDPA needs, and it is
    # block-diagonal by construction because varlen attention never crosses a document boundary.
    flat_T = B * T
    ref = torch.zeros((flat_T, flat_T), dtype=torch.bool)
    starts = cu[:-1].tolist()
    lens = (cu[1:] - cu[:-1]).tolist()
    for d, (s, L) in enumerate(zip(starts, lens, strict=True)):
        P = int(plens[d])
        for q in range(L):
            for k in range(L):
                if reference_mask(q, k, P, prefix=True):
                    ref[s + q, s + k] = True
    print(f"  dense reference mask: {int(ref.sum())} allowed pairs of {flat_T * flat_T}")

    _causal_mod, prefix_mod = build_mask_mods()

    def grads(path):
        """(loss, {name: grad}) for 'kernel' or 'sdpa'. Same weights, same batch, bf16, no fp8."""
        mdl.zero_grad(set_to_none=True)
        orig = M.flash_attn_varlen_func
        targets = [mdl.blocks[li].mixer for li in layers]
        depth = [0]

        def patched(q, k, v, **kw):
            if depth[0] == 0:
                return orig(q, k, v, **kw)
            if path == "kernel":
                kw.pop("causal", None)
                return orig(q, k, v, mask_mod=prefix_mod,
                            aux_tensors=[plens.to(torch.int32)], **kw)
            # SDPA WITH THE DENSE MASK. q/k/v arrive as [B*T, h, hd] on the varlen path, so they
            # are reshaped to [1, h, B*T, hd] -- one sequence, the block-diagonal mask supplying
            # every boundary the varlen kernel would have got from cu.
            qq, kk, vv = (t.unsqueeze(0).transpose(1, 2) for t in (q, k, v))
            m = ref.to(q.device).unsqueeze(0).unsqueeze(0)
            y = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, attn_mask=m)
            return y.transpose(1, 2).squeeze(0)

        def wrap(mod):
            inner = mod.forward

            def fwd(*a, **k):
                depth[0] += 1
                try:
                    return inner(*a, **k)
                finally:
                    depth[0] -= 1
            mod.forward = fwd
            return inner

        inners = [wrap(t) for t in targets]
        M.flash_attn_varlen_func = patched
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = mdl(ids, cu=cu)[0]
                lv = torch.nn.functional.cross_entropy(
                    logits.float().view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
            lv.backward()
            g = {n: p.grad.detach().float().clone()
                 for n, p in mdl.named_parameters() if p.grad is not None}
            return lv.item(), g
        finally:
            M.flash_attn_varlen_func = orig
            for t, inner in zip(targets, inners, strict=True):
                t.forward = inner

    la, ga = grads("kernel")
    lb, gb = grads("sdpa")
    print(f"\nforward loss: kernel {la:.10f}  sdpa {lb:.10f}  diff {la - lb:+.3e}")
    print("  bf16 tolerance is ~1e-2 relative; a larger gap means the two masks are not the "
          "same mask and the gradient comparison below is meaningless.")

    keys = sorted(set(ga) & set(gb))
    rows = []
    for n in keys:
        a, b = ga[n].flatten(), gb[n].flatten()
        na, nb = a.norm().item(), b.norm().item()
        cos = float(torch.dot(a, b) / (a.norm() * b.norm())) if na > 0 and nb > 0 else float("nan")
        rows.append((cos, n, na, nb))
    rows.sort()
    print(f"\nper-tensor gradient cosine over {len(rows)} tensors:")
    print(f"  min {rows[0][0]:.6f} ({rows[0][1]})")
    print(f"  max {rows[-1][0]:.6f}")
    bad = [r for r in rows if not (r[0] > 0.99)]
    print(f"  tensors with cosine <= 0.99: {len(bad)}")
    for cos, n, na, nb in bad[:12]:
        print(f"    {cos:+.6f}  {n}  |kernel| {na:.4e}  |sdpa| {nb:.4e}")
    print("\nVERDICT: " + (
        "GRADIENTS AGREE -- the backward is not the defect; the divergence is an optimisation "
        "effect, so read the --no_fp8 probe and the LR question."
        if not bad else
        f"GRADIENTS DISAGREE on {len(bad)} tensor(s) -- the names above locate the defect."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
