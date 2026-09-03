#!/usr/bin/env python3
"""Where does a packed document stop matching the same document scored alone?

THE INVARIANT UNDER TEST, 6e's and it is the right one: with cu_seqlens reaching both chunk_kda
(model.py:131) and the varlen attention (model.py:191), a document's logits inside a packed row must
equal that document scored alone, up to bf16 noise. Measured on card 4 (pod, 2026-09-03 21:5xZ) they
do not: six HumanEval tasks packed into one row differ from the same six scored one-per-row by 3-4
in max|logit| at solution positions, WITH NO MASK INVOLVED -- a plain causal control disagreed by
3.9619. Every pretraining row in this repo is packed, so if isolation is not holding, it is not
holding for the training runs either.

THIS SCRIPT ONLY NAMES THE FIRST LAYER THAT DIVERGES. It does not fix anything and does not touch
model.py. A fix chosen before the mechanism is named is the cause-named-one-site-too-narrow error.

A CANDIDATE FOUND BY READING, stated up front so the measurement can refute it rather than be
steered by it -- model.py:102-114, DeltaRecurrence.forward:

    w, K = self.short_conv.weight, self.short_conv.kernel_size[0]
    h = F.pad(x.transpose(1, 2), (K - 1, 0))          # left-pad the WHOLE ROW, once
    y = h[:, :, :T] * w[:, 0, 0].unsqueeze(-1)
    for i in range(1, K):
        y = y + h[:, :, i : i + T] * w[:, 0, i].unsqueeze(-1)

The depthwise k=4 causal convolution runs over the whole [B, T, D] row and never sees `cu`. The
first three tokens of document 2 therefore convolve with the LAST tokens of document 1, whereas the
same document scored alone convolves with the zero pad. cu is passed to chunk_kda AFTER this, so it
cannot undo it. That predicts a specific signature: divergence starting at the FIRST KDA layer, and
concentrated in the first K-1 = 3 positions of each document rather than growing with distance from
the row start. A recurrent-state leak predicts the opposite -- divergence that grows with distance.
The two are distinguishable, which is why the per-position profile is printed and not just a max.

WHAT IS ALSO CHECKED, because "the first layer differs" is not by itself the mechanism:
  - a two-document control (the invariant on the smallest packing that can violate it)
  - per-position profile within each document, first 3 positions called out separately
  - the same probe with the short_conv fed document-by-document, which is a MEASUREMENT of the
    candidate, not a fix: if that alone restores agreement at layer 0, the site is named.

USAGE
    CUDA_VISIBLE_DEVICES=4 python3 scripts/n7c_pack_isolation.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CKPT = "ckpt_n7c_p3.pt"
N_TASKS = 6


def main():
    import json  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415

    import model as M  # noqa: PLC0415
    from scripts.loader import load_checkpoint  # noqa: PLC0415
    from train import doc_cu_seqlens  # noqa: PLC0415

    if not M.HAS_FA:
        raise SystemExit("REFUSING: HAS_FA is False; the varlen path would not run.")

    mdl, _cfg = load_checkpoint(CKPT, dtype=torch.bfloat16)
    mdl = mdl.cuda().eval()
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    eos = tok.token_to_id("<eos>")

    # WHICH MIXER EACH BLOCK HAS, printed with the divergence table: "layer 3 diverges" is not
    # actionable without knowing whether layer 3 is KDA or MLA, and that is the whole question of
    # which kernel is failing to honour cu.
    kinds = ["MLA" if isinstance(b.mixer, M.GatedMLA) else "KDA" for b in mdl.blocks]
    print(f"{len(mdl.blocks)} blocks: " + " ".join(f"{i}:{k}" for i, k in enumerate(kinds)))

    data = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
    docs = []
    with open(data, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            ids = tok.encode(r["prompt"]).ids + tok.encode(r["canonical_solution"]).ids
            if ids and len(ids) < 900:
                docs.append((r["task_id"], ids))
            if len(docs) >= N_TASKS:
                break

    # CAPTURE EVERY BLOCK'S OUTPUT. Wrapping forward rather than registering a hook: a hook does not
    # fire inside a grad_ckpt recompute (memory/hooks-dont-fire-in-recompute), and while this script
    # runs under no_grad, wrapping is the form that stays correct if it is ever reused under one.
    caught = {}

    def wrap(i, b):
        inner = b.forward

        def fwd(*a, **k):
            out = inner(*a, **k)
            caught[i] = (out[0] if isinstance(out, tuple) else out).detach().float()
            return out
        b.forward = fwd
        return inner

    originals = [wrap(i, b) for i, b in enumerate(mdl.blocks)]

    def states(ids_row, cu):
        caught.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            mdl(ids_row, cu=cu)
        return {i: v[0].clone() for i, v in caught.items()}

    def pack(sel):
        """(ids, cu, spans) for the given documents packed into one row, sft_math.py's construction."""
        flat, spans = [], []
        for _tid, ids in sel:
            spans.append((len(flat), len(ids)))
            flat += [*ids, eos]
        if len(flat) % 2:  # odd length misaligns chunk_kda (model.py:125): a crash, not a warning
            flat.append(eos)
        t = torch.tensor([flat], device="cuda")
        return t, doc_cu_seqlens(t, eos), spans

    def alone(ids):
        flat = [*ids, eos]
        if len(flat) % 2:
            flat.append(eos)
        t = torch.tensor([flat], device="cuda")
        return t, doc_cu_seqlens(t, eos)

    def first_divergence(sel, label, tol=0.05):
        ids_p, cu_p, spans = pack(sel)
        sp = states(ids_p, cu_p)
        solo = [states(*alone(ids)) for _tid, ids in sel]
        print(f"\n== {label}: {len(sel)} documents, {ids_p.shape[1]} tokens, "
              f"{cu_p.numel() - 1} segments")
        print(f"  {'blk':>3s} {'kind':4s} {'maxdiff':>9s} {'first3':>9s} {'rest':>9s}   "
              f"where it lives")
        first = None
        for i in sorted(sp):
            worst = f3 = rest = 0.0
            for d, (start, n) in enumerate(spans):
                a = sp[i][start:start + n]
                b = solo[d][:n]
                diff = (a - b).abs().amax(dim=-1)  # per position
                worst = max(worst, diff.max().item())
                f3 = max(f3, diff[:3].max().item())
                if n > 3:
                    rest = max(rest, diff[3:].max().item())
            # first3 vs rest is what separates the two mechanisms: a k=4 causal conv crossing the
            # boundary can only touch the first 3 positions of a document; leaked recurrent state
            # or an attention mask that does not honour cu would show up throughout.
            where = ("boundary only (<=3 positions, k=4 conv shape)" if f3 > tol >= rest
                     else "throughout the document" if rest > tol
                     else "agrees")
            print(f"  {i:3d} {kinds[i]:4s} {worst:9.4f} {f3:9.4f} {rest:9.4f}   {where}")
            if first is None and worst > tol:
                first = (i, kinds[i], worst, f3, rest)
        if first is None:
            print(f"  ALL BLOCKS AGREE within {tol} -- isolation holds for this packing.")
        else:
            i, kind, worst, f3, rest = first
            print(f"  FIRST DIVERGENCE: block {i}, mixer {kind}, max {worst:.4f} "
                  f"(first3 {f3:.4f}, rest {rest:.4f})")
        return first

    # TWO DOCUMENTS FIRST: the smallest packing that can violate the invariant. If it already fails
    # here, nothing about six-way packing or long rows is needed to explain the 3-4 gap.
    first_divergence(docs[:2], "control, two documents")
    six = first_divergence(docs[:N_TASKS], f"the reported case, {N_TASKS} documents")

    # POSITION PROFILE at the first diverging block, printed as actual numbers rather than a verdict:
    # a k=4 conv crossing the boundary gives a spike at positions 0-2 and zero after; a recurrent
    # leak grows with distance from the row start.
    if six:
        blk = six[0]
        ids_p, cu_p, spans = pack(docs[:N_TASKS])
        sp = states(ids_p, cu_p)
        print(f"\n== per-position profile at block {blk} ({kinds[blk]}), "
              f"max|diff| by position within each document")
        for d, ((tid, ids), (start, n)) in enumerate(zip(docs[:N_TASKS], spans, strict=True)):
            solo = states(*alone(ids))
            diff = (sp[blk][start:start + n] - solo[blk][:n]).abs().amax(dim=-1)
            head = " ".join(f"{v:.3f}" for v in diff[:6].tolist())
            tail = " ".join(f"{v:.3f}" for v in diff[-3:].tolist())
            print(f"  doc {d} {tid:14s} start {start:4d}  pos0-5: {head}  ...  last3: {tail}")
        print("  A k=4 causal conv crossing the boundary can only move positions 0-2 of a document\n"
              "  (model.py:109-113 left-pads the WHOLE ROW and never sees cu). Leaked recurrent\n"
              "  state or an attention mask ignoring cu would move later positions too, and doc 0\n"
              "  should be clean either way since nothing precedes it.")

    for i, b in enumerate(mdl.blocks):
        b.forward = originals[i]
    return 0


if __name__ == "__main__":
    sys.exit(main())
