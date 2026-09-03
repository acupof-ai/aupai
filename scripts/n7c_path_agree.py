#!/usr/bin/env python3
"""Do the TRAINING-side and EVAL-side prefix paths agree on a HumanEval-shaped batch?

6e's second hypothesis for P3's +0.0228 penalty, and the one the prompt-length binning does not
settle. The binning refutes the FIRST hypothesis (prompt distribution): the penalty is flat across
prompt-length quartiles -- +0.0350 / +0.0287 / +0.0346 / +0.0296 shortest to longest, 37-39 of 41
worse in every bin -- so HumanEval's long docstrings are not the cause. What it cannot rule out is
that the two call sites disagree.

THE TWO SITES ARE DIFFERENT SHAPES, which is why this is not paranoia:
  training  B=16, T=4096, ~250 documents packed per batch, prompt lengths 12-1143 within documents,
            cu from doc_cu_seqlens, aux from doc_prompt_lengths
  eval      B=1, ONE document of ~200-1400 tokens, cu = [0, T], aux = [prompt_len]
Same function, wildly different arguments. scripts/n7c_grad_check.py compared them on the TRAINING
shape only. A bug that needs a single-document batch -- an off-by-one at the row boundary, a
degenerate cu2, the P == L edge -- would be invisible there and would land entirely on the eval
number, which is exactly the observed pattern: training loss identical to the twin, eval 0.0228
worse.

THE TEST. Take real HumanEval prompt+solution pairs, run each through the eval path (one row) and
through the training path (the same tokens packed as documents in a B>1 batch with doc_cu_seqlens),
and require the per-position logits at the SOLUTION positions to agree. Both paths run the same
prefix_two_call, so any difference is in how cu and aux are built at the two sites, which is the
thing in question.

WHAT AGREEMENT WOULD MEAN: the eval path is not the explanation, and P3's +0.0228 is a real
property of applying this mask at inference to these weights. WHAT DISAGREEMENT WOULD MEAN: the
eval number is measuring a wiring difference and the row must not be written from it.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CKPT = "ckpt_n7c_p3.pt"
ARM = "p3"
N_TASKS = 6  # six real tasks; the question is agreement, not a mean over the set


def main():
    import json  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415

    import model as M  # noqa: PLC0415
    from eval.prefix_mask import PREFIX_ARMS, doc_prompt_lengths, prefix_two_call  # noqa: PLC0415
    from scripts.loader import load_checkpoint  # noqa: PLC0415
    from train import doc_cu_seqlens  # noqa: PLC0415

    if not M.HAS_FA:
        raise SystemExit("REFUSING: HAS_FA is False; neither path would run the real kernel.")

    mdl, cfg = load_checkpoint(CKPT, dtype=torch.bfloat16)
    mdl = mdl.cuda().eval()
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    eos = tok.token_to_id("<eos>")
    layers = PREFIX_ARMS[ARM]
    targets = [mdl.blocks[i].mixer for i in layers]

    data = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
    pairs = []
    with open(data, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            p = tok.encode(r["prompt"]).ids
            s = tok.encode(r["canonical_solution"]).ids
            if s and len(p) + len(s) < 900:  # fits several per training row
                pairs.append((r["task_id"], p, s))
            if len(pairs) >= N_TASKS:
                break
    print(f"{len(pairs)} tasks, prompt+solution lengths "
          f"{[len(p) + len(s) for _, p, s in pairs]}")

    orig = M.flash_attn_varlen_func
    depth = [0]
    aux = [None]

    def patched(q, k, v, **kw):
        if aux[0] is None or depth[0] == 0:
            return orig(q, k, v, **kw)
        cu_in = kw.pop("cu_seqlens_q", None)
        kw.pop("cu_seqlens_k", None)
        return prefix_two_call(orig, q, k, v, cu_in, aux[0][0], **kw)

    def wrap(mod):
        inner = mod.forward

        def fwd(*a, **k):
            depth[0] += 1
            try:
                return inner(*a, **k)
            finally:
                depth[0] -= 1
        mod.forward = fwd

    for t in targets:
        wrap(t)
    M.flash_attn_varlen_func = patched

    def logits_for(ids_row, cu, plens, masked=True):
        # masked=False is THE CONTROL: aux stays None so prefix_two_call is never entered and both
        # sites run plain causal varlen. Without it this test cannot tell a prefix-wiring difference
        # from a PACKING difference -- and packing changes more than attention: the KDA layers are
        # recurrent (chunk_kda), so tokens of task 2 in a packed row may see state carried from task
        # 1 in a way a single-row eval forward never produces. That would make the two paths differ
        # for every mask including none, and reading it as a prefix defect would be the same error
        # as the gradient check's first run, which reported 166 of 175 tensors disagreeing with no
        # control row to attribute it to.
        aux[0] = [plens.to(torch.int32)] if masked else None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return mdl(ids_row, cu=cu)[0].float()

    # EVAL PATH: one row per task, cu = [0, T], aux = [prompt_len]. Exactly what
    # eval/humaneval_bpb.py's OursModel.logprobs builds.
    eval_logits, eval_causal = {}, {}
    for tid, p, s in pairs:
        ids = torch.tensor([p + s], device="cuda")
        cu = torch.tensor([0, len(p) + len(s)], dtype=torch.int32, device="cuda")
        pl = torch.tensor([len(p)], device="cuda")
        eval_logits[tid] = (logits_for(ids, cu, pl)[0], len(p), len(s))
        eval_causal[tid] = logits_for(ids, cu, pl, masked=False)[0]

    # TRAINING PATH: the same token sequences packed into ONE row separated by <eos>, with cu from
    # doc_cu_seqlens and aux from doc_prompt_lengths -- the sft_math.py construction. Labels mask the
    # prompt to -100 and supervise the solution, which is what doc_prompt_lengths reads.
    flat, labels = [], []
    spans = []
    for _tid, p, s in pairs:
        spans.append((len(flat), len(p), len(s)))
        flat += p + s + [eos]
        labels += [-100] * len(p) + s + [-100]
    # PAD TO AN EVEN LENGTH: an odd sequence misaligns chunk_kda (model.py:125), which is a real
    # crash and not a warning -- hit earlier today on the 4097-column pack.
    if len(flat) % 2:
        flat.append(eos)
        labels.append(-100)
    ids_t = torch.tensor([flat], device="cuda")
    lab_t = torch.tensor([labels], device="cuda")
    cu_t = doc_cu_seqlens(ids_t, eos)
    pl_t = doc_prompt_lengths(lab_t, cu_t).to("cuda")
    print(f"training row: {len(flat)} tokens, {cu_t.numel() - 1} documents, "
          f"per-document prompt lengths {pl_t.tolist()}")
    train_logits = logits_for(ids_t, cu_t, pl_t)[0]
    train_causal = logits_for(ids_t, cu_t, pl_t, masked=False)[0]

    M.flash_attn_varlen_func = orig

    # COMPARE AT THE SOLUTION POSITIONS ONLY, because those are the positions the BPB number is
    # computed from. A difference outside them cannot explain the eval delta.
    print("\nper-task max |logit difference| at solution positions, eval path vs training path:")
    print(f"  {'task':16s} {'prompt':>6s} {'sol':>4s} {'PREFIX':>9s} {'CAUSAL':>9s}   "
          f"reading")
    worst = worst_c = 0.0
    for (tid, _p, _s), (start, plen, slen) in zip(list(pairs), spans, strict=True):
        ev, _pl, _sl = eval_logits[tid]
        a = ev[plen - 1:plen - 1 + slen]
        b = train_logits[start + plen - 1:start + plen - 1 + slen]
        d = (a - b).abs().max().item()
        ac = eval_causal[tid][plen - 1:plen - 1 + slen]
        bc = train_causal[start + plen - 1:start + plen - 1 + slen]
        dc = (ac - bc).abs().max().item()
        worst, worst_c = max(worst, d), max(worst_c, dc)
        print(f"  {tid:16s} {plen:6d} {slen:4d} {d:9.4f} {dc:9.4f}   "
              f"{'packing, not the mask' if dc > 0.5 else 'mask-specific' if d > 0.5 else 'agree'}")
    # bf16 through 12 blocks accumulates; the question is whether the paths AGREE, and a
    # disagreement large enough to move BPB by 0.0228 would be far above numerical noise.
    print(f"\nworst across tasks: prefix {worst:.4f}  causal control {worst_c:.4f}")
    if worst_c > 0.5:
        print("VERDICT: THE CONTROL ALSO DISAGREES, so this test cannot attribute anything to the "
              "prefix wiring. Packing several tasks into one row is not equivalent to scoring them "
              "one at a time in THIS model regardless of mask -- the KDA layers are recurrent, so a "
              "packed row carries state a single-row forward never has. The comparison is invalid "
              "as designed and the eval-vs-training question needs a construction where the causal "
              "control agrees first.")
    elif worst > 0.5:
        print(f"VERDICT: PREFIX-SPECIFIC DISAGREEMENT of {worst:.4f} while the causal control "
              f"agrees at {worst_c:.4f} -- the eval number is measuring a wiring difference between "
              "the two call sites and the row must not be written from it.")
    else:
        print("VERDICT: BOTH PATHS AGREE -- the eval-side wiring is not the explanation for P3's "
              "+0.0228, so that penalty is a real property of the mask at inference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
