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

THE FIRST VERSION OF THIS TEST WAS INVALID AND ITS ANSWER IS RETRACTED. It packed six tasks into
one row and compared that against each task scored alone, then read max|dlogit| 4.6910 as "the
eval path is broken". Measured on card 4, pod 2026-09-03 21:5xZ, the causal control disagrees
almost as much -- 3.9619 with NO mask involved anywhere:

  task              prompt  sol    PREFIX    CAUSAL
  HumanEval/0          117   52    0.3654    0.2802
  HumanEval/1          131   99    4.6910    3.9619
  HumanEval/2           83    8    3.5368    2.7987
  HumanEval/3          118   27    3.9020    2.7733
  HumanEval/4          123   33    3.4644    2.1841
  HumanEval/5           92   48    3.3154    1.8609

So packing changes the logits by ~3-4 on its own and the comparison could never isolate the mask.
model.py:122-131 does pass cu_seqlens into chunk_kda, so the boundaries reach the kernel; that
argument being passed is evidently not sufficient for a packed row to equal an unpacked one here.
Whatever the mechanism, a packed-vs-unpacked comparison cannot answer a question about cu/aux
CONSTRUCTION, because the packing itself moves the number more than the thing being tested.

WHAT THE INVALID RUN DID SETTLE, and it kills the leading hypothesis: the training-side
doc_prompt_lengths returned [117, 131, 83, 118, 123, 92] and the eval side's len(prompt) for the
same six tasks is [117, 131, 83, 118, 123, 92]. Identical. The two sites derive the SAME prompt
length, so "doc_prompt_lengths reads the first supervised position while eval uses len(p_ids), and
they disagree" is refuted.

THE TEST NOW. Hold the shape fixed and vary ONLY the wiring: one task, one row, one document,
scored twice --
  eval construction     cu = [0, T] built by hand, aux = [len(prompt)]      (humaneval_bpb.py)
  training construction cu = doc_cu_seqlens(ids), aux = doc_prompt_lengths(labels, cu)  (sft_math.py)
Same tokens, same batch shape, same single document. Nothing is packed, so there is no packing
difference left to confound it, and any disagreement is in the construction -- which is the
question. The causal control runs in both constructions too, so a difference that is not about the
mask is still separable.

WHAT AGREEMENT MEANS: the eval path is not the explanation, and P3's +0.0228 is a real property of
applying this mask at inference to these weights. WHAT DISAGREEMENT MEANS: the eval number is
measuring a wiring difference and the row must not be written from it.
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
    from scripts.loader import claim_my_cards, load_checkpoint  # noqa: PLC0415
    from train import doc_cu_seqlens  # noqa: PLC0415

    # de-55 step 3: loads to CPU and moves at `mdl.cuda()`, so load_checkpoint's cuda-gated claim
    # does not fire.
    claim_my_cards("n7c_path_agree", note="two constructions compared in BPB")

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

    # TRAINING CONSTRUCTION, SAME SHAPE: still one task per row, one document, but cu and aux come
    # from the training helpers instead of being built by hand. Labels mask the prompt to -100 and
    # supervise the solution, which is what doc_prompt_lengths reads. The trailing <eos> is what
    # makes it a document to doc_cu_seqlens, so the row is one token longer than the eval row and
    # the comparison is taken over the solution positions the two rows share.
    train_logits, train_causal, derived_plen = {}, {}, {}
    for tid, p, s in pairs:
        flat = p + s + [eos]
        labels = [-100] * len(p) + s + [-100]
        if len(flat) % 2:  # odd length misaligns chunk_kda (model.py:125): a crash, not a warning
            flat.append(eos)
            labels.append(-100)
        ids_t = torch.tensor([flat], device="cuda")
        lab_t = torch.tensor([labels], device="cuda")
        cu_t = doc_cu_seqlens(ids_t, eos)
        pl_t = doc_prompt_lengths(lab_t, cu_t).to("cuda")
        derived_plen[tid] = pl_t.tolist()
        train_logits[tid] = logits_for(ids_t, cu_t, pl_t)[0]
        train_causal[tid] = logits_for(ids_t, cu_t, pl_t, masked=False)[0]

    # THE AUX MUST MATCH FIRST, and it is checked rather than eyeballed: if doc_prompt_lengths
    # returns a different prompt length than the eval site uses, the two paths are masking
    # different spans and a logit difference would be that, not a wiring bug in the kernel call.
    bad_aux = [(tid, derived_plen[tid], len(p)) for tid, p, _s in pairs
               if derived_plen[tid] != [len(p)]]
    print("\nderived aux vs eval aux: " + ("ALL MATCH" if not bad_aux else f"MISMATCH {bad_aux}"))

    M.flash_attn_varlen_func = orig

    # COMPARE AT THE SOLUTION POSITIONS ONLY, because those are the positions the BPB number is
    # computed from. A difference outside them cannot explain the eval delta.
    print("\nper-task max |logit difference| at solution positions, "
          "eval construction vs training construction (same shape):")
    print(f"  {'task':16s} {'prompt':>6s} {'sol':>4s} {'PREFIX':>9s} {'CAUSAL':>9s}   reading")
    worst = worst_c = 0.0
    for tid, p, s in pairs:
        plen, slen = len(p), len(s)
        ev, _pl, _sl = eval_logits[tid]
        a = ev[plen - 1:plen - 1 + slen]
        b = train_logits[tid][plen - 1:plen - 1 + slen]
        d = (a - b).abs().max().item()
        ac = eval_causal[tid][plen - 1:plen - 1 + slen]
        bc = train_causal[tid][plen - 1:plen - 1 + slen]
        dc = (ac - bc).abs().max().item()
        worst, worst_c = max(worst, d), max(worst_c, dc)
        print(f"  {tid:16s} {plen:6d} {slen:4d} {d:9.4f} {dc:9.4f}   "
              f"{'not the mask' if dc > 0.5 else 'MASK-SPECIFIC' if d > 0.5 else 'agree'}")
    # bf16 through 12 blocks accumulates; the question is whether the paths AGREE, and a
    # disagreement large enough to move BPB by 0.0228 would be far above numerical noise.
    # A max|dlogit| IS NOT A BOUND ON BPB, and 0.5 was my threshold, not a derived one. The question
    # the P3 row turns on is whether a construction difference can move gold BPB by 0.0228, so
    # compute BPB both ways on these tasks and print the difference in the SAME units. Gold BPB is
    # sum(-log2 p(gold token)) / solution bytes, which is what eval/humaneval_bpb.py reports.
    import torch.nn.functional as F  # noqa: PLC0415
    print("\ngold BPB on these tasks, computed from each construction (the units the row uses):")
    tot = {"eval": 0.0, "train": 0.0}
    nbytes = 0
    for tid, p_ids, s_ids in pairs:
        plen = len(p_ids)
        gold = torch.tensor(s_ids, device="cuda")
        nb = len(tok.decode(s_ids).encode("utf-8"))
        nbytes += nb
        for lbl, src in (("eval", eval_logits[tid][0]), ("train", train_logits[tid])):
            lg = src[plen - 1:plen - 1 + len(s_ids)]
            nats = F.cross_entropy(lg.float(), gold, reduction="sum").item()
            tot[lbl] += nats / 0.6931471805599453  # nats -> bits
    be, bt = tot["eval"] / nbytes, tot["train"] / nbytes
    print(f"  eval construction  {be:.4f}")
    print(f"  train construction {bt:.4f}")
    print(f"  difference         {bt - be:+.4f}   against the P3 delta under test, +0.0228")
    print(f"  ({nbytes} solution bytes over {len(pairs)} tasks; the row's number is 164 tasks / "
          f"29662 bytes, so this is a spot check on the construction, not a re-score)")
    if abs(bt - be) > 0.0228:
        print("  THIS EXCEEDS THE EFFECT UNDER TEST: the construction difference alone can produce "
              "a delta the size of P3's, so the max|dlogit| agreement above is not sufficient and "
              "the row still must not be written from the eval number.")
    else:
        print(f"  Bounded BELOW the effect under test by "
              f"{0.0228 / max(abs(bt - be), 1e-9):.0f}x, so the construction cannot account for "
              f"P3's delta.")

    print(f"\nworst across tasks: prefix {worst:.4f}  causal control {worst_c:.4f}")
    if bad_aux:
        print("VERDICT: THE TWO SITES MASK DIFFERENT SPANS -- doc_prompt_lengths and the eval "
              f"site's len(prompt) disagree: {bad_aux}. Fix that before reading the logits, "
              "because a different mask span explains any difference on its own.")
    elif worst_c > 0.5:
        print("VERDICT: THE CONTROL DISAGREES WITH THE SHAPE HELD FIXED, which the packing "
              "explanation no longer covers -- the only remaining differences are the trailing "
              "<eos> and cu being derived rather than hand-built. Attribute nothing to the prefix "
              "wiring until that is explained; it is now the cheaper thing to chase.")
    elif worst > 0.5:
        print(f"VERDICT: PREFIX-SPECIFIC DISAGREEMENT of {worst:.4f} while the causal control "
              f"agrees at {worst_c:.4f} -- with shape and aux held equal this is the mask wiring "
              "itself, so the eval number is measuring a bug and the P3 row must not be written "
              "from it.")
    else:
        print(f"VERDICT: BOTH CONSTRUCTIONS AGREE (prefix {worst:.4f}, causal {worst_c:.4f}) -- "
              "with the shape held fixed the eval and training sites build the same mask, so the "
              "eval-side wiring is NOT the explanation for P3's +0.0228 and that penalty is a real "
              "property of applying this mask at inference to these weights. NOTE THE SCOPE: this "
              "says the two call sites agree, not that the eval shape is the training shape -- "
              "packing alone moves these logits by 3-4 (measured above), so a single-document eval "
              "is a different regime from a packed training batch no matter which mask is used.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
