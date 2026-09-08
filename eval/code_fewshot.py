#!/usr/bin/env python3
"""Code-500 few-shot continuation on a BASE checkpoint (companion to L1 math).

Same instrument-existence logic as eval/l1_fewshot.py: plain-text continuation,
3 demos (rows 0-2 of code_holdout_500, excluded -> N=497), no ChatML -- the
base saw no chat template, and zero-shot ChatML would confound format with
capability. The base continues the python code; the code is executed in the
sandbox and stdout matched against the recorded oracle.

Why this exists (fb 2026-08-30): zero-shot code-500 and math-500 both hang off
the single SFT checkpoint -- the whole generative axis is one point of
failure. Few-shot on base separates "can the model write code" from "did SFT
teach the format"; the SFT number then has a control.

Known-answer (--selfcheck, no GPU): every reference solution scores 1 on its
own oracle, wrong solutions score 0, and the prompt/extraction contract is
verified on mock continuations.

Usage: CUDA_VISIBLE_DEVICES=X python3 eval/code_fewshot.py --ckpt ckpt_p324.pt
"""
import argparse
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "scripts"))
from eval_artifacts import attest, open_artifact  # noqa: E402
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# FLA_FLASH_KDA deliberately NOT set: see l1_fewshot.py:26-29.

import torch  # noqa: E402
from eval.gsm8k import generate_batch  # noqa: E402
from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402
from datagen.sandbox_exec import run_sandboxed  # noqa: E402

TEST_PATH = os.path.join(ROOT, "data", "eval", "code_holdout_500.jsonl")
TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
N_DEMOS = 3


def _norm_lines(s):
    return [ln.rstrip() for ln in s.split("\n") if ln.strip() != ""]


def score_code(code, expected_output, timeout=10):
    rc, out, _ = run_sandboxed(code, timeout=timeout)
    return rc == 0 and _norm_lines(out) == _norm_lines(expected_output)


def build_prompt(demos, target_q):
    """demos: [(instruction, reference_code, expected_output)]. Plain-text
    continuation format, pinned before first run (same discipline as L1)."""
    parts = [f"题目：{q}\n```python\n{code}\n```\n运行输出：\n{out}"
             for q, code, out in demos]
    parts.append(f"题目：{target_q}\n```python\n")
    return "\n\n".join(parts)


def extract_code(cont):
    """The continuation follows the prompt's opening ```python fence; take
    everything up to the first closing fence. No fence (truncated gen) ->
    the whole continuation, which execution will judge honestly."""
    end = cont.find("```")
    return cont[:end] if end >= 0 else cont


def continuation_code(gen_ids, vals, tok, num_id, fone_on):
    """generate_batch's ids for one row -> the code to execute.

    generate_batch RETURNS ONLY THE GENERATED IDS -- its last statement slices
    x[i, lengths[i]:ends[i]] per row, and its docstring says "Returns generated ids".
    Until 2026-09-08 this file sliced `ids[len(prompt):]` on top of that, cutting the
    prompt off a SECOND time. Found by 4c; eval/l1_fewshot.py always decoded `ids`
    directly and is the pattern followed here.

    WHAT THAT DID TO THE RUNS ALREADY PUBLISHED, and what the surviving artifacts can
    and cannot establish. The predictions on the pod
    (data/eval/preds_code_fewshot{,_0shot,_1shot}.jsonl, 497 rows each) were written as
    `cont[-300:]` -- A TAIL WINDOW, so they are not the cut continuation and cannot be
    read as one. 491 of 497 sit at exactly the 300-char cap, and a control on a file
    written by another tool shows the window ALONE moves parseability, so any statement
    about mid-statement starts or parse rates from these files is confounded.

    What they do establish: NOT EMPTY. An empty continuation stores as empty, and 0 of
    497 are, in every arm -- matching the logs' own 2.2% at 0-shot and "non-empty 98.4%"
    at 3-shot. So the runs did NOT decode to the empty string.

    What the arithmetic establishes: the runs used max_new=512 with rep_stop off, and a
    3-shot prompt is 319-335 tokens, so the cut removed roughly the first two thirds of
    what was generated and scoring saw the tail. Head absent, tail present.

    That is the dangerous shape: 0/497 with almost everything a syntax error reads
    exactly like "the model cannot write code", which is the conclusion those runs drew.
    An empty string would have been noticed in a day.

    Split out of main()'s batch loop so --selfcheck can drive it. The defect
    survived because the decode lived inline where no case could reach it, and it
    needs no GPU and no sandbox to check -- only the tokenizer.
    """
    if fone_on:
        import fone  # local, as in main(): the module pulls torch in
        cont = fone.decode_text(gen_ids, vals, tok, num_id)
    else:
        cont = tok.decode(gen_ids)
    return cont, extract_code(cont)


def score_row(gen_ids, vals, expected_output, tok, num_id, fone_on):
    """One row -> (continuation text, scored, empty). Executes; pod only."""
    cont, code = continuation_code(gen_ids, vals, tok, num_id, fone_on)
    if not code.strip():
        return cont, False, True
    return cont, score_code(code, expected_output), False


def selfcheck():
    rows = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8")]
    demos = [(r["instruction"], r["reference_code"], r["expected_output"])
             for r in rows[:N_DEMOS]]
    evals = rows[N_DEMOS:]

    # prompt/extraction contract on mock continuations
    p = build_prompt(demos, "写一个函数返回 1")
    assert "题目：" in p and p.endswith("```python\n"), "prompt format drift"
    code = extract_code("def f():\n    return 1\n```\n运行输出：\n1\n")
    assert code == "def f():\n    return 1\n", f"extraction drift: {code!r}"
    assert extract_code("print(1)\n") == "print(1)\n", "no-fence extraction drift"
    print("prompt/extraction contract: OK")

    # THE KNOWN ANSWER FOR THE DECODE PATH, and its own negative control.
    #
    # continuation_code is fed exactly what generate_batch returns: the tokens of the
    # continuation, prompt already stripped by generate_batch. Six reference solutions
    # round-tripped through the real tokenizer must come back byte-identical.
    #
    # Under the defect this file carried until 2026-09-08 -- a second
    # `ids[len(prompt):]` on top of generate_batch's own slice -- this same case reads
    # 0/6. So it needs no separate broken-world fixture: the number it prints IS the
    # difference between the two versions, and 6/6 is unreachable for the old code.
    #
    # ROUND TRIP, NOT "NON-EMPTY", and that distinction is the whole case. The empty
    # rate is reported but is NOT the criterion: on the real published runs the defect
    # beheaded rather than emptied (max_new=512, so subtracting a 335-token prompt still
    # left ~180 tokens), and 0 of 497 stored continuations are empty in any arm. A
    # non-empty assertion passes on every one of those rows,
    # and non-empty is exactly what the caller reads as success.
    #
    # It runs BEFORE the two execution cases and needs no sandbox and no GPU, so it is
    # a defence that works on a laptop. The defect survived precisely because the decode
    # lived inline in main()'s batch loop, where --selfcheck could not reach it and the
    # only way to run it was a pod with a checkpoint.
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK_PATH)
    ka = evals[:6]
    ka_ok = ka_empty = 0
    for r in ka:
        # what generate_batch hands back for a perfect greedy continuation:
        # the reference solution followed by the closing fence
        gen_ids = tok.encode(r["reference_code"] + "\n```\n").ids
        _, code = continuation_code(gen_ids, None, tok, None, False)
        if not code.strip():
            ka_empty += 1
        elif code.strip() == r["reference_code"].strip():
            ka_ok += 1
        else:
            print(f"  KNOWN-ANSWER FAIL: {r['instruction'][:36]} -> {code[:60]!r}")
    print(f"decode known-answer: {ka_ok}/{len(ka)} recovered, "
          f"empty-continuation rate {ka_empty / len(ka):.0%}")
    ka_fails = (len(ka) - ka_ok) + ka_empty
    if os.geteuid() != 0:
        # The two cases below execute code, and datagen/sandbox_exec.py:169 refuses without
        # root. Skipping them off-pod is what lets the decode case above run in the hook on a
        # laptop -- the defect it guards was a decode defect, and gating a decode check behind
        # a sandbox is how it stayed unchecked for a week of runs.
        print("gold round-trip / wrong-solution: SKIPPED (sandbox_exec needs root; pod only)")
        return ka_fails

    fails = 0
    for i, r in enumerate(evals):
        if not score_code(r["reference_code"], r["expected_output"]):
            fails += 1
            print(f"  GOLD FAIL row {i}")
    print(f"gold round-trip: {len(evals) - fails}/{len(evals)} pass")

    wrong = [
        ("print('this is not the answer')", "definitely wrong output"),
        ("while True:\n    pass", "anything (timeout must not score)"),
        ("def f(:\n    pass", "anything (syntax error)"),
    ]
    wfails = sum(1 for code, exp in wrong if score_code(code, exp))
    print(f"wrong-solution zero: {len(wrong) - wfails}/{len(wrong)} pass")

    return fails + wfails + ka_fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokenizer", default=TOK_PATH)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature; 0 = greedy")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--demos", type=int, default=N_DEMOS, choices=[0, 1, 3, 8],
                    help="number of few-shot demos (0 = pure continuation, tests whether "
                         "demos help or hurt; fb 2026-08-30)")
    ap.add_argument("--eval-from", type=int, default=8,
                    help="eval rows start at this index; demo pool is rows[:eval-from]. "
                         "Pinned at 8 so the 0/1/3/8 sweep scores the same 492 problems "
                         "(honest_measurement_prereg §2)")
    ap.add_argument("--force", action="store_true",
        help="overwrite an existing predictions file (default: refuse; the rows are the only copy)")
    ap.add_argument("--run", default=None,
        help="name this run so predictions version instead of colliding: preds_x.<run>.jsonl")
    args = ap.parse_args()

    if args.selfcheck:
        sys.exit(1 if selfcheck() else 0)
    if not args.ckpt:
        ap.error("--ckpt required (unless --selfcheck)")

    import fone  # noqa: F401

    model, cfg = load_checkpoint(args.ckpt, device=args.device, dtype=torch.bfloat16)
    tok = load_tokenizer(args.tokenizer, cfg)
    fone_on = getattr(cfg, "fone", False)
    num_id = getattr(cfg, "num_id", None)

    rows = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8")]
    # Demo pool = rows[:eval_from], eval = rows[eval_from:]. Every arm (0/1/3/8)
    # scores the same 492 problems; --demos <= --eval-from keeps demos out of eval.
    # Pinned 2026-09-01 (honest_measurement_prereg §2): the 3-vs-8 byte-identical
    # defect in l1_fewshot came from a pool sized to the smallest arm.
    assert args.demos <= args.eval_from, (
        f"--demos {args.demos} would show rows also scored as eval "
        f"(eval starts at {args.eval_from})")
    demos = [(r["instruction"], r["reference_code"], r["expected_output"])
             for r in rows[:args.demos]]
    evals = rows[args.eval_from:]
    print(f"code few-shot: {len(demos)} demos, {len(evals)} eval problems", flush=True)

    # The checkpoint's name is IN the path, for the reason l1_fewshot.py:168 records: a
    # path without it collides across checkpoints and the second scoring gets
    # ArtifactExists instead of a number. Found on l1_fewshot (fb, 2026-09-02); this file
    # carried the same defect and no one had hit it yet. --ckpt is optional here (--selfcheck
    # loads nothing), so the empty case gets a literal rather than "None".
    preds_path = os.path.join(
        ROOT, f"data/eval/preds_code_fewshot_{args.demos}shot"
        f"_{os.path.basename(args.ckpt) if args.ckpt else 'nockpt'}"
        + (f".t{args.temperature}" if args.temperature else "")
        + ".jsonl")
    correct = total = no_fence = 0
    with open_artifact(preds_path, force=args.force, run=args.run) as fout:
        # --run versions the path, so the handle's name is the file that exists.
        out_path = fout.name
        for s in range(0, len(evals), args.batch):
            batch = evals[s : s + args.batch]
            texts_in = [build_prompt(demos, r["instruction"]) for r in batch]
            if fone_on:
                prompts, pvals = fone.encode_prompts(texts_in, tok, num_id)
            else:
                prompts, pvals = [tok.encode(t).ids for t in texts_in], None
            with torch.no_grad():
                out = generate_batch(model, prompts, args.max_new, args.device,
                                     args.temperature, pvals, rep_stop=False)
            out_ids, out_vals = out if fone_on else (out, [None] * len(batch))
            for r, ids, vs in zip(batch, out_ids, out_vals):
                cont, ok, empty = score_row(ids, vs, r["expected_output"],
                                            tok, num_id, fone_on)
                no_fence += int(empty)
                correct += int(ok)
                total += 1
                fout.write(json.dumps({"q": r["instruction"], "gen": cont,
                                       "ok": ok}, ensure_ascii=False) + "\n")
            if total % 64 < args.batch or total == len(evals):
                print(f"  {total}/{len(evals)} acc={correct / total:.1%}", flush=True)

    # attest what was WRITTEN, not what was requested: --run versions the path, and
    # attesting preds_path recorded a hash for a file this run never touched.
    attest(out_path)  # the citation contract: the writer proves these bytes existed
    delta = 1.4 / (total ** 0.5)
    print(f"code-500 few-shot ({args.demos}-shot, t={args.temperature}): {correct}/{total} = {correct / total:.1%}")
    print(f"binomial delta={delta:.1%} -> 2*delta={2 * delta:.1%}; "
          f"instrument exists iff acc > {2 * delta:.1%}")
    print(f"empty-continuation rate {no_fence / total:.1%}")
    print(f"preds saved: {out_path}")


if __name__ == "__main__":
    main()
