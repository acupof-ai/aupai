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
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# FLA_FLASH_KDA deliberately NOT set: see l1_fewshot.py:26-29.

import torch  # noqa: E402
from eval.gsm8k import generate_batch  # noqa: E402
from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402
from scripts.sandbox_exec import run_sandboxed  # noqa: E402

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
    return fails + wfails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokenizer", default=TOK_PATH)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--demos", type=int, default=N_DEMOS, choices=[0, 1, 3],
                    help="number of few-shot demos (0 = pure continuation, tests whether "
                         "demos help or hurt; fb 2026-08-30)")
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
    demos = [(r["instruction"], r["reference_code"], r["expected_output"])
             for r in rows[:args.demos]]
    evals = rows[args.demos:]
    print(f"code few-shot: {len(demos)} demos, {len(evals)} eval problems", flush=True)

    preds_path = os.path.join(ROOT, f"data/eval/preds_code_fewshot_{args.demos}shot.jsonl")
    correct = total = no_fence = 0
    with open(preds_path, "w", encoding="utf-8") as fout:
        for s in range(0, len(evals), args.batch):
            batch = evals[s : s + args.batch]
            texts_in = [build_prompt(demos, r["instruction"]) for r in batch]
            if fone_on:
                prompts, pvals = fone.encode_prompts(texts_in, tok, num_id)
            else:
                prompts, pvals = [tok.encode(t).ids for t in texts_in], None
            with torch.no_grad():
                out = generate_batch(model, prompts, args.max_new, args.device, 0.0, pvals)
            out_ids, out_vals = out if fone_on else (out, [None] * len(batch))
            for r, ids, vs, pr in zip(batch, out_ids, out_vals, prompts):
                # slice the continuation off the prompt BEFORE decode: the
                # demos contain ```python blocks, and token_len != char_len
                cont_ids = ids[len(pr):]
                cont = (fone.decode_text(cont_ids, vs[len(pr):], tok, num_id)
                        if fone_on else tok.decode(cont_ids))
                code = extract_code(cont)
                if not code.strip():
                    no_fence += 1
                    ok = False
                else:
                    ok = score_code(code, r["expected_output"])
                correct += int(ok)
                total += 1
                fout.write(json.dumps({"q": r["instruction"], "gen": cont[-300:],
                                       "ok": ok}, ensure_ascii=False) + "\n")
            if total % 64 < args.batch or total == len(evals):
                print(f"  {total}/{len(evals)} acc={correct / total:.1%}", flush=True)

    delta = 1.4 / (total ** 0.5)
    print(f"code-500 few-shot ({args.demos}-shot): {correct}/{total} = {correct / total:.1%}")
    print(f"binomial delta={delta:.1%} -> 2*delta={2 * delta:.1%}; "
          f"instrument exists iff acc > {2 * delta:.1%}")
    print(f"empty-continuation rate {no_fence / total:.1%}")
    print(f"preds saved: {preds_path}")


if __name__ == "__main__":
    main()
