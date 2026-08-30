#!/usr/bin/env python3
"""L1: few-shot continuation math on a base checkpoint (reasoning_panel.md S2).

Pre-registered 2026-08-30 (before p324 landed): N>=500, exact-match final answer,
non-zero > 2*delta (delta=1.4/sqrt(N)) => the production instrument exists.

Format (pinned before running): plain-text continuation -- 3 solved demos (problems
0-2 of math_test_500, full gold solutions) then the target problem; the base model
continues the solution. Demos excluded from eval (N=497). ChatML is NOT used: the
base saw 1.18% chat-domain data and the zero-shot ChatML 0/500 confounds format
with capability; plain continuation is the clean bridge.

Usage: CUDA_VISIBLE_DEVICES=7 python3 eval/l1_fewshot.py --ckpt ckpt_p324.pt
"""
import argparse
import json
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# FLA_FLASH_KDA must stay unset: the new-arch ladder checkpoints (attn_every 4)
# route 9/12 layers through chunk_kda, and the eval runners' "0" default makes
# that import fail (train.py:107 -> chunk_kda=None -> forward crash). score_matrix
# leaves it unset and scores the same checkpoints fine.
import fone  # noqa: E402
from eval.gsm8k import generate_batch  # noqa: E402
from algorithms.rlvr_reward import reward_fn, extract_boxed  # noqa: E402
from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402

TEST_PATH = os.path.join(ROOT, "data", "eval", "math_test_500.jsonl")
TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
N_DEMOS = 3

# Inlined from eval/math_zh.py: importing that module sets FLA_FLASH_KDA=0,
# which kills chunk_kda for the new-arch ladder checkpoints.
ANS_RE = re.compile(r"答案是[:：]\s*(.+?)(?:[。\n]|$)")


def score(gen, gold):
    gold_ans = extract_boxed(gold)
    if gold_ans is None:
        return 0.0
    if extract_boxed(gen) is not None:
        return reward_fn(gen, gold_ans)
    m = ANS_RE.search(gen)
    if m:
        return reward_fn(f"\\boxed{{{m.group(1).strip()}}}", gold_ans)
    return 0.0


def build_prompt(demos, target_q):
    parts = [f"题目：{q}\n解答：{a}" for q, a in demos]
    parts.append(f"题目：{target_q}\n解答：")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokenizer", default=TOK_PATH)
    args = ap.parse_args()

    # dtype goes through load_checkpoint (a3a0de0 upcasts KDA A_log/dt_bias to
    # fp32 after the cast); a separate .to(bf16) here would undo the upcast.
    model, cfg = load_checkpoint(args.ckpt, device=args.device, dtype=torch.bfloat16)
    tok = load_tokenizer(args.tokenizer, cfg)
    fone_on = getattr(cfg, "fone", False)
    num_id = getattr(cfg, "num_id", None)

    rows = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8")]
    demos = [(r["instruction"], r["output"]) for r in rows[:N_DEMOS]]
    evals = rows[N_DEMOS:]
    print(f"L1 few-shot: {len(demos)} demos, {len(evals)} eval problems", flush=True)

    preds_path = os.path.join(ROOT, "data", "eval", "preds_l1.jsonl")
    correct = total = 0
    n_box = 0
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
            for r, ids, vs in zip(batch, out_ids, out_vals):
                gen = fone.decode_text(ids, vs, tok, num_id) if fone_on else tok.decode(ids)
                ok = score(gen, r["output"])
                correct += int(ok)
                total += 1
                n_box += int("\\boxed" in gen or "答案是" in gen)
                fout.write(json.dumps({"q": r["instruction"], "gen": gen[-300:], "ok": ok},
                                      ensure_ascii=False) + "\n")
            if total % 64 < args.batch or total == len(evals):
                print(f"  {total}/{len(evals)} acc={correct / total:.1%}", flush=True)

    delta = 1.4 / (total ** 0.5)
    print(f"L1 math-500 few-shot: {correct}/{total} = {correct / total:.1%}")
    print(f"binomial delta={delta:.1%} -> 2*delta={2 * delta:.1%}; "
          f"instrument exists iff acc > {2 * delta:.1%}")
    print(f"answer-present rate {n_box / total:.1%}")
    print(f"preds saved: {preds_path}")


if __name__ == "__main__":
    main()
