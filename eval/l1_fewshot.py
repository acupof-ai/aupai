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
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "scripts"))
from eval_artifacts import attest, open_artifact  # noqa: E402
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
    ap.add_argument("--demos", type=int, default=3,
                    help="number of solved demonstrations (fb's demo-count sweep: "
                         "0-shot continuation vs 3-shot; pre-registered reading: "
                         "0-shot > 3-shot by >2delta=12.6pt => demos interfere; "
                         "both zero => the zero is not a prompt-format artefact)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokenizer", default=TOK_PATH)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature; 0 = greedy")
    ap.add_argument("--out", default=None, help="write JSON summary to this path")
    ap.add_argument("--force", action="store_true",
        help="overwrite an existing predictions file (default: refuse; the rows are the only copy)")
    ap.add_argument("--run", default=None,
        help="name this run so predictions version instead of colliding: preds_x.<run>.jsonl")
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
    print(f"L1 few-shot: {args.demos} demos, {len(evals)} eval problems", flush=True)

    preds_path = os.path.join(ROOT, "data", "eval",
                              f"preds_l1_d{args.demos}"
                              + (f".t{args.temperature}" if args.temperature else "")
                              + ".jsonl")
    correct = total = 0
    n_box = 0
    with open_artifact(preds_path, force=args.force, run=args.run) as fout:
        # --run versions the path, so the handle's name is the file that exists.
        out_path = fout.name
        for s in range(0, len(evals), args.batch):
            batch = evals[s : s + args.batch]
            texts_in = [build_prompt(demos[:args.demos], r["instruction"]) for r in batch]
            if fone_on:
                prompts, pvals = fone.encode_prompts(texts_in, tok, num_id)
            else:
                prompts, pvals = [tok.encode(t).ids for t in texts_in], None
            with torch.no_grad():
                out = generate_batch(model, prompts, args.max_new, args.device, args.temperature, pvals)
            out_ids, out_vals = out if fone_on else (out, [None] * len(batch))
            for r, ids, vs in zip(batch, out_ids, out_vals):
                gen = fone.decode_text(ids, vs, tok, num_id) if fone_on else tok.decode(ids)
                ok = score(gen, r["output"])
                correct += int(ok)
                total += 1
                n_box += int("\\boxed" in gen or "答案是" in gen)
                fout.write(json.dumps({"q": r["instruction"], "gen": gen, "ok": ok},
                                      ensure_ascii=False) + "\n")
            if total % 64 < args.batch or total == len(evals):
                print(f"  {total}/{len(evals)} acc={correct / total:.1%}", flush=True)

    # attest what was WRITTEN, not what was requested: --run versions the path, and
    # attesting preds_path recorded a hash for a file this run never touched.
    attest(out_path)  # the citation contract: the writer proves these bytes existed
    delta = 1.4 / (total ** 0.5)
    acc = correct / total
    print(f"L1 math-500 few-shot: {correct}/{total} = {acc:.1%}")
    print(f"binomial delta={delta:.1%} -> 2*delta={2 * delta:.1%}; "
          f"instrument exists iff acc > {2 * delta:.1%}")
    print(f"answer-present rate {n_box / total:.1%}")
    print(f"preds saved: {preds_path}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"correct": correct, "total": total, "acc": acc,
                        "binomial_delta": delta, "answer_present_rate": n_box / total,
                        "demos": args.demos, "temperature": args.temperature,
                        "preds_path": preds_path}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
