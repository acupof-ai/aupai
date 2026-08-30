#!/usr/bin/env python3
"""Math eval on the 500-problem held-out set (data/eval/math_test_500.jsonl).

Greedy generation with the SFT prompt format, \\boxed{} extraction via
algorithms.rlvr_reward (falls back to 答案是：...), exact/numeric match.

Usage: python eval/math_zh.py --ckpt ckpt_sft_math.pt [--max_new 512] [--batch 16]
"""
import argparse
import json
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("FLA_FLASH_KDA", "0")

import fone  # noqa: E402
from eval.gsm8k import generate_batch  # noqa: E402
from algorithms.rlvr_reward import reward_fn, extract_boxed  # noqa: E402
from scripts.loader import format_prompt, load_checkpoint, load_tokenizer  # noqa: E402

TEST_PATH = os.path.join(ROOT, "data", "eval", "math_test_500.jsonl")
TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
from scripts.eqcheck import check_steps  # noqa: E402

ANS_RE = re.compile(r"答案是[:：]\s*(.+?)(?:[。\n]|$)")


def score(gen, gold):
    """gold is the full solution text; extract its boxed answer first."""
    gold_ans = extract_boxed(gold)
    if gold_ans is None:
        return 0.0
    if extract_boxed(gen) is not None:
        return reward_fn(gen, gold_ans)
    m = ANS_RE.search(gen)
    if m:
        return reward_fn(f"\\boxed{{{m.group(1).strip()}}}", gold_ans)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--max_new", type=int, default=512)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--shards", type=int, default=1, help="split the test set across N processes")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer", default=TOK_PATH, help="vocabulary the checkpoint was trained on")
    args = parser.parse_args()

    # dtype through load_checkpoint (a3a0de0 upcasts KDA A_log/dt_bias to fp32
    # after the cast); a separate .to(bf16) here would undo the upcast.
    model, cfg = load_checkpoint(args.ckpt, device=args.device, dtype=torch.bfloat16)
    tok = load_tokenizer(args.tokenizer, cfg)
    # A FoNE checkpoint must decode through fone: tok.decode emits the [NUM] token
    # itself, so no answer parses and the score collapses for a non-model reason.
    fone_on = getattr(cfg, "fone", False)
    num_id = getattr(cfg, "num_id", None)

    rows = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8")]
    rows = rows[args.shard :: args.shards]
    preds_path = os.path.join(
        ROOT, "data", "eval", f"preds_{os.path.basename(args.ckpt)}"
        + (f".{args.shard}" if args.shards > 1 else "")
        + ".jsonl"
    )
    correct = total = 0
    n_box = n_eq = n_bad = n_rewrite = tot_len = 0
    by_steps = {}
    with open(preds_path, "w", encoding="utf-8") as fout:
        for s in range(0, len(rows), args.batch):
            batch = rows[s : s + args.batch]
            texts_in = [format_prompt(r["instruction"]) for r in batch]
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
                n_box += extract_boxed(gen) is not None
                n_rewrite += "解答" in gen
                tot_len += len(ids)
                e, b = check_steps(gen)
                n_eq += e; n_bad += b
                k = min(check_steps(r["output"])[0], 3)  # difficulty bucket by gold step count
                by_steps.setdefault(k, [0, 0]); by_steps[k][0] += int(ok); by_steps[k][1] += 1
                fout.write(json.dumps({"q": r["instruction"], "gold": r["output"][-80:],
                                       "gen": gen[-300:], "ok": ok}, ensure_ascii=False) + "\n")
            if total % 64 == 0 or total == len(rows):
                print(f"  {total}/{len(rows)} acc={correct / total:.1%}", flush=True)

    print(f"math-500: {correct}/{total} = {correct / total:.1%}")
    print(f"boxed rate {n_box / total:.1%} | rewrite('解答') rate {n_rewrite / total:.1%} | "
          f"avg gen tokens {tot_len / total:.0f} | step-eq wrong {n_bad}/{n_eq} = {n_bad / max(n_eq, 1):.1%}")
    print("acc by gold steps: " + ", ".join(f"{k}{'+' if k == 3 else ''}: {c}/{n}={c / n:.0%}" for k, (c, n) in sorted(by_steps.items())))
    print(f"preds saved: {preds_path}")


if __name__ == "__main__":
    main()
