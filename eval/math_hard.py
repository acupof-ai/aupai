#!/usr/bin/env python3
"""Hard math eval (L3/L4) on data/synthetic/math_hard_eval_1k.jsonl.

Same greedy generation and \\boxed{} scoring as eval/math_zh.py, but the gold
answer comes from the verified `answer` field and results are broken out by level.

Usage: python eval/math_hard.py --ckpt ckpt_sft_v5.pt [--shards N --shard I]
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("FLA_FLASH_KDA", "0")

from train import HybridLM  # noqa: E402
from eval.gsm8k import generate_batch  # noqa: E402
from eval.math_zh import ANS_RE, check_steps  # noqa: E402
from algorithms.rlvr_reward import reward_fn, extract_boxed  # noqa: E402

TEST_PATH = os.path.join(ROOT, "data", "synthetic", "math_hard_eval_1k.jsonl")
TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")


def score(gen, gold_ans):
    if extract_boxed(gen) is not None:
        return reward_fn(gen, gold_ans)
    m = ANS_RE.search(gen)
    return reward_fn(f"\\boxed{{{m.group(1).strip()}}}", gold_ans) if m else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    a = p.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    cfg.grad_ckpt = False
    model = HybridLM(cfg).to(a.device)
    model.load_state_dict(ck["model"])
    model = model.to(torch.bfloat16).eval()
    tok = Tokenizer.from_file(TOK_PATH)

    rows = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8")][a.shard :: a.shards]
    preds = os.path.join(ROOT, "data", "eval", f"hard_{os.path.basename(a.ckpt)}"
                         + (f".{a.shard}" if a.shards > 1 else "") + ".jsonl")
    by = {}
    n_eq = n_bad = tot_len = 0
    with open(preds, "w", encoding="utf-8") as f:
        for s in range(0, len(rows), a.batch):
            batch = rows[s : s + a.batch]
            prompts = [tok.encode(f"问：{r['instruction']}\n答：").ids for r in batch]
            with torch.no_grad():
                outs = generate_batch(model, prompts, a.max_new, a.device)
            for r, ids in zip(batch, outs):
                gen = tok.decode(ids)
                ok = score(gen, str(r["answer"]))
                by.setdefault(r["level"], [0, 0])
                by[r["level"]][0] += int(ok)
                by[r["level"]][1] += 1
                e, b = check_steps(gen)
                n_eq += e
                n_bad += b
                tot_len += len(ids)
                f.write(json.dumps({"q": r["instruction"], "answer": r["answer"], "level": r["level"],
                                    "gen": gen[-300:], "ok": ok}, ensure_ascii=False) + "\n")
    c = sum(v[0] for v in by.values())
    n = sum(v[1] for v in by.values())
    print(f"math-hard: {c}/{n} = {c / n:.1%}")
    print("  " + ", ".join(f"{k}: {v[0]}/{v[1]}={v[0] / v[1]:.0%}" for k, v in sorted(by.items())))
    print(f"  avg gen tokens {tot_len / n:.0f} | step-eq wrong {n_bad}/{n_eq}")


if __name__ == "__main__":
    main()
