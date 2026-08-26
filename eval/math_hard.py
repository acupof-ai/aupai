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
    p.add_argument("--k", type=int, default=1, help="samples per problem; reports pass@1 (mean) and pass@k")
    p.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (k>1 needs >0)")
    a = p.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    cfg.grad_ckpt = False
    model = HybridLM(cfg).to(a.device)
    model.load_state_dict(ck["model"])
    model = model.to(torch.bfloat16).eval()
    tok = Tokenizer.from_file(TOK_PATH)

    rows = [json.loads(l) for l in open(TEST_PATH, encoding="utf-8")][a.shard :: a.shards]
    preds = os.path.join(
        ROOT,
        "data",
        "eval",
        f"hard_{os.path.basename(a.ckpt)}" + (f".{a.shard}" if a.shards > 1 else "") + ".jsonl",
    )
    k = max(1, a.k)
    temp = a.temperature if k > 1 or a.temperature > 0 else 0.0
    by = {}  # level -> [sum of per-problem pass@1, any-correct count, n problems]
    n_eq = n_bad = tot_len = 0
    per_batch = max(1, a.batch // k)  # k samples of one problem are laid out consecutively
    with open(preds, "w", encoding="utf-8") as f:
        for s in range(0, len(rows), per_batch):
            batch = rows[s : s + per_batch]
            prompts = [tok.encode(f"问：{r['instruction']}\n答：").ids for r in batch for _ in range(k)]
            with torch.no_grad():
                outs = generate_batch(model, prompts, a.max_new, a.device, temp)
            for i, r in enumerate(batch):
                oks = []
                for ids in outs[i * k : (i + 1) * k]:
                    gen = tok.decode(ids)
                    ok = score(gen, str(r["answer"]))
                    oks.append(ok)
                    e, b = check_steps(gen)
                    n_eq += e
                    n_bad += b
                    tot_len += len(ids)
                    f.write(
                        json.dumps(
                            {
                                "q": r["instruction"],
                                "answer": r["answer"],
                                "level": r["level"],
                                "gen": gen[-300:],
                                "ok": ok,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                lv = by.setdefault(r["level"], [0.0, 0, 0])
                lv[0] += sum(oks) / k
                lv[1] += int(any(oks))
                lv[2] += 1
    n = sum(v[2] for v in by.values())
    p1 = sum(v[0] for v in by.values())
    pk = sum(v[1] for v in by.values())
    print(
        f"math-hard: pass@1 {p1 / n:.1%} ({p1:.1f}/{n})"
        + (f" | pass@{k} {pk / n:.1%} ({pk}/{n})" if k > 1 else "")
    )
    print(
        "  "
        + ", ".join(
            f"{lvl}: p@1 {v[0] / v[2]:.0%}" + (f" p@{k} {v[1] / v[2]:.0%}" if k > 1 else "") + f" (n={v[2]})"
            for lvl, v in sorted(by.items())
        )
    )
    print(f"  avg gen tokens {tot_len / (n * k):.0f} | step-eq wrong {n_bad}/{n_eq}")


if __name__ == "__main__":
    main()
