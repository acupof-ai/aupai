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

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("FLA_FLASH_KDA", "0")

import fone  # noqa: E402
from eval.gsm8k import generate_batch  # noqa: E402
from eval.math_zh import ANS_RE, check_steps  # noqa: E402
from algorithms.rlvr_reward import reward_fn, extract_boxed  # noqa: E402
from scripts.loader import format_prompt, load_checkpoint, load_tokenizer  # noqa: E402

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
    p.add_argument("--data", default=None, help="problem jsonl (default: the math-hard holdout)")
    p.add_argument(
        "--tokenizer",
        default=TOK_PATH,
        help="a checkpoint must be scored with the vocabulary it was trained on. data/tokenizer.json "
        "is rebuilt in place, so an older checkpoint needs its own file passed here -- ids do not "
        "survive a vocabulary rebuild, and nothing in the checkpoint detects the mismatch beyond "
        "cfg.vocab.",
    )
    p.add_argument(
        "--dump",
        default=None,
        help="write {instruction, greedy, gens} per problem here, for the solve-rate probe",
    )
    a = p.parse_args()

    model, cfg = load_checkpoint(a.ckpt, device=a.device)
    model = model.to(torch.bfloat16)
    tok = load_tokenizer(a.tokenizer, cfg)
    # A FoNE checkpoint writes numbers as [NUM] carrying a value, so prompt and
    # output go through fone, not the tokenizer alone.
    fone_on = getattr(cfg, "fone", False)
    num_id = getattr(cfg, "num_id", None)

    rows = [json.loads(l) for l in open(a.data or TEST_PATH, encoding="utf-8")][a.shard :: a.shards]
    preds = os.path.join(
        ROOT,
        "data",
        "eval",
        f"hard_{os.path.basename(a.ckpt)}" + (f".{a.shard}" if a.shards > 1 else "") + ".jsonl",
    )
    k = max(1, a.k)
    temp = a.temperature if k > 1 or a.temperature > 0 else 0.0
    # pass@k at temperature 0 draws k identical greedy answers, so pass@k == pass@1
    # by construction; eval_hard.sh defaults TEMP=0.
    assert not (k > 1 and temp <= 0), (
        f"--k {k} at temperature {temp}: the k samples would be identical to the greedy "
        "answer and pass@k would equal pass@1 by construction. Pass --temperature (0.8 is "
        "the project's pass@k setting) or TEMP=0.8 through eval/eval_hard.sh."
    )
    # pass@1 is the greedy answer; the k samples feed only pass@k and the sampled
    # mean, so pass@k - pass@1 has no sampling noise on the pass@1 side.
    by = {}  # level -> [greedy correct, sum of sampled acc, any-correct, n]
    n_eq = n_bad = tot_len = n_gen = 0
    per_batch = max(1, a.batch // k)
    dump = open(a.dump, "w", encoding="utf-8") if a.dump else None
    with open(preds, "w", encoding="utf-8") as f:
        for s in range(0, len(rows), per_batch):
            batch = rows[s : s + per_batch]
            texts_in = [format_prompt(r["instruction"]) for r in batch]
            if fone_on:
                base, base_v = fone.encode_prompts(texts_in, tok, num_id)
            else:
                base, base_v = [tok.encode(t).ids for t in texts_in], None
            with torch.no_grad():
                greedy = generate_batch(model, base, a.max_new, a.device, 0.0, base_v)
                sampled = []
                if k > 1:
                    rep = [p for p in base for _ in range(k)]
                    rep_v = [v for v in base_v for _ in range(k)] if fone_on else None
                    sampled = generate_batch(model, rep, a.max_new, a.device, temp, rep_v)
            if fone_on:
                greedy, greedy_v = greedy
                sampled, sampled_v = sampled if k > 1 else ([], [])
            for i, r in enumerate(batch):
                oks, texts = [], []
                pairs = [(greedy[i], greedy_v[i] if fone_on else None)] + [
                    (sampled[j], sampled_v[j] if fone_on else None)
                    for j in range(i * k, (i + 1) * k)
                    if k > 1
                ]
                for ids, vs in pairs:
                    gen = fone.decode_text(ids, vs, tok, num_id) if fone_on else tok.decode(ids)
                    ok = score(gen, str(r["answer"]))
                    oks.append(ok)
                    texts.append(gen)
                    e, b = check_steps(gen)
                    n_eq += e
                    n_bad += b
                    tot_len += len(ids)
                    n_gen += 1
                    f.write(
                        json.dumps(
                            {
                                "q": r["instruction"],
                                "answer": r["answer"],
                                "level": r["level"],
                                "gen": gen,
                                "ok": ok,
                                "greedy": len(oks) == 1,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if dump:  # grouped per problem: greedy separate from the sampled ones
                    dump.write(
                        json.dumps(
                            {k2: r[k2] for k2 in ("program_id", "level", "answer") if k2 in r}
                            | {"instruction": r["instruction"], "greedy": texts[0], "gens": texts[1:]},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                lv = by.setdefault(r["level"], [0, 0.0, 0, 0])
                lv[0] += int(oks[0])
                lv[1] += sum(oks[1:]) / k if k > 1 else 0.0
                lv[2] += int(any(oks[1:]))  # SAMPLED only: greedy is not one of the k draws
                lv[3] += 1
            # restartable + progress: a 3-hour pass@k eval silent until the summary line
            # both stalls the harness log-silent monitor and hides where it is.
            _done = sum(v[3] for v in by.values())
            if _done % 64 == 0 or _done == len(rows):
                print(f"  {_done}/{len(rows)} pass@1={sum(v[0] for v in by.values()) / _done:.1%}", flush=True)
    if dump:
        dump.close()
    n = sum(v[3] for v in by.values())
    p1 = sum(v[0] for v in by.values())
    ps = sum(v[1] for v in by.values())
    pk = sum(v[2] for v in by.values())
    line = f"math-hard: pass@1(greedy) {p1 / n:.1%} ({p1}/{n})"
    if k > 1:
        line += (
            f" | sampled@T={temp} mean {ps / n:.1%} | pass@{k} {pk / n:.1%} ({pk}/{n})"
            f" | gap {(pk - p1) / n:+.1%} | T={temp}"
        )
    print(line)
    print(
        "  "
        + ", ".join(
            f"{lvl}: p@1 {v[0] / v[3]:.0%}" + (f" p@{k} {v[2] / v[3]:.0%}" if k > 1 else "") + f" (n={v[3]})"
            for lvl, v in sorted(by.items())
        )
    )
    print(f"  avg gen tokens {tot_len / n_gen:.0f} | step-eq wrong {n_bad}/{n_eq}")


if __name__ == "__main__":
    main()
