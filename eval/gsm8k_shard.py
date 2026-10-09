#!/usr/bin/env python3
"""Sharded GSM8K ChatML eval for one SFT checkpoint. Mirrors gsm8k.evaluate's scoring
(CUTS truncation, last-number match) but scores only a fixed row shard and writes a
{correct,total} partial the bash driver merges. Usage:
  gsm8k_shard.py --ckpt X --shard_i i --shard_n N --out runs/.../gsm_part_i.json
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from scripts.loader import load_checkpoint, load_tokenizer, prompt_fn  # noqa: E402
from score_matrix import classify  # noqa: E402
from eval.gsm8k import load_dataset, extract_number, CUTS, generate_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard_i", type=int, required=True)
    ap.add_argument("--shard_n", type=int, required=True)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model, cfg = load_checkpoint(a.ckpt, device="cuda")
    model = model.to(torch.bfloat16)
    tok = load_tokenizer("data/tokenizer.json", cfg)
    fmt = prompt_fn(classify(cfg, os.path.basename(a.ckpt)))
    if fmt.__name__ != "format_prompt":
        print(f"WARN fmt={fmt.__name__}, expected format_prompt (ChatML)", flush=True)

    rows = list(load_dataset())
    rows = [r for idx, r in enumerate(rows) if idx % a.shard_n == a.shard_i]
    correct = total = 0
    BS = 64
    for s in range(0, len(rows), BS):
        batch = rows[s:s + BS]
        p_ids = [tok.encode(fmt(r["question"])).ids for r in batch]
        golds = [float(r["answer"].split("####")[-1].replace(",", "").strip()) for r in batch]
        outs = generate_batch(model, p_ids, a.max_new, "cuda", 0.0)
        for out_ids, gold in zip(outs, golds, strict=True):
            text = tok.decode(out_ids)
            for c in CUTS:
                text = text.split(c)[0]
            pred = extract_number(text)
            total += 1
            if pred is not None and abs(pred - gold) < 1e-4:
                correct += 1
        print(f"shard{a.shard_i} {total}/{len(rows)} acc={correct/max(1,total):.2%}", flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"shard_i": a.shard_i, "correct": correct, "total": total}, f)
    print(f"shard{a.shard_i} DONE {correct}/{total}")


if __name__ == "__main__":
    main()
