#!/usr/bin/env python3
"""OpenBookQA eval: 4-option log-likelihood scoring.

Usage: python eval/openbookqa.py [--n 500] [--device cuda]
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval import log_likelihood_joint
from scripts.loader import load_checkpoint, load_tokenizer

TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")


def load_model(ckpt_path, device="cuda"):
    model, cfg = load_checkpoint(ckpt_path, device=device)
    tok = load_tokenizer(TOK_PATH, cfg)
    return model, tok, cfg


def load_dataset():
    from datasets import load_dataset as hf_load_dataset

    # Test split has no public labels; validation is the standard labeled eval set.
    return hf_load_dataset("allenai/openbookqa", "main", split="validation", streaming=True)


def evaluate(model, tok, device="cuda", n=None):
    ds = load_dataset()
    correct = total = 0
    for d in ds:
        if n is not None and total >= n:
            break
        labels = d["choices"]["label"]
        texts = d["choices"]["text"]
        prompt = (
            f"Question: {d['question_stem']}\n"
            + "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            + "\nAnswer:"
        )
        scores = [log_likelihood_joint(model, tok, prompt, f" {l}", device) for l in labels]
        pred = labels[scores.index(max(scores))]
        if pred == d["answerKey"]:
            correct += 1
        total += 1
        if total % 100 == 0:
            print(f"  {total} questions, acc={correct / total:.1%}", flush=True)

    acc = correct / max(total, 1)
    print(f"\n=== OpenBookQA ===\n{acc:.1%} ({correct}/{total})")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join(ROOT, "ckpt.pt"))
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, tok, _ = load_model(args.ckpt, args.device)
    print(f"Model loaded, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    evaluate(model, tok, args.device, args.n)
