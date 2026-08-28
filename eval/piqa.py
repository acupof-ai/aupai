#!/usr/bin/env python3
"""PIQA eval: log-likelihood scoring over 2 physical-reasoning solutions.

Usage: python eval/piqa.py [--limit N] [--device cuda]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.loader import load_checkpoint, load_tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "ckpt.pt")
TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")


def load_model(device="cuda"):
    model, cfg = load_checkpoint(CKPT, device=device)
    tok = load_tokenizer(TOK_PATH, cfg)
    return model, tok


@torch.no_grad()
def log_likelihood(model, tok, context, continuation, device):
    ctx_ids = tok.encode(context).ids
    cont_ids = tok.encode(continuation).ids
    full = torch.tensor([ctx_ids + cont_ids], device=device)
    logits, _ = model(full)
    log_probs = torch.log_softmax(logits[0].float(), dim=-1)
    cont_log_probs = log_probs[range(len(ctx_ids) - 1, len(ctx_ids) + len(cont_ids) - 1), cont_ids]
    return cont_log_probs.sum().item()


def load_dataset():
    from datasets import load_dataset as hf_load

    return hf_load("lighteval/piqa", "plain_text", split="validation")


def evaluate(model, tok, device="cuda", limit=None):
    ds = load_dataset()
    correct = 0
    total = 0
    for d in ds:
        if limit and total >= limit:
            break
        goal = d["goal"].rstrip()
        scores = [
            log_likelihood(model, tok, goal, " " + d["sol1"], device),
            log_likelihood(model, tok, goal, " " + d["sol2"], device),
        ]
        pred = max(range(2), key=lambda i: scores[i])
        correct += int(pred == d["label"])
        total += 1
        if total % 200 == 0:
            print(f"  {total} questions, acc={correct / total:.1%}", flush=True)
    acc = correct / max(total, 1)
    print(f"PIQA: {acc:.1%} ({correct}/{total})")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, tok = load_model(args.device)
    print(f"Model loaded, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    evaluate(model, tok, args.device, args.limit)
