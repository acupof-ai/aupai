#!/usr/bin/env python3
"""ARC eval: log-likelihood scoring for ARC-Easy and ARC-Challenge.

Usage: python eval/arc.py [--limit N] [--device cuda]
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

CONFIGS = {"ARC-E": "ARC-Easy", "ARC-C": "ARC-Challenge"}


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


def load_dataset(config):
    from datasets import load_dataset as hf_load

    return hf_load("allenai/ai2_arc", config, split="test")


def evaluate_config(model, tok, config, device="cuda", limit=None):
    ds = load_dataset(config)
    correct = 0
    total = 0
    for d in ds:
        if limit and total >= limit:
            break
        question = d["question"].rstrip()
        choices = d["choices"]["text"]
        labels = d["choices"]["label"]
        scores = [log_likelihood(model, tok, question, " " + c, device) for c in choices]
        pred = labels[max(range(len(choices)), key=lambda i: scores[i])]
        correct += int(pred == d["answerKey"])
        total += 1
        if total % 200 == 0:
            print(f"  {config}: {total} questions, acc={correct / total:.1%}", flush=True)
    return correct / max(total, 1)


def evaluate(model, tok, device="cuda", limit=None):
    results = {}
    for name, config in CONFIGS.items():
        acc = evaluate_config(model, tok, config, device, limit)
        print(f"{name}: {acc:.1%}")
        results[name] = acc
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, tok = load_model(args.device)
    print(f"Model loaded, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(evaluate(model, tok, args.device, args.limit))
