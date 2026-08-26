#!/usr/bin/env python3
"""OpenBookQA eval: 4-option log-likelihood scoring.

Usage: python eval/openbookqa.py [--n 500] [--device cuda]
"""

import argparse
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from train import HybridLM

TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|think|>", "<|/think|>"]


def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = HybridLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(TOK_PATH)
    for t in SPECIAL_TOKENS:
        if tok.token_to_id(t) is None:
            tok.add_special_tokens([t])
    return model, tok, cfg


def log_likelihood(model, tok, prompt, choice, device="cuda"):
    """Log-likelihood of choice tokens given prompt."""
    ids_p = tok.encode(prompt).ids
    ids_f = tok.encode(prompt + choice).ids
    if len(ids_f) <= len(ids_p):
        return -1e9
    x = torch.tensor([ids_f], device=device)
    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        log_probs = F.log_softmax(logits[0], dim=-1)
    return sum(
        log_probs[len(ids_p) + i - 1, tid].item()
        for i, tid in enumerate(ids_f[len(ids_p):])
        if len(ids_p) + i - 1 >= 0
    )


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
        scores = [log_likelihood(model, tok, prompt, f" {l}", device) for l in labels]
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
