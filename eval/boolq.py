#!/usr/bin/env python3
"""BoolQ eval: log-likelihood scoring for yes/no questions.

Usage: python eval/boolq.py [--n 1000] [--device cuda]
"""

import argparse
import os
import sys
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval import log_likelihood_joint
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


def load_dataset():
    from datasets import load_dataset as hf_load_dataset

    return hf_load_dataset("google/boolq", split="validation", streaming=True)


def evaluate(model, tok, device="cuda", n=None):
    ds = load_dataset()
    correct = total = 0
    for d in ds:
        if n is not None and total >= n:
            break
        prompt = f"Passage: {d['passage']}\nQuestion: {d['question']}?\nAnswer:"
        scores = [log_likelihood_joint(model, tok, prompt, f" {a}", device) for a in ("no", "yes")]
        pred = bool(scores.index(max(scores)))  # 0=no, 1=yes
        if pred == d["answer"]:
            correct += 1
        total += 1
        if total % 100 == 0:
            print(f"  {total} questions, acc={correct / total:.1%}", flush=True)

    acc = correct / max(total, 1)
    print(f"\n=== BoolQ ===\n{acc:.1%} ({correct}/{total})")
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
