"""GSM8K evaluation via greedy generation (Chinese prompts).

1319 grade-school math problems. Prompt = "问：{q}\n答：", generate up to 256
tokens greedily, take the last number in the response, compare to "#### N".
Batched generation (batch of 8); prompts are right-padded, which is safe
because pad tokens always sit right of real content and causal attention
never looks right.
"""

import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import generate_batch  # noqa: F401  (re-exported: math_hard.py and math_zh.py import it from here)
from scripts.loader import load_checkpoint, load_tokenizer, prompt_fn

EOS_ID = 1
MAX_CTX = 4096  # the model's trained seq len; smaller truncates the model's own long reasoning away
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def load_dataset():
    from datasets import load_dataset

    return load_dataset("openai/gsm8k", "main", split="test")


def extract_number(text):
    """Last number in text (commas stripped); None if absent."""
    nums = NUM_RE.findall(text)
    return float(nums[-1].replace(",", "")) if nums else None


@torch.no_grad()
def evaluate(model, tok, device, batch_size=8, temperature=0.0, fmt=None):
    # No default format. `fmt=format_prompt` would be the defect with a friendlier face:
    # a base checkpoint scored in ChatML reads zero on an unseen prefix, not on capability
    # (AGENTS.md:200; 1.6% vs 94.4% fence rate, eval/score_code_exec.py:9-31). The caller
    # holds the cfg, so the caller decides -- prompt_fn(classify(cfg, name)).
    if fmt is None:
        raise ValueError("evaluate() needs fmt=prompt_fn(classify(cfg, name)); a default "
                         "would silently score a base checkpoint in ChatML")
    rows = list(load_dataset())
    correct = total = 0

    for s in range(0, len(rows), batch_size):
        batch = rows[s : s + batch_size]
        p_ids = [tok.encode(fmt(r["question"])).ids for r in batch]
        golds = [float(r["answer"].split("####")[-1].replace(",", "").strip()) for r in batch]

        for out_ids, gold in zip(generate_batch(model, p_ids, 256, device, temperature), golds, strict=True):
            pred = extract_number(tok.decode(out_ids))
            total += 1
            if pred is not None and abs(pred - gold) < 1e-4:
                correct += 1

        if total % 128 == 0 or total == len(rows):
            print(f"  {total}/{len(rows)} acc={correct / total:.2%}", flush=True)

    acc = correct / total
    print(f"GSM8K: {correct}/{total} = {acc:.2%} (t={temperature}, fmt={fmt.__name__})")
    return acc


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from score_matrix import classify

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "ckpt_sft.pt"
    model, cfg = load_checkpoint(ckpt, device="cuda")
    model = model.to(torch.bfloat16)
    tok = load_tokenizer("data/tokenizer.json", cfg)
    # classify, not an assumption: the old default was ckpt_sft.pt, so pointing this at a
    # base checkpoint by hand silently scored it in ChatML.
    evaluate(model, tok, "cuda", fmt=prompt_fn(classify(cfg, os.path.basename(ckpt))))
