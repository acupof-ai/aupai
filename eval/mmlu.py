"""MMLU evaluation via multiple-choice log-likelihood scoring.

57 subjects, 4 options each. Prompt = question + "A. .. B. .. C. .. D. ..",
score the log-likelihood of each continuation letter, pick argmax.
"""
import sys
from collections import defaultdict
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer

sys.path.insert(0, "/work/aupai")
from train import HybridLM

LETTERS = ["A", "B", "C", "D"]


def load_dataset():
    from datasets import load_dataset
    return load_dataset("cais/mmlu", "all", split="test")


@torch.no_grad()
def evaluate(model, tok, device):
    ds = load_dataset()
    per_subj = defaultdict(lambda: [0, 0])  # subject -> [correct, total]
    correct = total = 0

    for q in ds:
        opts = " ".join(f"{L}. {c}" for L, c in zip(LETTERS, q["choices"]))
        prompt = f"{q['question']} {opts}"
        p_ids = tok.encode(prompt).ids

        scores = {}
        for L in LETTERS:
            a_ids = tok.encode(L).ids
            x = torch.tensor([p_ids + a_ids], device=device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            log_probs = torch.log_softmax(logits[0], dim=-1)
            scores[L] = sum(
                log_probs[len(p_ids) + i - 1, t].item() for i, t in enumerate(a_ids)
            )

        pred = LETTERS.index(max(scores, key=scores.get))
        gold = q["answer"]
        subj = q["subject"]
        per_subj[subj][1] += 1
        total += 1
        if pred == gold:
            per_subj[subj][0] += 1
            correct += 1

    acc = correct / total
    print(f"MMLU overall: {correct}/{total} = {acc:.2%}")
    print("Top 5 subjects:")
    for subj, (c, n) in sorted(
        per_subj.items(), key=lambda kv: kv[1][0] / kv[1][1], reverse=True
    )[:5]:
        print(f"  {subj}: {c}/{n} = {c / n:.2%}")
    return acc


if __name__ == "__main__":
    ck = torch.load("ckpt_sft.pt", map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = HybridLM(cfg).cuda().bfloat16()
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file("data/tokenizer.json")
    for t in ["<|im_start|>", "<|im_end|>"]:
        if tok.token_to_id(t) is None:
            tok.add_special_tokens([t])
    evaluate(model, tok, "cuda")
