"""MMLU evaluation via multiple-choice log-likelihood scoring.

57 subjects, 4 options each. Prompt = question + "A. .. B. .. C. .. D. ..",
score the log-likelihood of each continuation letter, pick argmax.
"""
import sys
import os
from collections import defaultdict

import torch

sys.path.insert(0, "/work/aupai")
from scripts.loader import load_checkpoint, load_tokenizer

LETTERS = ["A", "B", "C", "D"]


MMLU_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "eval", "mmlu_test.jsonl")
# cais/mmlu "all" test split, 14,042 rows, 13-whitespace-gram screened against every r3 _dc
# domain 2026-09-14 (runs/contam_mmlu_r3.json). Auxiliary metric, deliberately NOT in the
# datagen holdout registry: that path forces the 5MiB holdout_hashes.txt over the tracked-blob
# cap (fb ruling 2026-09-14, option B), and the SFT pool is decontaminated independently.
MMLU_SHA1 = "d9c4079e4e04aec3ffcb0e636a77f43ab5f5f022"


def load_dataset():
    import hashlib
    import json

    with open(MMLU_PATH, "rb") as fh:
        raw = fh.read()
    got = hashlib.sha1(raw).hexdigest()
    if got != MMLU_SHA1:
        raise RuntimeError(
            f"{MMLU_PATH} sha1 {got} != screened {MMLU_SHA1}; refusing to score an "
            "unscreened MMLU copy. Rebuild from cais/mmlu 'all' test and rerun the 13-gram "
            "audit before changing MMLU_SHA1")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


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
    model, cfg = load_checkpoint("ckpt_sft.pt", device="cuda")
    model = model.to(torch.bfloat16)
    tok = load_tokenizer("data/tokenizer.json", cfg)
    evaluate(model, tok, "cuda")
