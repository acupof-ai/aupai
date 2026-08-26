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
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import HybridLM

EOS_ID = 1
MAX_CTX = 1024
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def load_dataset():
    from datasets import load_dataset

    return load_dataset("openai/gsm8k", "main", split="test")


def extract_number(text):
    """Last number in text (commas stripped); None if absent."""
    nums = NUM_RE.findall(text)
    return float(nums[-1].replace(",", "")) if nums else None


@torch.no_grad()
def generate_batch(model, prompts, max_new, device, temperature=0.0):
    """Greedy (temperature=0) or sampled decoding for a list of token-id lists. Returns generated ids."""
    B = len(prompts)
    prompts = [p[-(MAX_CTX - max_new) :] for p in prompts]
    lengths = [len(p) for p in prompts]
    x = torch.full((B, max(lengths)), EOS_ID, dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        x[i, : lengths[i]] = torch.tensor(p, device=device)
    ends = torch.tensor(lengths, device=device)  # next write position per row
    done = torch.zeros(B, dtype=torch.bool, device=device)
    ar = torch.arange(B, device=device)

    for _ in range(max_new):
        logits, _ = model(x[:, -MAX_CTX:])
        step_logits = logits[ar, ends - 1]
        if temperature > 0:
            nxt = torch.multinomial(torch.softmax(step_logits.float() / temperature, dim=-1), 1).squeeze(1)
        else:
            nxt = step_logits.argmax(dim=-1)
        if x.size(1) <= int(ends.max()):
            x = torch.cat([x, torch.full((B, 1), EOS_ID, dtype=torch.long, device=device)], dim=1)
        x[ar, ends] = torch.where(done, torch.full_like(nxt, EOS_ID), nxt)
        ends += (~done).long()
        done |= nxt == EOS_ID
        if bool(done.all()):
            break
    return [x[i, lengths[i] : ends[i]].tolist() for i in range(B)]


@torch.no_grad()
def evaluate(model, tok, device, batch_size=8):
    rows = list(load_dataset())
    correct = total = 0

    for s in range(0, len(rows), batch_size):
        batch = rows[s : s + batch_size]
        p_ids = [tok.encode(f"问：{r['question']}\n答：").ids for r in batch]
        golds = [float(r["answer"].split("####")[-1].replace(",", "").strip()) for r in batch]

        for out_ids, gold in zip(generate_batch(model, p_ids, 256, device), golds):
            pred = extract_number(tok.decode(out_ids))
            total += 1
            if pred is not None and abs(pred - gold) < 1e-4:
                correct += 1

        if total % 128 == 0 or total == len(rows):
            print(f"  {total}/{len(rows)} acc={correct / total:.2%}", flush=True)

    acc = correct / total
    print(f"GSM8K: {correct}/{total} = {acc:.2%}")
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
