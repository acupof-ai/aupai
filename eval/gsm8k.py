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
import fone
from scripts.loader import format_prompt, load_checkpoint, load_tokenizer

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
def generate_batch(model, prompts, max_new, device, temperature=0.0, prompt_values=None):
    """Greedy (temperature=0) or sampled decoding for a list of token-id lists. Returns generated ids.

    prompt_values switches on the FoNE path: a per-position value list for each prompt.
    Then the return is (ids, values) per row, because a [NUM] token carries no number
    of its own -- the digit head reads it off the same hidden state that predicted the
    token, and fone.decode_text writes it back into the text.
    """
    B = len(prompts)
    keep = max(0, MAX_CTX - max_new)  # prompt budget; 0 means "keep all" (p[-0:] == p[0:])
    fone_on = prompt_values is not None
    if fone_on:
        prompt_values = [v[-keep:] for v in prompt_values]
    prompts = [p[-keep:] for p in prompts]
    lengths = [len(p) for p in prompts]
    x = torch.full((B, max(lengths)), EOS_ID, dtype=torch.long, device=device)
    v = torch.zeros((B, max(lengths)), device=device) if fone_on else None
    for i, p in enumerate(prompts):
        x[i, : lengths[i]] = torch.tensor(p, device=device)
        if fone_on:
            v[i, : lengths[i]] = torch.tensor(prompt_values[i], device=device)
    ends = torch.tensor(lengths, device=device)  # next write position per row
    done = torch.zeros(B, dtype=torch.bool, device=device)
    ar = torch.arange(B, device=device)
    num_id = model.cfg.num_id if fone_on else None

    for _ in range(max_new):
        logits, hidden = model(
            x[:, -MAX_CTX:], num_vals=v[:, -MAX_CTX:] if fone_on else None, return_hidden=fone_on
        )
        # logits only covers the last MAX_CTX positions; index relative to that slice
        off = max(0, x.size(1) - MAX_CTX)
        step_logits = logits[ar, ends - off - 1]
        if temperature > 0:
            nxt = torch.multinomial(torch.softmax(step_logits.float() / temperature, dim=-1), 1).squeeze(1)
        else:
            nxt = step_logits.argmax(dim=-1)
        if x.size(1) <= int(ends.max()):
            x = torch.cat([x, torch.full((B, 1), EOS_ID, dtype=torch.long, device=device)], dim=1)
            if fone_on:
                v = torch.cat([v, torch.zeros((B, 1), device=device)], dim=1)
        x[ar, ends] = torch.where(done, torch.full_like(nxt, EOS_ID), nxt)
        if fone_on:
            val = fone.decode(model.num_logits(hidden[ar, ends - off - 1].float())).to(v.dtype)
            v[ar, ends] = torch.where(nxt == num_id, val, torch.zeros_like(val))
        ends += (~done).long()
        done |= nxt == EOS_ID
        if bool(done.all()):
            break
    ids = [x[i, lengths[i] : ends[i]].tolist() for i in range(B)]
    if not fone_on:
        return ids
    vals = [v[i, lengths[i] : ends[i]][x[i, lengths[i] : ends[i]] == num_id].tolist() for i in range(B)]
    return ids, vals


@torch.no_grad()
def evaluate(model, tok, device, batch_size=8):
    rows = list(load_dataset())
    correct = total = 0

    for s in range(0, len(rows), batch_size):
        batch = rows[s : s + batch_size]
        p_ids = [tok.encode(format_prompt(r["question"])).ids for r in batch]
        golds = [float(r["answer"].split("####")[-1].replace(",", "").strip()) for r in batch]

        for out_ids, gold in zip(generate_batch(model, p_ids, 256, device), golds, strict=True):
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
    model, cfg = load_checkpoint("ckpt_sft.pt", device="cuda")
    model = model.to(torch.bfloat16)
    tok = load_tokenizer("data/tokenizer.json", cfg)
    evaluate(model, tok, "cuda")
