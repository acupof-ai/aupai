#!/usr/bin/env python3
"""Prepare SFT data: 问：{instruction}\n答：{output}<eos>, prompt-masked, packed.

Reads all SFT sources, tokenizes with data/tokenizer.json, masks instruction
tokens (labels=-100), packs into (seq+1)-token rows, and saves
data/sft/sft_all.pt as {"input_ids": int32 (N, seq+1), "labels": int32 (N, seq+1)}.
labels[t] = input_ids[t] for output/eos tokens, -100 for prompt tokens.
Training slices x=[:, :-1], y=labels[:, 1:].
"""

import json
import os
import random

import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TOK_PATH = os.path.join(DATA, "tokenizer.json")
OUT_PATH = os.path.join(DATA, "sft", "sft_all.pt")

SEQ = 4096  # model context; rows are SEQ+1 (input + 1 to predict)
MAX_EXAMPLES = 500_000
ENC_BATCH = 8192

# (path, question_key, answer_key)
SOURCES = [
    (os.path.join(DATA, "alpaca_gpt4_zh.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "coig.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "openo1_sft.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "gsm8k_zh.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "school_math_r1_zh.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "s1k.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "sft", "fable5_cot.jsonl"), "prompt", "response"),
    (os.path.join(DATA, "synthetic", "code_python_zh.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "synthetic", "knowledge_qa_zh.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "synthetic", "math_gsm8k_zh.jsonl"), "instruction", "output"),
]


def read_examples():
    """Yield (prompt, output) text pairs from all sources."""
    for path, qk, ak in SOURCES:
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                q = (d.get(qk) or "").strip()
                a = (d.get(ak) or "").strip()
                inp = (d.get("input") or "").strip()
                if inp:
                    q = f"{q}\n{inp}"
                if not q or not a:
                    continue
                yield f"问：{q}\n答：", a
                n += 1
        print(f"  {os.path.basename(path)}: {n}", flush=True)


def main():
    random.seed(42)
    tok = Tokenizer.from_file(TOK_PATH)
    eos = tok.token_to_id("<eos>")
    assert eos is not None, "tokenizer has no <eos>"

    examples = list(read_examples())
    random.shuffle(examples)
    if len(examples) > MAX_EXAMPLES:
        examples = examples[:MAX_EXAMPLES]
    print(f"total examples: {len(examples)}", flush=True)

    ids_stream = []
    lab_stream = []
    n_trunc = 0
    n_mismatch = 0
    row_len = SEQ + 1

    for i in range(0, len(examples), ENC_BATCH):
        batch = examples[i : i + ENC_BATCH]
        prompts = [p for p, _ in batch]
        fulls = [p + a for p, a in batch]
        enc_p = tok.encode_batch(prompts)
        enc_f = tok.encode_batch(fulls)
        for ep, ef in zip(enc_p, enc_f):
            ids_p, ids_f = ep.ids, ef.ids + [eos]
            # byte-level BPE without prefix space: prompt is an exact prefix of full
            plen = len(ids_p)
            if ids_f[:plen] != ids_p:
                n_mismatch += 1
                plen = 0
                for a, b in zip(ids_p, ids_f):
                    if a != b:
                        break
                    plen += 1
            if len(ids_f) > row_len:
                # keep the tail (answer end + eos); adjust mask boundary
                plen = max(0, plen - (len(ids_f) - row_len))
                ids_f = ids_f[-row_len:]
                n_trunc += 1
            ids_stream.extend(ids_f)
            lab_stream.extend([-100] * plen + ids_f[plen:])
        if (i // ENC_BATCH) % 10 == 0:
            print(f"  tokenized {min(i + ENC_BATCH, len(examples))}/{len(examples)}", flush=True)

    n_rows = len(ids_stream) // row_len
    ids_stream = ids_stream[: n_rows * row_len]
    lab_stream = lab_stream[: n_rows * row_len]
    input_ids = torch.tensor(ids_stream, dtype=torch.int32).view(n_rows, row_len)
    labels = torch.tensor(lab_stream, dtype=torch.int32).view(n_rows, row_len)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    tmp = OUT_PATH + ".tmp"
    torch.save({"input_ids": input_ids, "labels": labels}, tmp)
    os.replace(tmp, OUT_PATH)

    total_tokens = input_ids.numel()
    loss_tokens = (labels != -100).sum().item()
    print(f"kept examples: {len(examples)}", flush=True)
    print(f"truncated (> {row_len} tokens): {n_trunc}", flush=True)
    print(f"prefix mismatches: {n_mismatch}", flush=True)
    print(f"packed rows: {n_rows} ({row_len} tokens each)", flush=True)
    print(f"total tokens: {total_tokens / 1e6:.2f}M", flush=True)
    print(f"loss tokens: {loss_tokens / 1e6:.2f}M ({100 * loss_tokens / total_tokens:.1f}%)", flush=True)
    print(f"saved {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
