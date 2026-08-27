#!/usr/bin/env python3
"""Prepare SFT data: 问：{instruction}\n答：{output}<eos>, prompt-masked, packed.

Reads all SFT sources, tokenizes with data/tokenizer.json, masks instruction
tokens (labels=-100), greedily packs whole examples (never split across a row,
over-length dropped) into (seq+1)-token rows right-padded with <eos>, and saves
data/sft/sft_all.pt as {"input_ids": int32 (N, seq+1), "labels": int32 (N, seq+1)}.
labels[t] = input_ids[t] for output/eos tokens, -100 for prompt/pad tokens.
sft.py resets KDA state + SWA attention at every <eos> (Cfg.doc_mask), so the
clean per-document <eos> boundary this produces is what the doc mask keys on.
Training slices x=[:, :-1], y=labels[:, 1:].
"""

import json
import os
import random
import sys

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holdout import is_holdout  # noqa: E402

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
    """Yield (prompt, output) text pairs from all sources, excluding eval-holdout questions."""
    n_holdout = 0
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
                if is_holdout(q):  # never train on a question the eval holds out
                    n_holdout += 1
                    continue
                yield f"问：{q}\n答：", a
                n += 1
        print(f"  {os.path.basename(path)}: {n}", flush=True)
    if n_holdout:
        print(f"  excluded {n_holdout} eval-holdout questions", flush=True)


def pack_and_save(examples, tok, eos, out_path, seq):
    """Greedily pack (prompt, output) text pairs into (seq+1)-token rows and save.

    One example never split across rows; over-length examples dropped; rows are
    prompt-masked (labels=-100) and right-padded with <eos>. Saves out_path as
    {"input_ids": int32 (N, seq+1), "labels": int32 (N, seq+1)}.
    """
    # The old scheme concatenated a flat token stream, cut it every row_len tokens,
    # and tail-truncated over-length examples (which deletes the prompt and leaves
    # plen=0). sft.py doc-masks by <eos> (doc_cu_seqlens, Cfg.doc_mask) so within-row
    # cross-example attention is already blocked -- but the mask cannot supply a
    # missing question: truncation left 3.9% of loss tokens as headless answer
    # fragments at full weight, and the flat cut orphaned 16.4% of answer tails
    # across a row boundary (18.9% of rows opened mid-answer). Dropping over-length
    # examples and never splitting one across a row removes both, and every row now
    # opens with a real (masked) prompt.
    rows_ids, rows_lab = [], []
    cur_ids, cur_lab = [], []
    n_drop = 0
    n_mismatch = 0
    n_pad = 0
    row_len = seq + 1

    def flush():
        """Emit the current row, right-padded with <eos> that carry no loss."""
        nonlocal cur_ids, cur_lab, n_pad
        if not cur_ids:
            return
        pad = row_len - len(cur_ids)
        n_pad += pad
        rows_ids.append(cur_ids + [eos] * pad)
        rows_lab.append(cur_lab + [-100] * pad)
        cur_ids, cur_lab = [], []

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
                n_drop += 1  # drop rather than truncate: a truncated head has no question
                continue
            if len(cur_ids) + len(ids_f) > row_len:
                flush()
            cur_ids.extend(ids_f)
            cur_lab.extend([-100] * plen + ids_f[plen:])
        if (i // ENC_BATCH) % 10 == 0:
            print(f"  tokenized {min(i + ENC_BATCH, len(examples))}/{len(examples)}", flush=True)
    flush()

    n_rows = len(rows_ids)
    input_ids = torch.tensor(rows_ids, dtype=torch.int32)
    labels = torch.tensor(rows_lab, dtype=torch.int32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    torch.save({"input_ids": input_ids, "labels": labels}, tmp)
    os.replace(tmp, out_path)

    total_tokens = input_ids.numel()
    loss_tokens = (labels != -100).sum().item()
    print(f"kept examples: {len(examples)}", flush=True)
    print(f"dropped (> {row_len} tokens): {n_drop}", flush=True)
    print(
        f"padding tokens: {n_pad / 1e6:.2f}M ({100 * n_pad / max(input_ids.numel(), 1):.1f}% of the pack)",
        flush=True,
    )
    print(f"prefix mismatches: {n_mismatch}", flush=True)
    print(f"packed rows: {n_rows} ({row_len} tokens each)", flush=True)
    print(f"total tokens: {total_tokens / 1e6:.2f}M", flush=True)
    print(f"loss tokens: {loss_tokens / 1e6:.2f}M ({100 * loss_tokens / total_tokens:.1f}%)", flush=True)
    print(f"saved {out_path} ({os.path.getsize(out_path) / 1e9:.2f} GB)", flush=True)


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

    pack_and_save(examples, tok, eos, OUT_PATH, SEQ)


if __name__ == "__main__":
    main()
