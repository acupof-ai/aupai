#!/usr/bin/env python3
"""Prepare stage-2 math SFT data: same packing as prepare_sft.py, math-heavy mix.

Mix (~69% math / 31% general by rows):
  school_math_train (~220K, deduped, 500 held out)
  gsm8k_zh_train (7.5K, #### normalized to 答案是：)
  alpaca_gpt4_zh (52K, general replay)
  coig_50k (50K sampled, exam/reading replay)
"""

import json
import os
import random
import sys

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from holdout import is_holdout  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
TOK_PATH = os.path.join(DATA, "tokenizer.json")
OUT_PATH = os.path.join(DATA, "sft", "sft_math.pt")

SEQ = 4096
MAX_EXAMPLES = 3_000_000
ENC_BATCH = 8192

SOURCES = [
    (os.path.join(DATA, "workbatch", "school_math_train.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "workbatch", "gsm8k_zh_train.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "alpaca_gpt4_zh.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "workbatch", "coig_50k.jsonl"), "instruction", "output"),
]


def read_examples(sources=SOURCES):
    for path, qk, ak in sources:
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                q = (d.get(qk) or "").strip()
                a = (d.get(ak) or "").strip()
                if not q or not a or is_holdout(q):
                    continue
                yield f"问：{q}\n答：", a
                n += 1
        print(f"  {os.path.basename(path)}: {n}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", help="comma-separated jsonl paths (instruction/output keys)")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    sources = [(p, "instruction", "output") for p in args.sources.split(",")] if args.sources else SOURCES
    random.seed(42)
    tok = Tokenizer.from_file(TOK_PATH)
    eos = tok.token_to_id("<eos>")
    assert eos is not None, "tokenizer has no <eos>"

    examples = list(read_examples(sources))
    random.shuffle(examples)
    if len(examples) > MAX_EXAMPLES:
        examples = examples[:MAX_EXAMPLES]
    print(f"total examples: {len(examples)}", flush=True)

    # Greedy packing, one example never split across rows.
    #
    # The old scheme concatenated a flat token stream and cut it every 4097 tokens,
    # and tail-truncated over-length examples (which deletes the prompt and leaves
    # plen=0). Measured on the shipped sft_all.pt (REVIEW_2026-08-26.md #3):
    # 3.9% of loss tokens were headless reasoning fragments at full weight, 16.4%
    # were answer tails orphaned across a row boundary, and 18.9% of rows opened
    # mid-answer. There is no document mask -- sliding-window attention spans 1023
    # tokens and KDA carries recurrent state across the whole row -- so the model
    # was conditioned on an unrelated problem while answering, then met a bare
    # "问：...\n答：" at eval.
    rows_ids, rows_lab = [], []
    cur_ids, cur_lab = [], []
    n_drop = 0
    n_mismatch = 0
    n_pad = 0
    row_len = SEQ + 1

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

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    torch.save({"input_ids": input_ids, "labels": labels}, tmp)
    os.replace(tmp, args.out)

    total_tokens = input_ids.numel()
    loss_tokens = (labels != -100).sum().item()
    print(f"kept examples: {len(examples)}", flush=True)
    print(f"dropped (> {row_len} tokens): {n_drop}", flush=True)
    print(f"padding tokens: {n_pad / 1e6:.2f}M ({100 * n_pad / max(input_ids.numel(), 1):.1f}% of the pack)", flush=True)
    print(f"prefix mismatches: {n_mismatch}", flush=True)
    print(f"packed rows: {n_rows} ({row_len} tokens each)", flush=True)
    print(f"total tokens: {total_tokens / 1e6:.2f}M", flush=True)
    print(f"loss tokens: {loss_tokens / 1e6:.2f}M ({100 * loss_tokens / total_tokens:.1f}%)", flush=True)
    print(f"saved {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
