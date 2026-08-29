#!/usr/bin/env python3
"""Prepare SFT data: ChatML (scripts/loader.format_example), prompt-masked, packed.

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
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from holdout import is_holdout  # noqa: E402
from loader import format_example  # noqa: E402

import fone  # noqa: E402

DATA = os.path.join(ROOT, "data")
TOK_PATH = os.path.join(DATA, "tokenizer.json")
OUT_PATH = os.path.join(DATA, "sft", "sft_all.pt")

SEQ = 4096  # model context; rows are SEQ+1 (input + 1 to predict)
MAX_EXAMPLES = 500_000
ENC_BATCH = 8192

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
                yield format_example(q, a)
                n += 1
        print(f"  {os.path.basename(path)}: {n}", flush=True)
    if n_holdout:
        print(f"  excluded {n_holdout} eval-holdout questions", flush=True)


def _encode_pairs(batch, tok, num_id):
    """(prompt, answer) pairs -> [(prompt_ids, full_ids, full_values)], one per pair.

    num_id None is plain BPE and the values come back empty. Otherwise numbers
    collapse to one [NUM] each and carry a value per position, exactly as train.py
    encodes the pretraining corpus -- a FoNE model has never seen a number written
    any other way, so packing SFT data the old way would fine-tune it out of its
    own input distribution.
    """
    prompts = [p for p, _ in batch]
    fulls = [p + a for p, a in batch]
    if num_id is None:
        return [(ep.ids, ef.ids, ()) for ep, ef in zip(tok.encode_batch(prompts), tok.encode_batch(fulls))]
    # Only the full text's values are kept. A prompt ending mid-number reads a
    # different value there than the full text does, and the full text is the one
    # the row actually contains.
    pp, _ = fone.encode_text(prompts, tok, num_id)
    fp, fv = fone.encode_text(fulls, tok, num_id)
    out, fi = [], 0
    for p_ids, f_ids in zip(pp, fp):
        k = int((f_ids == num_id).sum())
        out.append((p_ids.tolist(), f_ids.tolist(), fv[fi : fi + k].tolist()))
        fi += k
    return out


# Imported, not re-implemented: a pack's fingerprint must equal the vocab_id train.py
# stamps into every checkpoint, or sft_math.py's equality check can never fire.
from loader import vocab_fingerprint as _vocab_fingerprint  # noqa: E402


def pack_and_save(examples, tok, eos, out_path, seq, num_id=None):
    """Greedily pack (prompt, output) text pairs into (seq+1)-token rows and save.

    One example never split across rows; over-length examples dropped; rows are
    prompt-masked (labels=-100) and right-padded with <eos>. Saves out_path as
    {"input_ids": int32 (N, seq+1), "labels": int32 (N, seq+1)}.

    num_id set adds "values": float32 (N, seq+1), the number at every [NUM]
    position and 0 elsewhere, which sft_math.py feeds to the FoNE embedding.
    """
    # Never split an example across rows; drop over-length ones. sft.py doc-masks by
    # <eos>, so within-row cross-example attention is already blocked, but a truncated
    # example has no prompt for the mask to supply.
    rows_ids, rows_lab, rows_val = [], [], []
    cur_ids, cur_lab, cur_val = [], [], []
    n_drop = 0
    n_mismatch = 0
    n_pad = 0
    row_len = seq + 1

    def flush():
        """Emit the current row, right-padded with <eos> that carry no loss."""
        nonlocal cur_ids, cur_lab, cur_val, n_pad
        if not cur_ids:
            return
        pad = row_len - len(cur_ids)
        n_pad += pad
        rows_ids.append(cur_ids + [eos] * pad)
        rows_lab.append(cur_lab + [-100] * pad)
        rows_val.append(cur_val + [0.0] * pad)
        cur_ids, cur_lab, cur_val = [], [], []

    for i in range(0, len(examples), ENC_BATCH):
        batch = examples[i : i + ENC_BATCH]
        for ids_p, ids_f, vals_f in _encode_pairs(batch, tok, num_id):
            ids_f = ids_f + [eos]
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
            if num_id is not None:
                # values back onto their own positions: the k-th [NUM] takes the k-th value
                dense, k = [], 0
                for t in ids_f:
                    dense.append(vals_f[k] if t == num_id else 0.0)
                    k += t == num_id
                assert k == len(vals_f), f"{k} [NUM] but {len(vals_f)} values"
                cur_val.extend(dense)
        if (i // ENC_BATCH) % 10 == 0:
            print(f"  tokenized {min(i + ENC_BATCH, len(examples))}/{len(examples)}", flush=True)
    flush()

    n_rows = len(rows_ids)
    input_ids = torch.tensor(rows_ids, dtype=torch.int32)
    labels = torch.tensor(rows_lab, dtype=torch.int32)

    # "vocab_id": the same key checkpoints carry, or the fingerprint check compares
    # two differently-named fields.
    blob = {"input_ids": input_ids, "labels": labels, "vocab_id": _vocab_fingerprint(tok)}
    if num_id is not None:
        blob["values"] = torch.tensor(rows_val, dtype=torch.float32)
        n_num = int((input_ids == num_id).sum())
        print(f"[NUM] tokens: {n_num / 1e6:.2f}M ({100 * n_num / input_ids.numel():.2f}% of the pack)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    torch.save(blob, tmp)
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
