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

import hashlib
import json
import os
import random
import sys
from collections import deque

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
#: examples the packer may look past to fill a row's tail. Large enough that some
#: example fits almost any remaining room, small enough that order stays locally
#: shuffled -- length-sorted packing would bias what the model sees first.
LOOKAHEAD = 512
ENC_BATCH = 8192

SOURCES = [
    (os.path.join(DATA, "alpaca_gpt4_zh.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "coig.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "openo1_sft.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "gsm8k_zh.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "school_math_r1_zh.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "s1k.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "sft", "fable5_cot.jsonl"), "prompt", "response"),
    # t29 (2026-08-31): dropped -- this IS the code-500 carve source. Its 2413
    # same-template sibling rows made SFT code-500 measure in-distribution
    # template recall, not capability (be.sft_v3_code500, dose-acc r=0.69).
    # The family-clean pack builds without it; the file stays on disk for the
    # eval's provenance (cont.code_holdout_carved).
    # t43 (2026-08-31): v5 addon -- English Evol-Instruct Python tasks, a
    # different generator and language family from the dropped Chinese carve
    # source. >12.6pt on code-500 = cross-generator transfer (capability);
    # ~0 = strong template recall.
    (os.path.join(DATA, "sft", "v5_evol_code_2300.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "synthetic", "knowledge_qa_zh.jsonl"), "instruction", "output"),
    (os.path.join(DATA, "synthetic", "math_gsm8k_zh.jsonl"), "instruction", "output"),
]


def _fp_file(path):
    """Content hash of a single file. Content-based, not git sha: uncommitted edits
    change what a pack contains, and a sha would not see them."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _fp_sources():
    """Content hash of all SOURCES files. A source swap that does not trigger a
    repack is invisible without this."""
    h = hashlib.sha256()
    for path, _, _ in SOURCES:
        with open(path, "rb") as f:
            h.update(os.path.basename(path).encode() + b"\0" + hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:16]


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

    def place(item):
        ids_f, plen, dense = item
        cur_ids.extend(ids_f)
        cur_lab.extend([-100] * plen + ids_f[plen:])
        if num_id is not None:
            cur_val.extend(dense)

    pending = deque()
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
            dense = None
            if num_id is not None:
                # values back onto their own positions: the k-th [NUM] takes the k-th value
                dense, k = [], 0
                for t in ids_f:
                    dense.append(vals_f[k] if t == num_id else 0.0)
                    k += t == num_id
                assert k == len(vals_f), f"{k} [NUM] but {len(vals_f)} values"
            pending.append((ids_f, plen, dense))
        # Pack with a bounded lookahead. Plain first-fit closes a row as soon as the NEXT
        # example does not fit, so the leftover is whatever that example's length happened
        # to be -- 11.8% of the 3.24b pack was tail padding, all of it forward and backward
        # on nothing. Scanning ahead for one that fits recovers most of it. Sorting by
        # length (classic FFD) would pack tighter still, but it puts every long example at
        # the front of training, so the shuffle has to survive: a window keeps the order
        # locally random. Drain only what a full window can see, so the tail of one encode
        # batch still packs against the head of the next.
        while len(pending) > LOOKAHEAD:
            room = row_len - len(cur_ids)
            j = next((k for k in range(min(LOOKAHEAD, len(pending))) if len(pending[k][0]) <= room), None)
            if j is None:
                flush()
                continue
            place(pending[j])
            del pending[j]
        if (i // ENC_BATCH) % 10 == 0:
            print(f"  tokenized {min(i + ENC_BATCH, len(examples))}/{len(examples)}", flush=True)
    while pending:
        room = row_len - len(cur_ids)
        j = next((k for k in range(len(pending)) if len(pending[k][0]) <= room), None)
        if j is None:
            flush()
            continue
        place(pending[j])
        del pending[j]
    flush()

    n_rows = len(rows_ids)
    input_ids = torch.tensor(rows_ids, dtype=torch.int32)
    labels = torch.tensor(rows_lab, dtype=torch.int32)

    # "vocab_id": the same key checkpoints carry, or the fingerprint check compares
    # two differently-named fields. The other three fingerprints make the pack's
    # provenance self-describing: which packer built it, which sources it read, and
    # which holdout set it was checked against. A stale holdout_fp is the
    # contamination that nothing currently catches.
    blob = {
        "input_ids": input_ids,
        "labels": labels,
        "vocab_id": _vocab_fingerprint(tok),
        "packer_fp": _fp_file(os.path.join(ROOT, "scripts", "prepare_sft.py")),
        "sources_fp": _fp_sources(),
        "holdout_fp": _fp_file(os.path.join(ROOT, "data", "eval", "holdout_hashes.txt")),
    }
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



def _selftest():
    """The packer reorders examples to fill row tails; the failure it can hide is losing or
    duplicating one, so assert conservation on real pack_and_save output, not on a model of it."""
    import tempfile

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    class _FakeTok:  # ids are lengths made unique per example, so conservation is checkable
        def encode_batch(self, texts):
            return [_Enc([hash(t) % 900 + 100] * len(t)) for t in texts]

    global _vocab_fingerprint
    real_fp, _vocab_fingerprint = _vocab_fingerprint, lambda _t: "selftest"
    random.seed(0)
    pairs = [("q" * random.randint(5, 60), "a" * random.randint(5, 400)) for _ in range(4000)]
    try:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "p.pt")
            pack_and_save(pairs, _FakeTok(), 0, out, 511)
            blob = torch.load(out, weights_only=True)
    finally:
        _vocab_fingerprint = real_fp

    ids, lab = blob["input_ids"], blob["labels"]
    assert ids.shape == lab.shape and ids.shape[1] == 512, ids.shape
    # Every example is one contiguous run of a single id; count runs per id and compare to
    # how many examples carry that id. A dropped or duplicated example moves a count.
    from collections import Counter

    want = Counter()
    for q, a in pairs:
        want[hash(q + a) % 900 + 100] += 1
    got = Counter()
    for row in ids.tolist():
        prev = None
        for t in row:
            if t != prev and t != 0:
                got[t] += 1
            prev = t
    assert got == want, f"packing lost or duplicated examples: {len(want - got)} missing"
    pad = int((lab == -100).sum()) - int(((lab == -100) & (ids != 0)).sum())
    assert pad >= 0
    return f"{len(pairs)} examples conserved across {ids.shape[0]} rows"


def main():
    out_path = OUT_PATH
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
    random.seed(42)
    tok = Tokenizer.from_file(TOK_PATH)
    eos = tok.token_to_id("<eos>")
    assert eos is not None, "tokenizer has no <eos>"

    examples = list(read_examples())
    random.shuffle(examples)
    if len(examples) > MAX_EXAMPLES:
        examples = examples[:MAX_EXAMPLES]
    print(f"total examples: {len(examples)}", flush=True)

    pack_and_save(examples, tok, eos, out_path, SEQ)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print("prepare_sft selftest OK:", _selftest())
    else:
        main()
