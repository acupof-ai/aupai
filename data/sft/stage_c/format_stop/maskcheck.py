#!/usr/bin/env python3
"""Mask check for format_stop_v1.pt: the loss boundary and the supervised terminator.

scripts/test_sft_pack.py checks the packer on 40 synthetic examples. This checks THE
SHIPPED PACK, row by row, against what the builder intended:

  * every row opens with a masked prompt (labels == -100), and every masked span ends at
    `assistant\\n` -- the boundary sft_math.py's loss depends on;
  * every supervised span is an answer of the length the JSONL holds, decoded, ending at
    <|im_end|>: the turn terminator IS supervised, so the model can learn to stop;
  * no supervised token is <|im_start|>, so the pack never trains the model to write a
    question or to open a second turn;
  * the pack's vocab_id equals the tokenizer's fingerprint and sft_math.py's required
    keys are all present, because a pack missing them is refused at load.

Counted, not sampled: the supervised spans are matched against the multiset of answer
lengths (one example never splits across rows, so each answer is one contiguous run).

    python3 data/sft/stage_c/format_stop/maskcheck.py [--jsonl PATH] [--pack PATH]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from loader import IM_END, IM_START, vocab_fingerprint  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

REQUIRED_KEYS = ("input_ids", "labels", "vocab_id", "holdout_fp", "packer_fp", "build_stats")


def runs(labels):
    """[(kind, start, end)] over a label row; None terminates the last run."""
    out, cur, start = [], labels[0] == -100, 0
    for i, y in enumerate(list(labels) + [None]):
        m = (y == -100) if y is not None else not cur
        if m != cur:
            out.append(("masked" if cur else "supervised", start, i))
            cur, start = m, i
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=os.path.join(HERE, "format_stop_v2.jsonl"))
    ap.add_argument("--pack", default=os.path.join(HERE, "format_stop_v2.pt"))
    args = ap.parse_args()

    import torch
    for path in (args.jsonl, args.pack):
        if not os.path.exists(path):
            print(f"REFUSE: {path} does not exist; run build.sh first")
            return 2

    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    im_start_id = tok.token_to_id(IM_START)
    im_end_id = tok.token_to_id(IM_END)
    eos_id = tok.token_to_id("<eos>")

    rows = [json.loads(line) for line in open(args.jsonl, encoding="utf-8") if line.strip()]
    want = collections.Counter(len(tok.encode(r["answer"]).ids) for r in rows)
    want_list = sorted(want)

    blob = torch.load(args.pack, map_location="cpu", weights_only=True)
    missing = [k for k in REQUIRED_KEYS if k not in blob]
    if missing:
        print(f"FAIL: pack lacks {missing}; sft_math.py --sft_path would refuse it")
        return 1
    if blob["vocab_id"] != vocab_fingerprint(tok):
        print(f"FAIL: pack vocab_id {blob['vocab_id']} != tokenizer {vocab_fingerprint(tok)}")
        return 1
    if "values" in blob:
        print("FAIL: pack carries FoNE values")
        return 1

    ids, labels = blob["input_ids"], blob["labels"]
    n_rows = int(ids.shape[0])
    checked_masked, checked_sup, bad = 0, 0, []
    sup_len_seen = collections.Counter()
    im_start_supervised = 0

    for r in range(n_rows):
        row, lab = ids[r].tolist(), labels[r].tolist()
        spans = runs(lab)
        if not spans or spans[0][0] != "masked":
            bad.append((r, "row does not open with a masked prompt"))
            continue
        for kind, a, b in spans:
            chunk = row[a:b]
            text = tok.decode(chunk, skip_special_tokens=False)
            if kind == "masked":
                if set(chunk) == {eos_id}:
                    continue  # right padding, not a prompt
                checked_masked += 1
                if not text.endswith("assistant\n"):
                    bad.append((r, f"masked span does not end at the assistant turn: {text[-40:]!r}"))
                if text.count(f"{IM_START}user") != 1:
                    bad.append((r, f"masked span holds {text.count(IM_START + 'user')} user turns"))
            else:
                checked_sup += 1
                # The packer appends <eos> to every example (prepare_sft.pack_and_save),
                # and it is supervised. The answer therefore ends at
                # <|im_end|><eos>: the terminator is supervised, and one extra position
                # after it tells the model the turn is over. Strip the <eos> and the
                # remainder must end at the turn terminator; the count below is the
                # answer's own length, <eos> excluded, so it can be matched to the JSONL.
                body_ids = chunk[:-1] if chunk and chunk[-1] == eos_id else chunk
                body_text = tok.decode(body_ids, skip_special_tokens=False)
                sup_len_seen[len(body_ids)] += 1
                if im_start_id in chunk:
                    im_start_supervised += 1
                    bad.append((r, "a role marker is supervised"))
                if not body_ids or body_ids[-1] != im_end_id:
                    bad.append((r, f"supervised span does not end at token id {im_end_id} "
                                   f"({IM_END}): last id {body_ids[-1] if body_ids else None}"))
                elif not body_text.endswith(IM_END):
                    bad.append((r, f"decoded span does not end at {IM_END}: {text[-30:]!r}"))
                elif body_text[: -len(IM_END)].endswith(IM_START):
                    bad.append((r, "supervised span is empty"))

    # every shipped answer appears as one supervised run of its own token length
    missing_lengths = []
    for length, count in want.items():
        if sup_len_seen[length] < count:
            missing_lengths.append((length, count, sup_len_seen[length]))
    extra_lengths = {L: c - want.get(L, 0) for L, c in sup_len_seen.items() if c > want.get(L, 0)}
    if missing_lengths:
        bad.append((-1, f"{len(missing_lengths)} answer lengths are missing from the pack, "
                       f"e.g. {missing_lengths[0]} (length, jsonl rows, pack rows)"))
    if extra_lengths:
        bad.append((-1, f"supervised runs with no matching answer length: "
                       f"{sorted(extra_lengths.items())[:5]}"))

    print(f"rows={n_rows} masked_spans={checked_masked} supervised_spans={checked_sup} "
          f"jsonl_rows={len(rows)} distinct_answer_lengths={len(want_list)}")
    print(f"vocab_id={blob['vocab_id']} holdout_fp={blob['holdout_fp']} "
          f"packer_fp={blob['packer_fp']}")
    if bad:
        for r, why in bad[:10]:
            print(f"FAIL row {r}: {why}")
        print(f"FAIL maskcheck: {len(bad)} problem(s)")
        return 1
    print(f"PASS maskcheck: every masked span ends at assistant\\n, every supervised span "
          f"ends at {IM_END}, {im_start_supervised} supervised role markers, "
          f"answer lengths match the JSONL exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
