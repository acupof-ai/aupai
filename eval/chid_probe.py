#!/usr/bin/env python3
"""CHID Chinese-idiom cloze probe — the one Chinese eval with resolution at 200M.

10-choice idiom cloze (random 10%). A decoder-only LM scores it by continuation
log-likelihood with the left context only: the blank sits mid-paragraph and idiom
choice is driven by the preceding syntax. Standalone from run_eval.score_mc.

Needs the HF proxy + httpx[socks] on the host; the pod has neither by default.

Usage:  python eval/chid_probe.py --ckpt X
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402
from eval.run_eval import score_mc  # noqa: E402


def load_chid(split="dev", max_rows=None):
    """Load clue/chid records into score_mc items.

    schema: `content` = paragraph split into segments with the idiom blanks inline;
    `candidates` = candidate idiom strings; `answers` = per-blank {text, candidate_id}.
    One item per blank: prompt = full left context up to the blank, options = the
    candidates, label = that blank's candidate_id.
    """
    from datasets import load_dataset

    ds = load_dataset("clue/chid", split=split)
    items = []
    for row in ds:
        content = row["content"]
        cands = row["candidates"]
        answers = row["answers"]
        # A blank is a placeholder segment ("" or "#idiom#"); the prompt is the
        # segments before it, joined.
        n_blank = 0
        left = []
        for seg in content:
            if seg in ("", "#idiom#"):
                if n_blank < len(answers):
                    a = answers[n_blank]
                    cid = a["candidate_id"] if isinstance(a, dict) else a
                    items.append({
                        "prompt": "".join(left),
                        "options": cands,
                        "label": int(cid),
                    })
                    n_blank += 1
            left.append(seg)
    if max_rows:
        items = items[:max_rows]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tokenizer.json"))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max_rows", type=int, default=None, help="probe on a small slice to check schema before scoring")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    model, cfg = load_checkpoint(a.ckpt, device="cpu")
    tok = load_tokenizer(a.tokenizer, cfg)
    items = load_chid(split=a.split, max_rows=a.max_rows)
    if not items:
        raise SystemExit("no items built — schema of clue/chid differs from the probe's assumption; dump one row and adjust load_chid")
    acc = score_mc(model, tok, items, "cpu", a.batch,
                   num_id=getattr(cfg, "num_id", None))
    print(f"CHID {a.split}: {len(items)} items, acc {acc:.1%} (random 10%)")


if __name__ == "__main__":
    main()