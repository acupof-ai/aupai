#!/usr/bin/env python3
"""CHID Chinese-idiom cloze probe — the one Chinese eval with resolution at 200M.

CHID (clue/chid) is 10-choice idiom cloze: random line 10% vs MMLU's 25%, and a
decoder-only LM scores it directly by continuation log-likelihood (the empty
sits mid-paragraph, so we score with the left context only — idiom choice is
driven by the preceding syntax, which is exactly what 200M learns from the
Chinese corpus). standalone from run_eval.score_mc, so no benchmarking harness
change is needed to probe it.

NOTE on running this: the pod has no HF route and needs aupai-fb's socks5 proxy,
and clue/chid is pulled via `datasets` — which needs httpx[socks]/socksio on the
host. On the PEP-668 pod that's a `--break-system-packages` pip (refused by
default). Run where the proxy + socks dep are available; this file only defines
the loader and the scorer, not the infra.

Expected decision gate (aupai-fb): dev 3,218, k4 vs k5. ~12% both -> no signal,
abandon; ~20%+ with a real spread -> wire into CI.

Usage:  python eval/chid_probe.py --ckpt ckpt_sft.pt [--k4 --k5 for the A/B]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.loader import load_checkpoint, load_tokenizer  # noqa: E402
from eval.run_eval import score_mc  # noqa: E402


def load_chid(split="dev", max_rows=None):
    """Load clue/chid records into score_mc items.

    schema (verified-on-first-run): `content` = paragraph split into segments
    with the idiom blanks inline; `candidates` = candidate idiom strings;
    `answers` = per-blank {text, candidate_id}. We build one item per blank:
      prompt  = the full left context up to the blank (blank is a delimiter)
      options = the candidate idioms
      label   = that blank's candidate_id (index into candidates)
    """
    from datasets import load_dataset

    ds = load_dataset("clue/chid", split=split)
    items = []
    for row in ds:
        content = row["content"]
        cands = row["candidates"]
        answers = row["answers"]
        # content is a list of segments; a blank is marked by a placeholder-like
        # segment (empty or the idiom removed). Prompt = all segments before the
        # i-th answer's blank, joined.
        n_blank = 0
        left = []
        for seg in content:
            if seg in ("", "#idiom#"):  # placeholder markers CHID uses
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