#!/usr/bin/env python3
"""C-Eval: 1,050 Chinese multiple-choice questions across 52 school and professional subjects.

This repo trains a Chinese model and, until now, scored it only on MMLU, ARC,
HellaSwag, PIQA, BoolQ, OpenBookQA and WinoGrande -- all English, all sitting at
the 25% chance line -- plus our own synthetic Chinese math. So no instrument in
the repo could see a change in Chinese knowledge or reasoning, which is most of
what the pretrain corpus contains. data/ceval.jsonl has been on the pod since
2026-08-24 and nothing ever read it.

Scored the same way as MMLU: the prompt carries the four options and the model
picks between the bare letter tokens A/B/C/D. Chance is 25%. A 200M model is
expected near chance at first; the value is that a real move now has somewhere
to show up.

    python eval/run_eval.py --ckpt X --tokenizer T --benchmarks ceval
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "data", "ceval.jsonl")
LETTERS = "ABCD"


def load_dataset(path=PATH):
    assert os.path.exists(path), (
        f"{path} is missing. It is the C-Eval validation split (the test split has no "
        "public labels); 1,050 rows of {question, A, B, C, D, answer, subject}."
    )
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_items(path=PATH):
    """run_eval's shape: prompt with the options inline, score the bare letters."""
    items = []
    for d in load_dataset(path):
        opts = " ".join(f"{L}. {d[L]}" for L in LETTERS)
        items.append(
            {
                "prompt": f"{d['question']}\n{opts}\n答案：",
                "options": list(LETTERS),
                "label": LETTERS.index(d["answer"].strip()),
            }
        )
    return items


def _demo():
    """The label must index the option the file names, or the score is a permutation
    of the truth and still looks plausible."""
    items = load_items()
    assert len(items) == 1050, len(items)
    raw = list(load_dataset())
    for d, it in zip(raw[:50], items[:50]):
        assert it["options"][it["label"]] == d["answer"].strip(), (d, it)
        assert d[d["answer"].strip()] in it["prompt"], "the correct option text is not in the prompt"
    from collections import Counter

    c = Counter(it["label"] for it in items)
    print(f"ceval self-test OK: {len(items)} items, answer spread {dict(sorted(c.items()))}")


if __name__ == "__main__":
    _demo()
