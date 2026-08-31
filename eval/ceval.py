#!/usr/bin/env python3
"""C-Eval: 1,050 Chinese multiple-choice questions across 52 school and professional subjects.

Two scorings of the same 1,050 questions, both chance 25%:

- MCF (default): the prompt carries the four options and the model picks between
  the bare letter tokens A/B/C/D. This is how MMLU is scored here.
- cloze (--cloze / load_items(cloze=True)): the prompt carries the question only
  and the four OPTION TEXTS are the continuations, scored by per-character
  normalised log-likelihood.

The split matters at our scale. OLMES (arXiv 2406.08446, Fig. 1 caption) puts the
acquisition of the MCF format at ~400B training tokens; stage 1 is 15-30B. Its
Tables 6-7 show Pythia-1B, OLMo-1B and TinyLlama-1.1B at chance under MCF and well
above it under cloze on the same questions -- at 5x our parameter count. A model
that has not learned "answer with a letter" scores at chance under MCF whatever it
knows, so an MCF reading cannot distinguish no knowledge from no format.

Per-character normalisation, not per-token: OLMES's acc_norm, and DataDecide
(arXiv 2504.11393 §3.3) reports character normalisation empirically optimal for
most tasks. Option texts differ in length, so unnormalised sums favour short ones.

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


def load_items(path=PATH, cloze=False):
    """run_eval's shape. cloze=False scores the bare letters (MCF); cloze=True
    scores the option texts as continuations, per-character normalised."""
    items = []
    for d in load_dataset(path):
        label = LETTERS.index(d["answer"].strip())
        if cloze:
            items.append(
                {
                    "prompt": f"{d['question']}\n答案：",
                    "options": [str(d[L]) for L in LETTERS],
                    "label": label,
                    "norm": "char",
                }
            )
        else:
            opts = " ".join(f"{L}. {d[L]}" for L in LETTERS)
            items.append(
                {
                    "prompt": f"{d['question']}\n{opts}\n答案：",
                    "options": list(LETTERS),
                    "label": label,
                }
            )
    return items


def _demo():
    """The label must index the option the file names, or the score is a silently permuted truth."""
    items = load_items()
    assert len(items) == 1050, len(items)
    raw = list(load_dataset())
    for d, it in zip(raw[:50], items[:50]):
        assert it["options"][it["label"]] == d["answer"].strip(), (d, it)
        assert d[d["answer"].strip()] in it["prompt"], "the correct option text is not in the prompt"

    # Cloze: the same label indexes the same answer, and the options are the TEXTS.
    cz = load_items(cloze=True)
    assert len(cz) == 1050, len(cz)
    for d, it in zip(raw[:50], cz[:50]):
        assert it["label"] == LETTERS.index(d["answer"].strip()), (d, it)
        assert it["options"][it["label"]] == str(d[d["answer"].strip()]), (d, it)
        assert it["norm"] == "char"
    # The cloze prompt must not carry the option LIST -- that is the MCF prompt.
    # A bare substring test is the wrong check: 15/1050 answers legitimately occur
    # inside their own question ("500" in "500字节"), which is the question's
    # content, not the answer being given away. What would be a real leak is the
    # rendered option list, so test for that.
    for it in cz:
        assert "A. " not in it["prompt"], "the option list leaked into the cloze prompt"
    # The two scorings must disagree in shape, or one of them is not what it claims.
    assert cz[0]["options"] != items[0]["options"], "cloze options are still the bare letters"
    assert len(set(len(o) for it in cz for o in it["options"])) > 1, "cloze options are all one length"

    from collections import Counter

    c = Counter(it["label"] for it in items)
    print(f"ceval self-test OK: {len(items)} items (MCF + cloze), answer spread {dict(sorted(c.items()))}")


if __name__ == "__main__":
    _demo()
