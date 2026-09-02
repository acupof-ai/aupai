#!/usr/bin/env python3
"""BoolQ dataset loader for eval/run_eval.py. Load-only: scoring lives in
run_eval.score_mc (separate prompt/option encoding). The former standalone
evaluate() carried a second, joint-tokenization scorer; deleted 2026-09-02
(44-14.1) -- two scorers for one checkpoint is how numbers drift apart, and
the tokenization divergence the audit suspected does not reproduce
(0/400 boolq items, measured 2026-09-02).
"""


def load_dataset():
    from datasets import load_dataset as hf_load_dataset

    return hf_load_dataset("google/boolq", split="validation", streaming=True)
