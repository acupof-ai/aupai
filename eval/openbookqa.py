#!/usr/bin/env python3
"""OpenBookQA dataset loader for eval/run_eval.py. Load-only: scoring lives
in run_eval.score_mc. The former standalone evaluate() carried a second,
joint-tokenization scorer; deleted 2026-09-02 (44-14.1) -- two scorers for
one checkpoint is how numbers drift apart.
"""


def load_dataset():
    from datasets import load_dataset as hf_load_dataset

    # Test split has no public labels; validation is the standard labeled eval set.
    return hf_load_dataset("allenai/openbookqa", "main", split="validation", streaming=True)
