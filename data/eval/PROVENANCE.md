# Eval split provenance

The two files behind `holdout.EVAL_FILES` are **frozen reference artifacts**. No upstream
generator survived (REVIEW_2026-08-26 finding #1: nothing in the repo or in
`git log --diff-filter=A` ever created them), so they cannot be faithfully regenerated and
must not be edited. They are pinned here by checksum instead.

| file | rows | origin (best known) |
|------|------|---------------------|
| `data/eval/math_test_500.jsonl` | 500 | Belle-derived Chinese grade-school math; the 500-problem holdout the SFT ledger reports against. |
| `data/synthetic/math_hard_eval_1k.jsonl` | 1032 | Synthetic hard-math eval (mathbank), the `eval_hard.sh` metric of record. |

Both carry `instruction` (+ `output`, and `level`/`answer` for the hard set). All 1532
questions feed `scripts/holdout.py` -> `data/eval/holdout_hashes.txt`.

## Integrity

Verify the frozen files are unchanged:

```
shasum -a 256 -c <<'SUMS'
609a5a5af31d666a01e2a82c692d1d54f5e13a081a2a98c929e02670941b34e5  data/eval/math_test_500.jsonl
3ce9b0ff7fc6253c0d23c41cd360f09242f8a9a67ed187f4f56f812957ac703b  data/synthetic/math_hard_eval_1k.jsonl
SUMS
```

If either sum changes, the split changed: re-run `python scripts/holdout.py` and treat every
accuracy reported before the change as measured against a different holdout.
