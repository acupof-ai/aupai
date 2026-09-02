"""Shared helpers for the eval harness."""

import os

# Offline before any `datasets` import in this package. Nine modules here call
# load_dataset (mmlu, gsm8k, winogrande, hellaswag, arc, boolq, piqa, openbookqa,
# chid_probe), and `datasets` contacts the Hub to check for updates even on a cache
# hit -- on the pod, huggingface.co times out and hf-mirror's API 403s, so the call
# blocks rather than fails. Measured 2026-08-30: a run_eval on a warm MMLU cache sat
# 18 minutes at 0% GPU with its CPU time frozen at 3m37s, holding 26.7GB, while the
# log showed nothing. Earlier the same day the same call "merely" took 176.6s and read
# as slow.
#
# The point is fast failure, not offline: with the cache present nothing changes, and
# with it missing load_dataset now raises immediately into run_eval.py's per-benchmark
# SKIPPED branch instead of hanging. This belongs here rather than in run_eval.py --
# knowledge parked in one metric's branch cannot save the metric beside it that imports
# the same library. scripts/fetch_*.py are deliberately outside this package: a fetcher
# must reach the network, an eval must never need to.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
