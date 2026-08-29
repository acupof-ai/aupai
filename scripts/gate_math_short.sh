#!/usr/bin/env bash
# Gate: a math_short_* batch must be clean vs the active eval holdouts before ingest.
# Finding 2026-08-30: the mathbank shares template DNA with math_hard_eval_1k's
# old generator -- every batch measured (v3/v5/v6/v7/v8/v10/v11) REJECTed at 0.8.
# math_hard_eval_1k was retired that day and replaced by math_hard_eval_v2_1k,
# whose types are disjoint from the bank (symbolic algebra/geometry the arithmetic
# bank cannot generate); all six local batches pass the v2 gate with 0 hits
# (facts/contamination.json#cont.math_hard_v2). This gate keeps it that way:
# it scans against the scanner's default holdout set (math-500 + v2), so a bank
# extension that drifts into v2's types stops here instead of at the next audit.
#
# Usage: scripts/gate_math_short.sh <batch.jsonl> [batch.jsonl ...]
# Exit: 0 = all clean (ingest allowed); 1 = at least one REJECT (do not ingest).
set -euo pipefail
cd "$(dirname "$0")/.."
rc=0
for src in "$@"; do
    echo "--- gate: $src"
    # baseline: gsm8k_zh -- the accepted clean math shard (same distribution family,
    # 0 hits). The scanner REQUIRES a same-scale baseline; without one a FPR number
    # has no binding power (cont.cci3_scale_failure).
    python3 scripts/scan_math_contamination.py "$src" \
        --fpr-baseline data/corpus/math/gsm8k_zh_000.jsonl || rc=1
done
if [ "$rc" -ne 0 ]; then
    echo "GATE REFUSED: fix the bank (or the batch), do not ingest contaminated rows"
fi
exit "$rc"
