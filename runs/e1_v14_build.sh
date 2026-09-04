#!/bin/bash
# v14 agentic SFT pack, rebuilt under the write gate with the CLI-flag credential tier (107524cc).
# Same shape as v13: default --limit 10000, --max-tokens 4096, no --subagents.
# restartable: the pack is staged to .unscanned before the secret scan and renamed only if the
# gate passes, so a kill costs the scan and leaves a file whose name says it was never cleared.
set -uo pipefail
cd /Users/bytedance/code/aupai-e1
export CUDA_VISIBLE_DEVICES=""
python3 -u scripts/build_agentic_sft.py --out data/sft/agentic_v14.jsonl
echo "build exit=$?"
