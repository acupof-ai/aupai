#!/bin/bash
# Launch the cot_ot3 fetch detached (aupai-6e 2026-09-04): .part + Content-Length
# size check + per-shard host recording are all inside fetch_corpus.t37; this
# script only detaches it and writes a DONE marker pod-side. Run only after
# b0's arm-2 (runs/b0_se_looped_2b.log) prints its first step line.
# Usage on the pod: bash scripts/launch_cot_ot3_fetch.sh
set -euo pipefail
cd /work/aupai
rm -f runs/cot_ot3_fetch_DONE
setsid nohup bash -c 'cd /work/aupai && python3 datagen/fetch_corpus.py \
  --source cot_ot3 --target_bytes 9000000000 \
  > runs/cot_ot3_fetch.log 2>&1 && echo DONE > runs/cot_ot3_fetch_DONE' \
  </dev/null >/dev/null 2>&1 &
sleep 2
ls -la runs/cot_ot3_fetch.log
echo "launched"