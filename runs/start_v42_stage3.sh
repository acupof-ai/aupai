#!/bin/bash
set -euo pipefail
cd /work/aupai
mkdir -p runs
setsid bash /work/aupai/runs/v42_stage3_1011.sh \
  > /work/aupai/runs/v42_stage3.log 2>&1 < /dev/null &
echo "STAGE3_STARTED $!"
