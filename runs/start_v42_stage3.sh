#!/bin/bash
set -euo pipefail
cd /work/aupai
mkdir -p runs
rm -f runs/v42_stage3.exit
setsid bash -c '
  bash /work/aupai/runs/v42_stage3_1011.sh
  rc=$?
  printf "%s rc=%s\n" "$(date -Is)" "$rc" > /work/aupai/runs/v42_stage3.exit
  exit "$rc"
' > /work/aupai/runs/v42_stage3.log 2>&1 < /dev/null &
echo "STAGE3_STARTED $!"
