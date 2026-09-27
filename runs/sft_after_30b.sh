#!/bin/bash
# Queue the reasoning SFT behind the 30B run: waits for the final checkpoint AND an empty card table
# on two consecutive polls, then runs the committed launcher (which re-checks pack, claims, cards).
cd /work/aupai || exit 1
i=0; clear=0
while [ "$i" -lt 600 ]; do
  i=$((i + 1))
  if grep -q 'Saved final\|saved ckpt_v41_ced_0926.pt\b' runs/v41_ced_0926.log 2>/dev/null || [ -f ckpt_v41_ced_0926.pt ]; then
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
    if [ "$n" -eq 0 ]; then clear=$((clear + 1)); else clear=0; fi
    [ "$clear" -ge 2 ] && break
  fi
  sleep 60
done
[ "$clear" -ge 2 ] || { echo "GAVE UP after $i polls ($(date -u +%H:%MZ))"; exit 1; }
echo "30B done, cards free at $(date -u +%H:%MZ); launching SFT"
export HYPOTHESIS="reasoning SFT v1 (cot 40.8% / sandbox-verified code 38.7% / code_if 20.4%, 61.37M tok, pack sft_reason_v1) on the v41_ced_0926 30B final raises HumanEval pass@1 above the base final's own number, toward the >=30% gate; 0926 base read 35/164 at step26000"
exec bash runs/v41_sft_reason_v1.sh
