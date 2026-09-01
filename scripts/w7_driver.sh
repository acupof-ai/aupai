#!/bin/bash
# Wait until every card in the caller's set is free, then run the peak probe on it.
#
# The driver does NOT pick cards. An earlier draft chose the first seven free ones off
# nvidia-smi, and device_set_honoured refused it: because "free" is read live, a computed
# choice can land outside whatever lane the caller set -- the same escape as a hardcoded
# index, just calculated instead of typed. So the caller names the seven and this only
# waits for them.
#
#   CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 bash scripts/w7_driver.sh
#
# It waits rather than claims: other sessions' evals come and go on cards 0 and 4, and
# nothing here should take a card from a running job. Readiness is read off the cards,
# not off a process list -- a SIGTERMed rank held 72 GiB after ps had lost it.
#
# A file rather than a quoted -c string: the first attempt nested four levels
# (tn -> crictl -> bash -lc -> setsid bash -c) and could not be read back to confirm
# what the pod received. An unverifiable launcher aimed at seven cards is how this
# morning's collision happened.
set -u
cd /work/aupai

source eval/_devs.sh 7 || exit 1
WANT=",$(IFS=,; echo "${_DEVS[*]}"),"

for _ in $(seq 1 240); do
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' -v want="$WANT" 'index(want, ","$1",") && $2 > 1024 {print $1}' \
         | tr '\n' ' ')
  if [ -z "$busy" ]; then
    echo "cards ${_DEVS[*]} free at $(date -u +%H:%M:%S) -- launching"
    bash scripts/w7_peak.sh
    exit $?
  fi
  sleep 20
done
echo "TIMEOUT: still busy after 80 min: $busy"
exit 1
