#!/bin/bash
set -euo pipefail
cd /work/aupai
rm -f runs/s3a_w16_wait_switch.rc
setsid bash runs/s3a_w16_wait_and_switch.sh > runs/s3a_w16_wait_switch.log 2>&1 &
echo $! > runs/s3a_w16_wait_switch.pid
sleep 2
cat runs/s3a_w16_wait_switch.pid
