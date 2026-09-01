#!/usr/bin/env bash
# Poll a pod log until a terminal line appears. `pod` runs through crictl exec and the
# tn tunnel drops every ~5 minutes, which kills a `tail -f` mid-stream -- so the watch
# has to re-establish the connection each tick rather than hold one open. A dropped
# tunnel is not a finished job, and a monitor that treats it as one reports silence.
#
# Usage: scripts/watch_pod_log.sh <logfile-relative-to-/work/aupai> [pattern]
set -uo pipefail
LOG=${1:?log path relative to /work/aupai}
PAT=${2:-"L1 math-500|answer-present|Traceback|Error|RuntimeError|refus"}
for _ in $(seq 1 240); do
  out=$(~/bin/pod "cd /work/aupai && tail -4 $LOG" 2>/dev/null || true)
  if printf '%s\n' "$out" | grep -E "$PAT"; then
    exit 0
  fi
  sleep 25
done
echo "TIMEOUT: $LOG produced no terminal line"
exit 1
