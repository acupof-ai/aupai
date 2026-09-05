#!/bin/bash
# Copy all 22 token cache groups (247.8 GB) from the overlay to NVMe, sha256 both sides.
#
# Written to a file and run with `setsid bash`, never passed inline to `pod`: shape 166, a
# `cd X && cmd &` leaves the cd out of the backgrounded half and the redirect silently does not run.
set -u
cd /work/aupai || exit 1

OUT=runs/token_cache_move.json
CAP_BYTES=$((64 * 1024 * 1024))

# QUEUE BEHIND THE SECRET SCAN, because both read the same rotational disk. The scan reads
# data/corpus off /work (vda2) and this copy reads /data00, which is a directory on the overlay --
# also vda2. Interleaving two sequential streams on rust is worse than half each: the arm seeks
# between them, so the pair can run 3-5x slower than either alone. Serial is strictly better here
# and it costs nothing, because neither job has a deadline.
#
# POLLED ON A LOG STRING, never on [ -d /proc/<pid> ]: /proc answers "has this pid been reaped",
# so a zombie keeps the guard true after the job exits (fb waited 31 minutes on a dead pid on
# 2026-09-03) and a pid read in the wrong namespace does not resolve at all. The scan's launcher
# writes `exit=<rc>` as its last line, which is namespace-independent and reap-independent.
#
# THE WAIT IS BOUNDED. 240 iterations x 60 s = 4 h, well past the scan's remaining ~1 h at three of
# six domains. On timeout this job REFUSES rather than starting anyway: an unbounded wait and a
# silent fallthrough are the two ways this kind of guard fails.
SCAN_LOG=runs/corpus_secret_report_0905.log
waited=0
for _ in $(seq 1 240); do
  if [ ! -f "$SCAN_LOG" ] || grep -q '^exit=' "$SCAN_LOG" 2>/dev/null; then
    break
  fi
  sleep 60
  waited=$((waited + 1))
done
if [ -f "$SCAN_LOG" ] && ! grep -q '^exit=' "$SCAN_LOG" 2>/dev/null; then
  echo "REFUSING: the secret scan has not finished after ${waited} min; not starting a 740 GB"
  echo "  read against it. Re-run this script when $SCAN_LOG ends with an exit= line."
  exit 1
fi
echo "scan finished after ${waited} min of waiting; starting the copy"

# --skip-done resumes: tokens_sample.pt is already verified from the rate measurement.
#
# THE HASH IS THE EXPENSIVE HALF, not the copy: every group is read three times (copy read, source
# hash, destination hash), so 247.8 GB of copy is ~740 GB of reads. At the overlay's 193 MB/s for
# the two source passes and NVMe's 1.3 GB/s for the destination pass, the floor is about 43 min.
# That is the price of knowing the copy is byte-exact, and it is paid once.
python3 -u scripts/move_token_caches.py \
  --src /data00 \
  --dst /mnt/data02/tokens \
  --out "$OUT" \
  --skip-done \
  2>&1 | head -c "$CAP_BYTES"
rc=${PIPESTATUS[0]}

echo "exit=$rc out=$OUT"
exit "$rc"
