#!/bin/bash
# Launch the corpus secret report on the pod. Report-only: no deletion, no filter change.
#
# Written to a FILE and run with `setsid bash runs/<name>.sh`, not passed inline to `pod`:
# AGENTS.md's shape 166 -- `pod "cd X && cmd & "` leaves the cd out of the backgrounded half, so
# the relative log path resolves under the container default and the redirect silently does not
# run, giving no log file at all rather than an empty one. scripts/launch_30b.sh:141 is the same
# pattern.
set -u
cd /work/aupai || exit 1

OUT=runs/corpus_secret_report_0905.json
LOG=runs/corpus_secret_report_0905.log
CAP_BYTES=$((256 * 1024 * 1024))   # 256 MiB, per 4c: the log is under a byte cap

# All six code dirs, ~81 GB (4c chose (ii) over the mix's single code_rp1t: a filter decision made
# on 28% of the code is a decision about the wrong denominator).
#
# THE CAP IS A PIPE INTO head -c, which means SIGPIPE kills the scan if it ever produces 256 MiB.
# That is the intended behaviour, not a flaw: normal output is ~10 lines (one per domain plus the
# totals), so reaching the cap means the job has gone haywire per-row, and the 2026-09-02 incident
# was a 123 GiB task output filling the disk. A run killed at the cap has already written every
# completed domain to $OUT, because the report flushes atomically per domain.
#
# ${PIPESTATUS[0]}, NOT $? -- after a pipeline $? is head's status, so a python crash would report
# exit=0 with the log ending mid-scan. The exit code is the one thing this line exists to record.
python3 -u scripts/corpus_secret_report.py \
  --domains code code_dedup08 code_py_rp1t code_py_starcoder code_rp1t code_rp1t_rest \
  --root /work/aupai \
  --out "$OUT" \
  --skip-done \
  2>&1 | head -c "$CAP_BYTES"
rc=${PIPESTATUS[0]}

echo "exit=$rc out=$OUT"
exit "$rc"
