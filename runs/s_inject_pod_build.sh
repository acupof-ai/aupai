#!/usr/bin/env bash
# Build experiment-1's injection shards ON THE POD, so the arms read pod-built data and the
# shas can be compared against a local build. A SCRIPT rather than an inline pod command
# because ~/bin/pod re-parses argv in a shell: parentheses are a syntax error and quoted
# strings word-split (memory: pod-argv-cannot-carry-prose).
set -u
cd /work/aupai || exit 1
OUT=runs/s_inject_build.log
: > "$OUT"
/usr/bin/python3 datagen/build_s_inject.py --n 1 8 64 256 >> "$OUT" 2>&1
echo "BUILD_RC=$?" >> "$OUT"
# The shas the local build is compared against. Printed into the same log so one file
# carries both the build and its fingerprints.
for f in s_inject_n1 s_inject_n8 s_inject_n64 s_inject_n256 p_format; do
  p="data/corpus/$f/${f}_000.jsonl"
  if [ -f "$p" ]; then
    printf 'SHA %-16s %s %s\n' "$f" "$(sha256sum "$p" | cut -d" " -f1)" "$(wc -l < "$p")" >> "$OUT"
  else
    printf 'SHA %-16s MISSING\n' "$f" >> "$OUT"
  fi
done
echo "DONE" >> "$OUT"
