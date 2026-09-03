#!/bin/bash
# de-14: run_ddp.sh's sync-stamp guard, on known answers.
#
# The guard is four lines of case/test in run_ddp.sh:41-56. This file exists because those
# four lines are the last thing between a 66-hour run and a stamp nobody verified, and a
# shell conditional has no selftest of its own.
#
# THE LOGIC IS DUPLICATED HERE, and that is a real cost: an edit to run_ddp.sh does not
# change this file. The alternative was extracting the guard into a sourced function, which
# adds a file the launch path must find on the pod -- a new failure mode on the exact path
# being protected. Six readings against a copy is the cheaper trade, and the copy is short
# enough to compare by eye. If run_ddp.sh's guard changes, change this too.
#
# The reading that earned it: `read -r _sha _dirty _when` COLLAPSES whitespace, so a stamp
# whose sha field is empty shifts `dirty` into `_sha`. "0" is valid hex and clears the case
# pattern, so only the LENGTH test refuses it -- the ordering of the two tests is load-bearing
# and was verified, not assumed.
#
#   bash scripts/test_stamp_guard.sh
mkdir -p /tmp/de14
check() {
  printf '%s 0 t\n' "$1" > /tmp/de14/s
  read -r _sha _dirty _when < /tmp/de14/s
  case "$_sha" in
    *[!0-9a-f]* | "") echo "REFUSE non-hex"; return 1 ;;
  esac
  if [ ${#_sha} -ne 40 ]; then echo "REFUSE ${#_sha}-char"; return 1; fi
  echo "ACCEPT 40-char"
}
fails=0
r=$(check "69c8bd87b48dc0cf1f509788de81a0636ecf2a62"); echo "  full sha        -> $r"
[ "$r" = "ACCEPT 40-char" ] || { echo "  FAIL: a real full sha was refused"; fails=1; }
r=$(check "8cd6834"); echo "  7-char (de-38)  -> $r"
[ "$r" = "REFUSE 7-char" ] || { echo "  FAIL: an abbreviated sha was accepted"; fails=1; }
r=$(check "8cd68340"); echo "  8-char (de-38)  -> $r"
[ "$r" = "REFUSE 8-char" ] || { echo "  FAIL: the 8-char form --short now emits was accepted"; fails=1; }
r=$(check "deadbeefzz"); echo "  non-hex         -> $r"
[ "$r" = "REFUSE non-hex" ] || { echo "  FAIL: a non-hex stamp was accepted"; fails=1; }
# An EMPTY first field is not the same test: `read` collapses whitespace, so a stamp whose
# sha is missing shifts dirty into _sha. That is worse than a malformed sha -- "0" is valid
# hex and clears the case pattern -- and only the LENGTH test catches it. Written as the real
# file shape (" 0 t"), not as an empty string handed to printf.
printf ' 0 t\n' > /tmp/de14/s
read -r _sha _d _w < /tmp/de14/s
echo "  sha field empty -> _sha=[$_sha] len=${#_sha}"
[ "$_sha" = "0" ] && [ ${#_sha} -ne 40 ] || { echo "  FAIL: the shift is not caught by length"; fails=1; }
# and a wholly empty file leaves _sha unset
: > /tmp/de14/s
_sha=; read -r _sha _d _w < /tmp/de14/s
[ -z "$_sha" ] && echo "  empty file      -> _sha empty, caught by the case pattern" || { echo "  FAIL: empty file"; fails=1; }
rm -rf /tmp/de14
[ $fails -eq 0 ] && echo "stamp guard: OK (6 readings: full sha, 7-char, 8-char, non-hex, empty sha field, empty file)" || echo "stamp guard: DEFECT"
exit $fails
