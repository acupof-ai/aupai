#!/usr/bin/env bash
# infra-layer: moves to aupai-infra at the split (6e ruling 2026-09-04).
#
# The vendored wrappers' refusals, against real command shapes. A refusal that
# fires on everything is as useless as one that fires on nothing, so every case
# below names which side it proves -- the negative cases are the load-bearing
# half (a guard with no negative control is not verified).
#
# restartable: pure string tests, no pod contact, <1s. Rerun freely.
set -u
POD="$(cd "$(dirname "$0")" && pwd)/pod"
PODPUT="$(cd "$(dirname "$0")" && pwd)/podput"
fail=0

# TN_STUB makes the wrappers testable without a pod: `pod` execs `tn exec`, so a
# stub on PATH turns a would-be network call into an echo. Without this the
# negative cases could not run at all, and only the refusals would be tested --
# which is exactly the blind spot this file exists to close.
stub=$(mktemp -d); trap 'rm -rf "$stub"' EXIT
printf '#!/bin/sh\necho "TN-CALLED: $*"\n' > "$stub/tn"; chmod +x "$stub/tn"

refuses() {  # refuses "<desc>" "<cmd>" -- must exit nonzero and say why
  local desc=$1 cmd=$2 out rc
  out=$(PATH="$stub:$PATH" "$POD" "$cmd" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then
    echo "FAIL [$desc]: accepted, should refuse -- $cmd"; fail=1
  elif ! printf '%s' "$out" | grep -q 'refusing'; then
    echo "FAIL [$desc]: refused without saying 'refusing' -- $out"; fail=1
  fi
}
accepts() {  # accepts "<desc>" "<cmd>" -- must reach tn
  local desc=$1 cmd=$2 out
  out=$(PATH="$stub:$PATH" "$POD" "$cmd" 2>&1)
  if ! printf '%s' "$out" | grep -q 'TN-CALLED'; then
    echo "FAIL [$desc]: did not reach tn -- $out"; fail=1
  fi
}

# MUST REFUSE: the shape that produces no log file at all (shape 166), and the
# two 17-hour loops that were live when this was written.
refuses "cd && bg"      'cd /work/aupai && python3 x.py > log 2>&1 &'
refuses "cd; setsid bg" 'cd /work/aupai; setsid nohup python3 x.py > log 2>&1 &'
refuses "cd && loop bg" 'cd /work/aupai && until grep -q DONE runs/x.log; do sleep 5; done &'

# MUST ACCEPT: these are the cases that decide whether the guard discriminates
# or just matches "cd". A cd with no backgrounding is the single most common
# thing anyone types at this tool.
accepts "cd, foreground"     'cd /work/aupai && python3 scripts/exp.py list'
accepts "no cd, background"  'setsid bash runs/foo.sh > runs/foo.log 2>&1 &'
accepts "no cd at all"       'nvidia-smi'
accepts "cd, pipeline"       'cd /work/aupai && ls runs | head -3'
accepts "ampersand inside"   'cd /work/aupai && grep -c "a && b" AGENTS.md'

# The two 17-hour loops live on the pod 2026-09-04, verbatim. They ACCEPT, and
# that is the honest result: their argv carries no cd and no trailing &, so no
# launch-time inspection of the command string can see the missing cd. Asserted
# here so nobody later reads the refusal as covering them -- they are class (b)
# in env_hygiene, caught at runtime by resolving the target against the process's
# own cwd. Two different halves of one failure; neither guard subsumes the other.
accepts "live loop, no cd"   'until grep -q ALL-DONE runs/count_en_c4_both.log; do sleep 5; done; echo DONE'

# --view must not be swallowed into the command (it was, in the first draft).
out=$("$POD" --view 2>&1)
printf '%s' "$out" | grep -q 'container view' || { echo "FAIL: --view not handled: $out"; fail=1; }

# podput: an absolute remote path is the whole point of the refusal.
out=$("$PODPUT" /etc/hosts relative/path.txt 2>&1); rc=$?
[ $rc -ne 0 ] && printf '%s' "$out" | grep -q 'absolute' || { echo "FAIL: podput took a relative remote path"; fail=1; }

[ $fail -eq 0 ] && echo "pod/podput refusals: all cases pass (3 refuse, 6 accept, 2 flag)"
exit $fail
