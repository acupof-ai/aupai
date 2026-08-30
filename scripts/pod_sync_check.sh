#!/bin/bash
# Minimal drift check: sha256 of the files that execute on the pod (pretrain ->
# score flow), local working tree vs /work/aupai. The pod is not a git repo and
# code arrives by hand-push, so without this the two diverge silently.
# Exit 1 on any DIFF or MISSING. The file set comes from scripts/pod_drift.py
# (the same set the pod's manifest gate enforces) -- one scope, two directions.
cd "$(dirname "$0")/.."
FILES=$(python scripts/pod_drift.py --list-scoped | tr '\n' ' ')
REMOTE=$(~/bin/pod "cd /work/aupai && sha256sum $FILES 2>/dev/null" < /dev/null)
fail=0
for f in $FILES; do
  lh=$(shasum -a 256 "$f" | cut -d' ' -f1)
  rh=$(printf '%s\n' "$REMOTE" | awk -v f="$f" '$2==f{print $1}')
  if [ -z "$rh" ]; then echo "MISSING on pod: $f"; fail=1
  elif [ "$lh" != "$rh" ]; then echo "DIFF: $f"; fail=1
  fi
done
[ $fail -eq 0 ] && echo "pod in sync ($(echo "$FILES" | wc -w | tr -d ' ') files)"
exit $fail
