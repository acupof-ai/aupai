#!/usr/bin/env bash
# Move data/corpus/web (the UNFILTERED 2.99M-document crawl) out of the domain
# namespace, to data/_quarantine/web_unfiltered.
#
# Why this exists: train.py globs data/corpus/<domain>/, so a mix naming `web`
# instead of `web_hq` trains on the corpus the quality filter removed and the run
# looks perfectly ordinary while it does it. build_tokenizer.py had the same hole
# -- its stratified sample drowned in the documents the filter had just dropped.
# Both failures are silent. A name outside data/corpus/ cannot be resolved as a
# domain by anything, which is a stronger guarantee than remembering not to type it.
#
# Why RENAME and never delete: the unfiltered corpus is the only copy of the input
# the filter ran on, and a different quality threshold has to be re-cuttable from it.
# 13GB deleted is not recoverable; a rename is undone with one mv.
#
#   scripts/quarantine_web.sh [--dry|--selftest]
#
# Idempotent: a second run reports the corpus is already quarantined and exits 0.
set -euo pipefail
cd "$(dirname "$0")/.."

# One runnable check: drives this same script against a throwaway data root.
selftest() {
  local tmp
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  mkdir -p "$tmp/data/corpus/web"
  printf '{"a":1}\n{"a":2}\n' > "$tmp/data/corpus/web/shard_000.jsonl"

  printf '{"domains": {"web": {"weight": 1.0}}}' > "$tmp/data/mix.json"
  if AUPAI_ROOT="$tmp" "$0" --dry >/dev/null 2>&1; then
    echo "FAIL: ran while a mix still named 'web'" >&2; exit 1
  fi

  printf '{"domains": {"web_hq": {"weight": 1.0}}}' > "$tmp/data/mix.json"
  AUPAI_ROOT="$tmp" "$0" --dry | grep -q '2 documents' || { echo "FAIL: --dry doc count" >&2; exit 1; }
  [ -d "$tmp/data/corpus/web" ] || { echo "FAIL: --dry moved the corpus" >&2; exit 1; }

  AUPAI_ROOT="$tmp" "$0" >/dev/null
  [ -d "$tmp/data/_quarantine/web_unfiltered" ] || { echo "FAIL: not moved" >&2; exit 1; }
  [ ! -d "$tmp/data/corpus/web" ] || { echo "FAIL: source survived the move" >&2; exit 1; }

  AUPAI_ROOT="$tmp" "$0" | grep -q 'already quarantined' || { echo "FAIL: not idempotent" >&2; exit 1; }
  echo "selftest ok"
  exit 0
}

DRY=0
case "${1:-}" in
  --dry) DRY=1 ;;
  --selftest) selftest ;;
  "") ;;
  *) echo "usage: $0 [--dry|--selftest]" >&2; exit 2 ;;
esac

ROOT=${AUPAI_ROOT:-.}
SRC="$ROOT/data/corpus/web"
DST="$ROOT/data/_quarantine/web_unfiltered"

# Refuse while any mix still names the domain: quarantining first would turn a
# silent wrong-corpus run into a silent empty-domain run, which is not an
# improvement. Fix the mix, then quarantine.
BAD=$(
  python3 - "$ROOT" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
for path in sorted(glob.glob(os.path.join(root, "data", "**", "*.json"), recursive=True)):
    try:
        with open(path, encoding="utf-8") as fh:
            mix = json.load(fh)
    except (ValueError, OSError):
        continue
    if isinstance(mix, dict) and isinstance(mix.get("domains"), dict) and "web" in mix["domains"]:
        print(path)
PY
)
if [ -n "$BAD" ]; then
  echo "refusing: these mix files still name the domain 'web' (use web_hq):" >&2
  echo "$BAD" | sed 's/^/  /' >&2
  exit 1
fi

if [ ! -d "$SRC" ]; then
  if [ -d "$DST" ]; then
    echo "already quarantined: $DST (nothing to do)"
  else
    echo "nothing to do: neither $SRC nor $DST exists"
  fi
  exit 0
fi
if [ -d "$DST" ]; then
  echo "refusing: both $SRC and $DST exist; merging them would mix two cuts" >&2
  exit 1
fi

SIZE=$(du -sh "$SRC" | cut -f1)
SHARDS=$(find "$SRC" -name '*.jsonl' | wc -l | tr -d ' ')
DOCS=$(find "$SRC" -name '*.jsonl' -exec cat {} + | wc -l | tr -d ' ')

if [ "$DRY" = 1 ]; then
  echo "[dry] would move $SRC -> $DST"
  echo "[dry] $SIZE, $SHARDS shards, $DOCS documents"
  exit 0
fi

mkdir -p "$(dirname "$DST")"
mv "$SRC" "$DST"
echo "moved $SRC -> $DST"
echo "$SIZE, $SHARDS shards, $DOCS documents"
echo "undo: mv $DST $SRC"
