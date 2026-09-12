#!/bin/bash
# Fetch the 0e-6 Chinese source: pleisto/wikipedia-cn-20230720-filtered.
# Outbound curl -4, follow the hf-mirror -> xet redirect, resume-capable.
set -u
DIR=${1:-/data00/aupai_raw/zhwiki}
mkdir -p "$DIR"
F="$DIR/wikipedia-cn.json"
URL="https://hf-mirror.com/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/wikipedia-cn-20230720-filtered.json"
ok=0
for i in $(seq 1 8); do
  if curl -4 -fSL -C - --retry 3 -m 1800 -o "$F" "$URL"; then ok=1; break; fi
  echo "fetch attempt $i failed ($(stat -c %s "$F" 2>/dev/null || echo 0) bytes), retry" >&2
  sleep 5
done
[ "$ok" = 1 ] || { echo FETCH_FAIL; exit 1; }
# JSON array must end with ] and parse
python3 - "$F" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
assert isinstance(d, list) and d and isinstance(d[0], dict) and "completion" in d[0]
print("FETCH_OK", len(d), "docs", __import__("os").path.getsize(p), "bytes")
PY
