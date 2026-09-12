#!/bin/bash
# Fetch N shards of AI-ModelScope/chinese-c4 (jsonl.zst) for 0e-6.
# curl -4, resume-capable. ~0.25B kept gate tokens per shard.
set -u
DIR=${1:-/data00/aupai_raw/zh_c4}
N=${2:-2}
mkdir -p "$DIR"
base="https://www.modelscope.cn/datasets/AI-ModelScope/chinese-c4/resolve/master/data"
ok=0
for i in $(seq 0 $((N - 1))); do
  f=$(printf "chinese-c4-%04d-of-0096.jsonl.zst" "$i")
  for t in 1 2 3; do
    if curl -4 -sL -C - --retry 3 -m 1800 -o "$DIR/$f" "$base/$f"; then ok=1; break; fi
    echo "fetch $f attempt $t failed ($(stat -c %s "$DIR/$f" 2>/dev/null || echo 0) bytes), retry" >&2
    sleep 5
  done
  [ "$ok" = 1 ] || { echo FETCH_FAIL "$f"; exit 1; }
done
echo "FETCH_OK $N shards in $DIR"
