#!/bin/bash
cd /work/aupai || exit 1
for d in wiki_chat math_seed textbook_30b zh_web; do
  echo "=== $d ==="
  python3 runs/count_dir.py "data/corpus/$d" 32 2>&1 | tail -3
done
echo "=== T44 DONE ==="
