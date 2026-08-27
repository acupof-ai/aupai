#!/usr/bin/env bash
# Build data/corpus/<domain>/ for data/mix.json, from the sources that exist on the pod.
#
#   bash scripts/build_domains.sh          # all domains
#   DOMAINS="math chat" bash scripts/build_domains.sh
#
# Order matters: the small domains are built first, then web is built with their documents
# pre-seeded into its dedup set (--exclude), so a code or English document is not counted
# once as `code` and again as `web`. data/pretrain_full.jsonl is a superset of every other
# jsonl here (measured 2026-08-26: skypile / cosmopedia / code_filtered / en_math_text /
# en_textbook are 100% contained in it), which is exactly why the exclusion is needed.
#
# Near-dedup is off throughout: the pure-python MinHash costs ~30ms/doc (24h for 11M docs) and
# both inputs are already deduplicated -- fineweb-2 by its own pipeline, pretrain_full when it
# was assembled. Exact dedup still runs.
set -euo pipefail
cd "$(dirname "$0")/.."
BC="python3 datagen/build_corpus.py --no_near_dedup"
FW2=${FW2:-/data00/fw2raw}
DOMAINS=${DOMAINS:-"code en math chat web"}
LOGS=${LOGS:-/tmp/corpus}
mkdir -p "$LOGS"

has() { [[ " $DOMAINS " == *" $1 "* ]]; }

if has code; then
  $BC --domain code --filters light --target_tokens 1e9 \
    --source jsonl:data/code_filtered.jsonl > "$LOGS/code.log" 2>&1 &
fi
if has en; then
  $BC --domain en --filters light --target_tokens 1e9 \
    --source jsonl:data/cosmopedia_extra.jsonl \
    --source jsonl:data/en_textbook.jsonl > "$LOGS/en.log" 2>&1 &
fi
if has math; then
  # Only sources whose answers are the publisher's own. The data/math/*.jsonl files were written
  # by an answer extractor that has since been fixed, so they are left out until re-fetched.
  # gsm8k_zh (meta-math/GSM8K_zh = the GSM8K *train* split machine-translated to Chinese) is excluded:
  # eval/gsm8k.py scores the GSM8K *test* split, and keeping that benchmark's train distribution out of
  # pretrain is what lets us call the score clean. ~7.5K rows, negligible for a 1e9-token domain.
  $BC --domain math --filters light --target_tokens 1e9 \
    --source jsonl:data/school_math_r1_zh.jsonl \
    --source jsonl:data/en_math_text.jsonl \
    --source "jsonl:data/synthetic/math_short_v*.jsonl" > "$LOGS/math.log" 2>&1 &
fi
if has chat; then
  $BC --domain chat --filters light --target_tokens 1e9 \
    --source jsonl:data/coig.jsonl \
    --source jsonl:data/alpaca_gpt4_zh.jsonl > "$LOGS/chat.log" 2>&1 &
fi
wait
echo "--- small domains done ---"
for d in code en math chat; do
  has "$d" && tail -1 "$LOGS/$d.log"
done

if has web; then
  # An array, and the pattern stays quoted: build_corpus.py globs it itself. Unquoted, the shell
  # expands it first and argparse sees one --exclude plus N stray positionals, which it rejects.
  EX=()
  for d in code en math chat; do
    [ -d "data/corpus/$d" ] && EX+=(--exclude "data/corpus/$d/*.jsonl")
  done
  # One process per input shard: a single pass over 11M documents at ~3.4K docs/s is ~55 min.
  # Each keeps its own exact-dedup set; cross-shard duplicates are what the upstream dedup
  # already removed, so the only cost is that the histograms are per-shard.
  # shellcheck disable=SC2086
  for p in "$FW2"/*.parquet; do
    n=$(basename "$p" .parquet)
    $BC --domain web --target_tokens 3e9 --source "parquet:$p" "${EX[@]}" > "$LOGS/web_$n.log" 2>&1 &
  done
  # shellcheck disable=SC2086
  $BC --domain web --target_tokens 3e9 --source jsonl:data/pretrain_full.jsonl "${EX[@]}" \
    > "$LOGS/web_pretrain_full.log" 2>&1 &
  wait
  echo "--- web done ---"
  for f in "$LOGS"/web_*.log; do echo "$(basename "$f"): $(grep -m1 'docs in' "$f")"; done
fi

echo "=== corpus sizes ==="
du -sh data/corpus/*/ 2>/dev/null
echo BUILD_DOMAINS_DONE
