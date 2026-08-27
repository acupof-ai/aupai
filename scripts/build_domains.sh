#!/usr/bin/env bash
# Build data/corpus/<domain>/ for data/mix.json, from the sources that exist on the pod.
#
#   bash scripts/build_domains.sh          # all domains
#   DOMAINS="math chat" bash scripts/build_domains.sh
#
# Order matters: the small domains are built first (sequentially, each excluding the ones already
# built so a document is not counted once as `code` and again as `en`/`chat`), then web is built
# with all of them pre-seeded into its dedup set (--exclude). data/pretrain_full.jsonl is a superset
# of every other jsonl here (measured 2026-08-26: skypile / cosmopedia / code_filtered / en_math_text /
# en_textbook are 100% contained in it), which is exactly why the exclusion is needed.
#
# Near-dedup is off for the large web build (pure-python MinHash ~30ms/doc = 24h for 11M docs; both
# web inputs are already deduplicated). It is ON for the small domains: they are small enough that the
# cost is trivial and template-heavy synthetic math benefits from near-dup removal across versions.
set -euo pipefail
cd "$(dirname "$0")/.."
BC=(python3 datagen/build_corpus.py)
BCW=(python3 datagen/build_corpus.py --no_near_dedup)   # web: near-dedup off (too slow at 11M docs)
FW2=${FW2:-/data00/fw2raw}
DOMAINS=${DOMAINS:-"code en math chat web"}
LOGS=${LOGS:-/tmp/corpus}
mkdir -p "$LOGS"

has() { [[ " $DOMAINS " == *" $1 "* ]]; }

# Small domains run in PARALLEL (they have no cross-domain overlap -- coig/school_math/gsm8k measured
# disjoint 2026-08-27 -- so independent exact-dedup sets are safe; the web build still --excludes them
# all). near-dedup is the slow part (pure-python MinHash ~30ms/doc), so it is applied ONLY where template
# overlap actually exists: the synthetic math_short_v* files. Everything else, incl. the big publisher
# math files (school_math_r1 214MB, en_math 307MB), runs exact-dedup only.

has code && "${BC[@]}" --domain code --filters light --target_tokens 1e9 --no_near_dedup \
  --source jsonl:data/code_filtered.jsonl > "$LOGS/code.log" 2>&1 &
has en && "${BC[@]}" --domain en --filters light --target_tokens 1e9 --no_near_dedup \
  --source jsonl:data/cosmopedia_extra.jsonl \
  --source jsonl:data/en_textbook.jsonl > "$LOGS/en.log" 2>&1 &
has chat && "${BC[@]}" --domain chat --filters light --target_tokens 1e9 --no_near_dedup \
  --source jsonl:data/coig.jsonl \
  --source jsonl:data/alpaca_gpt4_zh.jsonl > "$LOGS/chat.log" 2>&1 &
if has math; then
  # math in two passes into the same domain dir: the big publisher files exact-dedup only (fast), then
  # the small synthetic templates with near-dedup ON (collapses math_short_v2/v4's ~68% overlap), excluding
  # the first pass. gsm8k_zh (GSM8K train, MT'd zh) stays out -- eval/gsm8k.py scores the GSM8K test split.
  ( "${BC[@]}" --domain math --filters light --target_tokens 8e8 --no_near_dedup \
      --source jsonl:data/school_math_r1_zh.jsonl \
      --source jsonl:data/en_math_text.jsonl > "$LOGS/math.log" 2>&1
    "${BC[@]}" --domain math --filters light --target_tokens 2e8 \
      --exclude "data/corpus/math/*.jsonl" \
      --source "jsonl:data/synthetic/math_short_v*.jsonl" >> "$LOGS/math.log" 2>&1 ) &
fi
wait
echo "--- small domains done ---"
for d in code en math chat; do has "$d" && echo "[$d] $(grep -h 'docs in' "$LOGS/$d.log" 2>/dev/null | tail -1)"; done

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
    "${BCW[@]}" --domain web --target_tokens 3e9 --source "parquet:$p" "${EX[@]}" > "$LOGS/web_$n.log" 2>&1 &
  done
  "${BCW[@]}" --domain web --target_tokens 3e9 --source jsonl:data/pretrain_full.jsonl "${EX[@]}" \
    > "$LOGS/web_pretrain_full.log" 2>&1 &
  wait
  echo "--- web done ---"
  for f in "$LOGS"/web_*.log; do echo "$(basename "$f"): $(grep -m1 'docs in' "$f")"; done
fi

echo "=== corpus sizes ==="
du -sh data/corpus/*/ 2>/dev/null
echo BUILD_DOMAINS_DONE
