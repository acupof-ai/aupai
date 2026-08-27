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

# Small domains, sequential + cross-excluding. SMALL_EX accumulates every small domain already built,
# so each new one dedups against the earlier ones (a code row present in en's source is dropped from en).
SMALL_EX=()
build_small() {  # <domain> <extra build_corpus args...>
  local d=$1; shift
  "${BC[@]}" --domain "$d" --filters light --target_tokens 1e9 "${SMALL_EX[@]}" "$@" > "$LOGS/$d.log" 2>&1
  [ -d "data/corpus/$d" ] && SMALL_EX+=(--exclude "data/corpus/$d/*.jsonl")
  tail -1 "$LOGS/$d.log"
}

has code && build_small code --source jsonl:data/code_filtered.jsonl
has en && build_small en \
  --source jsonl:data/cosmopedia_extra.jsonl \
  --source jsonl:data/en_textbook.jsonl
if has math; then
  # Only sources whose answers are the publisher's own. The data/math/*.jsonl files were written
  # by an answer extractor that has since been fixed, so they are left out until re-fetched.
  # gsm8k_zh (meta-math/GSM8K_zh = the GSM8K *train* split machine-translated to Chinese) is excluded:
  # eval/gsm8k.py scores the GSM8K *test* split, and keeping that benchmark's train distribution out of
  # pretrain is what lets us call the score clean. ~7.5K rows, negligible for a 1e9-token domain.
  # near-dedup (default on) collapses the ~68% template overlap across math_short_v2/v4.
  build_small math \
    --source jsonl:data/school_math_r1_zh.jsonl \
    --source jsonl:data/en_math_text.jsonl \
    --source "jsonl:data/synthetic/math_short_v*.jsonl"
fi
has chat && build_small chat \
  --source jsonl:data/coig.jsonl \
  --source jsonl:data/alpaca_gpt4_zh.jsonl
echo "--- small domains done ---"

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
