#!/usr/bin/env bash
# Build data/corpus/<domain>/ for the default mix (train.py Cfg.mix), from the sources that exist on the pod.
#
#   bash scripts/build_domains.sh          # all domains
#   DOMAINS="math chat" bash scripts/build_domains.sh
#
# Order matters: small domains first, each excluding the ones already built (a document must not
# be counted once as `code` and again as `en`/`chat`); web last with all of them pre-seeded into
# its dedup set. data/pretrain_full.jsonl is a superset of every other jsonl here, so the
# exclusion is what keeps web from double-counting them.
#
# Near-dedup is off for web (pure-python MinHash ~30ms/doc = 24h for 11M docs; both web inputs
# are already deduplicated) and ON for the small domains (cheap at their size; synthetic math
# templates benefit from near-dup removal across versions).
set -euo pipefail
cd "$(dirname "$0")/.."
BC=(python3 datagen/build_corpus.py)
BCW=(python3 datagen/build_corpus.py --no_near_dedup)
FW2=${FW2:-/data00/fw2raw}
DOMAINS=${DOMAINS:-"code en math chat web"}
LOGS=${LOGS:-/tmp/corpus}
mkdir -p "$LOGS"

has() { [[ " $DOMAINS " == *" $1 "* ]]; }

# Small domains run in PARALLEL: no cross-domain overlap (coig/school_math/gsm8k are disjoint), so
# independent exact-dedup sets are safe; web still --excludes them all. near-dedup is the slow part
# (~30ms/doc), so it is applied ONLY where template overlap exists: the synthetic math_short_v* files.
# Everything else, incl. the big publisher math files, runs exact-dedup only.

has code && "${BC[@]}" --domain code --filters light --target_tokens 1e9 --no_near_dedup \
  --source jsonl:data/code_filtered.jsonl > "$LOGS/code.log" 2>&1 &
has en && "${BC[@]}" --domain en --filters light --target_tokens 1e9 --no_near_dedup \
  --source jsonl:data/cosmopedia_extra.jsonl \
  --source jsonl:data/en_textbook.jsonl > "$LOGS/en.log" 2>&1 &
has chat && "${BC[@]}" --domain chat --filters light --target_tokens 1e9 --no_near_dedup \
  --source jsonl:data/coig.jsonl \
  --source jsonl:data/alpaca_gpt4_zh.jsonl > "$LOGS/chat.log" 2>&1 &
if has math; then
  # math in two passes into the same dir: publisher files exact-dedup only (fast), then synthetic
  # templates with near-dedup ON, excluding the first pass. gsm8k_zh stays out: eval/gsm8k.py scores
  # the GSM8K test split.
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
  # The pattern stays quoted: build_corpus.py globs it itself; unquoted, the shell expands it and
  # argparse rejects the stray positionals.
  EX=()
  for d in code en math chat; do
    [ -d "data/corpus/$d" ] && EX+=(--exclude "data/corpus/$d/*.jsonl")
  done
  # One process per input shard (~55 min for 11M docs single-pass). Each keeps its own exact-dedup
  # set; cross-shard duplicates are what the upstream dedup already removed, so the only cost is
  # per-shard histograms.
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
