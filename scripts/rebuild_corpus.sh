#!/usr/bin/env bash
# Rebuild corpus domains from reproducible sources, dropping belle-derived data.
#
#   bash scripts/rebuild_corpus.sh math     # rebuild just one domain (fast, verifies)
#   bash scripts/rebuild_corpus.sh web      # the big fineweb2 pull
#   bash scripts/rebuild_corpus.sh          # all domains
#
# Each domain rm's its old data/corpus/<domain>/ first, so stale shards do not survive.
# Small domains run before web; web --excludes all of them.
set -euo pipefail
cd "$(dirname "$0")/.."
BC=(python3 datagen/build_corpus.py)
BCW=(python3 datagen/build_corpus.py --no_near_dedup)
DOMAINS=${DOMAINS:-}
JOBS=${*:-all}
LOGS=/tmp/corpus_rebuild
mkdir -p "$LOGS"
DIFFS=0
# accepts "$1" being a single domain or a space-separated list
has() {
  [ "$JOBS" = "all" ] && return 0
  case " $JOBS " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

echo "[rebuild] dropping belle/school_math_r1_zh; math keeps en_math_text + synthetic math_short_v*"
echo "[rebuild] web -> fineweb2 cmn_Hani (target 3e9 tok)"

# --- small domains -----------------------------------------------------------
if has code; then
  rm -rf data/corpus/code; mkdir -p data/corpus/code
  "${BC[@]}" --domain code --filters light --target_tokens 1e9 --no_near_dedup \
    --source jsonl:data/code_filtered.jsonl > "$LOGS/code.log" 2>&1
fi
if has en; then
  rm -rf data/corpus/en; mkdir -p data/corpus/en
  "${BC[@]}" --domain en --filters light --target_tokens 1e9 --no_near_dedup \
    --source jsonl:data/cosmopedia_extra.jsonl --source jsonl:data/en_textbook.jsonl \
    > "$LOGS/en.log" 2>&1
fi
if has chat; then
  rm -rf data/corpus/chat; mkdir -p data/corpus/chat
  "${BC[@]}" --domain chat --filters light --target_tokens 1e9 --no_near_dedup \
    --source jsonl:data/coig.jsonl --source jsonl:data/alpaca_gpt4_zh.jsonl \
    > "$LOGS/chat.log" 2>&1
fi
if has math; then
  rm -rf data/corpus/math; mkdir -p data/corpus/math
  # pass 1: publisher files, exact-dedup only, NO school_math_r1_zh (belle)
  "${BC[@]}" --domain math --filters light --target_tokens 8e8 --no_near_dedup \
    --source jsonl:data/en_math_text.jsonl > "$LOGS/math.log" 2>&1
  # pass 2: synthetic templates with near-dedup ON, excluding pass 1
  "${BC[@]}" --domain math --filters light --target_tokens 2e8 \
    --exclude "data/corpus/math/*.jsonl" \
    --source "jsonl:data/synthetic/math_short_v*.jsonl" >> "$LOGS/math.log" 2>&1
fi
for d in code en math chat; do
  has "$d" && echo "[$d] $(grep -h 'docs in' "$LOGS/$d.log" 2>/dev/null | tail -1)"
done

# --- web: fineweb2-cmn (reproducible), excludes the small domains -------
if has web; then
  rm -rf data/corpus/web; mkdir -p data/corpus/web
  EX=()
  for d in code en math chat; do
    [ -d "data/corpus/$d" ] && EX+=(--exclude "data/corpus/$d/*.jsonl")
  done
  # local fineweb2-cmn parquets, not a fresh HF pull: the pod has no HF route, and
  # these parquets are the reproducible fineweb2 source.
  FW2=${FW2:-/data00/fw2raw}
  "${BCW[@]}" --domain web --target_tokens 3e9 --source "parquet:$FW2/*.parquet" "${EX[@]}" \
    > "$LOGS/web.log" 2>&1
  echo "[web] $(grep -h 'docs in' "$LOGS/web.log" | tail -1)"
fi
echo REBUILD_DONE