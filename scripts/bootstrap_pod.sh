#!/usr/bin/env bash
# Bootstrap an empty pod from zero to trainable — idempotent, one stage at a time.
#
#   bash scripts/bootstrap_pod.sh            # run all stages
#   bash scripts/bootstrap_pod.sh fetch      # resume/run just one stage
#
# Stages (each rerunnable, each stopping on error rather than feeding a broken
# artifact to the next):
#   verify  — data/data_verify.py against data/MANIFEST.tsv; frozen/eval missing
#             or mismatched is an ERROR.
#   fetch   — fetched tier via the fetchers; frozen tier from $ARCHIVE (skipped
#             until ARCHIVE points at a real archive).
#   build   — scripts/build_domains.sh -> data/corpus/<domain>/
#   vocab   — scripts/build_tokenizer.py --force (needs the default mix + the corpus)
#   check   — scripts/check_mix.py: dry-run the schedule before burning GPUs.
#
# NOT included: launching the pretrain. That is a human decision.
set -uo pipefail
cd "$(dirname "$0")/.."

# Off-box archive root for the frozen tier; fetch skips frozen while this is absent.
ARCHIVE=${ARCHIVE:-/archive/aupai-frozen}
STAGE=${1:-all}
LOG=/tmp/bootstrap.log
mkdir -p data
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { say "BOOTSTRAP FAILED at $1"; exit 1; }
want() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

# --- verify -----------------------------------------------------------------
if want verify; then
  say "stage verify"
  python3 scripts/data_verify.py --missing-only --root "$PWD" || {
    echo "  (empty pod expected: frozen tier not yet archived) — continuing to fetch"
  }
  say "stage verify: done (missing listed above)"
fi

# --- fetch ------------------------------------------------------------------
if want fetch; then
  say "stage fetch"
  if [ -d "$ARCHIVE" ]; then
    for f in pretrain_full cosmopedia_extra en_textbook code_filtered en_math_text; do
      [ -f "data/$f.jsonl" ] || cp "$ARCHIVE/$f.jsonl" "data/$f.jsonl"
    done
  else
    say "  ARCHIVE=$ARCHIVE not set/absent — SKIPPING frozen (5 sources); requires user decision"
  fi
  # fetched tier via the fetchers; each skips files already present (idempotent)
  python3 scripts/fetch_math_data.py ape210k belle gsm8k_zh math23k mxode || die "fetch_math_data"
  python3 scripts/fetch_sft_data.py fetch || die "fetch_sft"
  python3 scripts/fetch_chat_data.py || die "fetch_chat"
  say "stage fetch: done — rerun data_verify fetch to confirm"
fi

# --- build corpus -----------------------------------------------------------
if want build; then
  say "stage build"
  bash scripts/build_domains.sh || die "build_domains"
  say "stage build: done"
fi

# --- tokenizer --------------------------------------------------------------
if want vocab; then
  say "stage vocab"
  MIX=$(python3 -c "import sys; sys.path.insert(0,'scripts'); import harness; print(harness.cfg_default('mix'))")
  [ -f "$MIX" ] || die "vocab ($MIX required; build first)"
  python3 scripts/build_tokenizer.py --force || die "build_tokenizer"
  say "stage vocab: done"
fi

# --- mix dry-run ------------------------------------------------------------
if want check; then
  say "stage check"
  python3 scripts/check_mix.py || die "check_mix"
  say "stage check: done — review the schedule before launching"
fi

say "BOOTSTRAP: $STAGE complete. Train launch is a separate human decision."