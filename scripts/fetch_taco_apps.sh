#!/bin/bash
# Fetch the TACO + APPS *train* splits for the reasoning SFT pack (de, 1e ruling 2026-09-27).
#
# Why: the existing SFT (sft_mixA_0924) is signature->code with no reasoning and no executable
# tests, and the disk held no problem-with-I/O-cases source (rl_code_apps.jsonl was empty).
# TACO (BAAI/TACO, 25,443 train problems) and APPS (codeparrot/apps, 5,000 train problems) both
# carry per-problem reference solutions and bundled stdin/stdout input_output cases, which the
# verifier runs in a CPU sandbox (datagen/... + algorithms/isolate.py) -- only solutions that
# pass their own cases are kept, then 13-gram decontaminated against HumanEval/MBPP.
#
# Network: host outbound IPv6/HF can fail; use curl -4 and the hf-mirror endpoint (then modelscope
# if a host moves), per the standing fetch rule. This script is idempotent and resumable: an
# already-complete file (size + sha256) is skipped, a partial/truncated one is re-fetched.
#
# Output dir (gitignored, not committed -- these are 2.2 GiB of raw data):
#   data/sft_raw/taco/train-NNNNN-of-00009.parquet
#   data/sft_raw/apps/train.jsonl
set -euo pipefail

MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"
DEST_ROOT="${SFT_RAW_DEST:-data/sft_raw}"
TACO_REPO="BAAI/TACO"
APPS_REPO="codeparrot/apps"
TACO_DIR="$DEST_ROOT/taco"
APPS_DIR="$DEST_ROOT/apps"

# sha256 captured from the 2026-09-27 mirror fetch; the verifier reads only files that match.
declare -A TACO_SHA=(
  [00000]=bee336c14dda183b1f700d54a149173418c7b3def295666159dd72c32aa8b326
  [00001]=1934cd4c4e8784bfc231b0294326a308dc2b7abf5e3c7f17622ece10ae8d8b56
  [00002]=1ad70829b190935c6cbded92d013b9b1fc10cd154dc0814c929716a1ea9ad1ae
  [00003]=934953c5693650a58ede19966db7ec1322f721e9c91b4d62e90a67dced86121d
  [00004]=d634726c27b85171494f7d38abfcd31d773a205002f9218bb54bc2cba49c99d4
  [00005]=c604f98dc0d1568939372f78fc6ed654c31b34ed3e0302636efcc8547802439c
  [00006]=e6ebb62153e93828e631f985d29466d7819262a6811a49b00a23a3cdb1904a11
  [00007]=e3dfc9a787b2234fbb25f31a7d15fad734c518c5fb6fc62a71616dae69944aef
  [00008]=16d88e9c6d4ddb5d28805dc6e03f2fb349ccf858436b86191f4c2b97170c1304
)
APPS_SHA="45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae"

sha_ok() { # sha_ok <expected> <path>
  [ -s "$2" ] && [ "$(sha256sum "$2" | cut -d' ' -f1)" = "$1" ]
}

fetch() { # fetch <url> <expected_sha> <out>
  if sha_ok "$2" "$3"; then echo "have $(basename "$3")"; return 0; fi
  echo "fetch $(basename "$3")"
  rm -f "$3"
  curl -4 -fL --retry 5 --retry-delay 5 --retry-all-errors --max-time 1800 "$1" -o "$3"
  sha_ok "$2" "$3" || { echo "sha mismatch for $3" >&2; exit 1; }
}

mkdir -p "$TACO_DIR" "$APPS_DIR"
for i in 00000 00001 00002 00003 00004 00005 00006 00007 00008; do
  fname="train-${i}-of-00009.parquet"
  fetch "$MIRROR/datasets/$TACO_REPO/resolve/main/ALL/$fname" "${TACO_SHA[$i]}" "$TACO_DIR/$fname"
done
fetch "$MIRROR/datasets/$APPS_REPO/resolve/main/train.jsonl" "$APPS_SHA" "$APPS_DIR/train.jsonl"
echo "fetched: $(du -sh "$TACO_DIR" "$APPS_DIR" | tr '\n' ' ')"
