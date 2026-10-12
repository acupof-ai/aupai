#!/usr/bin/env bash
# Rebuild every artifact of the C1 "format and stop" set, then check them.
#
#   bash data/sft/stage_c/format_stop/build.sh            # both versions (v1 then v2)
#   bash data/sft/stage_c/format_stop/build.sh v1         # only v1 (the set D1 trained on)
#   bash data/sft/stage_c/format_stop/build.sh v2         # only v2 (the scaled set)
#
# Deterministic: the seed is fixed, the generator is build.py, and nothing here reads the
# network. Names carry the version, so a v2 build never overwrites v1's pack. Outputs, per
# version <v>, land next to this script:
#   format_stop_<v>.jsonl  the rows      format_stop_<v>.pt  the pack
#   spotcheck_200_<v>.jsonl              manifest_<v>.json   report_<v>.json
# Checks, in order, all of which must pass for the version being built:
#   1. scripts/test_sft_pack.py   the packer's own loss-mask gate (CI)
#   2. selfcheck.py               structure, with four negative worlds
#   3. maskcheck.py               the shipped pack's mask boundary and terminator
#   4. sft_math.py --check_pack   optional (C1_RUN_CHECK_PACK=1): the loader accepts the pack
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if [ "$#" -gt 0 ]; then
  VERSIONS=("$@")
else
  VERSIONS=(v1 v2)
fi

for V in "${VERSIONS[@]}"; do
  echo "== [$V] build: $HERE/build.py"
  "$PY" "$HERE/build.py" --version "$V" --out-dir "$HERE"

  echo "== [$V] gate 1/4: scripts/test_sft_pack.py (packer loss-mask gate)"
  "$PY" scripts/test_sft_pack.py
  echo "== [$V] gate 2/4: selfcheck.py (structure + 4 negative worlds)"
  "$PY" "$HERE/selfcheck.py" --set "$HERE/format_stop_$V.jsonl" --quiet
  echo "== [$V] gate 3/4: maskcheck.py (shipped pack: boundary + terminator)"
  "$PY" "$HERE/maskcheck.py" --jsonl "$HERE/format_stop_$V.jsonl" --pack "$HERE/format_stop_$V.pt"

  echo "== [$V] gate 4/4: sft_math.py --check_pack (opt-in, needs a local checkpoint)"
  # The pack's required keys, vocab_id and holdout_fp are already checked by maskcheck.py;
  # this gate is the loader's own acceptance. It is OPT-IN because it loads a whole
  # checkpoint, and on a laptop that load is heavy: with ckpt_local/ckpt_v42_sft_run.pt the
  # command below succeeded once (CHECK_PACK ok ... vocab_id=f1f860970d15d623 holdout_fp=
  # b382de09159701c0) and was killed by the OS on later attempts at the same command, with
  # no output beyond the RequestsDependencyWarning. Gate 4 therefore does not gate the
  # build. To run it:
  #   C1_RUN_CHECK_PACK=1 C1_CKPT=ckpt_local/ckpt_v42_sft_run.pt bash data/sft/stage_c/format_stop/build.sh v2
  if [ "${C1_RUN_CHECK_PACK:-0}" = "1" ]; then
    CKPT="${C1_CKPT:-ckpt_local/ckpt_v42_sft_run.pt}"
    if [ -f "$CKPT" ]; then
      "$PY" sft_math.py --check_pack --sft_path "$HERE/format_stop_$V.pt" --resume "$CKPT" 2>&1 | tail -2
    else
      echo "SKIP check_pack: $CKPT is absent; set C1_CKPT to a checkpoint"
    fi
  else
    echo "SKIP check_pack by default; see this script's gate 4 comment"
  fi

  echo "== [$V] done: $(wc -l < "$HERE/format_stop_$V.jsonl") rows, $(du -h "$HERE/format_stop_$V.pt" | cut -f1) pack"
done
