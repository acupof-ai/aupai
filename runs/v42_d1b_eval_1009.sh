#!/bin/bash
# Evaluate the D1 checkpoint: the decisive read on H1 (does format/stop data fix E2?).
#
# E2 first (it is the level D1 targets), then E1 (must not regress: the base's E1 failure
# is repetition, and D1 trains exactly the behaviour that is supposed to suppress it).
# E0 runs first so the chain has its bottom rung: E2's prereq is E0, and check_prereq now
# refuses a stage whose rung is missing or failed.
#
# One card at a time; the card is named by CUDA_VISIBLE_DEVICES and the code takes cuda:0.
set -uo pipefail
cd /work/aupai
NAME=v42_d1b_format_stop_1009
CKPT=ckpt_v42_d1b_format_stop_1009.pt
LOG=runs/${NAME}_eval.log
: > "$LOG"

source eval/_devs.sh 4          # this driver uses up to four shard devices

run_on () {   # run_on <i> <stage args...>
  i="$1"; shift
  # `${_DEVS[$i]}` INLINE, unquoted: harness check device_set_honoured's _CVD_SAFE accepts
  # only `=${_DEVS[`, `=${CUDA_VISIBLE_DEVICES:-`, or an empty/-1 set. Going through a local
  # `card=` variable reads as a physical index to the check, and the check is right to flag
  # it -- the idiom it can prove is the one worth keeping.
  echo "=== card ${_DEVS[$i]}: eval_ladder.py $* : $(date -u '+%F %T') UTC" >> "$LOG"
  CUDA_VISIBLE_DEVICES=${_DEVS[$i]} python3 eval/eval_ladder.py "$@" >> "$LOG" 2>&1
  echo "rc=$? for: $*" >> "$LOG"
}

[ -f "$CKPT" ] || { echo "REFUSING_missing_D1_ckpt -- training still running or failed" | tee -a "$LOG"; exit 2; }

run_on 0 e0 --ckpt "$CKPT" --device cuda:0 --expect-engram 1,5,9,13,17,21 \
       --expect-engram-keys 24 --expect-vocab-id f1f860970d15d623
run_on 1 e2 --ckpt "$CKPT" --device cuda:0 --control
run_on 2 e1 --ckpt "$CKPT" --device cuda:0 --control --skip-prereq
echo "=== D1 eval done: $(date -u '+%F %T') UTC" >> "$LOG"
python3 - <<'PY' >> runs/v42_d1b_format_stop_1009_eval.log 2>&1
import json, pathlib
for stage in ("e0", "e2", "e1"):
    p = pathlib.Path("runs/ladder/ckpt_v42_d1b_format_stop_1009") / stage / "summary.json"
    if not p.exists():
        print(f"{stage}: NO SUMMARY at {p}"); continue
    d = json.loads(p.read_text())
    print(f"{stage}: n={d.get('n')} passed={d.get('passed')} exit={d.get('exit_code')} "
          f"gates={json.dumps(d.get('gates'), ensure_ascii=False)}")
PY
