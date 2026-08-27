#!/usr/bin/env bash
# End-to-end acceptance test for the aupai CLI. Drives the built release binary and asserts real
# behavior — no unit tests, no mocks. Dependency-free (bash + the binary), runnable on mac and pod.
#
#   cli/e2e.sh            # builds if needed, runs all assertions
#   AUPAI_BIN=/path/aupai cli/e2e.sh   # test a specific binary
#
# Exits non-zero if any assertion fails. Every assertion prints PASS/FAIL with a one-line reason.

set -uo pipefail
cd "$(dirname "$0")"

BIN="${AUPAI_BIN:-target/release/aupai}"
if [ ! -x "$BIN" ]; then
  echo "building $BIN ..."
  cargo build --release >/dev/null 2>&1 || { echo "build failed"; exit 1; }
fi

PASS=0
FAIL=0
pass() { echo "PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "FAIL: $1"; FAIL=$((FAIL + 1)); }

# check_exit <expected> <name> -- <cmd...>
check_exit() {
  local want=$1 name=$2
  shift 3 # drop want, name, and the literal `--`
  "$@" >/dev/null 2>&1
  local got=$?
  if [ "$got" -eq "$want" ]; then pass "$name (exit $got)"; else fail "$name (exit $got, want $want)"; fi
}

# --- aupai list exits 0 and names all commands ---
OUT=$("$BIN" list 2>&1); RC=$?
if [ $RC -eq 0 ] && grep -q "train " <<<"$OUT" && grep -q "pipeline " <<<"$OUT" &&
  grep -q "ckpt " <<<"$OUT" && grep -q "status " <<<"$OUT" && grep -q "eval " <<<"$OUT"; then
  pass "list exits 0 and names commands (train/pipeline/ckpt/status/eval)"
else
  fail "list missing commands or bad exit ($RC)"
fi

# --- no 'wandb' anywhere in list/help (trackio is the tracker) ---
if { "$BIN" list; "$BIN" train --help; "$BIN" --help; } 2>&1 | grep -qi wandb; then
  fail "'wandb' appears in list/help output"
else
  pass "no 'wandb' in list/help output"
fi

# --- dry-run train BEFORE flags: exit 0, resolved-config block + plan, NO files created ---
BEFORE=$(ls -1 | wc -l)
OUT=$("$BIN" --dry-run train --name e2e_probe 2>&1); RC=$?
AFTER=$(ls -1 | wc -l)
if [ $RC -eq 0 ] && grep -q "resolved config" <<<"$OUT" && grep -q "run_ddp.sh --name e2e_probe" <<<"$OUT"; then
  pass "dry-run train prints resolved-config block + plan (exit 0)"
else
  fail "dry-run train missing config block or plan (exit $RC)"
fi
if [ ! -e "../ckpt_e2e_probe.pt" ] && [ ! -e "../runs/e2e_probe.pipeline.json" ] && [ "$BEFORE" -eq "$AFTER" ]; then
  pass "dry-run train created no files"
else
  fail "dry-run train touched disk"
fi

# --- CRITICAL: --dry-run AFTER a passthrough token must NOT execute (last=true fix) ---
# `train --fp8 --name foo --dry-run` should be a clap parse error (exit 2), never launch torchrun.
OUT=$("$BIN" train --fp8 --name foo --dry-run 2>&1); RC=$?
if [ $RC -eq 2 ] && grep -qi "unexpected argument" <<<"$OUT"; then
  pass "dry-run after a flag ERRORS (does not execute) — footgun closed"
elif grep -qi "resolved config\|run_ddp.sh" <<<"$OUT" && ! grep -qi "torchrun\|saved ckpt" <<<"$OUT"; then
  pass "dry-run after a flag stayed a dry-run (did not execute)"
else
  fail "dry-run after a flag may have executed (exit $RC)"
fi

# --- correct passthrough form previews and injects the recipe ---
OUT=$("$BIN" --dry-run train --name foo -- --fp8 2>&1); RC=$?
if [ $RC -eq 0 ] && grep -q "run_ddp.sh" <<<"$OUT" && grep -q -- "--fp8" <<<"$OUT"; then
  pass "dry-run train --name foo -- --fp8 previews with recipe"
else
  fail "correct passthrough form did not preview (exit $RC)"
fi

# --- eval fail-fast: missing checkpoint -> exit 2, our message, not a python traceback ---
OUT=$("$BIN" eval /nonexistent_ckpt_xyz.pt 2>&1); RC=$?
if [ $RC -eq 2 ] && grep -q "checkpoint not found" <<<"$OUT" && ! grep -qi "traceback" <<<"$OUT"; then
  pass "eval /nonexistent.pt fails fast (exit 2, 'checkpoint not found')"
else
  fail "eval fail-fast wrong (exit $RC): $OUT"
fi

# --- eval with no checkpoint arg -> usage error, exit 2 ---
check_exit 2 "eval (missing required arg)" -- "$BIN" eval

# --- pipeline --status of a nonexistent run -> exit 0, clear message, runs nothing ---
OUT=$("$BIN" pipeline --status nonexistent_run_xyz 2>&1); RC=$?
if [ $RC -eq 0 ] && grep -qi "no state\|nothing has run" <<<"$OUT"; then
  pass "pipeline --status nonexistent exits 0 with clear message"
else
  fail "pipeline --status nonexistent wrong (exit $RC): $OUT"
fi

# --- ckpt list exits 0 ---
check_exit 0 "ckpt list" -- "$BIN" ckpt list

# --- status exits 0 ---
check_exit 0 "status" -- "$BIN" status

# --- status --json exits 0 and is valid-ish JSON (starts with {) ---
OUT=$("$BIN" status --json 2>&1); RC=$?
if [ $RC -eq 0 ] && [ "${OUT:0:1}" = "{" ]; then
  pass "status --json exits 0 and emits JSON"
else
  fail "status --json wrong (exit $RC)"
fi

# --- a bad subcommand -> exit 2 ---
check_exit 2 "unknown subcommand" -- "$BIN" frobnicate

echo
echo "e2e: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
