#!/bin/bash
# launch_30b.sh -- the staged 15B->30B pretrain (t22) as ONE reviewed command, not a line
# typed at 3am. Every recipe decision lives here with its source; the readiness board names
# this file. Runs through `harness launch --training` (block cards, NGPU from allocation,
# fa/doc_mask startup gate, monitor).
#
# WSD staged schedule (t47, verified 0.2b->0.3b: JOIN at stable lr, plan rebuilt from the
# second mix, step continues):
#   stage 1  mix_15b_stage1.json  --warmdown 0  (anneal_frac 0 asserted from the file)
#            -> warmup 300 + stable lr to 15B, NO anneal, ends at stable lr for the join
#   stage 2  --resume <15B ckpt> --mix mix_30b.json --warmdown 0.10
#            -> resumes at the stable lr, continues the absolute step, anneals the last 10% at 30B
# Shared recipe (docs/lessons/scale_36b_plan.md §1b, readout_30b_prereg.md):
#   seq 4096, warmup 300, save_every 500 (t38 eff.ckpt_resume_16h_interval), AttnRes on,
#   fp8, bf16 params NO fp32 master (t01: delta +0.068 < 0.24), NGPU from allocation (de 9da7333).
#
# Usage:
#   scripts/launch_30b.sh --stage 1 [--dry]              stage 1 (15B)
#   scripts/launch_30b.sh --stage 2 --resume <ckpt> [--dry]   stage 2 (resume into 30B)
#   --dry prints the resolved torchrun line + the STAGE's own readiness, exit non-zero while blocked.
#   --gate-timeout N overrides the startup gate, which `harness launch` otherwise derives
#     from the mix's own cache bytes; pass it only for a case that derivation cannot see.
set -uo pipefail
cd "$(dirname "$0")/.."

STAGE=""; RESUME=""; DRY=0; GATE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE=$2; shift 2 ;;
    --resume) RESUME=$2; shift 2 ;;
    --gate-timeout) GATE=$2; shift 2 ;;
    --dry) DRY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ "$STAGE" = 1 ] || [ "$STAGE" = 2 ] || { echo "usage: --stage 1|2 [--resume <ckpt>] [--dry]" >&2; exit 2; }

if [ "$STAGE" = 1 ]; then
  NAME=pretrain_15b_s1; MIX=data/mix_15b_stage1.json; WARMDOWN=0
  # stage 1 must not anneal: the file's anneal_frac is the contract, assert it here so a
  # mis-set file cannot silently anneal stage 1 and break the stable-lr handoff.
  af=$(python3 -c "import json;print(json.load(open('$MIX')).get('anneal_frac','MISSING'))" 2>&1)
  if [ "$af" != "0" ] && [ "$af" != "0.0" ]; then
    echo "refusing: $MIX anneal_frac=$af, stage 1 needs 0 (WSD: ends at stable lr for the join)" >&2
    exit 2
  fi
  EXTRA="--warmdown $WARMDOWN --anneal_frac 0"
else
  NAME=pretrain_30b_s2; MIX=data/mix_30b.json; WARMDOWN=0.10
  [ -n "$RESUME" ] || { echo "refusing: stage 2 needs --resume <stage-1 ckpt>" >&2; exit 2; }
  EXTRA="--warmdown $WARMDOWN --resume $RESUME"
fi

# --seed 42, not 0. train.py:1733 applies flags with `if hasattr(Cfg,k) and v` and 0 is
# falsy, so the --seed 0 this script used to pass was dropped and Cfg.seed kept its default
# 42. Stage 1 ran under 42 (b0 audit 2026-08-31); stage 2 states it so the two stages share
# one documented seed and the value in the command is the value in effect. de fixes the
# apply after stage 1 ends; until then no flag whose valid value is 0 or "" can be trusted.
FLAGS="--mix $MIX --seq 4096 --warmup 300 --save_every 500 --attn_res_blocks 0 --attn_every 4 \
--batch 16 --accum 2 --vocab 32784 --bucket_cap_mb 50 --seed 42 $EXTRA --name $NAME"

# Readiness: the mix contract + nothing still _blocked, read from THIS stage's own mix so
# the line a person reads at launch names the mix being launched (not always mix_30b.json).
contract=$(python3 -c "import sys; sys.path.insert(0,'scripts'); import harness; s,e=harness.check_mix_30b_contract('.', '$MIX'); print(f'  [{s}] mix_contract {e}')" 2>&1)
blocked=$(python3 -c "import json;m=json.load(open('$MIX'));print(' '.join(sorted(m.get('_blocked',{}))))")

echo "== launch_30b stage $STAGE readiness ($MIX) =="
echo "$contract"
READY=1
# Any FAIL blocks. The contract line was printed but never tested, so a mix whose weights
# summed to 1.69013 printed [FAIL] and READY on adjacent lines and exited 0 (b0 G0).
# Matched on the absence of PASS, not the presence of FAIL, so a crashed check that prints
# a traceback also blocks rather than passing for lack of the word.
case "$contract" in
  *"[PASS]"*) ;;
  *) echo "BLOCKED: mix contract did not pass."; READY=0 ;;
esac
if [ -n "$blocked" ]; then
  echo "BLOCKED: ${blocked// /, } still in _blocked -- not stamped yet."
  READY=0
fi
[ "$READY" = 1 ] && echo "READY: contract passed, all domains stamped, none blocked."
  echo "READY: all domains stamped, none blocked."
  READY=1
fi

# Startup gate: derived by `harness launch` from the mix's own cache bytes (de), so this
# script names no number. build_mix (train.py:1807) runs BEFORE the fa/doc_mask gate lines
# (train.py:1886) and train.py:1396 torch.loads every domain's FULL cache on every rank --
# 149 GiB x 7 ranks for stage 1, which took 386 s on 2026-08-31 and would have been killed
# by the 120 s default. --gate-timeout still overrides for a case the derivation cannot see.
if [ -n "$GATE" ]; then GATE_ARG="--gate-timeout $GATE"; else GATE_ARG=""; fi
echo "== resolved command =="
echo "python3 scripts/harness.py launch $NAME --training $GATE_ARG --auto-resume 2 \\"
echo "  --hypothesis 'staged 30B (t22) stage $STAGE: $(echo $FLAGS | tr -s ' ')' \\"
echo "  -- bash run_ddp.sh $FLAGS"

if [ "$DRY" = 1 ]; then
  [ "$READY" = 1 ] && exit 0 || exit 1
fi
if [ "$READY" != 1 ]; then
  echo "refusing to launch: mix has blocked domains. Re-run when they stamp." >&2
  exit 1
fi
# --auto-resume makes harness launch a BLOCKING supervisor that must outlive the child, so
# this whole script has to be detached (setsid nohup), not just the torchrun inside it.
exec python3 scripts/harness.py launch "$NAME" --training $GATE_ARG --auto-resume 2 \
  --hypothesis "staged 30B (t22) stage $STAGE: $(echo $FLAGS | tr -s ' ')" \
  -- bash run_ddp.sh $FLAGS
