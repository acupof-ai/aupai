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
set -uo pipefail
cd "$(dirname "$0")/.."

STAGE=""; RESUME=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE=$2; shift 2 ;;
    --resume) RESUME=$2; shift 2 ;;
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

FLAGS="--mix $MIX --seq 4096 --warmup 300 --save_every 500 --attn_res_blocks 0 --attn_every 4 \
--batch 16 --accum 2 --vocab 32784 --bucket_cap_mb 50 --seed 0 $EXTRA --name $NAME"

# Readiness: the mix contract + nothing still _blocked, read from THIS stage's own mix.
contract=$(python3 scripts/harness.py check --only mix_30b_contract 2>&1)
blocked=$(python3 -c "import json;m=json.load(open('$MIX'));print(' '.join(sorted(m.get('_blocked',{}))))")

echo "== launch_30b stage $STAGE readiness ($MIX) =="
echo "$contract" | grep -E "mix_30b_contract" || echo "$contract" | tail -2
if [ -n "$blocked" ]; then
  echo "NOT READY: ${blocked// /, } still in _blocked -- not stamped yet."
  READY=0
else
  echo "READY: all domains stamped, none blocked."
  READY=1
fi

GATE=120; [ -n "$RESUME" ] && GATE=300   # a 959MB+ ckpt load exceeds the default gate (t38)
echo "== resolved command =="
echo "python3 scripts/harness.py launch $NAME --training --gate-timeout $GATE \\"
echo "  --hypothesis 'staged 30B (t22) stage $STAGE: $(echo $FLAGS | tr -s ' ')' \\"
echo "  -- bash run_ddp.sh $FLAGS"

if [ "$DRY" = 1 ]; then
  [ "$READY" = 1 ] && exit 0 || exit 1
fi
if [ "$READY" != 1 ]; then
  echo "refusing to launch: mix has blocked domains. Re-run when they stamp." >&2
  exit 1
fi
exec python3 scripts/harness.py launch "$NAME" --training --gate-timeout "$GATE" \
  --hypothesis "staged 30B (t22) stage $STAGE: $(echo $FLAGS | tr -s ' ')" \
  -- bash run_ddp.sh $FLAGS
