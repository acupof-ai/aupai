#!/bin/bash
# launch_30b.sh -- the t22 30B pretrain as ONE reviewed command, not a line typed at 3am.
# Every recipe decision of today lives here with its source; the readiness board names
# this file. Runs through `harness launch --training`, which allocates the block cards,
# derives NGPU from the allocation, arms the fa/doc_mask startup gate, and monitors.
#
# Recipe (docs/lessons/scale_36b_plan.md §1b, docs/lessons/readout_30b_prereg.md):
#   mix          data/mix_30b.json   -- 8 capability domains (t30 contract)
#   seq          4096
#   warmup       300                 -- 0.9% of ~32.7k steps; the ladder's 20 is 0.06% here (t02 note)
#   save_every   500                 -- ~15 min, 0.55% wall; resume within t01 noise (t38, eff.ckpt_resume_16h_interval)
#   AttnRes      on (attn_res_blocks 0 = Full)
#   fp8          on                  -- run_ddp/train pass --fp8; bf16 params, NO fp32 master (t01: delta +0.068 < 0.24)
#   NGPU         from allocation     -- harness launch sets it (de 9da7333)
#
# Usage:
#   scripts/launch_30b.sh --dry            print the resolved torchrun line + readiness, exit non-zero while blocked
#   scripts/launch_30b.sh                   launch (refuses unless ready)
#   scripts/launch_30b.sh --resume <ckpt>   continue with identical flags (gate-timeout 300 for the large ckpt load)
set -uo pipefail
cd "$(dirname "$0")/.."

NAME=pretrain_30b
MIX=data/mix_30b.json
RESUME=""
DRY=0
for a in "$@"; do
  case "$a" in
    --dry) DRY=1 ;;
    --resume) RESUME_NEXT=1 ;;
    *) if [ "${RESUME_NEXT:-0}" = 1 ]; then RESUME=$a; RESUME_NEXT=0; fi ;;
  esac
done

# The recipe, one place. Bench/train flags only -- cards and NGPU come from the allocation.
FLAGS="--mix $MIX --seq 4096 --warmup 300 --save_every 500 --attn_res_blocks 0 --attn_every 4 \
--batch 16 --accum 2 --vocab 32784 --bucket_cap_mb 50 --seed 0 --name $NAME"
[ -n "$RESUME" ] && FLAGS="$FLAGS --resume $RESUME"

# Readiness: the mix contract (weights sum to 1.0, no ladder-name reuse, landed domains
# stamped) AND that nothing is still _blocked. A dry run reports; a real launch refuses.
contract=$(python3 scripts/harness.py check --only mix_30b_contract 2>&1)
blocked=$(python3 -c "import json;m=json.load(open('$MIX'));print(' '.join(sorted(m.get('_blocked',{}))))")

echo "== launch_30b readiness =="
echo "$contract" | grep -E "mix_30b_contract" || echo "$contract" | tail -2
if [ -n "$blocked" ]; then
  echo "NOT READY: ${blocked// /, } still in _blocked -- these 30B domains are not stamped yet."
  READY=0
else
  echo "READY: all domains stamped, none blocked."
  READY=1
fi

# The resolved command, always printed so a reviewer sees exactly what would run.
GATE=120; [ -n "$RESUME" ] && GATE=300   # a 959MB+ ckpt load exceeds the default gate (t38)
echo "== resolved command =="
echo "python3 scripts/harness.py launch $NAME --training --gate-timeout $GATE \\"
echo "  --hypothesis '30B pretrain (t22): $(echo $FLAGS | tr -s ' ')' \\"
echo "  -- bash run_ddp.sh $FLAGS"

if [ "$DRY" = 1 ]; then
  [ "$READY" = 1 ] && exit 0 || exit 1
fi
if [ "$READY" != 1 ]; then
  echo "refusing to launch: mix has blocked domains. Re-run when 3b stamps them." >&2
  exit 1
fi
exec python3 scripts/harness.py launch "$NAME" --training --gate-timeout "$GATE" \
  --hypothesis "30B pretrain (t22): $(echo $FLAGS | tr -s ' ')" \
  -- bash run_ddp.sh $FLAGS
