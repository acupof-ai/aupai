#!/bin/bash
# AttnRes A/B, serial, both arms in the same window: attn_res_blocks=0 (Full) vs 8.
#
# The first attempt was invalidated twice over. Both invocations passed --ar-blocks 0,
# so the arms were identical; and the surviving arm straddled a deadlocked job, an eval
# and a recompile while its counterpart ran in a quiet window, so it reported 2.76x
# slower for 6x LESS work. Neither defect is visible in a result that looks well-formed.
#
# So this runs both arms back to back in one window, from one script, with only
# --attn_res_blocks differing. The gap between the arms is minutes, not hours.
#
# Full reads 2145 sources at L=32 against Block-8's 353 -- the question is what that
# costs in throughput, and it is frozen at launch because AttnRes is a frozen key.
#
# Cards come from the caller. Card 7 is another session's eval, so this is world=6 --
# what matters is that both arms see the same world, not which world.
set -u
cd /work/aupai

source eval/_devs.sh 6 || exit 1
WANT=",$(IFS=,; echo "${_DEVS[*]}"),"
NW=${#_DEVS[@]}

busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F', ' -v want="$WANT" 'index(want, ","$1",") && $2 > 1024 {print $1}' | tr '\n' ' ')
if [ -n "$busy" ]; then
  echo "refusing: card(s) $busy are in use; an A/B whose arms see different load measures the load"
  exit 1
fi

rm -rf /tmp/torchinductor_root

for ar in 0 8; do
  echo "=== ARM attn_res_blocks=$ar world=$NW cards ${_DEVS[*]} $(date -u +%H:%M:%S)"
  NGPU="$NW" PORT=29541 ./run_ddp.sh \
    --mix data/mix_30b_stage2.json --name "ab_ar${ar}" \
    --dim 1024 --layers 32 --heads 8 --ffn_hidden 3072 \
    --batch 32 --accum 1 --grad_ckpt --lr_scale 0.85 \
    --attn_res_blocks "$ar" \
    --max_steps 40 --save_every 100000 --val_every 0 \
    > "runs/ab_ar${ar}.log" 2>&1
  echo "=== ARM ar${ar} exit=$? $(date -u +%H:%M:%S)"
  grep -E '^step |Error|out of memory|Traceback' "runs/ab_ar${ar}.log" | tail -4
done
echo AB_ALL_DONE
