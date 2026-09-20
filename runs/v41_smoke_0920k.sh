#!/bin/bash
# smoke k: the "does it taste right" run after the pod came back on a hostPath /work.
#
# SHAPE IS READ FROM THE RECORD, not composed: flags are runs/v41_smoke_0911j.sh (the
# recorded --csa2_win_flash arm, facts/v41.json#v41.smoke_compiled_flash_h_i_0911 for the
# batch ladder). Two deltas only, both from the controller's order:
#   1. --cards 0,1 / NGPU=2 (the j and i records are world 4 and world 8)
#   2. --mix data/mix_sample.json
#
# WHY mix_sample AND NOT mix_smoke_warmup (j's mix): measured on the pod 2026-09-20, every one
# of mix_smoke_warmup's six domains has 0 shards in data/corpus/ -- the run would refuse at
# startup. mix_sample's single 'sample' domain has 148 shards / 3180 docs and is tracked in git,
# so it needs no new corpus.
#
# WORLD CHANGES THE STEP BUDGET, NOT THE PER-RANK PEAK. The record's peak (72.64 GiB/rank,
# batch 4) is quoted for world 8; accum and world do not move per-rank peak, but they do move
# tokens/step, so --max_steps must not be used to bound this run -- the warmdown start would
# shift and the loss curve would sit on a different LR schedule, which is the trap the j
# launcher's own comment records (i: total 381, j: total 150, different schedules).
cd /work/aupai || exit 1
export NGPU=2
exec python3 scripts/harness.py launch v41_smoke_0920k --training --cards 0,1 --class infra-verification \
  --hypothesis "the V4.1 compiled flash-SWA stack trains on the rebuilt pod: first step's loss and throughput match the recorded smoke within their spread, at the recorded per-rank peak" \
  -- ./run_ddp.sh --mix data/mix_sample.json --name v41_smoke_0920k \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 4 \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.0 --warmup 20 --save_every 100000 --no-grad_ckpt \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --n_swa_only_layers 2 --no-attn_res \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11 --moe_arm v41smoke
