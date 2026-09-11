#!/bin/bash
# V4.1 HumanEval gate run (prereg v41_gate_0911). DRAFT -- NOT LAUNCHED; the go is the
# controller's. Launch preconditions are in runs/prereg.jsonl#v41_gate_0911.
#
# Flags are the i smoke flags (v41_smoke_0911i.sh), changed only where the gate differs:
#   - mix_v41_gate.json (ae PR #246) replaces the smoke warmup mix
#   - world 6 on block 0-5 (NGPU=6; the harness grant names the cards, this only sizes run_ddp)
#   - --save_every 2000 (smoke used 100000 to suppress saves)
#   - --moe_arm v41gate
# Batch/accum: B4 is the compiled+flash ceiling from the ladder -- h B8 OOM 94.6 GiB pre-step,
# i B4/accum4 measured 72.64 GiB/rank (facts/v41.json#v41.smoke_compiled_flash_h_i_0911). accum
# does not change peak bytes, so accum8 grows the effective batch to 4*8*6*4096 = 786,432
# tokens/step while the per-rank peak stays at i's 72.6 GiB, under the 80 GiB ceiling.
# 30.0B tokens (data/mix_v41_gate.json total_tokens, main 7ba4568f) / 786,432 ~= 38.1K steps.
#
# LR SCHEDULE RULED BY fb 2026-09-11 (prereg v41_gate_0911 amendment 1): warmup 500 absolute
# steps (1.3% of 38.1K; the 30B run of 09-07 used the same order; 20 is a smoke warmup),
# warmdown 0.65, anneal_frac 0.10 (last 10% uses the gate mix's per-domain anneal=weight).
# B4xaccum8 is the fixed recipe: when de-108 lands its gain goes to wall-clock, batch does not
# move, so the prereg measures one shape.
# --csa2_win_flash added per fb 2026-09-11 (prereg amendment 6): smoke j peak 43.5 vs
# 72.6 GiB (i); adopted on memory -- the 1.4x speed criterion FAILED (measured 1.11x).
# Recipe otherwise unchanged (B4/accum8, world 6); no B8 test before the go.
cd /work/aupai || exit 1
export NGPU=6
exec python3 scripts/harness.py launch v41_gate_0911 --training --class incremental --hypothesis "V4.1 flat CSA2 MoE on the UltraData gate mix clears HumanEval pass@1 >= 30% (prereg v41_gate_0911)" \
  -- ./run_ddp.sh --mix data/mix_v41_gate.json --name v41_gate_0911 \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 8 \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.10 --warmup 500 --save_every 2000 --no-grad_ckpt \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --n_swa_only_layers 2 --no-attn_res \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11 --moe_arm v41gate
