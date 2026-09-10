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
# 110.74B tokens / 786,432 ~= 140.8k steps.
#
# OPEN BEFORE LAUNCH (docs/standards/v41_pivot.md names no LR schedule): warmdown 0.65 /
# anneal_frac 0.10 / warmup 20 are Cfg defaults carried forward from the i smoke (0.65, 0.0,
# 20); the controller confirms or replaces them. anneal_frac 0.10 also matches the gate mix's
# anneal=weight two-phase convention.
cd /work/aupai || exit 1
export NGPU=6
exec python3 scripts/harness.py launch v41_gate_0911 --training --class incremental --hypothesis "V4.1 flat CSA2 MoE on the UltraData gate mix clears HumanEval pass@1 >= 30% (prereg v41_gate_0911)" \
  -- ./run_ddp.sh --mix data/mix_v41_gate.json --name v41_gate_0911 \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 8 \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.10 --warmup 20 --save_every 2000 --no-grad_ckpt \
  --attn_every 1 --csa --csa2 --rope_dims 64 --n_swa_only_layers 2 --no-attn_res \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11 --moe_arm v41gate
