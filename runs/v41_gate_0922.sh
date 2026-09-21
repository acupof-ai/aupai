#!/bin/bash
# V4.1 HumanEval gate run, FRESH world-8 start (2026-09-22).
#
# WHY A NEW NAME AND NOT v41_gate_0911. That run has ten amendments hanging on it and was stopped
# by the user once already (0916 pod loss). Reusing the name would make "which attempt is this"
# unanswerable from the ledger.
#
# NOT A RESUME, and this is not a choice: no v41_gate checkpoint survives the 2026-09-16 pod
# destruction (`find /data00 /work/aupai -name '*v41_gate*.pt*'` = empty), so
# runs/v41_gate_0911_resume_w8.sh cannot start at all -- its first line is
# `--resume ckpt_v41_gate_0911.pt.step6000`.
#
# Flags are runs/v41_gate_0911.sh's, changed ONLY where world 8 requires it:
#   - NGPU 6 -> 8, and the grant is block 0-7 with NO lane (runs/card_assignment.json, 2026-09-22)
#   - accum 8 -> 6, to HOLD THE EFFECTIVE BATCH IDENTICAL: 8*4*6*4096 = 6*4*8*4096 = 786,432
#     tokens/step, so the 30.0B / 786,432 = 38,146-step schedule and the LR schedule are unchanged.
#     accum does not move per-rank peak bytes (the i/h ladder showed peak is set by batch, not
#     accum), so the accum change buys wall-clock and nothing else.
# Everything else is byte-for-byte the 0911 line: --csa2_win_flash (the j/adopted win-flash peak
# 43.5 GiB vs 72.6 unflagged), B4, warmup 500 absolute / warmdown 0.65 / anneal_frac 0.10,
# --save_every 2000, --moe_arm v41gate, the same MoE/CSA2/RoPE64 flags.
#
# PRECONDITION, not checked here: harness launch reads the grant and refuses a card outside it.
cd /work/aupai || exit 1
export NGPU=8
exec python3 scripts/harness.py launch v41_gate_0922 --training --class incremental --hypothesis "V4.1 flat CSA2 MoE on the UltraData gate mix clears HumanEval pass@1 >= 30% (fresh world-8 start; supersedes the stopped v41_gate_0911)" \
  -- ./run_ddp.sh --mix data/mix_v41_gate.json --name v41_gate_0922 \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 6 \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.10 --warmup 500 --save_every 2000 --no-grad_ckpt \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --n_swa_only_layers 2 --no-attn_res \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 --moe_layers 0-11 --moe_arm v41gate
