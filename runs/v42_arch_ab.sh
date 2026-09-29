#!/bin/bash
# v42 architecture A/B, 1.0B tokens each from scratch on data/mix_v42_ab_1b.json (1e, 2026-09-29).
#   ARM=A  the v41_ced_0926 architecture: 12 layers (CED 6/6), all CSA2, 48 experts top-3 x 1728,
#          sigmoid router, tied head. 3.22B total / 355.4M active.
#   ARM=B  V4.1-aligned candidate: 24 layers (CED 12/12), first two pure SWA, encoder CSA2
#          F,R,R,R,R,R,F,R,R,R (Full every 6), decoders Full, 64 experts top-8 x 640 + 1 shared,
#          sqrtsoftplus router x1.5, untied head. 3.26B total / 620.7M active (meta build).
# Everything else is the same line: world 8, B4/accum6, seq 4096, warmup 100, warmdown 0.65,
# anneal 0.10, stochastic rounding, router lr 0.001, same mix, same seed.
# ARM B changes seven things at once. It answers "adopt the package or not", not which change
# carries the delta.
cd /work/aupai || exit 1
: "${ARM:?set ARM=A or ARM=B}"
NAME="v42_ab_${ARM}_$(date +%m%d)"
case "$ARM" in
  A) SHAPE="--layers 12 --ffn_hidden 6912 --ced_enc_layers 6 --moe_experts 48 --moe_top_k 3 --moe_expert_ffn 1728 --moe_layers 0-11 --router_score sigmoid" ;;
  B) SHAPE="--layers 24 --ffn_hidden 5760 --ced_enc_layers 12 --attn_hybrid --n_swa_only_layers 2 --csa2_modes F,R,R,R,R,R,F,R,R,R,F,F,F,F,F,F,F,F,F,F,F,F --moe_experts 64 --moe_top_k 8 --moe_expert_ffn 640 --moe_layers 0-23 --router_score sqrtsoftplus --moe_routed_scale 1.5 --untie_head" ;;
  *) echo "ARM must be A or B" >&2; exit 2 ;;
esac
# MB is the per-rank micro-batch; accum follows so every arm steps 4*6*8*4096 = 786,432 tokens.
# B has twice A's layers and so about twice its activations; if B4 does not fit, run MB=2.
MB=${MB:-4}
case "$MB" in 1|2|3|4|6) ;; *) echo "MB must divide 24" >&2; exit 2 ;; esac
ACC=$((24 / MB))
export NGPU=8
# shellcheck disable=SC2086
exec python3 scripts/harness.py launch "$NAME" \
  --training --class incremental --gate-timeout 3000 \
  --hypothesis "v42 arch A/B arm $ARM at 1.0B tokens: arm B (V4.1-aligned, 620.7M active) reaches lower val and lower HumanEval gold bpb than arm A (355.4M active) at equal tokens; step time is read beside it" \
  -- ./run_ddp.sh --mix data/mix_v42_ab_1b.json --name "$NAME" \
  --dim 1024 --heads 8 --batch "$MB" --accum "$ACC" \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.10 --warmup 100 --save_every 100000 --val_every 250 --no-grad_ckpt \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 --no-attn_res --ced \
  --moe_shared 1 --moe_arm "v42$ARM" --stochastic_round --moe_router_lr 0.001 \
  $SHAPE
