#!/bin/bash
# v42 gate, world-16 resume: node 0 of 2 (pod, 192.168.29.37), 8 cards each side.
# Tokens/step is held at 786,432 by MB 4 x ACC 3 x world 16 (was 4 x 6 x 8), so the
# step schedule, warmdown and anneal are unchanged; only wall clock halves.
# The resume checkpoint is the interrupt written when the world-8 run was SIGTERMed;
# scripts/rehearse_cursor.py gates the cursor before this script runs.
# node 1 is h20b (192.168.29.36), launched FIRST via /root/v42_w16_node1.sh there:
# torchrun rendezvous waits on the master, so start order is node1 then node0.
cd /work/aupai || exit 1
RESUME="${1:?usage: v42_w16_node0.sh <interrupt-ckpt>}"
[ -f "$RESUME" ] || { echo "resume checkpoint not found: $RESUME"; exit 1; }
export PREC_FLAG=--bf16
export NGPU=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.29.37 PORT=29500
export NCCL_SOCKET_IFNAME=eth0
MB=4
ACC=3
exec python3 scripts/harness.py launch v42_gate_1001r \
  --training --class incremental --gate-timeout 3000 \
  --hypothesis 'the world-16 resume preserves the world-8 trajectory (tokens/step identical at 786432 via accum 6->3) and halves wall clock; the falsifier is s/step >= 9 (interconnect-bound, revert to world-8) or val rising over any 3 consecutive marks' \
  -- ./run_ddp.sh --mix data/mix_v41_gate.json --name v42_gate_1001r \
  --arch v42 --moe_arm v42b \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch "$MB" --accum "$ACC" \
  --lr_scale 1.0 --warmdown 0.65 --anneal_frac 0.10 --warmup 500 --save_every 2000 --no-grad_ckpt \
  --v42_impl attn_impl=fused,rope_impl=real,moe_stacked=1,hc_impl=liger,norm_impl=liger,attn_logit_softcap=50 \
  --v42_lr 1e-3 --v42_record \
  --resume "$RESUME"
