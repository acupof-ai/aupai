#!/bin/bash
# Resume 1 of 1.5b-a0.2b-e48_30b, from ckpt_1.5b-a0.2b-e48_30b.pt.step22500 to step 25430.
#
# Launch with (the card list is the caller's, see below):
#   CUDA_VISIBLE_DEVICES=1,2,3,4,5,7 setsid bash /work/aupai/runs/resume1_30b.sh \
#     > /work/aupai/runs/resume1_30b.wrap.log 2>&1 &
#
# WHY .step22500 AND NOT A LATER SAVE. Warmdown starts at step 22887 and save_every is 500, so
# .step23000 already carries annealed weights; resuming there would re-anneal them. .step22500 is
# the last save before warmdown, which is why the parent run was stopped at it deliberately rather
# than run to its end. Cost: 387 steps (0.30B tokens) for a clean lr state.
#
# ulimit -c 0 IS LOAD-BEARING, not hygiene. The stop's SIGKILL teardown wrote /work/aupai/core at
# 79.5 GB (4c, 16:20Z), and /work is at 97% with 71 GB free. This run keeps three rolling saves at
# 6.06 GB each, so one more core dump of that size fills the filesystem and takes the run with it.
# A core file is worth nothing here anyway: the process is killed on purpose at the stop.
#
# THE MIX IS CURSOR-SPECIFIC AND NOT REUSABLE. data/mix_1.5b-a0.2b-e48_30b_resume1.json is written
# against .step22500's row_cursor, so every `epochs` field in it is the TOTAL the model will have
# read, and the file is correct for exactly that checkpoint and wrong for any other. A stale copy
# generated before the deriver fix sat on the pod at 16:24Z with _launch_blocked
# ['chat_qa','chatml','cot'] -- launch_gate refuses a mix with _launch_blocked set, which is the
# second net, but the first is that this file names the mix the fixed generator wrote.
set -u
cd /work/aupai || exit 1
# No core dumps: see above. Set before torchrun so every rank inherits it.
ulimit -c 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# THE CARDS COME FROM THE CALLER. CUDA_VISIBLE_DEVICES is not additive -- assigning it here would
# REPLACE the block the launcher confined this run to, which is how a lane launch landed on a
# training-block card on 2026-08-31 (2f97e4a). Resume 1's block is 1,2,3,4,5,7 under the 09-06 user
# order (cards 0 and 6 are tileRL's); runs/card_assignment.json's block_cards says the same. The
# grant is not a reading: runs/claims/ and nvidia-smi -i 1 are checked immediately before launch.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "refusing: resume1_30b needs CUDA_VISIBLE_DEVICES from the caller (block 1,2,3,4,5,7)." >&2
  exit 2
fi
NCARDS=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
if [ "$NCARDS" -ne 6 ]; then
  echo "refusing: this is a world-6 resume but CUDA_VISIBLE_DEVICES exposes $NCARDS device(s) ($CUDA_VISIBLE_DEVICES). The parent ran world 6 and the optimizer state is sharded for it." >&2
  exit 2
fi
case ",$CUDA_VISIBLE_DEVICES," in
  *,0,*|*,6,*) echo "refusing: cards 0 and 6 are tileRL's under the 09-06 user order; got $CUDA_VISIBLE_DEVICES." >&2; exit 2 ;;
esac
MIX=data/mix_1.5b-a0.2b-e48_30b_resume1.json
if [ ! -s "$MIX" ]; then
  echo "refusing: $MIX is missing or empty. It is generated per-cursor by scripts/write_mix_500m.py --resume-cursor ckpt_1.5b-a0.2b-e48_30b.pt.step22500." >&2
  exit 2
fi
export NGPU=6
export PORT=29628
# torchrun DIRECTLY, NOT ./run_ddp.sh, and this is the arm's precision rather than a style choice.
# run_ddp.sh:136 injects --fp8 into every command it wraps, and train.py REFUSES --bf16 with --fp8
# ("--fp8 already casts the model to bf16 and then converts the linears to fp8 compute; --bf16 is
# the cast WITHOUT that conversion"). The parent 30B run launched torchrun directly with --bf16 and
# no --fp8 -- read from its own ledger cmd, not remembered -- so wrapping this resume in run_ddp.sh
# does two wrong things at once: it fails outright on the conflict, and had train.py merely
# preferred one flag it would have silently continued the arm in a different numeric format at
# step 22500. Measured: all six ranks refused at 18:00Z, exit 1, no checkpoint, no orphan.
# A RESUME INHERITS ITS PARENT'S LAUNCH SHAPE, not the repo's default wrapper.
exec python3 scripts/harness.py launch 1.5b-a0.2b-e48_30b_resume1 --training \
  --class incremental \
  --gate-timeout 900 \
  --hypothesis "Resume 1 of the MoE-48 30B leg from .step22500 (17.70B tokens read) to step 38146, under a cursor-derived mix: cot/chatml/chat_qa held at 4.000 TOTAL epochs rather than 4.000 per segment, and the share they release placed on math_owm_stage2 and code_py_starcoder under a 2.10 total-epoch ceiling. Does the val descent measured at -0.00981/1k (t=-26.17 over steps 12400-20600) continue through the warmdown, and does the HumanEval tie break at the endpoint (prereg runs/prereg.jsonl#moe48_30b_0907@amended_7)" \
  -- torchrun --nproc_per_node=6 --master_port=29628 train.py \
     --mix "$MIX" --name 1.5b-a0.2b-e48_30b \
     --resume ckpt_1.5b-a0.2b-e48_30b.pt.step22500 \
     --dim 1024 --layers 12 --heads 8 --ffn_hidden 3072 --batch 8 --accum 4 --bf16 \
     --no-grad_ckpt --lr_scale 1.0 --warmdown 0.1 --anneal_frac 0 --warmup 300 \
     --val_every 200 --save_every 500 --seed 42 \
     --moe_experts 48 --moe_top_k 3 --moe_expert_ffn 768 --moe_shared 1 --moe_layers 0-11 \
     --moe_bias_gamma 0.001 --moe_arm moe48_8b --attn_every 4
