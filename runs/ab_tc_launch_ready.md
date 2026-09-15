# v42 textbook-continuation A/B — LAUNCH-READY commands

Prepared 2026-09-15 (de) against the r3 FINAL. **DRAFT ONLY — does not launch.**
Both arms start only on explicit user go, after E0. One arm at a time, each
needs all 8 cards (set `euo pipefail`, detached per AGENTS.md pod rules).

## Join point (verified)

- Join checkpoint: `ckpt_v41_r3_0914.pt` — step **38070**, total_steps 38070,
  seed 42 (pod, 12.9 GB, carries `opt`; it is the r3 FINAL).
- Segment: **137** steps to total step 38207; world 8 × batch 4 × accum 6
  = 192 rows/step = 26,304 plan rows.
- `--warmdown 0.003586` = 137 / (38070 + 137), baked into BOTH mixes and asserted
  equal by the dry-run. `--warmup 0` (absolute-step warmup is dead post-resume),
  `--lr_scale 0.20` (join peak = 0.20 × r3 base LR; end floor 0.01),
  `--anneal_frac 0` (single phase; anneal == weight).
- Mixes regenerated against the final cursor and carry `_derived_against`
  (six-domain triple + seed matched, 0 mismatches; textbook domain has no cursor
  and legally starts at row 0). Plain `*_dc` caches only, no `cache_exclude`.

## Mixes (regenerated on the pod against the final ckpt)

- T `data/mix_textbook_cont.json`: textbook_claude_v41_dc weight **0.2999**
  = exactly **7,888 segment rows = 4 epochs** of the 1,972-row trainable pool
  (2,075 packed seq-rows − 103 val); six r3 domains share the 18,416 residual.
- C `data/mix_cont_ctrl.json`: same six r3 domains at anneal proportions
  (0.45/0.25/0.18/0.05/0.02/0.05), no textbook.
- Dry-run T/C proportionality is one common scale ~0.70009 (largest-remainder ±1 row).

## Dry-run evidence (CPU, no GPU; rerun verbatim before launch)

```
python3 scripts/write_mix_v42_stage2.py --ckpt ckpt_v41_r3_0914.pt
CUDA_VISIBLE_DEVICES= python3 scripts/dryrun_v42_textbook_ab.py \
    --t data/mix_textbook_cont.json --c data/mix_cont_ctrl.json \
    --ckpt ckpt_v41_r3_0914.pt --require-final-step 38070
# rc=0, last line: "OK and LAUNCH-READY: cursor is the r3 final (step 38070);
# both plans are 137 steps, six-domain T/C is one common scale, triple matched,
# no cap/stale."
```

## Launch commands (run from /work/aupai, only on explicit go)

The two arms differ ONLY in `--mix`/`--name`; every other flag is identical and
matches the r3 run's geometry byte-for-byte. `runs/v42_textbook_ab.sh
ckpt_v41_r3_0914.pt` prints these exact lines.

TREATMENT (run first), detached:

```bash
setsid nohup bash -c 'cd /work/aupai && ./run_ddp.sh \
  --mix data/mix_textbook_cont.json --name v42_textbook_t \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 6 \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 \
  --n_swa_only_layers 2 --no-attn_res \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 \
  --moe_layers 0-11 --moe_arm v41r3 --no-grad_ckpt \
  --lr_scale 0.20 --warmup 0 --warmdown 0.003586 --anneal_frac 0 --save_every 100 \
  --resume ckpt_v41_r3_0914.pt \
  > runs/v42_textbook_t.log 2>&1' </dev/null >/dev/null 2>&1 &
```

CONTROL (only after T finishes and frees all 8 cards):

```bash
setsid nohup bash -c 'cd /work/aupai && ./run_ddp.sh \
  --mix data/mix_cont_ctrl.json --name v42_textbook_c \
  --dim 1024 --layers 12 --heads 8 --ffn_hidden 6912 --batch 4 --accum 6 \
  --attn_every 1 --csa --csa2 --csa2_win_flash --rope_dims 64 \
  --n_swa_only_layers 2 --no-attn_res \
  --moe_experts 48 --moe_top_k 3 --moe_shared 1 --moe_expert_ffn 1728 \
  --moe_layers 0-11 --moe_arm v41r3 --no-grad_ckpt \
  --lr_scale 0.20 --warmup 0 --warmdown 0.003586 --anneal_frac 0 --save_every 100 \
  --resume ckpt_v41_r3_0914.pt \
  > runs/v42_textbook_c.log 2>&1' </dev/null >/dev/null 2>&1 &
```

## Artifacts and rollback

- Products: `ckpt_v42_textbook_t.pt` and `ckpt_v42_textbook_c.pt`, each at step
  38207 (run_ddp.sh also writes `.stepN` rollers; final is the no-suffix name).
- Logs: `runs/v42_textbook_t.log`, `runs/v42_textbook_c.log`; exp rows
  `v42_textbook_t` / `v42_textbook_c`.
- Both arms resume the r3 FINAL with a FRESH optimizer via `--resume` and do not
  overwrite r3 or each other (different `--name`). No rollback of r3 is needed:
  r3 final `ckpt_v41_r3_0914.pt` is untouched. To abort an arm mid-run, kill its
  torchrun PID group by exact PID and delete that arm's `ckpt_v42_textbook_{t,c}*`
  only; never touch `ckpt_v41_r3_0914.pt`.
- Post-arm eval (separate go, 66 owns pairing): n=10 T0.2 HE CLEAN156 + MBPP
  CLEAN338 on each arm via `bash runs/e0_n10.sh ckpt_v42_textbook_t.pt et` and
  `... ckpt_v42_textbook_c.pt ec`, then `eval/paired_bootstrap.py` one-sided 95%
  CI over the 156 CLEAN HE tasks (criterion: lower bound of mean d_i > 0, and
  ET not worse than E0).
