# v42 phi post-r3 SFT — go/no-go dry check (2026-09-15, 3b)

Cardless, no-GPU, no-launch readiness check for `runs/v42_phi_sft.sh`. Nothing here started
training, touched a GPU, or changed script logic. The actual SFT needs an explicit user go.

## Gate results (all measured on the pod against current main)

| check | result |
|---|---|
| resume ckpt | `ckpt_v41_r3_0914.pt` present, step 38070 / total 38070 (r3 FINAL), vocab_id `f1f860970d15d623` |
| pack present | `data/sft/sft_phi_codeexercises_v42_65m_0914.pt` (774,729,109 B), 23,637 rows × 4097 |
| pack gate `--check_pack` | **RC=0** — vocab_id matches, 63.97M supervised tokens, fone ok, holdout accepted |
| holdout_fp | pack stamp `10d9c13fd4ffa359` == sha256(data/eval/holdout_hashes.txt)[:16] `10d9c13fd4ffa359`. Note: this is the **file** hash (what `sft_math.py` compares), distinct from the file's inner `# fp: 0dbff3db…` registry header — both correct, different dimensions. The 09-13 40M pack is a different file, not used. |
| scripts/test_sft_pack.py (CPU) | **RC=0** (loss mask: prompt −100 / output supervised; tool spans masked; failing case caught) |
| **External-benchmark decontam (go hard-gate, D5)** | **CLEAN — re-scanned directly, NOT inherited from the 0913 40M pack.** This 65M pack is a different/larger example set (build_stats 263,363 examples / 63.70M sup vs the 0913 pack's 165,622 / 40.0M; sources_fp 7afa47b8 same, but the example multiset differs), so the 0913 release result does not transfer. Independent whitespace-13-gram scan (normaliser 0aefe6a2) of every decoded token of the 23,637 rows vs HE-164 (prompt+solution+test) and MBPP-427 (prompt; code+test_list): **HE 0 rows/0 problems; MBPP 21 rows / 3 problems (mbpp427:97,164,296), all in the 89-problem r3 union, 0 new problems** → `runs/phi65m_pack_decontam_gate.json` clean=true. The builder also drops decon/holdout hits per rebuilt function (`datagen/build_phi_codeexercises_pack.py:197-200`, manifest 97,770 dropped). The 21 rows are the same 3 MBPP canonical-solution idioms seen in the smaller pack (14 rows there; the larger set surfaces more rows of the same 3 tasks), not new leakage. `--check_pack` alone does NOT prove this (it checks vocab/holdout/fone, not external benchmarks); the direct scan does. |
| bash syntax / flags | `bash -n` OK; sft_math accepts --epochs/--batch/--lr_scale/--save_every/--out/--resume |
| optimizer | FRESH — `sft_math.py:324` loads only `ck["model"]` weights; `:338 build_optimizers` rebuilds (the :563/:566 load is the in-step fp8-NaN rollback, not cross-run resume) |
| cards | caller-supplied `CUDA_VISIBLE_DEVICES`; `eval/_devs.sh 8` resolves 8 devices, no hard-coded indices |

## Schedule arithmetic (read from the built 23,637-row pack)

- Striping: 23,637 rows → 2,954/rank (floor) → 2,954 // batch 4 = **738 steps/epoch**.
- Default **N=6 → 4,428 steps**; cosine warmdown start = total − 0.65·total = **1,550**;
  inherited `Cfg.warmup=500` (sft_math has no --warmup) = **11.3%** warmup. Clean ramp→plateau→cosine.
- LR `--lr_scale 0.1` = 0.1× the r3 initial LR; fp8 ON, grad_ckpt ON (required for fp8 backward).
- Output is timestamped `ckpt_v42_phisft_n6_<UTC>.pt`, so an interrupted/rerun never silently overwrites a completed run.

## Final launch command (user go only)

```bash
# On the pod, from /work/aupai, AFTER the controller grants the 8-card block and user says go:
HYPOTHESIS='phi signature-continuation SFT lifts HumanEval rstrip pass@1 over the r3 (ET) base' \
  setsid nohup bash runs/v42_phi_sft.sh > runs/v42_phi_sft.<UTC>.log 2>&1 </dev/null &
# RESUME defaults to ckpt_v41_r3_0914.pt (the r3 final); override with RESUME= to resume a different base.
# The script itself card-claims, runs exp.py start/done, and on success scores HE --rstrip_nl.
```

- Duration: **~2.4–3.0 h** of one continuous 8-card hold (4,428 steps).
- GPU: full 8-card block, world 8 × batch 4 = 32 rows/step.

## Risk: NOT restartable mid-run

`--resume` loads pretrained weights only and starts a FRESH optimizer at step 0 with a fresh
per-epoch randperm; `--save_every 200` mid-run checkpoints cannot be continued by this path.
Any interrupt restarts the ENTIRE 4,428-step run from the r3 base at epoch 0. Confirm an
unbroken ~3 h 8-card window before launch; detach with setsid so a tn tunnel drop does not kill it.

## Failure rollback

- Non-zero train exit: script writes `exp.py done --status fail "sft exited <rc>"`, releases the
  card claim (trap EXIT), and returns the code. No base checkpoint is modified (`--resume` reads
  r3, output goes to a new timestamped file) — relaunch is a clean re-run, no rollback of r3 needed.
- Card-claim refusal or pack-gate failure: script exits 2 before any torchrun; nothing to undo.
- Eval failure post-train: the SFT checkpoint still exists at `$OUT`; it is recorded fail and
  simply is not promoted — the r3 base and all prior checkpoints are untouched.
