# psft n=10 eval + paired bootstrap vs E0 — pre-staged, zero-GPU

Prepared by 66, 2026-09-15. **Nothing here is launched and no GPU is touched until fb
orders it after phi SFT completes.** SFT was at step 2280/4428 (~19 s/10 step) when this
was written; ckpt lands in roughly an hour.

Checkpoint (date-suffixed basename, keep the `.pt`):

```
ckpt_v42_phisft_n6_20260915T063647Z.pt
```

## Static verification (done before the ckpt exists)

The #363-fixed globs in `runs/e0_n10.sh` (main/pod sha d56fa12e) were simulated against
the date-suffixed basename:

- HE glob matches the 8 expected shard files, exact, 8/8.
- MBPP glob matches the 8 expected shard files, exact, 8/8.
- Adding an `e0`-named shard file leaves the psft HE match count at 8 (tag-pinned glob
  stays arm-isolated; psft never merges E0 files).

`e0_n10.sh` itself refuses with exit 2 if the ckpt is not present (`[ -f "$CKPT" ]`), so
launching early cannot score a missing checkpoint. HE runs `--rstrip_nl`
(`e0_n10.sh:46`), correct for a continuation SFT; MBPP runs sig-docstring-rstrip native
with `--no_clean`, and the merger recomputes CLEAN.

## 1. 8-card eval (fb orders this after the SFT ends)

Caller exports the 8-card block; the script maps each fixed-position shard onto it through
`eval/_devs.sh` (never hard-code a physical index). Detach with `setsid`, two-level
redirect (AGENTS.md):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
pod "cd /work/aupai && setsid nohup bash -c 'bash runs/e0_n10.sh ckpt_v42_phisft_n6_20260915T063647Z.pt psft > runs/e0_psft.log 2>&1' </dev/null >/dev/null 2>&1 &"
```

One process per card, HE then MBPP serialised on the card, 8 shards fixed-position
(`idx % 8`). n=10, temp 0.2, max_new 280 (pinned in
`runs/prereg.jsonl#textbook_continuation_ab_0914`). Merger rejects shard gap, duplicate
(task,si), wrong task count, or per-task n != 10.

Artifacts:

| artifact | path |
|---|---|
| HE merged preds (bootstrap input) | `data/eval/psft_he_merged.n10temp0.2.jsonl` |
| MBPP merged preds (bootstrap input) | `data/eval/psft_mbpp_merged.n10temp0.2.jsonl` |
| summary JSON | `runs/e0_psft_result.json` |
| per-shard logs | `runs/psft_shard{0..7}.log`, `runs/e0_psft.log` |

`runs/e0_psft_result.json` has two top-level segments, **`humaneval`** and **`mbpp`**,
each with `full_pass/full_denom/full_rate`, `clean_pass/clean_denom/clean_rate`,
`tasks_total`, `n`, `empty`, `shards`. Denoms: HE FULL 1640 / CLEAN 1560; MBPP FULL 4270
/ CLEAN 3380.

E0 baselines already on the pod: `data/eval/e0_he_merged.n10temp0.2.jsonl`,
`data/eval/e0_mbpp_merged.n10temp0.2.jsonl`.

## 2. Paired bootstrap psft − E0 (CPU, after the eval)

Input is per-sample mean Σc/(tasks·n), NOT pass@10 coverage. Resampling unit is the task;
within-task sample_idx 0..9 are fixed draws, never independent. `--a` is psft so the
reported difference is psft − E0.

Prereq: `eval/paired_bootstrap.py --he_union` is PR #368 — it must be merged and
pod-pushed before the HE command below runs (the eval in step 1 does not need it).

HE CLEAN 156 (exclude the 8-id r3 HE union; MBPP's `--clean` is the opposite keep-list and
is refused if both are passed):

```bash
python3 eval/paired_bootstrap.py \
  --a data/eval/psft_he_merged.n10temp0.2.jsonl \
  --b data/eval/e0_he_merged.n10temp0.2.jsonl \
  --he_union runs/contam_r3_he_union.json \
  --label_a psft --label_b e0
```

MBPP CLEAN 338 (keep r3_mbpp_clean):

```bash
python3 eval/paired_bootstrap.py \
  --a data/eval/psft_mbpp_merged.n10temp0.2.jsonl \
  --b data/eval/e0_mbpp_merged.n10temp0.2.jsonl \
  --clean runs/contam_r3_mbpp_union.json \
  --label_a psft --label_b e0
```

Both commands refuse unless every shared task carries the identical sample_idx set on both
sides (the `(task_id, sample_idx)` pairing contract), so a misaligned run emits no CI.

## 3. Read-out / gate

Report, per benchmark on CLEAN:

- observed per-sample means `rate_a_observed` (psft), `rate_b_observed` (E0);
- observed mean diff `mean_diff_observed`;
- **one-sided 95% lower bound `diff_one_sided_lower_95` of psft − E0**;
- whether that lower bound is ≥ 0.30 (the gate fb named), and separately whether it is > 0
  (psft not worse than E0).

Reference E0 per-sample means (2026-09-15): HE CLEAN 292/1560 = 18.72%; MBPP CLEAN
909/3380 = 26.89%. Pass@10 coverage (HE 30.77%, MBPP 40.24%) is a different, larger number
and is not the gate — do not feed it to the bootstrap.
