# Disk inventory — `runs/` and non-data caches (de's section)

Read-only. Measured on the pod in the container (`~/bin/pod`, `/work/aupai`) and on the laptop
worktree, 2026-09-04 ~07:0xZ. `bytes` is `du -sb`; `newest mtime` is `stat -c %y`, UTC.

**Totals.** `/work/aupai/runs` is **24.47 GB in 1,053 files**. Four entries are 23.66 GB of it
(96.7%); everything else in `runs/` sums to 0.81 GB. `bench_eff` (outside `runs/`, also mine) is
**3.04 GB in 23 files**, 2.91 GB of which is seven per-rank DDP traces.

## `runs/` — the four that are the whole number

| path | bytes | files | newest mtime UTC | purpose | referenced by | disposition |
|---|---|---|---|---|---|---|
| `runs/owm_dedup_ck` | 8,767,305,884 | 6 | 2026-09-03 15:58 | MinHash signature + location caches for cross-domain near-dedup: `code_py_starcoder.sig.npy` 4.75 GB, `math_owm_stage2.sig.npy` 3.18 GB, `code_py_rp1t.sig.npy` 161 MB, plus three `.loc.json` | **nothing.** Zero repo references to the directory name. Writers are `datagen/near_dedup_scale.py`, `datagen/code_dedup_build.py`, `datagen/code_dedup_handread.py` (by `.sig.npy` suffix, not by this path) | **CANDIDATE, largest single item on the box.** Derived and resumable: it is a checkpoint of a dedup pass, not a result. Deleting costs a re-run of the signature pass over `code_py_starcoder` + `math_owm_stage2`, never data. NOT on fb's 09-03 candidate list (postdates it). Needs the 24h KEEP-claim broadcast — whoever is mid-dedup on starcoder loses their resume point |
| `runs/scan_math_ws.json` | 6,624,717,572 | 1 | 2026-09-03 23:57 | math-corpus containment scan output | 3 refs, incl. `facts/contamination.json`; **no script reads it** | **CANDIDATE** (6e's ruling 2026-09-04, promoted from "candidate after the fact is confirmed"): "no reader" is the same evidence used for `owm_dedup_ck` and `ddp_trace_rank1..6`, so it belongs in the same column. Same byte size as the next row **by coincidence** — md5 differs (`9912552c` vs `273282ad`), they are not duplicates, and that line stays here so nobody deletes one as a dup of the other |
| `runs/scan_eval_golds.json` | 6,624,717,572 | 1 | 2026-09-03 20:05 | eval-gold containment scan output | 7 refs, incl. `datagen/holdout.py`, `datagen/scan_eval_golds.py`, `facts/contamination.json` | **KEEP while `holdout.py` reads it.** Code reference, not just prose |
| `runs/control_lr_scan_v2` | 1,641,216,027 | 37 | 2026-09-02 21:33 | LR-scan arm outputs (v2) | 5 refs, docs only | CANDIDATE if the scan's conclusion is in `facts/`; no code reads it |
| `runs/control_lr_scan` | 656,499,403 | 18 | 2026-09-02 18:57 | LR-scan arm outputs (v1) | 11 refs incl. `scripts/eval_heldout.py`, `scripts/heldout_crossarm.sh` | KEEP — two scripts name it |

## `runs/` — the rest, in one block

`trace_p200m_3step.json` 59.4 MB (cited by `algorithms/attnres_fused.py`, `scripts/trace_classes.py`,
`facts/efficiency.json` — KEEP). Three build logs at ~18 MB each: `build_starcoder.log`,
`code_tests_v1_phase_a.log`, `code_tests_trial.log` (the middle one has **zero** references).
`audit_fineweb_edu_v2_scores.jsonl` 4.9 MB, 1 ref. Then 13 log/json files between 0.4 and 4 MB,
and every `runs/*/` subdirectory other than the two LR scans and `owm_dedup_ck` is **under
210 KB** — `e1_27_step0` 2.1 KB, `e1_27_sweep` 17.7 KB, `e1_28` 21.6 KB, `heldout_ctrl` 2.9 KB,
`heldout_v2` 206.5 KB, `inductor_src` 139.1 KB, `n7c_2x2` 74.2 KB, `n7c_prefix` 131.2 KB, `n8`
13.2 KB. None of these is worth a disposition decision.

**`runs/claims` is EMPTY on the pod** — 0 bytes, 0 entries. Not a disk finding; recorded because
the standing rule is that a claim lives in the tree the job runs from, so an empty claims dir on
the box where jobs run means either nothing holds a card through `card_claim` right now, or
claims are being written somewhere else.

ADJUDICATED (6e, 2026-09-04): expected right now — nothing aupai holds a card (arm 1 done, arm 2
killed, b0's card-5 rescore not yet launched). The open half is that "no claim" and "claims land
somewhere else" produce the same empty directory, so b0 was asked to confirm a claim row appears
**on the pod** when the rescore starts. If it does not, the second reading is the true one.

**No `.REFUSED` sidecars exist under `/work/aupai/runs`** (`find -name '*.REFUSED*'` → 0). The
inventory request listed them; they are not there.

## `bench_eff` — 3.04 GB, and the split is by rank

| path | bytes | files | newest mtime UTC | purpose | referenced by | disposition |
|---|---|---|---|---|---|---|
| `bench_eff/ddp_trace_rank0.json` | 415,798,937 | 1 | 2026-09-02 05:47 | torch profiler trace, rank 0 | **`facts/efficiency.json`, 7 occurrences**, incl. `eff.dynamo_recompile_not_a_lever` and `eff.fusion_and_elementwise_are_disjoint_but_the_trace_is_off_config`, which cite it by absolute pod path | **KEEP.** A fact's source must name a file that exists |
| `bench_eff/ddp_trace_rank{1..6}.json` | 2,493,851,568 total (6 files, 414–417 MB each) | 6 | 2026-09-02 05:47 | same trace, other ranks | **none.** `git grep ddp_trace_rank[1-6]` over `facts/` and `docs/` returns nothing; `parse_ddp.py:6` defaults to rank0 | **CANDIDATE, 2.49 GB, the cleanest large win in my section.** The per-rank copies were kept to compare ranks; no fact or script cites one. If rank comparison is still wanted, that is the KEEP-claim to make |
| `bench_eff/trace.json` | 135,382,179 | 1 | 2026-09-02 05:47 | single-card trace | `bench_eff/parse_trace.py` | KEEP |
| `bench_eff/__pycache__` | 125,110 | — | — | bytecode | — | delete, trivial |

Correction to the request as stated: it names "the 415 MB ddp_trace" as one file. There are
**seven**, one per rank, 2.91 GB together.

## Outside my section, named because it is large

`retired_pre0830v2` is **6.19 GB** at `/work/aupai`. Not `runs/`, not a cache, and not mine to
rule on — flagging it so it is not missed.

## Laptop-only

105 files under `runs/` exist on the laptop and not the pod, **4.0 MB total**. Largest are
`runs/l1_2x2/preds_l1_d3_*.jsonl` (978 KB, 931 KB), `runs/e1_28/e1_28_prov_chatml.json` (242 KB),
`runs/review.jsonl` (224 KB), then the `runs/audit_0904/*` reports. All tracked or eval
predictions; no disk concern, and the direction of the gap is the C4 transport item, not disk.

## What I did not measure

- `du -sb --exclude=data .` on `/work/aupai` reads **269.8 GB**, which includes checkpoints and
  is tilerl's column, not mine.
- No `*.unscanned` or `tmp`/`cache` files were found outside `data/` and the checkpoints. Stated
  as "none found by name", not as "none exist" — I searched `runs/` and the repo root, not every
  subdirectory of the container.
- Whether the two 6.6 GB scan outputs can be deleted depends on whether
  `facts/contamination.json` already carries every number derived from them. I read that the
  facts cite them; I did not verify that no future question needs the raw scan.
