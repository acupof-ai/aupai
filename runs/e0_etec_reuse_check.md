# e0_n10.sh reuse check: e0 / et / ec collision and glob correctness

Read-only audit, 2026-09-15 (66, fb order). E0 was mid-run; no GPU used, no file
under active generation touched. Subject: `runs/e0_n10.sh` on main (post #359,
pod stamp c67b7b84).

## The three commands

Prereg `textbook_continuation_ab_0914`: E0 = r3 FINAL; ET = T treatment
(`--name v42_textbook_t`); EC = C control (`--name v42_textbook_c`). ckpt file
bases (the script passes the path INCLUDING `.pt`, and basename keeps it):

| tag | checkpoint (arg) | CK basename used in files |
|---|---|---|
| e0 | `ckpt_v41_r3_0914.pt` | `ckpt_v41_r3_0914.pt` |
| et | `ckpt_v42_textbook_t.pt` | `ckpt_v42_textbook_t.pt` |
| ec | `ckpt_v42_textbook_c.pt` | `ckpt_v42_textbook_c.pt` |

```
CUDA_VISIBLE_DEVICES=<8-card block> bash runs/e0_n10.sh ckpt_v41_r3_0914.pt     e0
CUDA_VISIBLE_DEVICES=<8-card block> bash runs/e0_n10.sh ckpt_v42_textbook_t.pt  et
CUDA_VISIBLE_DEVICES=<8-card block> bash runs/e0_n10.sh ckpt_v42_textbook_c.pt  ec
```

## 1. Tag-to-tag collision: NONE (the artifact name spaces are disjoint)

Per tag/shard, reconstructed from the generators' actual path rules:

- HE shard (humaneval_gen, open_artifact `run=` versioning appends `.{run}`
  before the extension):
  `preds_humaneval_<CK>.rstripnl.n10temp0.2.shard{i}of8.{TAG}_he_n10_s{i}.jsonl`
- MBPP shard (mbpp_gen bakes `{run}` into the f-string AND open_artifact appends
  it again, so the run tag currently appears twice — see finding 2):
  `preds_mbpp_<CK>.{TAG}_mbpp_n10_s{i}.n10temp0.2.shard{i}of8.{TAG}_mbpp_n10_s{i}.jsonl`
- merged: `data/eval/{TAG}_{he,mbpp}_merged.n10temp0.2.jsonl`
- result: `runs/e0_{TAG}_result.json`
- logs: `runs/{TAG}_shard{i}.log`, `runs/e0_{TAG}.log`

Exhaustive set check over the three tags: e0∩et = e0∩ec = et∩ec = 0, 28
artifacts each. Isolation keys: (a) CK basename differs per arm
(r3_0914.pt vs v42_textbook_t.pt vs v42_textbook_c.pt), (b) every name carries
TAG. `--force` therefore cannot delete another tag's shard: the force target is
the exact per-(ckpt,tag,shard) path, which no other tag produces.

## 2. BLOCKING — BOTH merge globs match zero files, so the E0 merge step fails

This is not a cross-tag collision; it is an intra-tag naming/glob mismatch that
breaks the merge for EVERY tag. Ground truth from the live E0 run on the pod:

```
preds_humaneval_...r3_0914.pt.rstripnl.n10temp0.2.shard0of8.e0_he_n10_s0.jsonl
preds_mbpp_...r3_0914.pt.e0_mbpp_n10_s0.n10temp0.2.shard0of8.e0_mbpp_n10_s0.jsonl
```

Running the script's exact globs on the pod returns 0 for both
(`ls ... | wc -l` -> 0, 0). Three compounding causes:

1. **`.pt` stripped in the glob but present in the file.** `CK_BASE=$(basename
   "$CKPT" .pt)` yields `ckpt_v41_r3_0914`, while humaneval_gen/mbpp_gen take
   `os.path.basename(ckpt)` WITHOUT stripping, so files contain
   `ckpt_v41_r3_0914.pt.`. Glob: `...r3_0914.` vs file `...r3_0914.pt.`.
2. **HE: run segment is between shard label and extension.**
   open_artifact(run=...) inserts `.{run}` before `.jsonl`, so the file is
   `...shard{i}of8.{TAG}_he_n10_s{i}.jsonl`; HE_GLOB ends
   `...shard*of8.jsonl` and cannot match.
3. **MBPP: run tag doubled + same trailing-segment problem.** mbpp_gen already
   puts `{run}` in the f-string and open_artifact appends it a second time, so
   the file is `...shard{i}of8.{TAG}_mbpp_n10_s{i}.jsonl`; MB_GLOB ends
   `...shard*of8.jsonl` and also matches 0.

Consequence: after all 16 generation processes pass, e0_n10.sh calls
e0_merge_score.py with a glob matching nothing -> "no shard files match"
SystemExit -> no FULL/CLEAN matrix. The generation work is correct; only the
final assembly is mis-wired.

### Fix (globs only; generators and in-flight preds untouched)

Match the real names, keep `.pt`, and pin TAG so the glob stays arm-isolated:

```
CK_BASE=$(basename "$CKPT")                 # keep .pt, matches os.path.basename
HE_GLOB="data/eval/preds_humaneval_${CK_BASE}.rstripnl.n${N}temp${TEMP}.shard*of${SHARDS}.${TAG}_he_n10_s*.jsonl"
MB_GLOB="data/eval/preds_mbpp_${CK_BASE}.${TAG}_mbpp_n10_s*.n${N}temp${TEMP}.shard*of${SHARDS}.${TAG}_mbpp_n10_s*.jsonl"
```

Verified on the live pod against the real E0 shard files: each glob matches
exactly 8. A follow-up should remove MBPP's doubled run tag at the source (stop
passing `run=` in mbpp_gen when the f-string already names the run, or drop the
f-string segment), but that changes a generator and is deferred so it cannot
touch the in-flight E0 outputs; the glob fix is sufficient and arm-safe.

## 3. et/ec future globs

With the corrected globs, ET/EC match `ckpt_v42_textbook_t.pt` /
`..._c.pt` plus their own TAG segment, so once those shards exist each glob
selects exactly its own 8 per benchmark (same fake-tree set test: MB pattern
exact 8/8; HE exact once the `.{TAG}_he...` segment is added). No cross-arm
pickup.
