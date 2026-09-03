# tilerl-22: N6 runs/ prune — the listing

**This is the list, not the deletion.** Broadcast now, held 24 h per the deletion
rule; unclaimed entries are deleted 2026-09-05. A file whose reason cannot be
written is not on the list.

Owner tilerl, pair b0, 2026-09-04.

## Result: 43 files, 1021 KB. The other two categories are empty.

The task named three populations. Measured, two contain nothing safely
deletable, and reporting that is the deliverable — a prune list padded with rows
that carry unique information is worse than a short one.

---

## 1. score_matrix duplicate `(ckpt, profile)` keys — **none exist**

```
58 rows, 58 distinct (ckpt, profile) keys, 0 duplicates.
```

Five checkpoints do appear twice, which is what the premise saw, but under
**different profiles** — `ckpt_sft_p324_v3/v4/v5.pt`, `ckpt_p324.pt`,
`ckpt_pretrain_15b_s1.pt`, each once as `milestone` and once as `full`.
`PROFILES` (eval/score_matrix.py:52-71) gives those two different metric sets,
so the pair holds different measurements of one checkpoint. **Deleting either
loses metrics the other never had.**

## 2. Killed-duplicate exp rows — **12 candidates, all rejected**

A run is identified by `(name, started, ended)`; 278 rows collapse to 266 keys,
so 12 share a key. All 9 groups differ in content, and `experiments.jsonl` has
**0 byte-identical duplicate lines** — nothing here is a mechanical double-write.

Three reasons they stay, each measured:

- **The later row is not a superset.** `ab_untie_head` looks like a clean growth
  chain — decision 0 → 1042 → 1496 → 1851 chars, finding 0 → 1677 → 2460.
  Substring-testing each earlier row against the last: **L236 and L245 carry text
  the final row dropped.** Only L246's `decision` is contained. A chain that
  grows is not a chain that accumulates.
- **The second row is the correction.** `eval_p500m_step1500_{base,l1,ppl}`
  L251-253 replace a commit sha with `unknown` plus "names no object in this
  repository — a pod-side sha from a tree main never held". First row holds the
  wrong sha, second holds the retraction; **both are the record.**
- **The second row is a dated amendment.** `params_leg_438m_3p76b` L274 is e1's
  2026-09-04 note that steps 4400-4550 shared card 0. Deleting L269 orphans the
  amendment; deleting L274 deletes the disclosure.

## 3. Committed log snapshots — **43 of 63, by hash**

63 `.log` files are committed under `runs/`. Compared each against the pod copy
**by md5, not by existence**:

| | count | disposition |
|---|---|---|
| byte-identical to the pod copy | **43** | **deletable — the list below** |
| same path, different content | 7 | keep: the committed bytes exist nowhere else |
| not on the pod at all | 13 | keep: git is the only copy |

**Existence was not a sufficient test and nearly cost two files.** My first pass
proposed `runs/ab_base_a_first.log` because the path exists on the pod — but the
pod's copy is **3.8 MB against the committed 8.4 KB**, starts with torchrun
warnings where the committed file starts with the mix line, and the committed
content is not even a prefix. `runs/b0_17_readout.log` is 21 KB on the pod
against 5.6 KB committed. Same path, different file, both times.

The 7 that differ, all kept:

```
runs/ab_base_a_first.log   runs/b0_17_readout.log   runs/data_leg_206m_8b.log
runs/pretrain_15b_s1.log   runs/t56_profile.log
runs/t57_recompile.log     runs/t57_steady.log
```

The 13 that are git-only, all kept — **deleting these destroys the sole copy**:

```
runs/0830v1_0.2b.log          runs/l1_2x2_ctrl_en.log   runs/l1_2x2_ours_en.log
runs/ab_shipment.log          runs/l1_2x2_ctrl_zh.log   runs/l1_2x2_ours_zh.log
runs/e1_29_floor_by_class.log runs/n7c_2x2/n7c_2x2.log  runs/t57_pad_ab.log
runs/eval_v2_fmt.log          runs/t57_seam.log         runs/t57_twin.log
runs/heldout_v2/clean_ctrl_lr1e-3.log
```

### The 43 proposed for deletion

Reason, identical for every one: **byte-identical to `/work/aupai/<same path>` on
the pod, which is the live copy; the committed file is a snapshot that has
diverged from nothing and adds no information.**

```
runs/ab_chunk.log
runs/ab_vocab.log
runs/ab_vocab_32784.log
runs/ab_vocab_32784_lr003.log
runs/ab_vocab_32784_zeroinit.log
runs/b0_16_rescale.log
runs/b0_16_table.log
runs/b0_17_arm3_dl.log
runs/bf16_update_loss.log
runs/bf16_update_loss2.log
runs/code_base_0shot.log
runs/code_zero_sft_v1.log
runs/codezh_sft_v2.log
runs/e1_27_step0/repro_lr0.1.log
runs/e1_27_step0/repro_lr1.0.log
runs/e1_28_clean_run.log
runs/heldout_v2/clean_floor_control.log
runs/heldout_v2/clean_floor_ours.log
runs/heldout_v2/clean_ours_sft.log
runs/heldout_v2/ctrl_lr1e-3.log
runs/heldout_v2/ctrl_lr1e-4.log
runs/heldout_v2/ctrl_lr3e-3.log
runs/heldout_v2/ctrl_lr3e-4.log
runs/heldout_v2/ctrl_lr3e-5.log
runs/heldout_v2/floor_control.log
runs/heldout_v2/floor_ours.log
runs/heldout_v2/ours_lr0.01.log
runs/heldout_v2/ours_lr0.03.log
runs/heldout_v2/ours_lr0.3.log
runs/heldout_v2/ours_lr1.0.log
runs/heldout_v2/ours_sft_reguarded.log
runs/l1_p324.log
runs/p02_sc_base.log
runs/p02_sc_patched.log
runs/pretrain_30b_s2.log
runs/scan_code_holdout.log
runs/scan_code_holdout_sft.log
runs/t38_ref.log
runs/t38_resume.log
runs/t52_p1_b32.log
runs/t52_p1_base.log
runs/t52_p2_maxauto.log
runs/w7_b16a2.log
```

**Objection worth raising against my own list:** deleting these makes the repo
depend on the pod for 43 files, and the pod is not backed up. The counter-case is
that a snapshot nobody refreshes is a stale copy waiting to disagree with the
live one — which is exactly the 7 above. If b0 prefers keeping them, the reason
to state is "the pod is not a durable store", and that is a fair objection to the
whole category rather than to any file on the list.

## What the listing changed about the premise

Two of three populations were empty, and the third was mostly traps: **20 of 63
committed logs must not be deleted** — 13 because git holds the only copy, 7
because the pod's file at the same path is a different file. A prune that trusted
"the pod holds the live logs" would have destroyed content in both groups, with
nothing in the filename to distinguish them.

Same shape as §156 one directory up: two things share a name, and only comparing
each one against the other store separates them. Existence is not identity;
**the test has to be the hash.**
