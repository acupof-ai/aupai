# tilerl-22: N6 runs/ prune — the listing, and why nothing is deleted

Owner tilerl, pair b0, 2026-09-04. **Verdict: KEEP ALL 63. Nothing deleted from
any of the three categories.**

## The fact that decides it: /work is an emptyDir

`scripts/pod_push.sh:71` discovers the pod's `/work` as
`/var/lib/kubelet/pods/*/volumes/kubernetes.io~empty-dir/work` — a kubernetes
**emptyDir**, which is destroyed when the pod restarts. So for every committed
log, the git copy is the durable one and the pod copy is the volatile one.

That inverts the premise. "Byte-identical to the pod" is **a fact about today,
not a reason to delete**: the pod copy can vanish on any restart, and then the
deletion has destroyed the only remaining copy of all 43. I had proposed
deleting them and raised the objection against my own list; 6e ruled on the
objection with the emptyDir fact, which I confirmed at that line.

**What was done instead: the 7 stale snapshots were refreshed from the pod.**

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

## 3. Committed log snapshots — **63 of 63 kept, 7 refreshed**

63 `.log` files are committed under `runs/`. Compared each against the pod copy
**by md5, not by existence**:

| | count | disposition |
|---|---|---|
| byte-identical to the pod copy | 43 | **keep** — see the emptyDir fact above |
| same path, different content | 7 | **refreshed from the pod**, listed below |
| not on the pod at all | 13 | keep: git is the only copy, and always was |

**Existence was not a sufficient test and nearly cost two files.** A first pass
proposed `runs/ab_base_a_first.log` because the path exists on the pod — but the
pod's copy is **3.8 MB against a committed 8.4 KB**, starts with torchrun
warnings where the committed file starts with the mix line, and the committed
content is **not even a prefix** of it. `runs/b0_17_readout.log` was 21 KB on the
pod against 5.6 KB committed. Same path, different file, both times. Switching
the test from existence to md5 is what caught it.

### The 7 stale snapshots, refreshed 2026-09-04

Each was checked for growth first — two size reads 30 s apart, **all seven
unchanged**, so every one is a finished run and no refresh captured a
half-written log. After refreshing, all 7 md5-match the pod copy.

| file | was | now |
|---|---|---|
| runs/ab_base_a_first.log | 8,359 | 3,819,506 |
| runs/t57_recompile.log | 6,705 | 3,301,291 |
| runs/data_leg_206m_8b.log | 1,268 | 310,366 |
| runs/pretrain_15b_s1.log | 2,807 | 241,940 |
| runs/b0_17_readout.log | 5,573 | 20,992 |
| runs/t57_steady.log | 4,523 | 10,911 |
| runs/t56_profile.log | 2,693 | 7,584 |

The committed copies were truncated excerpts — `ab_base_a_first.log` held 0.2% of
the run. **A stale snapshot is worse than no snapshot: it looks like the record
and is not.**

### None growing, so none deferred

No committed log is attached to a live run right now; the params leg writes
`runs/params_leg_438m_3p76b.log`, which is not among the 63.

## What the listing changed about the premise

All three populations came back empty of anything deletable. Two were empty on
the data; the third was emptied by a fact about the storage — **`/work` is an
emptyDir, so "the pod has it too" is not durability**. A prune that trusted "the
pod holds the live logs" would have deleted 43 files whose remaining copy dies at
the next pod restart, plus, on the first pass, 2 whose pod file was a different
file with the same name.

What the task actually surfaced was the opposite of a prune: **7 committed logs
had silently decayed into truncated excerpts of the runs they claim to record**,
and refreshing them was the useful work.

Same shape as §156 one directory up: two things share a name, and only comparing
each one against the other store separates them. Existence is not identity;
**the test has to be the hash.**
