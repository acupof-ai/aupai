# Whole-machine disk inventory (pod), for tilerl

Measured 2026-09-04 08:4xZ on the pod, `/work` at **96% full: 1.9T used, 87G available** of 2.0T
(`df -h /work`; `/` is the same overlay). Every number below is read, not estimated; the method
is stated with each one because two of them are corrections of my own earlier figures.

## 1. What fills the disk

    /work/aupai              844 G      the aupai tree (this repo's pod copy + checkpoints)
    /work/Qwen3.8-27B-bf16    42 G      tileRL model weights
    /work/Qwen3.8-27B-NVFP4   22 G      tileRL, quantised
    /work/newdata             11 G
    /work/Qwen3.8-27B-DFlash2 3.6 G     tileRL
    /work/tilelang_cache      2.3 G     tileRL
    /work/fwe                 1.3 G
    /work/_env_restore        1.3 G

`du -sh /work/*` sorted. aupai is 844 G of the 1.9 T, so aupai is the largest single consumer but
NOT the majority: ~1.05 T is outside it and outside the eight entries above (nested paths and
other trees not broken out here).

## 2. aupai checkpoints, inode-aware

    inodes matching *.pt*            841
    distinct inodes                  819
    distinct bytes                307.87 GB = 286.75 GiB
    apparent bytes                330.42 GB = 307.75 GiB
    hardlink double-count      22,548,835,495 bytes = 21.00 GiB over 22 inodes
    name groups                      256

**Apparent minus distinct is 21.00 GiB of hardlinks, not 21.00 GiB of recoverable space.**
Deleting one name of a hardlinked pair frees nothing; both names must go. Any figure that sums
`ls -l` sizes over these files overstates the recoverable total by that amount.

## 3. The `.stepN` rollback buffers

    .stepN files                     158        132.42 GiB
      cited somewhere in-tree         14         15.72 GiB
      cited nowhere                  144        116.70 GiB

"Cited" = the exact filename appears in `facts/*.json`, `runs/*.jsonl`, or `docs/standards/*.md`.
The 144 uncited files are the prune candidate: **116.70 GiB, against 87 G free.** Deleting them
roughly doubles the headroom on this filesystem.

`.stepN` is a rollback buffer refreshed every `min(200, save_every)` steps, so a `.stepN` does NOT
hold the weights its number names — see the recorded finding that `.step2000`'s metadata matched
`.ep1` field-for-field while 174 of 176 tensors differed. That is why these are prunable at all:
they are not the checkpoints anyone cites, and their name does not describe their contents.

## 4. Two corrections to my own earlier numbers

I reported **346 inodes / 271.60 GiB distinct / 89 groups referenced by nothing / 159.29 GiB** and
separately **153 `.stepN` inodes, 13 pinned, 140 unpinned = 109.91 GiB**. Both were computed with a
filter of `f.endswith('.pt')`. A rollback buffer is named `ckpt_x.pt.step2000` — it does not END in
`.pt`, so that filter silently skipped every one of them. Re-run with `'.pt' in f`:

    346 inodes  ->  841          (and 819 distinct)
    271.60 GiB  ->  286.75 GiB
    153 .stepN  ->  158
    109.91 GiB  ->  116.70 GiB unpinned

The direction is consistent — the old filter could only undercount — but the shape of the error is
worth more than the size of it: `groups == inodes` in the first run (146 == 146), which is
impossible for a tree containing rollback buffers, because every buffer shares a base name with
its parent checkpoint. The invariant was visible in the output and I read past it. A count whose
group count equals its file count has found no families, and this tree is families.

## 5. What is NOT in this inventory

- **`/work/tl-rl`** — tileRL's own tree, not walked. It is not aupai's to inventory or prune.
- **`/data00`** — token caches, on a different filesystem; not part of the 87 G pressure.
- **Non-`.pt` files under `/work/aupai`** — the 844 G total includes them, but the 286.75 GiB
  breakdown is checkpoints only. The ~557 G difference is data, corpora and logs, not accounted
  for here.
- **Per-owner attribution of the 144 unpinned files.** Uncited is not unowned: a file can be live
  work whose owner has not written a fact yet. The prune broadcasts the list and deletes after 24 h
  unclaimed, which is the rule that covers this gap; this inventory does not shortcut it.
