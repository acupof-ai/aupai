# de-46 after-listing: the 12:00Z command, rehearsed 2026-09-04 against the live pod

Not a doc. A worksheet, deleted when de-46 closes.

## The command

    python3 scripts/gen_ckpt_listing.py --out runs/pod_ckpt_candidates_2026-09-04.txt

Run it AFTER b0's rm, not before. `--previous` defaults to the newest listing
(`runs/pod_ckpt_candidates_2026-09-03.txt`), which is what carries the KEEP lines and the
original deadlines forward; do not pass it by hand.

## Rehearsed state, 2026-09-04 (before b0's rm)

    on pod            335 files, 297 GB
    listed as rows    302
    held out           33   -- every one of them claimed, 0 unexplained
    deletion plan     192 files, 153.0 GB

The plan was 193 / 154.2 GB when I generated it on 09-03. One file left it because b0 pinned
`ckpt_pretrain_30b_s2.pt.step26500` in between. That is the pin mechanism working, not drift.

## What the reconciliation must still say after the rm

I checked the held-out set against the live claims rather than trusting the count:

    held out AND claimed        33
    UNEXPLAINED                  0   <- this is the one that matters

A nonzero UNEXPLAINED means files are vanishing from the inventory for a reason the script
does not state, which is the failure this listing exists to prevent. Recompute it, do not
assume it.

## The 15 "claimed but absent" entries are NOT 15 problems

13 are doubled-`.pt` artifacts: `_parse_ckpt_listing` deliberately emits both readings of a
shorthand claim, so `.pt.step1500` also yields `X.pt.pt.step1500`, impossible on any disk.
Over-protection is free for a READER and inverts for a GUARD, which is why the deletion plan
only ever names a file the SCAN found.

Two are parse variants whose real file is present, verified by stat on the pod:

    ckpt_p500m_20b_0902.interrupt.step83          -> .pt.interrupt.step83 PRESENT (ino 84186445)
    ckpt_pretrain_15b_s1.pt.milestone_8b_step8500 -> .milestone_8b_step8500.pt PRESENT (ino 84187881)

ONE IS A GENUINE LOSS AND ALREADY RECORDED: `ckpt_p200m_4b_0902.pt.step500` is absent,
which is what the RETIRED line in the 09-03 listing says -- the roller rotated it out while
a KEEP line in a text file could not bind it (§162). The rehearsal reproduces the known
state; it did not find anything new.

## After the run

1. Diff the two listings' plan totals. 192 planned minus whatever b0 actually removed should
   be what remains; a large residue means the rm did not run as planned, not that the
   generator is wrong.
2. `python3 scripts/harness.py check` -- `ckpt_facts_sources_present` reads the NEWEST
   listing, so a fact whose source b0 removed turns from `[deletion-candidate]` to
   `[absent]` and that is the check doing its job.
3. Commit the listing, then push it: it is in the manifest's scope.
