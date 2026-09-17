# Local repo cleanup approval list — list only (no deletions performed)

Status: INVENTORY for user line-item approval. genA 2026-09-18, zero deletions run. Nothing
here was removed; every row carries a recommendation and the basis for "safe". Full
machine-readable lists (exact paths, sizes) are attached in the author's scratch at
`~/aupai-textgen/genA/cleanup_lists_2026-09-18/{A..E}_*.txt`; this doc is the decision form.
The exact lists are committed alongside it for verification at
`docs/standards/cleanup_lists_0918/` (`A_clean_merged_home_worktrees.txt`,
`B_dirty_merged_worktrees.txt`, `C_live_worktrees.txt`,
`D_merged_branches_no_worktree.txt`, `E_detached_persistent.txt`).
The user answers yes/no per row before anything is deleted.

Measured on the canonical checkout `/Users/bytedance/code/aupai` (local main kept current with
origin/main; tip at audit `11bcde71`). Sizes are local working-tree `du`; the shared
`.git/` object store is not counted per worktree.

## TL;DR
| bucket | what | count | reclaim | recommendation |
|---|---|---|---|---|
| A | clean merged worktrees in `/Users/bytedance/code/...`, no uncommitted files | 58 at snapshot (re-enumerate; was 60 at re-count) | **~5.3 GiB** | delete after user yes (worktree + merged branch) |
| B | worktrees on MERGED branches but with uncommitted/dirty files | 9 | ~1.2 GiB (incl. one 1.1G) | **keep until each owner confirms the dirty files are junk** — do not auto-delete |
| C | worktrees on UNMERGED (live) branches | 22 | ~11.7 GiB (two big: enc-probe 9.2G, 98 1.1G) | **keep** — active/WIP; owner sign-off each |
| D | local branches already merged to main with NO worktree | 51 at snapshot (re-enumerate; was 54) | ~0 disk (refs only) | `git branch -d` safe; delete after user yes |
| E | detached worktrees in `/Users/...` (scratch reviews left behind) | 8 | ~0.6 GiB | delete after user yes (named by commit) |
| F | `/tmp` worktrees holding UNCOMMITTED work on a main-ancestor HEAD | 2 (98-owned design docs) | ~0.2 MiB text | **KEEP — owner 98 must commit/discard; a reboot DELETES them, it does not reap them** |
| — | other detached scratch worktrees under `/tmp` / system temp | 49 | ephemeral scratch | reboot-reaped; but see F — the dirty ones are NOT safe, re-verify status at action time |
| data | uncommitted/ignored local `data/` files | — | 723 MiB dir = 24.6 MiB tracked-present (223 files) + **697 MiB untracked-ignored (762 files)** + ~8 MiB tracked-but-ignored; **none of the 697 MiB is plainly expendable** | see §4; conservative-KEEP the corpus/quarantine bytes after the pod loss |

## Why merged-and-clean is safe to remove (A/D/E)
- A merged branch's tip is an ancestor of `origin/main` (`git merge-base --is-ancestor <b>
  origin/main` true), so every commit is already on the remote; deleting the local ref and the
  working directory loses no git object.
- "clean" means `git status --porcelain` is empty, so there are no uncommitted edits, untracked
  non-ignored files, or staged changes to lose. (Ignored build/data outputs inside a worktree
  are regenerable and are the same category as §4.)
- Removal is the reversible pair `git worktree remove <path>` then `git branch -d <branch>`
  (`-d` refuses unless merged — a second guard; never `-D`). Any mistake recreates the worktree
  from the remote branch in seconds.

## 1. A — clean merged persistent worktrees (recommend DELETE; 58, ~5.3 GiB)
Largest and fully safe: includes genB's two she called out —
`/Users/bytedance/code/aupai-l0dedup` (branch `genb-l0-dedup-selftest`, MERGED, clean) and
`/Users/bytedance/code/aupai-l0p11` (branch `genb-build-corpus-tokens-none`, MERGED, clean) —
plus the ae/de/3b/66/98/genA/0e/fb review and feature worktrees. One outlier by size:
`/Users/bytedance/code/aupai-66` (branch `66`, merged, clean) is **1.1 GiB** of local ignored
outputs; the other 57 are ~70-80 MiB each. Full list in `A_clean_merged_home_worktrees.txt`.

Representative (all satisfy MERGED + empty status):
`aupai-l0dedup`, `aupai-l0p11`, `aupai-l2chunk`, `aupai-l2test`, `aupai-de-eng493`
(de-engram-nograd, #501 merged), `aupai-de-idxgrid` (#500 merged), `aupai-gena-gpu41f`
(#497), `aupai-gena-ae10`, `aupai-gena-b3` (#473), `aupai-gena-r476`, `aupai-diskcheck`,
`aupai-66`, and the `v41f-*` de/fb feature worktrees that all landed.

Exact command (only after the user approves the row list):
`git worktree remove <path> && git branch -d <branch>` per row; never `--force`.

## 2. B — merged-branch worktrees WITH uncommitted files (recommend KEEP pending owner; 9)
The branch is merged but the working tree is not clean, so deletion could discard work:

| dirty files | branch | path | note |
|---|---|---|---|
| 13 | 0e-data-fp-readiness | `~/code/aupai-0e` | owner 0e to confirm |
| 9 | genb-l3-funnel-selftests | `~/code/aupai/.claude/worktrees/genB` | genB's own harness worktree; genB to confirm |
| 7 | fb-v41f-compressor | `~/code/aupai-v41f` | owner fb |
| 4 | ae-gate-mix | `~/code/aupai-ae` (1.0G) | owner ae |
| 4 | de-stepD-pre-v2 | `~/code/aupai-de-dpre3` | superseded by #507? de to confirm |
| 2 | 66-l3-locked-ledger | `~/code/aupai-66-l3label` | owner 66 |
| 1 | de-117-textbooks / de-v42-gate-followup / 66-review376 | … | owner each |
| 1 | 3b-review-rows-0917 | `/tmp/ledger3b` | review-row scratch; 3b to confirm |

Basis: even one untracked non-ignored file is unrecoverable once the dir is removed. These
need the owning session to say "those dirty files are throwaway"; only then are they as safe
as A.

## 3. C — live (unmerged) worktrees and D — merged ref-only branches
**C (KEEP, 22, ~11.7 GiB):** branches not ancestors of origin/main = open work. Two large:
`aupai-enc-probe` (`enc-probe-wip`, **9.2 GiB**, dirty=6 — probe data, owner must decide) and
`aupai-98` (`98-e0-readout-panel`, **1.1 GiB**, dirty=7). The rest are ~70-95 MiB active
de/3b/66/0e/fb branches (e.g. `de-stepD-pre`, `0e-review-484`, `3b-vet-stats-disktruth`,
`66-moe-grouped-mm`, `fb-v41f-{attn,engram,indexer}`, `pr454`). None should be deleted
without the owner; an unmerged branch may hold the only copy of work.

**D (safe ref deletion; 51):** local branches already on origin/main with no worktree —
`git branch -d` only, zero working-tree risk, negligible disk (the objects are shared in
`.git` and remain reachable from origin). Samples: `3b`, `3b-91`, `66-rev-470`, `98`, `ae`,
`de-98-guard-population`, `genb-fpfilters-pattern-hash`, `pr482`, `pr489`, `pr501`. Full
list in `D_merged_branches_no_worktree.txt`. Recommend delete after yes.

**E (detached persistent scratch; 8, ~0.6 GiB):** detached-HEAD review archives in
`/Users/...` from past second-reads (`aupai-0e-r500/r497/r506`, `aupai-0e-pr441`,
`v41f-model-read-wt`, `main-probe-wt`, `aupai-de-mg3/mg4`). Each is pinned to a specific
commit. These are safe to delete NOT because the commit sha is an ancestor of main in every
case, but because the patch content is already on main: e.g. `aupai-gena-gpu41f`'s detached
tip `97388c3a` is **not** an ancestor of `origin/main` and hangs off no ref (reflog-only), yet
`git patch-id` is `e51d01b8…`, identical to `c34170b3` which IS on main — same change. So the
safety basis is "the work is on main by content (patch-id / merged PR), re-pushable", and each
row should be confirmed by that content check (or the PR it reviewed) before delete. Full
list with shas in `E_detached_persistent.txt`.

## 3b. F — `/tmp` worktrees with UNCOMMITTED design work (KEEP, owner 98; NOT reboot-safe)

Two `/private/tmp/wt-*` worktrees have a clean main-ancestor HEAD but a **dirty tracked design
document** — real edits that exist in NO commit and on no branch. Calling these "ephemeral,
auto-reaped on reboot" is wrong: a reboot DESTROYS the file, it does not recycle tracked work.

| path | uncommitted diff | owner |
|---|---|---|
| `/private/tmp/wt-98-idx-trainability-design` | `docs/standards/v41f_indexer_trainability_design.md` +88/-43 | 98 |
| `/private/tmp/wt-98-v41f-train-ckpt-design` | `docs/standards/v41f_train_checkpoint_design.md` +102/-24 | 98 |

Both HEADs are ancestors of current `origin/main`, so these are edits pressed onto a fresh
base, not stale-base leftovers. Note for the owner: the working copies look like EARLIER
drafts of docs that have since landed in revised form on main — main's step-D design carries
the newer G1–G7 revision (#489) and the indexer design carries the refined sub-ULP STE text,
so the dirty content may already be superseded. That is a content call only 98 can make; the
safe action is for 98 to diff each against current main and either commit the still-wanted
parts or `git restore`/remove. This list must NOT delete them and must not assume the reboot
handles it. The other dirty `/tmp` review trees (`eng493`, `eng493base`, `idx494`) are
harmless: their only diffs are cosmetic re-wraps of files already fixed on main and untracked
test files that already landed — still confirmed with the owner, but no design content.

The 49 detached worktrees under `/private/tmp` and the system temp dir are scratch and are
reclaimed automatically on reboot; no action.

## 4. Local `data/` — 723 MiB, almost all conservative-KEEP after the 2026-09-16 pod loss
The repo's own rule (`docs/standards/data_pipeline_rebuild_0916.md`, `data/PROVENANCE.md`)
is "frozen sources have NO reproduction script; verify, do not delete — re-fetching introduces
contamination drift." After the pod was wiped, a local copy may be the only surviving copy.

Measured accounting (correct git split; the earlier draft's "~8 MiB ignored" used the wrong
flag and is withdrawn):
- **24.6 MiB tracked files present (223 files)** — committed, KEEP.
- **697.3 MiB UNTRACKED-IGNORED files (762 files, `git ls-files --others -i
  --exclude-standard`)** — the real bulk; none of it is plainly expendable today.
- ~8.2 MiB tracked-but-now-ignored (159 files, `-ci`) — already in git history.
- 0 untracked non-ignored (nothing would be lost silently by `git clean` semantics — but do
  NOT run `git clean`, the ignored corpus is exactly what it would erase).

| ignored path | size | untracked-ignored files | recommendation / basis |
|---|---|---|---|
| `data/sft/` (ignored content beside the tracked gate manifests) | 201 MiB | 10 | KEEP — generated gate/decontam SFT material; rebuild-then-delete |
| `data/synthetic/` (besides tracked `math_hard_eval_1k.jsonl`) | 190 MiB | 12 | KEEP — generated SFT/eval; same ordering |
| `data/_corpus_unsanitized/math_530k_20260830` | 134 MiB | 9 | KEEP pending ruling — deliberate UNSANITIZED pre-image; confirm still referenced by an audit |
| `data/math/` | 60 MiB | 3 | KEEP — regenerable via `fetch_math_data.py`/`build_math_expand.sh`, but post-loss refetch carries contamination drift |
| `data/_quarantine/rlvr_math_UNFILTERED.jsonl` | 38 MiB | 1 | KEEP pending owner — quarantine is a hold, not trash |
| `data/alpaca_gpt4_zh.jsonl` + other ignored roots | remainder | — | KEEP — frozen SFT source class, re-fetch drift rule |
| tracked: 8 `data/sft/*.gate_exclude_manifest.jsonl`, `data/synthetic/math_hard_eval_1k.jsonl`, `data/tokenizer*.json` | ~32 MiB | tracked | KEEP — committed manifests + the gate tokenizer (the 2.5M gate file named in the rebuild runbook) |

Conclusion: there is **no sizeable data bucket safe to delete today**. The 697 MiB should be
removed, if at all, as part of the post-provisioning rebuild in
`data_pipeline_rebuild_0916.md` (§0/§8), AFTER fresh copies exist on the new node and match
the recorded sha/facts. A local tidy-up that `git clean -x`s or `rm -rf`s these could erase a
sole surviving frozen file. Even the ~8 MiB tracked-but-ignored files should be left for that
pass rather than hand-deleted.

## Snapshot is moving — re-enumerate at action time, do not execute these lists as-is
The worktree set changed during the audit (155→159 worktrees in minutes): between the first
and second enumeration bucket A moved by a couple of rows, E by a few, and D gained newly
merged branches (e.g. `0e-packer-fp-a1`, `3b-review-rows-0917`, `pr507`). The attached
A–E lists are the evidence snapshot at `11bcde71`; the actual remove/delete step MUST
re-derive each set on the then-current `origin/main` with the same three predicates
(`worktree list --porcelain`, `merge-base --is-ancestor <branch> origin/main`,
`status --porcelain` empty) and re-check every `/tmp` tree for dirty files. Never pipe the
committed text list straight into a removal.

## What I need from the user (yes/no per bucket)
1. **A** — remove the clean merged worktrees + merged branches (re-enumerated; ~5.3 GiB at snapshot)?
2. **D** — `git branch -d` the merged ref-only branches (re-enumerated)?
3. **E** — remove the detached persistent scratch worktrees after a per-row content/PR check?
4. **F** — do NOT touch the two 98 dirty `/tmp` design worktrees; route to 98 to commit/discard.
5. **B/C** — route the 9 dirty-merged and 22 live worktrees to their owners (enc-probe 9.2G, aupai-98 1.1G).
6. **data** — defer all 697 MiB of untracked-ignored corpus/quarantine/SFT bytes to the post-provisioning rebuild; delete nothing locally.

No command in any bucket runs without the corresponding yes and a fresh enumeration.
