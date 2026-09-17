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
| A | clean worktrees in `/Users/bytedance/code/...` on branches already merged to origin/main, no uncommitted files | 58 | **~5.3 GiB** | delete after user yes (worktree + merged branch) |
| B | worktrees on MERGED branches but with uncommitted/dirty files | 9 | ~1.2 GiB (incl. one 1.1G) | **keep until each owner confirms the dirty files are junk** — do not auto-delete |
| C | worktrees on UNMERGED (live) branches | 22 | ~11.7 GiB (two big: enc-probe 9.2G, 98 1.1G) | **keep** — active/WIP; owner sign-off each |
| D | local branches already merged to main with NO worktree | 51 | ~0 disk (refs only) | `git branch -d` safe; delete after user yes |
| E | detached worktrees in `/Users/...` (scratch reviews left behind) | 8 | ~0.6 GiB | delete after user yes (named by commit) |
| — | detached scratch worktrees under `/tmp` and the system temp | 49 | ephemeral | auto-reaped on reboot; nothing to do |
| data | uncommitted/ignored local `data/` files | — | 722M dir, **only ~8 MiB git-ignored + ~0.1 MiB tracked manifests** is plainly expendable; **the big dirs are conservative-KEEP after the pod loss** | see §4; do not delete the 134M/190M/60M corpus copies without a provenance decision |

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
commit; all those commits are on main or on merged PRs. Recommend delete after a per-row
glance at the commit (`E_detached_persistent.txt` lists the sha). No branch is lost because
there was no branch.

The 49 detached worktrees under `/private/tmp` and the system temp dir are scratch and are
reclaimed automatically on reboot; no action.

## 4. Local `data/` — mostly conservative-KEEP after the 2026-09-16 pod loss
`data/` is 722 MiB locally, but most of it is **not** plainly disposable. The repo's own rule
(`docs/standards/data_pipeline_rebuild_0916.md`, `data/PROVENANCE.md`) is "frozen sources
have NO reproduction script; verify, do not delete — re-fetching introduces contamination
drift." After the pod was wiped, a local copy may be the only surviving copy, so the safe
default is to keep until the corpus machine is rebuilt and sha-verified.

| path | size | tracked? | recommendation | basis |
|---|---|---|---|---|
| `data/sft/*.gate_exclude_manifest.jsonl` (8 files) | 0.1 MiB | **tracked** | KEEP | committed; gate manifests |
| `data/synthetic/math_hard_eval_1k.jsonl` | 0.36 MiB | **tracked** | KEEP | the one tracked synthetic file |
| other git-ignored files under `data/` (incl. ignored `data/sft/*`, sample shards) | **~8 MiB** | ignored | may delete | regenerable/ignored; user yes |
| `data/tokenizer*.json` | ~7.6 MiB | local | **KEEP** | the gate tokenizer (2.5M) + variants; called out in the rebuild runbook |
| `data/math/` | 60 MiB | ignored, 0 tracked | KEEP for now | regenerable via `datagen/fetch_math_data.py`/`build_math_expand.sh`, but refetch after pod loss carries contamination drift; delete only once corpus machine re-fetches & sha-checks |
| `data/synthetic/` (rest) | 190 MiB | ignored except the eval file | KEEP for now | generated SFT/eval material; same rebuild-then-delete ordering |
| `data/_corpus_unsanitized/math_530k_20260830` | 134 MiB | ignored | KEEP pending ruling | a deliberately retained UNSANITIZED pre-image; confirm it is still referenced by an audit before deleting |
| `data/_quarantine/rlvr_math_UNFILTERED.jsonl` | 38 MiB | ignored | KEEP pending ruling | quarantine is a hold, not trash; owner confirms release/delete |
| `data/alpaca_gpt4_zh.jsonl` | 33 MiB | ignored | KEEP | frozen SFT source class; re-fetch drift rule |
| `data/s1k.jsonl` | 12 MiB | local | verify then decide | confirm tracked/regenerable before any delete |

So the "~722M reclaimable" figure from the earlier pass overstates what is safe TODAY: only
**~8 MiB of git-ignored regenerables** is unambiguous. The remaining corpus/quarantine bytes
should be deleted (if at all) as part of the post-provisioning data rebuild in
`data_pipeline_rebuild_0916.md` §8/§0, after fresh copies exist and match the recorded
facts — not by a local tidy-up that could erase a sole surviving frozen file.

## What I need from the user (yes/no per bucket)
1. **A** — remove the 58 clean merged worktrees + their merged branches (~5.3 GiB)? (list A)
2. **D** — `git branch -d` the 51 merged ref-only branches? (list D)
3. **E** — remove the 8 detached persistent scratch worktrees after a sha glance? (list E)
4. **B** — for each of the 9 dirty-merged worktrees, route to its owner for a keep/delete call.
5. **C** — leave all 22 live worktrees (recommended); separately decide enc-probe (9.2G) and
   aupai-98 (1.1G) with their owners.
6. **data** — approve deleting only the ~8 MiB ignored regenerables now; defer the
   134M/190M/60M/38M/33M corpus + quarantine bytes to the post-provisioning rebuild.

No command in any bucket runs without the corresponding yes.
