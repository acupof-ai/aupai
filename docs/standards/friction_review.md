# Friction review

Daily summary of `harness friction` (runs/friction.jsonl), ranked by count then minutes lost. Top two causes get a fix commit or a task; one line per cause. Owner: 44. Reviewer: de.

## 2026-09-04 (43 rows, 38 causes; ran 14:08Z)

Top two unfixed causes fixed this commit: (1) 3x near_miss/process_failure rows without minutes_lost → `check_friction_minutes_required` (baseline 3, FAILs on 4th violation); (2) 1x ff merge runs no pre-commit hook → `scripts/hooks/post-merge` (runs harness check after fast-forward, warns on failure).

| cause | n | min lost | resolution |
|---|---|---|---|
| manifest merge conflicts (pod_head_manifest regenerated per commit) | 4 | ~16 | fix 56fa71b5 + 89d86882 |
| near_miss/process_failure minutes not reported | 3 | n/r | **fix this commit** — check_friction_minutes_required |
| flash_attn mask_mod wrong gradients on SM 9.0 (forward correct) | 1 | ~95 | fix carried (1/1 rows) |
| merge_main.sh pathspec+pipe-filter hid commit failure | 1 | ~40 | fix by habit, controller side (6e) |
| lane card serialization (every non-training job routed to one lane) | 1 | ~40 | task de-44 |
| ghost running: pod experiments.jsonl lacked closes written on main | 1 | ~40 | fix carried (1/1 rows) |
| sft pack accepted with no holdout_fp on a WARNING | 1 | ~40 | fix carried (1/1 rows) |
| tasks_closed_by_commit / facts_well_formed subprocess-per-row cost | 1 | ~40 | fix carried (1/1 rows) |
| git reset --staged on merged file records a DELETION | 1 | ~25 | fix carried (1/1 rows) |
| score_matrix duplicate after cross-branch union merge | 1 | ~25 | fix carried (1/1 rows) |
| sft_math.py refuses pack whose holdout_fp mismatches live hashes | 1 | ~25 | fix carried (1/1 rows) |
| CUDA_VISIBLE_DEVICES in exp cmd not launch cmd; card_claim records assertion not open | 1 | ~20 | fix carried (1/1 rows) |
| podput tracked-path guard refused pod_push.sh's own calls | 1 | ~20 | fix carried (1/1 rows) |
| git fetch stale FETCH_HEAD (worktrees share one .git) | 1 | ~15 | fix carried (1/1 rows) |
| ff merge runs no pre-commit hook; wip lands on main unchecked | 1 | ~10 | **fix this commit** — scripts/hooks/post-merge |
| git restore --staged --worktree discarded uncommitted append | 1 | ~10 | fix carried (1/1 rows) |
| setsid zombie child claimed alive by card_claim (kill-0 says alive) | 1 | ~10 | fix carried (1/1 rows) |
| podput guard blocks the sanctioned path (PODPUT_TRACKED_OK unset) | 1 | ~12 | fix carried (1/1 rows) |
| selftest acquire() caller-dependent (ppid = shell or wrapper) | 1 | ~8 | fix carried (1/1 rows) |
| hand-written tasks.jsonl id reuse invisible to max+1 allocator | 1 | ~6 | fix carried (1/1 rows) |
| branch selftest registration gates nothing (hook runs main's copy) | 1 | ~6 | fix carried (1/1 rows) |
| dirty derived file aborts merge before drivers are consulted | 1 | ~5 | fix carried (1/1 rows) |
| manifest conflict before this worktree installed the driver | 1 | ~4 | fix carried (1/1 rows) |
| EXPERIMENTS.md render conflicts between branches | 1 | ~3 | fix: merge driver (regen, not conflict) |
| friction add --commit used --no-verify, manufactured a manifest refix | 1 | ~3 | fix carried (1/1 rows) |
| ff merge brings a branch manifest that no longer matches HEAD | 1 | ~2 | fix carried (1/1 rows) |
| concurrent peer merges raced controller's merge on HEAD/index | 1 | ~2 | fixed by scripts/merge_main.sh (mkdir lock) |
| merge tree loses a path only its second parent held | 1 | n/r | fix carried (1/1 rows) |
| merge that drops a path records D; --no-merges discriminates authorship | 1 | ~0 | fix carried (1/1 rows) |
| staged-dirty manifest after b0-ve-rownorms merge aborted the next | 1 | n/r | root-fixed by 56fa71b5 |
| e1's edits staged in integration tree blocked three-way merges | 1 | n/r | **unfixed** (0/1 rows) |
| harness task done refuses --commit not on main; owner cannot merge | 1 | n/r | fix carried (1/1 rows) |
| pod_push --all refused whole batch (run_ddp.sh executing past resume offset) | 1 | n/r | fix carried (1/1 rows) |
| launch_tests.json last-write-wins erased L12 arch rows | 1 | n/r | fix carried (f2bd7bc0 re-key) |
| 3b-14 prior= citation fails tasks_paired_and_prior; merge refused | 1 | n/r | **unfixed** (0/1 rows) |
| scan_eval_golds.py default use_char=True shredded code/math syntax | 1 | n/r | fix carried (1/1 rows) |
| gate_failure_shapes.md N collision on concurrent appends | 1 | n/r | fix carried (1/1 rows) |
| podput of tracked scan_eval_golds.py before merge; drift gate refused launch | 1 | n/r | fix carried (1/1 rows) |

## 2026-09-04 (21 rows, 18 causes; first summary, overdue from 12:00Z, ran 16:5xZ)

| cause | n | min lost | resolution |
|---|---|---|---|
| manifest merge conflicts (pod_head_manifest regenerated per commit) | 4 | ~16 | **fix 56fa71b5** — untracked + gitignored, pod_push regenerates from HEAD |
| lane card serialization (every non-training job routed to one lane) | 1 | ~40 | **task de-44** (opened 2026-09-04) |
| sft pack accepted with no holdout_fp on a WARNING | 1 | ~40 | fix carried (1/1 rows) |
| tasks_closed_by_commit / facts_well_formed subprocess-per-row cost | 1 | ~40 | fix carried (1/1 rows) |
| score_matrix duplicate after cross-branch union merge | 1 | ~25 | fix carried (1/1 rows) |
| sft_math.py refuses pack whose holdout_fp mismatches live hashes | 1 | ~25 | fix carried (1/1 rows) |
| ff merge runs no pre-commit hook; wip lands on main unchecked | 1 | ~10 | **unfixed** |
| setsid zombie child claimed alive by card_claim (kill-0 says alive) | 1 | ~10 | fix carried (1/1 rows) |
| selftest acquire() caller-dependent (ppid = shell or wrapper) | 1 | ~8 | fix carried (1/1 rows) |
| hand-written tasks.jsonl id reuse invisible to max+1 allocator | 1 | ~6 | fix carried (1/1 rows) |
| branch selftest registration gates nothing (hook runs main's copy) | 1 | ~6 | fix carried (1/1 rows) |
| dirty derived file aborts merge before drivers are consulted | 1 | ~5 | fix carried (1/1 rows) |
| manifest conflict before this worktree installed the driver | 1 | ~4 | fix carried (1/1 rows) |
| EXPERIMENTS.md render conflicts between branches | 1 | ~3 | fix: merge driver (regen, not conflict) |
| friction add --commit used --no-verify, manufactured a manifest refix | 1 | ~3 | fix carried (1/1 rows) |
| ff merge brings a branch manifest that no longer matches HEAD | 1 | ~2 | fix carried (1/1 rows) |
| gate_failure_shapes.md N collision on concurrent appends | 1 | n/r | fix carried (1/1 rows) |
| staged-dirty manifest after b0-ve-rownorms merge aborted the next | 1 | n/r | root-fixed by 56fa71b5 |

## 2026-09-04 (second run; 34 rows, 31 causes; ran 04:31Z)

| cause | n | min lost | resolution |
|---|---|---|---|
| flash_attn mask_mod wrong gradients on SM 9.0 (forward correct) | 1 | ~95 | fix carried (1/1 rows, 03:30) — fact eff.flash_attn_cute_mask_mod_backward_wrong_sm90, two-call workaround verified |
| merge_main.sh pathspec+pipe-filter hid commit failure (~40 min dirty integration tree) | 1 | ~40 | fix by habit, controller side (6e, 2026-09-04): writes unfiltered, exit status read |
| lane card serialization (every non-training job routed to one lane) | 1 | ~40 | task de-44 (opened 2026-09-04) |
| ghost running: pod experiments.jsonl lacked closes written on main | 1 | ~40 | fix carried (1/1 rows, 02:46) |
| sft pack accepted with no holdout_fp on a WARNING | 1 | ~40 | fix carried (1/1 rows) |
| tasks_closed_by_commit / facts_well_formed subprocess-per-row cost | 1 | ~40 | fix carried (1/1 rows) |
| score_matrix duplicate after cross-branch union merge | 1 | ~25 | fix carried (1/1 rows) |
| sft_math.py refuses pack whose holdout_fp mismatches live hashes | 1 | ~25 | fix carried (1/1 rows) |
| podput tracked-path guard refused pod_push.sh's own calls | 1 | ~20 | fix carried (1/1 rows) |
| manifest merge conflicts (pod_head_manifest regenerated per commit) | 4 | ~16 | fix 56fa71b5 + 89d86882 |
| podput guard blocks the sanctioned path (PODPUT_TRACKED_OK unset) | 1 | ~12 | fix carried (1/1 rows) |
| ff merge runs no pre-commit hook; wip lands on main unchecked | 1 | ~10 | **unfixed** |
| setsid zombie child claimed alive by card_claim (kill-0 says alive) | 1 | ~10 | fix carried (1/1 rows) |
| selftest acquire() caller-dependent (ppid = shell or wrapper) | 1 | ~8 | fix carried (1/1 rows) |
| hand-written tasks.jsonl id reuse invisible to max+1 allocator | 1 | ~6 | fix carried (1/1 rows) |
| branch selftest registration gates nothing (hook runs main's copy) | 1 | ~6 | fix carried (1/1 rows) |
| dirty derived file aborts merge before drivers are consulted | 1 | ~5 | fix carried (1/1 rows) |
| manifest conflict before this worktree installed the driver | 1 | ~4 | fix carried (1/1 rows) |
| EXPERIMENTS.md render conflicts between branches | 1 | ~3 | fix: merge driver (regen, not conflict) |
| friction add --commit used --no-verify, manufactured a manifest refix | 1 | ~3 | fix carried (1/1 rows) |
| ff merge brings a branch manifest that no longer matches HEAD | 1 | ~2 | fix carried (1/1 rows) |
| podput of tracked scan_eval_golds.py before merge; drift gate refused launch | 1 | n/r | fix carried (1/1 rows) |
| staged-dirty manifest after b0-ve-rownorms merge aborted the next | 1 | n/r | root-fixed by 56fa71b5 |
| e1's edits staged in integration tree blocked three-way merges | 1 | n/r | **unfixed** (0/1 rows) |
| harness task done refuses --commit not on main; owner cannot merge | 1 | n/r | fix carried (1/1 rows) |
| pod_push --all refused whole batch (run_ddp.sh executing past resume offset) | 1 | n/r | fix carried (1/1 rows) |
| launch_tests.json last-write-wins erased L12 arch rows | 1 | n/r | fix carried (f2bd7bc0 re-key) |
| 3b-14 prior= citation fails tasks_paired_and_prior; merge refused | 1 | n/r | **unfixed** (0/1 rows) |
| scan_eval_golds.py default use_char=True shredded code/math syntax | 1 | n/r | fix carried (1/1 rows) |
| concurrent peer merges raced controller's merge on HEAD/index | 1 | n/r | fixed by scripts/merge_main.sh (mkdir lock) |
| gate_failure_shapes.md N collision on concurrent appends | 1 | n/r | fix carried (1/1 rows) |
## 2026-09-05 (53 rows, 48 causes; ran 04:3xZ)

| cause | n | min lost | resolution |
|---|---|---|---|
| data/pod_head_manifest.txt is regenerated by the pre-commit hook on ever… | 4 | ~16 | fix 56fa71b5 + 89d86882 (verified untracked 2026-09-05) |
| near_miss,process_failure (minutes not reported) | 3 | n/r | fix check_friction_minutes_required (09-04); direct-launch → task de-60 |
| CUDA_VISIBLE_DEVICES=4 was written into the exp rows cmd field and NOT i… | 1 | ~20 | fix carried (1/1 rows) |
| EXPERIMENTS.md is rendered from runs/experiments.jsonl by exp.py render,… | 1 | ~3 | fix carried (1/1 rows) |
| I podput datagen/scan_eval_golds.py (f056e238 content) to the pod repo p… | 1 | n/r | fix carried (1/1 rows) |
| I ran `git reset HEAD runs/redaction_handread_v14.tsv` on a file that ha… | 1 | ~25 | fix carried (1/1 rows) |
| MY OWN SEQUENCE, not git merge. I appended the amendment to the worktree… | 1 | ~10 | fix carried (1/1 rows) |
| UnboundLocalError in run_checks timeout strike handling: strikes[name]=n… | 1 | ~10 | fix carried (1/1 rows) |
| a dirty runs/friction.jsonl or data/pod_head_manifest.txt aborts the mer… | 1 | ~5 | 56fa71b5 (manifest untracked) + friction add --commit for th |
| a fast-forward merge runs no pre-commit hook, so a wip commit that would… | 1 | ~10 | **unfixed** (0/1 rows) |
| a hand-written tasks.jsonl row reused an id, and the id allocator takes … | 1 | ~6 | fix carried (1/1 rows) |
| a merge commit's tree can lose a path that only its SECOND parent held, … | 1 | n/r | fix carried (1/1 rows) |
| a merge that drops a path DOES record a D against the parent that held i… | 1 | ~0 | fix carried (1/1 rows) |
| a setsid backgrounded child whose wrapper exits immediately leaves nobod… | 1 | ~10 | fix carried (1/1 rows) |
| after a three-way merge of b0-ve-rownorms the manifest sat staged-dirty … | 1 | n/r | 56fa71b5: no regen in the hook at all |
| behind-main refusal livelocks against hook duration: the pre-commit hook… | 1 | n/r | **unfixed** (0/1 rows) |
| broken world _broken_friction_minutes_required wrote 4 fixture rows to t… | 1 | ~10 | fix carried (1/1 rows) |
| check_shared_file_claim FAILs during `git merge`, naming INCOMING files … | 1 | ~9 | fix carried (1/1 rows) |
| circular, and the circle is the hook-resolution rule. The hook refused t… | 1 | ~15 | fix carried (1/1 rows) |
| controller's three merge_main.sh commits never landed: 'git commit <path… | 1 | ~40 | habit, controller side: writes run unfiltered and $? or the  |
| data/pod_head_manifest.txt conflicted on a merge I started before instal… | 1 | ~4 | fix carried (1/1 rows) |
| deadlock between two hooks: the behind-main guard refuses a commit while… | 1 | ~12 | fix carried (1/1 rows) |
| docs/lessons/gate_failure_shapes.md: new shapes are appended as '## N.' … | 1 | n/r | fix carried (1/1 rows) |
| e1 committed ed93d879/7ecc860e directly on main while the controller's m… | 1 | ~2 | scripts/merge_main.sh (mkdir lock), AGENTS.md:374 and the ho |
| e1's edits staged in the integration tree (A scripts/e1_n8_row_edge_prob… | 1 | n/r | **unfixed** (0/1 rows) |
| every worktree runs MAIN's copy of scripts/hooks/pre-commit (.git/hooks/… | 1 | ~6 | fix carried (1/1 rows) |
| fast-forward merge brings a branch manifest that no longer matches HEAD;… | 1 | ~2 | fix carried (1/1 rows) |
| file_claim.py uses positional 'selftest' but the hook runs it as '--self… | 1 | ~15 | fix carried (1/1 rows) |
| flash_attn_4-4.0.0b15's mask_mod produces a correct FORWARD and wrong GR… | 1 | ~95 | fix carried (1/1 rows) |
| friction add --commit used --no-verify and printed an OWED: pod_drift --… | 1 | ~3 | fix carried (1/1 rows) |
| git fetch /Users/bytedance/code/aupai main fetched a ref three commits b… | 1 | ~15 | fix carried (1/1 rows) |
| harness launch routes every non-training job to the single lane card, so… | 1 | ~40 | **unfixed** (0/1 rows) |
| harness task done refuses a --commit that has not reached main, and the … | 1 | n/r | fix carried (1/1 rows) |
| measured a temp-dir leak fix THROUGH the hook and read 0 leaked, which w… | 1 | ~6 | fix carried (1/1 rows) |
| merge_main.sh could not take the lock in 120 s and exited 1; the same co… | 1 | n/r | **unfixed** (0/1 rows) |
| merge_main.sh:126-128 removes .git/merge_main.lock when a WAITER finds i… | 1 | n/r | **unfixed** (0/1 rows) |
| pod's runs/experiments.jsonl lacked the closes written on main (pod_push… | 1 | ~40 | fix carried (1/1 rows) |
| pod_push --all refused the whole 5-file batch because run_ddp.sh is exec… | 1 | n/r | fix carried (1/1 rows) |
| runs/launch_tests.json keeps one row per test path (last write wins); St… | 1 | n/r | fix carried (1/1 rows) |
| runs/score_matrix.jsonl held two rows for one (ckpt, profile) after a cr… | 1 | ~25 | fix carried (1/1 rows) |
| runs/tasks.jsonl row 3b-14 prior='facts/corpus_supply.json' fails tasks_… | 1 | n/r | **unfixed** (0/1 rows) |
| scan_eval_golds.py defaulted use_char=True, applying e1-28's char-13 win… | 1 | n/r | fix carried (1/1 rows) |
| sft_math.py accepted a pack with NO holdout_fp on a printed WARNING, so … | 1 | ~40 | fix carried (1/1 rows) |
| sft_math.py:168 refuses a pack whose holdout_fp mismatches the live data… | 1 | ~25 | fix carried (1/1 rows) |
| tasks_closed_by_commit and facts_well_formed spawned one subprocess per … | 1 | ~40 | fix carried (1/1 rows) |
| the selftest's in-process acquire() calls passed no pid, so the holder d… | 1 | ~8 | fix carried (1/1 rows) |
| ~/bin/podput (mtime 2026-09-04 06:45) refuses any path tracked on main a… | 1 | ~12 | fix carried (1/1 rows) |
| ~/bin/podput tracked-path guard (fb, 06:45Z) refused pod_push.sh's own p… | 1 | ~20 | fix carried (1/1 rows) |


open tasks per owner (check one_deliverable_per_owner, WARN threshold >1): de 6, e1 4, 3b 1, 44 1, b0 1, tilerl 1 (14 total)

committer-pushes rule not firing: six committers' files on main had not reached the pod; 4c pushed them with --all (2026-09-05). The rule says each committer pushes their own files; nothing detected the drift until a manual diff.
