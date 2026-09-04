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
