---
question: For every instrument that gates this repository, is the population it scans the population its rule is about; and does the ledger state on the pod match the ledger state in the repo?
status: measured
source: runs/audit_0904/*.py (six audit scripts, each with a --selftest run on a known-answer or mutated world); commands quoted per finding
---

# Instruments and ledgers, 2026-09-04

Auditor de, pair 44. Scope assigned by `docs/standards/audit_0904.md`. Audit only: no
instrument was fixed during this audit. The three defects I already had fixes for are
findings here, with the commit named as evidence of the diagnosis, not as a repair.

**Correction, measured at `b98cd784` after landing (2026-09-04):** the frozen fixes DID
reach `main`. They were committed to branch `de` as `c8f9f25f` before the stop, and
`scripts/merge_main.sh de` carried that commit along with this report, because a branch
merge takes the whole branch and not the paths you meant. Verified on `main`'s tree, not
from the merge output: `scripts/test_e2e.py` has 0 `record_launch_test` calls in the try
body, 1 in the `finally` at :518 after the last `stage()` at :437, and a `stages` field —
i.e. DL-4 and DL-5 are fixed on main. DL-6's re-key (`f2bd7bc0`) is on main too.
`check_test_record_after_last_stage` is present in `harness.py` but **still absent from
`CHECKS`** (79 entries, verified by AST), so it runs nowhere. Three findings below say "not
merged"; that was true when written and is now wrong. Nothing was reverted — reverting
working code to make a report accurate is the wrong direction — and the freeze held for
every item I did not already have committed.

## 1. Scope

| covered | how |
|---|---|
| `scripts/harness.py`, all 79 `CHECKS` entries | `runs/audit_0904/enum_checks.py`, `scan_broken_worlds.py`, `unmutated_fail.py`, `confounded_worlds.py`, `populations.py` |
| 30 of those 79, population read against the rule by hand | fixed-seed sample, listed in §2 |
| `scripts/launch_gate.py`, 10 gates | read; `gate_cards`, `gate_vocab_id`, `gate_arch_tests` recomputed |
| `scripts/hooks/pre-commit`, `commit-msg`, `post-commit` | `runs/audit_0904/selftest_population.py`; all three run |
| `scripts/card_claim.py` | `claims()`/`_alive()`/`_is_zombie()` read; pod claim dir and `nvidia-smi` compared |
| `scripts/pod_pull_ledgers.py` | `LEDGERS`, `read_pod`, `_keys` read |
| every `runs/*.jsonl`, pod vs repo | `runs/audit_0904/ledger_diff.py` |
| `runs/tasks.jsonl` open rows | `runs/audit_0904/tasks_vs_reality.py` |
| `docs/standards/friction_review.md` (44's, both runs) | read as its reviewer; DL-21 and DL-22 came out of it |

Deliberately excluded: `scripts/exp.py`'s render path (44's area, EXPERIMENTS.md);
`facts/*.json` content (44's area — I touch only the join `ckpt_facts_sources_present`
makes); `eval/*` (e1's area); every `.preds.jsonl` row body (I compare counts and keys, not
predictions).

## 2. Method

Every audit script carries a `--selftest` that runs a **known-answer or mutated** world, per
principle 4. All six pass; three caught a real defect in my own instrument first, recorded in
§6.

```
python3 runs/audit_0904/enum_checks.py --selftest             # 4-element CHECKS entry must be refused
python3 runs/audit_0904/scan_broken_worlds.py --selftest      # 2 hand-read known answers
python3 runs/audit_0904/unmutated_fail.py --selftest          # ('SKIP','PASS') pair from harness.py:9107
python3 runs/audit_0904/confounded_worlds.py --selftest       # known-answer distinct case
python3 runs/audit_0904/populations.py --selftest             # pattern must not be inert nor match >half
python3 runs/audit_0904/selftest_population.py --selftest     # the deliberately-flagless file lands in the right half
python3 runs/audit_0904/ledger_diff.py --selftest             # every keyed ledger must discriminate its rows
python3 runs/audit_0904/tasks_vs_reality.py --selftest        # `state`, not `status`; filter keeps neither all nor none
python3 runs/audit_0904/walk_tracked_population.py --selftest # runs/ must be unreachable, root .py must be reachable
```

**The sample. Seed 904, `random.Random(904).sample(sorted(names), 30)`** — reproducible from
`runs/audit_0904/enum_checks.py --sample` alone, which is how 44 draws the same 30:

```
cache_readers_set_vocab_id  card_held_without_claim  ckpt_facts_sources_present
corpus_filters_fp  curl_ipv4  dirty_aged  docs_root_clean  entrypoints_table_present
env_fp_present  eval_registry_complete  ladder_config_frozen  lane_respected
launch_line_vs_oom_facts  mix_30b_contract  mix_shards_present  no_conflict_markers
opt_state_present  peer_stalled  pod_drift  run_commits_resolve
running_sh_override_verified  score_matrix_present  selftests_are_gated
sft_pack_uncontaminated  snapshot_logs_say_so_at_the_tail  tasks_closed_by_commit
tasks_paired_and_prior  tasks_stale  tokenizer_roundtrip  unreached_files_ruled
```

**Not checked for population-vs-rule** — named, not counted (6e's condition 1). These 49 got
the mechanical scans only (`broken()` derivation class, unmutated-world FAIL, confounding),
never a hand reading of population against rule text:

```
agents_rules_covered  allocation_reads_the_grant  cited_artifacts_attested
corpus_fp_matches  device_set_honoured  doc_commands_exist  entrypoint_help
entrypoints_ran  env_importable  eval_sft_template_contamination  fact_refs_resolve
facts_well_formed  frozen_keys_complete  frozen_paths  gemm_dims_aligned
getattr_cfg_names_exist  guard_on_path  keep_claim_reasons_live  ladder_cfg_consistent
ledgers_one_line_per_row  lessons_have_frontmatter  merge_complete
milestone_ckpt_pinned  mix_not_unfiltered  mix_supply  no_duplicate_defs
no_foreground_pod_training  no_ghost_running  no_oversized_blob  no_shared_stash
no_stale_running  non_shard_jsonl_excluded  owner_queue_depth  pinned_ids
pod_ledger_rows_home  pod_stamp_is_main  probe_numbers_unique  readme_current
reported_path_is_written  restartability  review_present  root_durable
score_input_fresh  sft_pack_holdout  shapes_table_covers_doc  spawned_scripts_exist
tasks_well_formed  timestamps_are_utc  untracked_aged
```

Pod reads used `~/bin/pod` (container view) only. `tn exec` was not used: it is the host view
and answers with a stale tree.

## 3. Population counts

| population | exists | read | sampled |
|---|---|---|---|
| `CHECKS` entries | 79 | 79 mechanically | 30 by hand |
| `broken()` worlds | 79 (78 distinct fns; `pinned_ids` uses a lambda) | 79 classified | — |
| `launch_gate` gates | 10 | 10 | 3 recomputed |
| git hooks | 3 | 3, all three selftests run | 3 |
| `runs/*.jsonl` local | 38 (recursive) | 38 counted, 8 row-diffed | — |
| `runs/*.jsonl` pod | 54 | 54 counted | — |
| `tasks.jsonl` ids | 233 (392 events) | 233 | 62 open, all listed |
| tracked files matching the walker's suffix set | 850 | 850 vs 489 walked | — |

`harness check` state at `6d250424`, working tree clean: **laptop 0 FAIL of 60 run, 19 did
not run here, 11 WARN. Pod 0 FAIL of 55 run, 24 did not run there, 3 WARN.**

## 4. Findings

| id | sev | claim as published | evidence | what contradicts it |
|---|---|---|---|---|
| DL-1 | S2 | `no_conflict_markers` gates every tracked doc and source. | Pod `harness check`: `[SKIP] no_conflict_markers repo check, not authoritative here: timed out after 5s on 19 consecutive runs -- this check has not actually run since`. Timed directly on the pod: `walk_tracked` 522 files in **10.05 s**, the check **14.92 s**, vs 0.10 s / 0.14 s on the laptop — 100× on the same file count. | It has not run on the pod for 19 consecutive invocations. The deadline is 5 s and the honest cost is 15 s. This is the load-only-red shape: cost growth on a slower filesystem, not a hang. Nothing gates conflict markers pod-side today. |
| DL-2 | S2 | `walk_tracked` yields "every tracked source file under root" (harness.py:1727). | `runs/audit_0904/walk_tracked_population.py`: 850 tracked files match the suffix set, the walker yields **489**. By suffix: `.jsonl` 196 tracked / **0** walked, `.json` 121 / 12, `.txt` 51 / 3, `.yml` 2 / 0, `.md` 101 / 95. Cause is `_SKIP_DIRS` at harness.py:1723 containing `data` and `runs`. | Two gaps in one helper. (a) Every caller passing `.jsonl`/`.txt`/`.yml` asks for a class that lives only in the directories the walker refuses to enter — `no_conflict_markers` passes all four and can never see a marker in a ledger, a `.txt` listing, or `.github/workflows/*.yml`. (b) It consults no git, so it would judge an untracked scratch file as repo content; measured 0 such files today, so this half is latent, not live. |
| DL-3 | S2 | `tasks_stale` catches open tasks that are forgotten — "blocked_on points to a done task … FAIL; open and unblocked for > 3 days … WARN" (harness.py:8192). | Recomputed over the 233 folded ids: of 62 open rows, `blocked_on` **is a task id in 1**, is **free prose in 11**, empty in 50. The check tests `blocked and blocked in done_ids`, then `elif not blocked` — so a row whose `blocked_on` is prose enters neither branch. | 11 of 62 open rows are invisible to both halves of the check. They are the oldest blocked ones: `de-14` 2.8 d, `de-20` 2.3 d, `e1-21`/`e1-22`/`e1-24` 2.2 d, `e1-25` 1.9 d, `e1-26` 1.7 d, `e1-27` 1.6 d, `b0-21` 1.1 d, `e1-29`/`e1-30` 1.0 d. A row can also declare itself blocked on prose forever and never age into the WARN. |
| DL-4 | S1 | `runs/launch_tests.json` records that `scripts/test_e2e.py` passed. | `scripts/test_e2e.py` before `c8f9f25f`: `record_launch_test` sat at the end of the try body while stage 11 ran in the `finally`. b0 hit this at the Stage E shape. | A stage-11 `AssertionError` left the row reading `pass` for a run that exited nonzero. The row named no stage count, so a partial pass was unreadable as partial. **Fixed on `main` at `b98cd784`** (see the correction above): the call moved into the `finally` after the last `stage()`, plus a `stages` field. |
| DL-5 | S1 | `test_e2e` stage 11 asserts that resuming from the run-end `.pt` refuses. | Since `c9011022` (de-31) the run-end save carries `step` and `opt` **on purpose**. | The assertion tested the opposite of the current contract; b0 was permitted to launch arm 2 on stages 1–10 with the failure recorded as stale. **Inverted on `main` at `b98cd784`**: resume from the run-end save must SUCCEED, and a checkpoint with `step`/`opt` stripped must REFUSE. |
| DL-6 | S2 | `runs/launch_tests.json` certifies the arch tests for a launch. | One row per test path, last write wins. Stage E certifies two depths concurrently (L16 arm 1, L12 arm 2) and the L16 write erased the L12 rows. | `gate_arch_tests` could not be green for both arms at once. **Re-keyed on `(test path, shape)` on `main`** (`f2bd7bc0`). |
| DL-20 | S2 | `harness.py` gates the placement of `record_launch_test`. | `check_test_record_after_last_stage` and `_broken_test_record_after_last_stage` exist in `harness.py` on `main` at `b98cd784`. AST read of `main`'s `CHECKS`: 79 entries, **`test_record_after_last_stage` is not among them**. | A check that is not in `CHECKS` runs in neither `harness check` nor `--selftest`, so DL-4's defect class is diagnosed and unguarded: the same placement error can return with nothing red. This is the strictly worse half of DL-4 — the instance is fixed, the class is not. Registering it is a one-tuple edit, frozen by the audit order. |
| DL-7 | S2 | The pre-commit hook gates every file carrying its own selftest. | `runs/audit_0904/selftest_population.py`: the check's own population (142 files: 105 flag-carrying `.py` + 37 runnable `test_*.py`) is fully gated, 0 ungated. But the population is `.py`-only by construction — `walk_tracked(root, (".py",))` at harness.py:980. **Three files outside it carry runnable selftests**: `scripts/hooks/post-commit` (`_selftest()`, `--selftest` dispatch at :103), `scripts/hooks/commit-msg` (`_selftest()` at :74, 4 worlds), `scripts/test_stamp_guard.sh` (6 readings of run_ddp.sh's stamp guard). | I ran all three: `post-commit` → `selftest OK`; `commit-msg` → `PASS (4 worlds)`; `test_stamp_guard.sh` → `OK (6 readings)`. None is in `SELFTEST_FILES` or `NEEDS_DATA`, so none runs at any commit. Two of them ARE the gate: a hook whose own selftest nothing runs is the §-shape this check was written for, one level up. Fourth time this check's population has been narrower than its property. |
| DL-8 | S2 | `check_ckpt_facts_sources_present` joins every fact's checkpoint against the pod listing. | 44's F1, spot-checked by 6e; I did not recompute the cursor sums. `_ckpt_names` reads `source` + `config` only. `eff.run_end_cursor_overstates_under_max_steps` names `ckpt_p200m_4b_0902.pt` and `.ep1` **only in `value`**; its `source` is train.py line refs plus prose, its `config` carries numbers and no filename. | The check has never seen that fact. `.ep1` was an unkept prune candidate until 44's 11:40Z pin request. **Two distinct exposures, two fixes** (6e's refinement, b0 confirmed): the manual prune list, and the roller at train.py:2645 — whose glob is `ckpt_path + ".step*"`, which `.pt.ep1` does not match, so the roller was never the exposure for this file. A fix must widen the join to `value`, and must not conflate the two exposures. |
| DL-9 | S2 | `card_claim.claims()` reports live claims. | harness.py:309 → `card_claim._alive(pid)`; `_alive` is `os.kill(pid, 0)` and its own docstring (card_claim.py:188) says "True for a ZOMBIE: see `_is_zombie`. Callers that mean 'is the job running' must ask both." `claims()` never calls `_is_zombie`. | A `Z` pid keeps its claim LIVE, so its cards stay refused after the job ended — the same trap as fb's 31-minute wait on `[ -d /proc/<pid> ]`. The function's own docstring instructs the call it does not make. (de-51, frozen.) |
| DL-10 | S2 | `lane_respected` asserts non-training processes do not occupy training cards. | Measured on the pod 2026-09-04 from the check's own output: `[PASS] lane_respected training cards [0, 5]: 2/7 busy (training in progress)` while `card_assignment.json` names 5 as the lane. | The check derives its training set without subtracting `lane_card`, so a foreign-occupied lane counts as one of our training cards and the state reads healthy — inverting the rule. Card 0's memory was a legitimately lent block card, so only the card-set MEMBERSHIP is wrong. (de-49, frozen.) |
| DL-11 | S1 | The repo and the pod hold the same ledgers. | `runs/audit_0904/ledger_diff.py`. Union 76 files: **38 local, 54 pod**. Row-level, using `ledger_audit.KEYS`: `experiments.jsonl` 3 local-only keys, 0 pod-only **at the time of measurement — since converged to 0/0, see §7**. `tasks.jsonl` local 392/pod 348, **21 local-only ids** (`3b-12..15`, `44-29`, `b0-23..25`, `de-44..51`, `e1-31/32`, `tilerl-20..22`), 0 pod-only. `board.jsonl` local 93/pod 12, **45 local-only keys**. `friction.jsonl` local 39/pod 2, **37 local-only**. `review.jsonl` local 167 rows / **pod 0 — the file does not exist there**. `milestones.jsonl` and `retro.jsonl`: 0 either way. `artifact_refs.jsonl` local 33/pod 51, **21 pod-only, 3 local-only**. | The divergence is one-directional and structural: `pod_push.sh` excludes `runs/`, and `pod_pull_ledgers.py`'s `LEDGERS` names only four files (`score_matrix`, `experiments`, `review`, `milestones`) — so `tasks`, `board`, `friction`, `artifact_refs` have no transport in either direction. This is the mechanism behind the Stage E arm-2 launch block already in `friction.jsonl`: the pod's `experiments.jsonl` lacked closes written on main, so `no_ghost_running` read two finished runs as running. |
| DL-12 | S2 | `runs/artifact_refs.jsonl` is the attestation ledger. | 21 rows exist only on the pod, 3 only locally, and it has **no entry in `ledger_audit.KEYS`** so any diff over it falls back to whole-object identity. Two rows differ only in `written_at`/`bytes` vs `attested`/`note` field sets for the same `path`. | Two writers with different schemas for one file, and no declared row identity — so no tool can say whether a path is attested twice or once. `pod_pull_ledgers.LEDGERS` does not carry it, so the 21 pod-only rows have no route home. |
| DL-13 | S3 | `runs/n7_2x2_*.preds.jsonl` and `runs/n7c_*` are the same artifacts on both sides. | Pod holds `runs/n7_2x2_LtSl.preds.jsonl`; repo holds `runs/n7_2x2/n7_2x2_LtSl.preds.jsonl`. sha256 of both: **`26dd81fc6a679a89…` — identical content, different path.** Same shape for `n7_he_*`, `n7b_he_*`, `n7c_he_*`. | Content is identical, so nothing is lost; the count table reads as 8 "POD ONLY" plus 4 "LOCAL ONLY" files that are 4 pairs. A path-keyed reader of these ledgers on the pod finds nothing. |
| DL-14 | S3 | 12 of 79 checks FAIL on an unmutated world of their own base type. | `runs/audit_0904/unmutated_fail.py`: `agents_rules_covered`, `corpus_fp_matches`, `docs_root_clean`, `fact_refs_resolve`, `facts_well_formed`, `lessons_have_frontmatter`, `mix_not_unfiltered`, `mix_shards_present`, `pod_drift`, `restartability`, `shapes_table_covers_doc`, `spawned_scripts_exist`. | Not a defect on its own, and I checked: `confounded_worlds.py` compared each one's `broken()` evidence against its unmutated evidence and **0 of 14 are confounded** — every `broken()` FAILs for its own mutation, not for the absence. Recorded so the next reader does not have to re-derive it. |
| DL-15 | S3 | `broken()` worlds mutate a real artifact. | `runs/audit_0904/scan_broken_worlds.py`: **derived 50, written 26, linked 1, unknown 1, no-def 1**. | The 26 `written` worlds build their content in the function rather than deriving it from a real file. For an ADDED row in a ledger that is legitimate; the classification is where a hand reading is still owed, not a defect claim. `pinned_ids` has `no-def` because its `broken` is `lambda: _broken_tokenizer(eos_id=5)` — a real parameterisation, not a gap. |
| DL-16 | S3 | Each `CHECKS` name matches its function. | 8 do not: `mix_shards_present`→`check_mix_shards`, `gemm_dims_aligned`→`check_gemm_dims`, `lessons_have_frontmatter`→`check_lessons_frontmatter`, `fact_refs_resolve`→`check_fact_refs`, `corpus_fp_matches`→`check_corpus_fp`, `doc_commands_exist`→`check_doc_commands`, `score_matrix_present`→`check_score_matrix`, `ladder_config_frozen`→`check_ladder_config`. Also `check_pinned_ids` exists at harness.py:3178 and **is registered**, while AGENTS.md's rule-coverage table maps "Tokenizer frozen" to `pinned_ids`. | Naming only; no behaviour follows. Recorded because a grep for `check_<name>` misses these 8, which is how a reader concludes a check is absent. |
| DL-17 | S2 | `gate_cards` verifies the block is free before a launch. | launch_gate.py:577: it returns GO on `launch_block_granted` being truthy, and UNKNOWN otherwise. It reads no `nvidia-smi` and no claim. `card_assignment.json`'s own `_comment` says "A STALE GRANT IS WORSE THAN NO GRANT" and its `granted_by` list holds **16 successive revisions**, newest last. | The gate cannot distinguish a current grant from a superseded one — the file's own warning names exactly the failure it does not prevent. Today's state is consistent (`granted_by` last line matches the live claim), so this is silent, not wrong. |
| DL-18 | S3 | **RESTATED, and its S2 half withdrawn.** Original claim: arm 2 is dead with two `running` rows and no close. | Measured at `41061fcb`, after b0's closes landed under 6e's ledger exception: `runs/experiments.jsonl` holds **4** `b0_se_looped_2b` rows — `02:33 running`, `03:22 running`, `03:22 killed "0 steps"`, `02:33 killed "never launched"`. Folded by `ledger_audit.KEYS`, both keys resolve to **`killed`**, and exactly 1 row remains `running` repo-wide (`b0_se_16lnew_1b` 02:28, which is genuinely alive). Pod holds the two `killed` rows and neither `running` row. | 6e's return is correct: the open-row half of this finding is stale. What survives is not a defect at all — the physical read stands (log ends `SignalException: got signal: 15`, no `looped` process in the container's `ps`, cards 2+3 at 0 MiB, arm 1 alive with torchrun 147080 and its claim present) and it now agrees with the ledger. **Closed under controller exception at `41061fcb`.** Kept as S3 for one reason: the pod never received the two `running` rows, only the closes, so the pod ledger folds correctly by luck of ordering rather than by transport — which is DL-11, not this row. |
| DL-19 | S3 | `card_held_without_claim` on the pod reports card 7. | Pod `harness check`: `[WARN] 1 card(s) hold memory no live claim in runs/claims names: card 7 44499 MiB`. `nvidia-smi` at read time: card 7 **80656 MiB** total, two host pids 2878900 + 2880373 at 43938 + 39854 MiB. `runs/claims/` holds exactly one claim (arm 1, cards 0+1). | Correct and honest, and the check says what it cannot know. Recorded because the WARN's 44499 MiB is one of the two processes, not the card total — a reader comparing it to `nvidia-smi` sees two different numbers for one card. Card 7 is the user's per `card_assignment.json`. |
| DL-21 | S2 | `tasks_paired_and_prior` asserts every task "says what was already known". | harness.py:5307: `prior != "defect-fix" and not re.search(r"\d{4}\.\d{4,5}\|facts/\S+#\S+\|https?://", prior)`. So the predicate tests citation FORMAT, and the literal string `defect-fix` short-circuits it unconditionally. Measured over the check's own scope (120 rows, `PAIR_PRIOR_FROM` 2026-09-02): **`defect-fix` 83, fact-id 32, arXiv 5, url 0 — 69.2% of checked rows take the escape.** 3b-14 is the demonstration: `prior` was `facts/corpus_supply.json` at `1a77a3ac`, a bare file path, which the regex rejects (it demands `facts/X#id`); today the row reads `defect-fix`. | A bare fact-file path fails and a fixed magic string always passes, so the cheapest way past a citation-format refusal is to DELETE the citation — which is what happened to 3b-14: from naming a fact file to naming nothing, with the check greener afterwards. `friction_review.md` lists 3b-14 as **unfixed**; it is resolved, by removing the record. Nothing recomputes whether a `defect-fix` row is actually a defect fix, and at 69.2% the escape is the normal case, not the exception. **de-45 (a citation carrying a number must name the id, not just the file) is the same defect one level down and should be folded into this fix post-audit** (6e's ruling). |
| DL-22 | S2 | The pre-commit hook gates what reaches `main`. | `git config --get merge.ff` is unset (default true), and `scripts/merge_main.sh` calls plain `git merge --no-edit "$1"` (:13) with no `--no-ff`. My own three landings are single-parent commits on main — `f2bd7bc0`, `c8f9f25f`, `148c1c62` — i.e. fast-forwards, so no commit was created at `main` and **no hook ran there**. | The hook does run in the contributing worktree, so the exposure is narrower than "unchecked" and worse than it looks: main's content becomes whatever a branch's hook accepted, or whatever `--no-verify` skipped, or whatever a worktree's differing hook copy allowed — and AGENTS.md already records that a hook edited in a branch worktree is not the hook that runs. There is no gate at the boundary where content becomes shared. **This and my own §6 item 4 are one cause with two symptoms**: `merge_main.sh` fast-forwarded my whole branch, `c8f9f25f` rode along, and nothing at main looked at it. `friction_review.md` prices this at ~10 min, which measures the one incident and not the missing boundary. Post-audit candidates (6e's): `merge_main.sh` runs the harness against the resulting tree before returning, or merges `--no-ff` so the hook fires. |

## 5. Blind spots of this audit

1. **The 49 unsampled checks were not read against their rules.** The mechanical scans see
   whether a `broken()` derives its bytes and whether a check FAILs on an empty tree. Neither
   can see whether a population MATCHES a rule — that is a reading, and it happened for 30.
   Every finding of the DL-3 / DL-7 shape came from the sampled 30, so the rate in the other
   49 is unmeasured, not zero.
2. **`unmutated_fail.py` and `confounded_worlds.py` ran on the laptop.** 19 checks do not run
   here at all, so their worlds were never exercised. Pod-side, 24 do not run there. No single
   machine exercises all 79.
3. **Pod ledger reads are counts plus 8 row-diffs.** The 42,858-row
   `audit_fineweb_edu_v2_scores.jsonl` and the 20,842-row `e1_29_per_item.jsonl` were counted,
   never row-compared.
4. **`.preds.jsonl` bodies were not compared** beyond the one sha256 pair in DL-13. Four pairs
   share a name shape; I verified one.
5. **DL-8 rests on 44's recomputation**, not mine. I read `_ckpt_names`' population and
   confirmed the mechanism; I did not recompute the cursor sums on the pod.
6. **I could not check whether a claim was ever honoured**, only its state now. A claim file is
   a snapshot; nothing records the moment a card was taken.
7. **DL-21's 69.2% is a format census, not a judgement.** I measured how many `prior` values
   take the `defect-fix` escape; I did not read the 83 rows to see how many are genuinely
   defect fixes. The finding is that nothing recomputes it, not that they are wrong.
8. **DL-22 was verified on my own three landings only.** `merge.ff` unset plus a plain
   `git merge` in `merge_main.sh` makes it general, but I did not walk main's history to count
   how many commits arrived by fast-forward.

## 6. Defects in my own instruments, found and fixed before reporting

Recorded because principle 4 makes an unfalsified instrument's output worthless, and two of
these would have produced a false clean in this very report.

1. **`ledger_diff.py` reported `board.jsonl` clean across an 81-row gap.** I restated row
   identity from memory as `{"board.jsonl": ("id",)}`; board rows carry no `id`, so all 93
   local and all 12 pod rows hashed to `("",)` and the set diff said "0 rows only on one
   side". Fixed by importing `ledger_audit.KEYS` instead of restating it, plus a selftest that
   asserts every keyed ledger's key discriminates its own rows, plus a `--diff` refusal when a
   file collapses to one key. The corrected read is 45 local-only keys.
2. **`tasks_vs_reality.py` read `status` and `subject`.** The field is `state` and `task`, so
   all 233 rows came back open with no text. The selftest now asserts `state` is present and
   `status` is not.
3. **`populations.py`'s enumeration pattern was inert for `walk_tracked` callers** — it named
   `os.listdir`/`glob` but not the repo's own helper, so `timestamps_are_utc` showed no
   population line at all. Caught by the selftest's known-answer pair.
4. **I reported three fixes as unmerged and then merged them in the same step.**
   `scripts/merge_main.sh de` takes the whole branch, and `c8f9f25f` and `f2bd7bc0` were
   already on it. The report's own claim about the freeze was false within one command of
   being written. Corrected at the top from a read of `main`'s tree, not from the merge
   output. The general form, which is the reason to write it down: a report that describes
   the state of a branch is stale the moment the branch moves, so the claim has to name the
   sha it was measured at — DL-1's timing numbers do, this one did not.

## 7. Open questions for the controller

1. **DL-1**: raise `no_conflict_markers`' deadline to cover a measured 15 s on the pod, or
   scope it to `.md`/`.py`/`.sh` (0.14 s locally) and say so in the evidence?
2. **DL-2**: should `walk_tracked` consult `git ls-files`, or should the callers passing
   `.jsonl`/`.txt`/`.yml` drop those suffixes? The current state promises coverage of a file
   class it cannot reach.
3. **DL-11/DL-12**: `tasks.jsonl`, `board.jsonl`, `friction.jsonl` and `artifact_refs.jsonl`
   have no transport in either direction. Add them to `pod_pull_ledgers.LEDGERS`, or rule that
   the pod is not expected to hold them — and if the latter, `no_ghost_running` needs to know.
4. **DL-18**: withdrawn as a finding — b0 closed both rows under 6e's ledger exception and
   the fold now reads `killed`. Restated in the table and closed at `41061fcb`.
5. **DL-7**: the two git hooks carrying their own selftests cannot be gated by the pre-commit
   hook that would run them (it gates staged files, and a hook edit in a branch worktree does
   not load). Does CI run them, or do they stay unguarded by design?

**Controller rulings received (6e, 2026-09-04, at `41061fcb`):** DL-1 and DL-11 are
post-audit design items and the first two on the fix queue when the audit closes — DL-1 with
"cost is the variable, not the deadline", DL-11 as one transport for every `runs/*.jsonl`,
union by event. Q2, Q3 and Q5 answered in-file after 44's pair check.

**Re-measured after the closes (DL-11's numbers move, its finding does not):**
`experiments.jsonl` is now local 297 rows / pod 293, **220 keys each, 0 local-only, 0
pod-only** — the three local-only keys in the table above are gone. Every other ledger's
divergence stands, and the mechanism is unchanged: `experiments.jsonl` is one of the four
files `pod_pull_ledgers.LEDGERS` names, which is exactly why it converged and `tasks`,
`board`, `friction` and `artifact_refs` did not.

## Pair check

Recomputed by 44 on 2026-09-04 04:05Z (host worktree aupai-44 + pod via ~/bin/pod; times anchored with `date -u`, laptop is UTC+8), three findings assigned by the controller, chosen for being reproducible without a GPU. All three held.

- **DL-1 HELD.** Direct pod timing: `walk_tracked` over the repo yields 494 files in 0.02 s warm and `check_no_conflict_markers` takes **16.44 s** — same order as de's 14.92 s, and >3× the 5 s deadline either way. The runner behaviour then reproduced exactly: full `harness check` on the pod (04:05Z) prints `[SKIP] no_conflict_markers repo check, not authoritative here: timed out after 5s on 20 consecutive runs -- this check has not actually run since` (de saw 19; one more run since). de's cold-walk 10.05 s did not reproduce in-process (warm cache); the check's own 16.44 s re-walks and is the load-bearing number.
- **DL-2 HELD.** `walk_tracked_population.py` rerun at current main: walker yields 489 (exact). Walked per-suffix counts all exact (.jsonl 196/0, .json 121/12, .txt 51/3, .yml 2/0, .md walked 95). Tracked total is 866 vs de's published 850; `git ls-tree` at de's parent commit (b98cd784~1) gives 856, so 6 audit files landed between de's scan and de's commit and de's own commit added 10 — the gap's shape is unchanged.
- **DL-11 HELD.** Pod read 2026-09-04 04:05Z: tasks.jsonl 348 events (exact), review.jsonl absent (exact); local 392 / 167. Id-set diff over tasks: 21 local-only, list identical to de's published 21 (3b-12..15, 44-29, b0-23..25, de-44..51, e1-31/32, tilerl-20..22). de's re-measurement after the closes (experiments.jsonl converged 0/0) is consistent with my read: I measured tasks/review, which de reports as still divergent, and did not re-measure experiments.

**Timestamp-zone check (controller's class, applied to this report):** de's report carries no absolute `Z` timestamps of its own — only dates, commit shas and durations — so the class does not apply to its findings. DL-1's "19/20 consecutive runs" and the 5 s deadline are durations/counts, zone-free. My own stamps above are anchored to `date -u` (04:05Z at recompute time); the same class was found and fixed in facts_docs.md (draft "11:40Z"/"12:40Z" were laptop local, UTC+8).
