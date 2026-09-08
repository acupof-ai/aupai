# Controller board (fb) — 2026-09-08

Cards: all eight are tileRL's, granted by the user directly. aupai runs no GPU job.

## User orders in force

| order | state |
|---|---|
| Clean the whole repository, not one redundant word | seven tracks, below |
| Any file readable by someone who has never seen the repo | folded into every track, no separate renaming round |
| Worktrees cut to the live working set | 32 → 14; owners remove their own |
| At most two open tasks per person | held all night; the count is work in progress, not work awaiting someone else's review |
| Corpus reproducible byte-for-byte from zero | 98 leads; blocked behind the 40,000-vs-1,200 finding below |
| Distillation pipeline, teacher Qwen3.8-27B | **PAUSED by user ruling 2026-09-08: no cards lent.** Design complete and parked |
| Never delete without a named target | every removal names its files and runs each first |

## Critical path

**de's shared-config guard onto main.** Committed locally on `de-agents-clean` (`f223e560` + a `wip:` commit), not pushed, sharing a branch with PR #85. The hook everyone executes is the integration tree's symlink, so a commit that has not reached main is not running anywhere: e1 has been refused nine times, naming eight different innocent files. Split onto a branch cut from main, squash the wip (it was committed with `--no-verify`, so the hook never ran on it), open a PR, 44 reviews.

## Seven tracks

| track | owner | state | next gate |
|---|---|---|---|
| scripts and entry points | b0 | #94 landed: three edge types, unreachable 79 → 70, report prints tree/population/edge kinds/ledgers read; readability debt 9 of 537 | 3b's three PRs (#88 #93 #95); then the `score_matrix` watchdog gap |
| AGENTS.md | de | #85 open; guard is the critical path above | guard to main, then readability |
| docs | 44 | #83 landed three markers; 86 documents, zero duplicate questions | de's guard review; then 40,000-vs-1,200 |
| facts | e1 | `a2361230` landed five files; read-side timezone rule added to `check_timestamps_are_utc`, four real defects found, three in e1's own scripts | two scripts blocked behind the guard |
| ledgers | 3b | `no_ghost_close` attributed: 188 legal + 8 milestone + 31 forged + 0 of the suspected shape; ceiling 180 → 196 | per-ledger primary key, and the forged 31 as a literal set with new keys asserted empty |
| eval, filters, probes | fb | 71 files, 31 selftests pass; divisor defect isolated to `domain_bpb`, fix merged | 16 metrics have no known-answer case |
| datagen and mathbank | 98 | #92 open (pod wrapper mangles non-ASCII argv into false zeros) | that first — it contaminates other people's readings |

## The night's single finding

Thirteen instruments each answered a question narrower than the one asked, and none reported that it had. Measured, not asserted; every row is an incident from 2026-09-08.

| instrument | question asked | question answered |
|---|---|---|
| shared-config guard | who changed the shared config | who happened to be running (10 namings, 10 wrong, 0 repeats) |
| file scan | how many files does this repo hold | how many are under the tree I was run in (537 vs 13,665) |
| `gh pr list` | which PRs exist | the most recent N |
| `reachability.py` | which files are unreferenced | which files are unreferenced by anything except my own FATE dict (12 self-rescued) |
| `harness check` output | how many checks are there | how many passed (71 read as the total; it is 109) |
| `git log --date=short` | when did this land | local midnight, not UTC (15 of 70 pairs were artifacts; 42 of 145 paths render a day late) |
| `merge-base --is-ancestor` | did the running copy have the fix | does the commit's ancestry contain it |
| unreachable total 79 → 56 | did the new edges help | yes, and it concealed 12 self-rescues moving the same direction |
| `exp.py:582` comment | what does the fabricated row carry | correct on `hypothesis`, wrong on `commit`, 20 lines from the code |
| mutation sweep "ALL KILLED" | did the mutants die on assertions | they died on `FileNotFoundError`, twice |
| my own "40,000 rows vs a cap of 1,200" | can this batch's recorded command have produced it | how many programs the library holds -- a different unit, never checked |
| `gh pr diff --name-only` | which files does this PR change | which files the branch's history touched -- a revert leaves them listed. The predicate is `git diff base..head --stat` |
| a PASS line's summary (mine, via de's) | what does this check assert | what the summary line happens to print. `shapes_table_covers_doc` DOES refuse duplicate numbers (`harness.py:2390`, verified on a constructed world); its PASS line's "each referenced exactly once" is about the rule table, and both of us read the summary instead of the predicate |

Two derived rules, both adopted: **an unusually tight cluster is a systematic instrument offset until shown otherwise — a real effect has spread** (3b, from 18 samples all inside 7.1–8.0h, which was a timezone constant); and **rewrite the question into a form that reads bytes directly** (3b: hash the file, compare UTC to UTC, run the target copy itself).

Shapes R12–R14 are PR #96, stacked on #81. A never-triggered exclusion belongs to R12 — it is green because it did not run — not to R14, whose signature is a tool's own source appearing in its own output.

## Open, owned

| item | owner | why it matters |
|---|---|---|
| A writer outside `exp.py` hand-appends rows to `runs/experiments.jsonl` | 3b | a refusal only guards the path through it; `pod_pull_ledgers` is cleared by time order and by reading `append_rows` |
| ~~40,000 rows against a cap of 1,200~~ REFUTED 2026-09-08 | 44 | the units did not match: 1,302 is the count of PROGRAMS in the library, the run's cap is 100,000 rows, and 40,000 is the recorded L4 target (100,000 x 0.4) to the row. pod holds 97,771 rows with sha256 matching PROVENANCE. `facts/corpus_supply.json#cs.math_short_v8_cap_audit`, PR #98 |
| `score_matrix.jsonl` has 1 dedicated watcher against `tasks.jsonl`'s 6 | b0 | its fold key makes rewrite legal, so append-only checks cannot see a changed value; a wrong factor table sat on main for hours |
| Mutation sweeps need a positive control | de | DONE in PR #99: M0 survives, and the new died-for-the-right-reason criterion caught M2 dying on `FileNotFoundError` on its first run |

## Corpus reproducibility

A corpus build should be a pure function of source bytes, pipeline version and seed.

| finding | consequence |
|---|---|
| `filters_fp` hashes exactly three files: `filters/pass{1,2,3}_garbage.py` | 15 of 50 domains can say the garbage filters were identical. **Zero of 50 are demonstrated byte-reproducible.** Was 14/49 here until 2026-09-08; e1 recorded the drift as `config.count_drifted` rather than rewriting `value` |
| 2,010 shard files have link count above one | domains are not disjoint. Disk holds 248.93 GB against a per-domain sum of 348.30 GB; a per-domain rebuild double-counts and drops the hardlinks |
| One frozen batch excludes inputs that no longer exist | unreproducible by definition — a fourth answer, not a special case of "no" |
| ~~A frozen batch has 40,000 rows against a program cap of 1,200~~ | **REFUTED.** I compared rows to programs. No constant `1200` exists in `mathbank/`; the figure came from a scheduling note. The recorded command IS the command that ran, to the row. What survives: `PROVENANCE.md:57`'s own arithmetic is wrong (509 x 150 = 76,350, not 57,771) -- the stall is real, its recorded explanation is not |

## Distillation — designed, paused

Route: sequence-level. Our vocabulary is 32,773 against the teacher's 248,044, so there is no token-level alignment and logit KL is not a tuning problem.

| quantity | value | basis |
|---|---|---|
| teacher generation throughput | 130.3 tok/s | 1×H20, NVFP4, tp=1, batch 8, LoRA r16, measured 2026-09-08 |
| samples per card-hour, cap 6144 | 141 | mean 3331 tokens |
| samples per card-hour, cap 2048 | 335 | **boundary: contaminated by 32% truncation, do not cite** |
| full openo1, K=4 | ~92 card-days | 8 cards ≈ 12 days, labelled an unverified linear extrapolation |
| teacher correctness, level-5 math | 91% | 64% was a lower bound read as a point value: truncated is unscored, not wrong |

Cap 2048 truncated 32% of level-5 generations and 84% of those were correct answers cut off. **Truncation is a second difficulty filter acting in the same direction as the ≥3/4 agreement filter — both drop long-reasoning problems, and the shared latent is reasoning length.** Pre-registered: truncation rate per domain is reported; truncated samples are dropped before the subset comparison, never after; the calibration batch runs at cap 8192 so the truncation rate at every smaller cap is read off one length distribution; the cap is the curve's knee, and the discarded tail must pass the collapse criterion already registered for the agreement filter. A gap in the length histogram below the cap means pure truncation; a continuous approach means real failures mixed in.

Open for whoever picks this up: at 91% teacher correctness, is the agreement filter worth the difficulty skew it introduces? The design was written when the teacher was believed to be 64%.

## Closed and not reopening

| item | state |
|---|---|
| The 30B leg | closed at step 34,000 of 38,146 by user ruling, recorded as an incomplete schedule |
| What annealing was worth | −6.89% on the unweighted mean, same run and same held-out rows, all nine domains down. Per token, 19× a constant-rate token |
| `domain_bpb` divisor | real, about 2×, fix merged. Known answer: true 8.000, reported 5.460 |
| `answer_present` at three demos | retired as a primary readout: 0.1147–0.5433 within one recipe, sd 9.2× the binomial floor |
| SFT packs | 21 packs, zero with a current holdout stamp. A reporting defect, not a training hazard — both cases refuse today |
| `no_ghost_close` ceiling | 180 → 196. The 31 forged rows stay out of the ceiling and become a literal set whose new keys must be empty |

## Open user decisions

1. Corpus composition for the next full run.
2. Pod disk at 95%.
