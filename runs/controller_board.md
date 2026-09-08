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

## Queue state, 2026-09-08 12:05Z

28 PRs merged today, 18 still open, CI green. **Reviews are now being picked up without dispatch** — de cleared 44's four, e1 cleared 98's three, b0 and tilerl are on the rest. The standing rule changed at 11:50Z: **a PR's roster pair reviews it without waiting to be assigned.** The earlier version made me watch the queue faster; de's correction is the right one — the bottleneck was not that only I could see it, it was that only I was looking.

Merge order still binds three: **#81 (§276-278) → #96 (§279-282) → #85 (§283)**, and **#98, #100 before #101** (its two fact references are forward references that no check would catch — `_commit_delivers` parses `facts/<f>.json#<id>` only from `runs/tasks.jsonl` evidence, never from prose).

## Critical path — cleared 2026-09-08 11:45Z

The shared-config guard is on main (#99, `11c8a89c`) and **verified on the execution side, not only the merge side**: `executed_hook_matches_main` PASS, the integration tree's `pre-commit` byte-identical to main's. Production evidence in the two hours after: the shared config went from 88 branch sections to 98 — **ten pushes, ten misfire opportunities, and the branch-excluded digest never moved from `eec48396`.** Eleven false accusations, eleven different innocent files, zero repeats, ended.

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

## Known-answer audit of the 16 unguarded eval metrics — 2026-09-08 12:20Z, fb

11 groups (grouped by shared scoring path), every one run on CPU against the repo's own functions with only the model stubbed, each with a negative control, each claimed defect sent to two independent verifiers. **7 defects confirmed, 4 metrics reproduce their known answer, 0 blocked.** No metric was judged by reading it.

| # | metric | defect | consequence |
|---|---|---|---|
| 1 | `eval/code_fewshot.py:178`, `eval/code_l0prime.py:217` | `cont_ids = ids[len(pr):]` strips the prompt length off a value that already excludes the prompt (`train.py:1662` returns generated ids only; `eval/l1_fewshot.py:596` has it right) | **At 3-shot it discards 319–335 tokens — more than a whole solution — and scores the empty string. Measured on six gold rows that must score 6/6: 0/6, empty-continuation rate 100%.** Every number these two tools ever produced is invalid |
| 2 | `eval/l1_fewshot.py:60` | `ANS_RE`'s terminator class `(?:[。.\n]|$)` contains the ASCII full stop, which is also the decimal point, so the lazy capture stops there: "答案是 3.5。" yields "3". **Two retractions, and the second is the useful one.** (i) My fix — drop the `.`, as the monolingual sibling `math_zh.py:35` has it — looked wrong when scored on CAPTURED STRINGS: 4/10 current, 8/10 mine, 10/10 for e1's `(?:[。\n]|(?<!\d)\.|\.(?!\d)|$)`. (ii) e1 then retracted that: scored on `score()`'s RETURN VALUE, which is what anyone acts on, the tally is 4 / **11** / **12** of 12, because `algorithms/rlvr_reward.py:46` already does `s.rstrip("。.,，")`. The one real divergence is `The answer is 1.5. Next sentence.` — mine runs to end of line. **A captured string that looks obviously broken (`'3.5.'`) can score correctly; asserting the capture instead of the score reports a 100-point difference where the real one is 1 case in 12.** e1's version still wins, on a better reason: it keeps the terminator's meaning identical in both languages instead of leaning on a downstream `rstrip` that does not know it is covering for anyone | A verbatim-correct decimal answer scores 0.0 while still counting as answer-present. **Regression introduced 2026-09-03 in `8ab15148`; the sibling `eval/math_zh.py:35` terminates on `[。\n]` only and is correct.** Bounded by 23/500 = 4.6% decimal golds. `be.l1_fewshot_p324` predates it; any rerun today does not |
| 3 | `eval/ceval.py:58` vs `eval/run_eval.py:281` | items are tagged `"norm": "char"` and the module documents per-character scoring, but the only scorer sums token log-probs with no divisor, and **no file in the repo ever reads the `norm` key** | The declared metric is inverted into a shortest-option bias. Verifier found it wider: `run_eval.py:146` registers ceval with `cloze=False`, so the per-character path is unreachable from the runner at all. Known answer: 100.0% declared vs 0.0% observed |
| 4 | `eval/winogrande.py:14-15` | `prefix.strip()` deletes the separator and the option is built with no leading space | Every item is scored on `"...brown suitcase becausethe trophy is too large."`, and the first scored token flips to the no-leading-space form on both options |
| 5 | `eval/ppl.py:69` | computes the held-out split from the global `train.Cfg.val_frac` only; `train.py:2695-2699` honours a per-domain `val_frac` from the mix | For the five `data/mix_e1_*.json` mixes that set `val_frac: 0`, ppl scores rows the run **trained on** and reports them as held-out, contradicting its own docstring. The ladder mixes carry no per-domain key, so figures taken with them are unaffected. The arithmetic itself is correct |
| 6 | `eval/code_zh.py:43` | `_norm_lines` drops **every** blank line, not the trailing ones its docstring at :42 promises | stdout with leading or interior blank lines the oracle does not have is accepted: 500/500 where the stated contract requires 0/500 |
| 7 | `eval/gsm8k.py:55` | never reads `cfg.fone`, so `skip_special_tokens=True` deletes `[NUM]` (id 32772) before `:31` extracts a number | **Latent**, not active: a correct answer would score 0.00% silently on a FoNE checkpoint, but no `--fone` run appears in `runs/experiments.jsonl`. `run_eval.py:384` guards this; `gsm8k.py`'s own `__main__` does not |

Reproduce their known answer, with the divisor and the alignment pinned analytically: **`mmlu`, `math_hard`, `math_zh`, `fone`.** The MC likelihood scorer itself is correct — the divisor is exactly 1 (raw sum, crossing bisected to 1e-12) and option token k is scored at logit `pl-1+k`, verified on 2376 real ARC-Easy items where the count-derived known answer 0.2492 matched to 1e-9.

**One sub-claim was refuted by verification and is not in the table**: that `chid_probe` shows the same defect on a chance-level baseline. A uniform-logit model is deterministic, not chance — its ranking is entirely `-T·lnV`, so it always picks the fewest-token candidate. The unequal-length contract violation is real; the way it was demonstrated was not.

**A sibling implementation answers the question for its own scope, not yours.** Rolling back to it looks like the safest default and here it was not — I read `math_zh.py:35`, confirmed the character was absent, and did not ask why it did not need to be there. Sixteenth instance of the night's shape, and the first where the narrowed answer came from a correct piece of code rather than a tool.

**Seventeenth, from e1's own retraction, and it is about who gets verified rather than what:** in the same hour e1 ran the control for `check_ckpt_facts_sources_present` against a baseline, ran the `[protected]` positive control and caught it deleting zero rows — then reasoned about `reward_fn` instead of calling it. **Own artefacts got a measurement; a peer's got an inference.** The asymmetry is invisible from inside because both feel like diligence.

**Why this audit existed:** `eval/domain_bpb.py` truncated its input while dividing by the untruncated length and reported 5.460 where the true value is 8.000. One known-answer case found it. These 16 metrics had no such case. **Six of the seven defects are in the same family — a value computed over one population and divided, compared, or sliced against another.**

## A criterion a degenerate input also satisfies — three instances in one hour, 2026-09-08

`non-empty`, `not-all-identical`, `no error raised`: garbage satisfies each of them, so none of them can fail on the input they exist to catch. The fix is the same every time — assert the value, not a property that a broken value also has.

| where | the criterion that could not fail | what it became |
|---|---|---|
| `eval/code_l0prime.py` (b0) | `non-empty` — and under the double-strip 45 of 60 truncated fragments ARE non-empty, all 45 failing `ast.parse`. `freeze_hard` keeps the first execution failure, so every fragment qualified as a distractor **by that tool's own criterion, inside a world truncation had built** | round-trip byte-identical |
| pass@k degeneration guard (tilerl) | `at least one sample differs` | record the `distinct` count itself |
| a tileRL test's seed assertion (tilerl) | the assertion **copied production's seed formula**, so it changed whenever production changed and could never fail | read the seed the engine actually submitted; mutation (step by `rows` instead of `group`) now turns it red, and was green before |

**And the mirror of it, from the same hour:** `pod_drift` reports that two sides differ and never which side has the evidence. Those are different pieces of information, and only the second one tells you which way to fix. Two sessions each picked a direction — b0 aligned the pod to git, tilerl aligned git to the callers — and **both could say they had fixed it**. The criterion (all three call sites invoke it as `python3 runs/count_dir.py`, so the executable bit was never read) was not in the red, not in the hint, and not in the file. Resolved at `5bf7eb38`, pod `chmod 644`, stamp on, 836 files match.

So the fix to that report is three items, and the third is the one that matters: the hint must not name a command that cannot work (`pod_push.sh` only ADDs content); it must print `sha256 identical (…)` rather than the bare assertion `content matches`; and **it must say where the criterion is found** — for a mode drift, grep the call sites. The first two save one wasted command and one repeated investigation. **Only the third stops two people fixing the same drift in opposite directions.**

## Unowned, ready to pick up

**A check that `facts/<f>.json#<id>` references in prose resolve.** Today only `runs/tasks.jsonl` evidence is parsed (`_commit_delivers`); a reference in a doc dangles silently. e1 measured the real population: **493 references repo-wide, 2 genuinely dangling** — `runs/controller_board.md` → `cs.math_short_v8_cap_audit` (resolves once #98 lands) and `runs/tasks.jsonl` → `dq.t24`, whose closing row put a bare file path in `evidence`, so the id was never parsed at all.

**The value is not the 2. It is the path to them: 24 → 20 → 11 → 2, false positives at ten times the real defects.** The middle step is the instructive one — a two-segment regex truncated every three-segment id (`mlm.ratio.sub1b_optimum` → `mlm.ratio`) and then reported that the truncated prefix did not exist. All 20 carried a complete evidence shape: the reference really is in the file, the prefix really is absent.

Acceptance conditions, from e1 and not negotiable, because without them it reports 20 false rows on day one and gets turned off (a permanent red is the same as no signal): positive assertions for three- and four-segment ids; fixture ids whitelisted or excluded by directory (`_broken_*` worlds contain deliberately absent ids); one case for a reference split across a line break.

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
