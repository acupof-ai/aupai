# Controller board (fb) — 2026-09-08, 17:4xZ

**The night's one sentence: N1 is 20% through its 7,629 steps on four cards with val falling 2.577 -> 2.348 -> 2.256, and the two silent-push defects that killed its first launch are now fixed on main and on the pod.**

## Cards

| | |
|---|---|
| aupai | **2, 4, 5, 7** — granted by the user 2026-09-08 ("做呗"), machine fields set (`launch_block_granted=true`, `block_cards="2,4,5,7"`, `lane_card=""`), on the pod at stamp `5c2a091c` |
| tileRL | 0, 1, 3, 6 — 0 and 6 by the STANDING order of 2026-09-06 ("0,6 tileRL"), which today's four-card grant names no index for and therefore does not override |
| running | 0 (level-5 eval) and 3 (GSM8K steps_to_score) at ~100%; 1, 2, 4, 5, 6, 7 at 0 MiB as of the last read |

**No lane card, on purpose.** All four go to one serial chain, one arm at a time, so no small job may run beside it.

**Two grant defects, both mine, both caught by someone else before a launch:**

1. **The grant was written into `granted_by` prose and the machine fields were left untouched** — `launch_block_granted` stayed `false` and `block_cards` stayed `""`, which is what `_allocation_cards(block=True)` actually reads. The human half of the file said one thing and the machine half said the opposite, **inside a file whose own comment reads "A STALE GRANT IS WORSE THAN NO GRANT"** (3b caught it).
2. **The first corrected version took 4,5,6,7** and `allocation_reads_the_grant` refused it, correctly: card 6 is tileRL's by standing order, and a grant of "four cards" that names no index does not supersede an order that names two.

3b declined to route around the gate with `CUDA_VISIBLE_DEVICES`, on a better ground than the rule: **that fall-through means "a human specified these", not "the controller granted these"** — it would have run, and nothing would have recorded that it was ever authorised.

## Running now — N1, cards 2,4,5,7

| | |
|---|---|
| run | N1, the anneal-arms control, `runs/anneal_n1_0908.log` |
| launched | 2026-09-08 17:00Z, alive since |
| progress | step 1500 / 7629 (20%), 0.79B tok |
| val | 2.577 (500) -> 2.348 (1000) -> **2.256 (1500)** |
| train loss | 5.010 (70) -> 2.700 (500) -> 2.056 (1000) -> **1.955 (1500)** |
| throughput | 51-77K tok/s/gpu, MFU 21-32%, peak 49.5 GiB/card |
| ETA | 3.0h remaining; **~3.6h per arm, ~10.8h serial for N1+N2+R** |
| cfg proof | `sample_seed 42 (pinned)`, `retokenizing` appears **0 times** in the log |

**The ETA number to use is 3.6h/arm, not the 2h13m in the proposal.** That reference was
p200m_4b_0902 on eight cards; this is four cards and 7,629 steps. The difference is card
count, not efficiency.

**Monitor `bdy8b3rxc`** fires on val lines, every 500th step, arm transitions and failure
signatures. N2 starts when N1 ends (3b owns the launch), R last.

## Landed tonight

| PR | what | evidence |
|---|---|---|
| #112 (`df5ffd98`) | merge_main stamps the session marker into both commit messages it generates | reviewed and merged by 44; `merge_main.sh --selftest` green |
| #113 (`3b292330`) | `pod_drift`'s runs/ predicate split by extension; both `pod_push.sh` copies now call one `--ship-paths` | merged + `--all` in the same step; pod stamp `3b292330`, dirty=0; the 25 formerly-absent scripts verified present |
| #114 | 44's active-params gate | **BLOCKED by fb** — see below |

**The marker census, corrected.** My first count said 203 of 644 commits today lacked a
`(session)` marker. The regex was wrong: it required the subject to END with `(name)` and
so rejected `(b0-37)`, `(e1, PR #67)`, `(3b, §267)`, `(de; reviewed 44)`, `(#45)`. Recounted:
449 marked, 57 gh default merge titles, 99 git default merge messages, 36 friction drains,
**3 written by a person**. The population was machine-written, so the fix was the generators
(#112), not the history.

**The `runs/` push defect, measured.** `_pod_written` was `startswith("runs/")` plus a
one-file allowlist, so 44 of 45 tracked `runs/*.sh|*.py` were unshippable by
`pod_push.sh --all` — exit 0, zero `refusing`, stamp advanced, pod kept the old copy.
Against the live pod: 25 absent entirely, 20 byte-identical (each pushed by name), 0
differing. `runs/anneal_arms.sh` was one of the 20 and N1 died on the stale copy.
**A directory name answers where a file is, not who wrote it.** Second defect, same root:
`pod_push.sh:124` and `:538` each carried a `grep -v '^runs/'` without the `PUSHED_RUNS`
exception, so `--all` also dropped `runs/card_assignment.json` — latent only because it had
been pushed by name. This is the drift b0 predicted in that constant's own comment on
2026-09-03.

## #114 blocked — the merge deletes a flag

44's active-params gate is correct in its own terms; merging it removes `--sample_seed`
from `train.py`'s parser. That dict is the flag registry, not a help table — `train.py:2981`
runs `parser.add_argument(f"--{name}", ...)` over its `.items()`.

| | `"sample_seed":` in train.py |
|---|---|
| merge-base `100f9b6f` | 1 |
| branch `44-active-params-gate` | 0 |
| `origin/main` | 1 |
| `git merge-tree --write-tree` (tree `a0c12835`) | **0** |

So the merge takes the deletion; it is not merge-base noise. `984bce9a` (3b) added the flag.
`runs/anneal_arms.sh` passes `--sample_seed 42` on all three arms, so N2 and R would die at
`unrecognized arguments` — the `--rg_mod` shape of 2026-08-30. One line restores it.

## N1's first death — 14:49Z, and what it cost

**N1 launched at 14:49Z and was dead by 14:51Z.** `SignalException: Process 1203764 got signal: 15` — killed by `harness launch`'s 120 s startup gate while it was still doing legitimate work: `mix: tokenizing math_owm_stage2 (4,135,793 docs, workers=1)`.

**The cause was in the first screen of its own log, and I classified it as a side cost:**

```
mix: math_owm_stage2 cache was shuffled at sample_seed 42, now 1337: retokenizing
cache read: 158,471 MiB (154.76 GiB) over 9 cache(s)
```

`Cfg.sample_seed` is `None` (`train.py:317`) and `_sample_seed()` falls back to `Cfg.seed` (`:2069`), so `--seed 1337/1338/1337` gave the three arms **two different corpus orders**:

| arm | seed | sample_seed | corpus order |
|---|---|---|---|
| N1 | 1337 | 1337 | A |
| N2 | 1338 | **1338** | **B** |
| R | 1337 | 1337 | A |

**`|N1 − N2|` would have carried init variance PLUS corpus-order variance while `|R − N1|` carries only the reweight — the floor measured on a superset of what the comparison holds constant.** Overstated, so the error direction is a false negative: a real effect masked, written up as "no measurable effect at this budget", which reads exactly like a clean null.

**`_sample_seed`'s own docstring already carried this and its remedy**, from de-7: `Cfg.seed` also drives weight init, so binding the cache to it "would change their training data and fold data variance into `ds.seed_variance_0p2b`" — **pin `sample_seed` and a seed sweep shares one cache.**

**Fix (3b, in flight):** add `--sample_seed` to train.py's int-flag dict, leave `Cfg.sample_seed` defaulting to `None` — changing the default would move the corpus order of every run that does not pass the flag, including the p02_s* arms `ds.seed_variance_0p2b` rests on. Arms become `--seed 1337/1338/1337 --sample_seed 42`. **Nothing retokenizes, so the gate cannot fire, and all three arms read the 2026-09-05 cache other runs have already exercised rather than two freshly-cut ones nobody has read** — 3b's addition, and the better half of the argument: an unread cache is itself an untested variable in the experiment.

**My error, and it is the one to keep:** I reported "N1 is up" from cards at 383 MiB, a claim file, and a log that had just been written. **All three are equally consistent with "starting" and "died thirty seconds ago", and I checked none of them against liveness** — I did not `tail` to the end of the log I had already opened. Memory occupancy and a claim file answer "this job existed", never "it is alive now". The two probes that do answer it are the log's last line and `nvidia-smi --query-compute-apps`; 3b ran both, I ran neither.

**Verified after the death:** cards 2,4,5,7 at 0 MiB, `--query-compute-apps` shows only tileRL's two pids, claim released, and **the cache is intact** — `tokens_math_owm_stage2.pt.seed` still 42, `.pt` still the 2026-09-05 03:59 file; it died before writing its tmp.

**Ticket, not blocking:** the 120 s startup gate kills a job for doing legitimate work. It intends to catch a job that never claims a device; what it measures is whether one claimed within 120 s, and a first-time cache build cannot. Pinning the seed hides it tonight — **the next person building a cache for the first time hits it, and the symptom is `signal: 15`, which reads as "somebody killed me".**

## The v2 model, decomposed

15 agents, 7 dimensions each surveyed then adversarially verified. **The headline is that three of the four architecture components do not exist in the tree**, and one of them has an unresolved specification.

| package | size | today's failing acceptance |
|---|---|---|
| `csa-doc-cu` | large | CSA raises `NotImplementedError` on packed input (`model.py:277`) while `train.py:3843` passes exactly that `cu` |
| `hca-module` | medium | HCA appears **0 times in any .py on all 318 branches** |
| `partial-rope` | large, **scope unknown** | 473 .py searched; the only rope string is an error message at `model.py:1564` |
| `v2-cfg-surface` | small | 6 of the v2 knobs are unreachable from any launch line |

**`partial-rope`'s scope is unknown for a reason worth stating: the spec says "the last 64 dimensions" without saying 64 of what** — per query head, per compressed-KV latent, or per residual channel — and gives no theta. Three readings, three different position resolutions. **That is a design decision nobody has made, not implementation work.**

**`csa-doc-cu` is a design decision too, not a port:** our rows pack ~10 documents per 4096 tokens, so the residual tokens at each block boundary need a rule, and a rule that differs per branch brings back the causal-leak class — the previous version leaked 1.44 max|delta| and it was invisible in the loss.

**What must NOT be rebuilt:** MoE and the rest of the 1.5b-a0.2b-e48 stack trained to 26.74B tokens at val 1.824. It works end to end.

## Open decisions that are the user's

| question | consequence of deferring |
|---|---|
| **Tokenizer unfreeze** (non-hanzi slots 11,487 vs MiniCPM5's 103,883 = 9.04x; our code fertility 1.248x worse; ref fertility 1.4286 vs 1.0519) | a rebuild invalidates every checkpoint, so it must be decided BEFORE a v2 pretrain, not after. Deferring silently chooses "do not rebuild" |
| **`partial-rope`'s 64 dimensions of what** | the package cannot be scoped, let alone started |
| **Cards for a production v2 run** | the last 30B run used six; four cannot host it |

## Queue state, 2026-09-08 13:20Z

**15 open. CI is not the bottleneck and neither is dispatch -- six green PRs are deadlocked behind one unmerged branch that contains the rows unblocking them.**

de has already reviewed **#75, #83, #89, #95, #98, #100**. All six review rows sit in commits `a5d530fd` and `6912878b`, which are on `de-agents-clean` -- that is **#85, itself awaiting review**. On main those six PRs read as zero review rows, so `review_present` cannot see work that was actually done, and nobody merges. **A PR awaiting review holds the key to six others.**

Mechanism, and it is the general one: **`runs/review.jsonl` is a ledger and merges by union via `merge_main.sh` in seconds; riding a code branch makes a reviewer's latency equal to that code PR's review latency.** Today that was six PRs times several hours. de writes the row on a ledger-only branch and merges it immediately from now on; 44 is reviewing #85 to drain the six.

Red, one assertion, one fix: **#102 and #105 both fail `EVIDENCE stale: []; undeclared: ['score_matrix_rewrites_traced']`** -- the new check entered `CHECKS` without an entry in `EVIDENCE` (`harness.py:17100`), and #105 contains #102's `0ab846c2`. Chain: b0 adds the line -> #102 green; #96 lands -> #102 merges -> #105 merges.

**#96 is held by tilerl and the hold is correct.** `scripts/test_sft_holdout_gate.py:64` is still live while §279 describes it as fixed, in a PR with no code. An entry naming R12's sharpest instance, leaving that instance in the tree, lets a reader cite §279 as evidence the check is fixed -- which is what R12 condemns, inside the paragraph describing R12. Requirement: a reader cannot take §279 as evidence of a fix. b0 picks the landing.

Landed since the last board: **#81 `c0944b06`, #94 `a50823d4`, #104 `5bf7eb38`** (tilerl merged and pushed the pod in the same step; drift OK, 837 files match, stamp `a50823d4`).

Merge order still binds: **#81 -> #96 -> #85**, and **#98, #100 before #101**.
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

Nineteen instruments each answered a question narrower than the one asked, and none reported that it had. Measured, not asserted; every row is an incident from 2026-09-08.

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
| `pod_push --check`'s UNREGISTERED line (mine) | how many .py on the pod are not in the manifest | how many that line had room for. It ends in `...`; I read 56 and relayed 56 to tilerl. It is **179** (`runs/` 120, `_e1tmp/` 25, root 11, `_b0tmp/` 11, `scripts/` 6, other 5) |
| my "56 UNREGISTERED, which should join the manifest" | which of these need a manifest entry | none of them -- b0's inversion is the right one: manifest means "on main, and the pod must match", and these do not exist on main. **For a file that should not persist, seeing it drift is worthless; seeing it still there is what matters.** The missing thing is a check that the pod root holds no untracked `_*.py` |
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

## MiniCPM5-2B research — 2026-09-08, 3b + 44, fb reviewing

Two sessions, assigned on the user's order to research it thoroughly. **One finding changes a decision; the rest close doors, which is also worth having.**

**The decision-changing one (3b).** MiniCPM's own ablation (arXiv 2404.06395, Table 1) measures annealing with high-quality and SFT data mixed into the pretraining data against annealing on pretraining data alone: **+8 to 12 points**, with **B-2 as the negative control** -- doubling SFT tokens 6B to 12B moves 40.9 to 41.2, i.e. nothing, so the gain is the mixing and not the token count. **We copied the 10% anneal length (`train.py:406`, comment `(MiniCPM-style)`) and changed no data: all 25 `data/mix_*.json` have `anneal == weight` for every domain.** Our anneal lowers the learning rate over a distribution identical to the one before it.

Cost of the gap, from our own store: the anneal is worth **-6.89%** unweighted mean loss with all nine domains down, at **19x** per-token value against a constant-LR token. That multiplier is what the data change would act on.

Prerequisite before any proposal is executable, and 44 owns it: **do we hold instruction/SFT data we can mix in.** 21 SFT packs exist; **17 carry no holdout stamp, 4 stamp two superseded holdout sets, 0 stamp the current one.** And the contamination side decides whether the proposal is legal at all: **30% of math-500 questions already have a containment hit in the math SFT corpus** (`facts/contamination.json#cont.split`), so mixing that same data into the anneal makes every post-anneal math reading uninterpretable. Answer is three sentences: what we hold, whether it is usable, which readings die if we mix it.

**Transfer caveat, raised by tilerl-27 and adopted before any run.** The +8 to 12 points was measured on *their* mix, not ours; four separate readings failed that way in tileRL today, each a correct number carried onto a different population. **So the first run carries our own control arm, not "do what they did and see how much it moves".** Our baseline is `anneal == weight`, and that baseline is itself the thing under test: if our normal-phase mix is already cleaner than theirs, the headroom the mixing buys may already be spent. **The criterion is written before the run -- how many points count, how wide the noise band, how many seeds** (tileRL lost 171 minutes today to a curve whose criterion was written after).

**Tokenizer (3b).** Non-hanzi slots **11,487 (ours) against 103,883 (theirs) = 9.04x**, not the 4.0x the size ratio suggests; our code fertility is **1.248x worse**; ref fertility **1.4286 against 1.0519**; they carry FIM and tool-call tokens, we carry none. The unfreeze decision is the user's and is open.

**Architecture (44), first-hand from `config.json` and the safetensors index, not the card.** No loop, no weight sharing (42 independent layers, no aliases, `lm_head` separate from `embed`); dense 2.5B; full attention GQA 16/2, head_dim 128; RoPE theta=5M unscaled, 131K context; vocabulary 130,560. **Every row of the transfer column reads "not transferable", and the two strongest rows are strong for different reasons**: MiniCPM4's InfLLM v2 sparse attention (81% sparse) was **dropped in gen 5, and the README's stated reason is deployment compatibility -- no custom kernel, no fork -- not capability**; theta=5M has no published reason in any of five sources checked. **The kernel one is a cost datum we have never priced: a team able to build sparse attention, and that shipped it, gave it up to avoid depending on a custom kernel -- and our v2 is entirely custom kernels (KDA, MoE, CSA).** Recorded in the fact's `boundary`, PR #106. Verdict for v2 architecture: change nothing.

**Two rules adopted from tonight's peer disagreements, both about how a measurement is recorded rather than taken:**

**A mutation record must let a reader build the same mutant.** b0 and tilerl each ran an "M6" on #102, got different survival, and both believed they were discussing one measurement. tilerl's under-report mutant emptied `changed`, so `bool(got)` was also False and it could not separate exact-set from non-empty; the mutant that separates them omits some leaves and keeps others (recurse dicts but not lists). b0's entry recorded the conclusion, not the construction, so a reproducer necessarily built a different shape. **"Four mutants, all killed" is a count; what carries information is which case kills which** -- the same ruling as this morning's on kill-set equality, arrived at from the other side. b0 is pinning all four by name.

**A conflict between two peers' first-hand readings is the controller's to resolve, not to forward.** 3b's anneal proposal raises `cot` 22.7x on the ground that it is instruction-shaped; e1 read the bytes the same evening and it is `f"{problem}\n\n{solution}"` plain text, by the generator's own comment. The proposal may still be right, but its reason has to change from "instruction data" to "high-quality reasoning text" -- and with it the expectation, since MiniCPM's +8-12 came from mixing SFT *format*. Second conflict, and it gates the two 26x rows: the proposal has no contamination column, while `facts/contamination.json#cont.split` records 30% of math-500 questions with a containment hit in the math SFT corpus. **Raising a domain 26x inside the 19x-per-token anneal window, without knowing its containment rate, manufactures an uninterpretable reading rather than inheriting one.** Requirement sent: a containment column, measured, for all four raised domains before any of them moves. The quantity is measurable in the existing pipeline -- e1's cot run produced `eval_contaminated=36` from it.
