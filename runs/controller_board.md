# Controller board (fb) — 2026-09-09, 01:2xZ

**The night's one sentence: the noise floor is measured — N1 1.823, N2 1.871, so `F = |N1 - N2| = 0.048` on the epoch-end val, and every same-step gap quoted earlier tonight (0.067-0.088) was measured on a 20-batch estimator, not the 100-batch one the criterion reads.**

## The number this round exists to produce

| arm | seed | final val (epoch-end, 100 batches) | steps | tokens |
|---|---|---|---|---|
| N1 | 1337 | **1.823** | 7,629 | 4.00B |
| N2 | 1338 | **1.871** | 7,629 | 4.00B |
| R | 1337 | not launched | — | — |

**F = 0.048.** Everything else about the two arms is identical: same mix
(`mix_200m_4b_annealN.json`), same `--sample_seed 42` so one corpus order, same recipe. The pair
isolates weight init and dropout.

**The correction that matters, and it lands against my own reporting.** I quoted the same-step
gaps all night — 0.088 at step 500, 0.071 at 7000, 0.072 at 7500, "mean 0.076, no trend" — and
treated them as the floor's scale. They are not the same quantity. `train.py:408-409`:

```
val_batches = 20
val_batches_full = 100  # fixed prefix, so the epoch-end number is comparable across runs
```

The periodic `step N val` line is a **20-batch** estimate (fifteen reads, min 0.067, max 0.088, mean 0.076; series committed as `runs/anneal_null_val_series_0908.tsv`). The `ep 1/1 ... val` line is a
**100-batch** one, and train.py's own comment says which of the two is comparable across runs.
Five times the data, so roughly half the sampling noise — which is the whole of the drop from
0.072 at step 7500 to 0.048 at the end. **Had I read the floor off the periodic series, I would
have published a floor ~50% too large and buried any true effect between 0.048 and 0.076.**
The criterion in `runs/anneal_arms.sh` said "final val" and was right for a reason nobody had
stated: it names the estimator, not just the time.

**What the floor means for R.** N1's entire anneal tail moved val 1.854 -> 1.823 = **0.031**,
which is *below* the 0.048 floor. So a reweight of the anneal phase whose effect is the same
order as the phase's own contribution cannot be read at this budget by construction. If
`|R - N1| <= 0.048`, that is the answer — a bound, reported as a bound — and the reweight does
not enter the 30B mix on this evidence.

**What two points cannot say.** F is a range over two draws, not a standard deviation. Any sigma,
p-value, or confidence interval quoted off this pair is fabricated. Written into the prereg row's
`will_not_claim` before R has a number.

## The floor is per metric, and on three metrics it is a sign flip

Both null arms are now scored (`runs/score_matrix.jsonl`, N1 and N2, 10 metrics each). The pair
differs only in `Cfg.seed`, so **every difference below is init noise with no effect in it.**

| metric | N1 | N2 | N2 - N1 | basis |
|---|---|---|---|---|
| final val (nats/tok) | 1.8230 | 1.8710 | **+0.0480** | 100 val batches |
| domain_loss unweighted mean | 2.0241 | 1.9956 | **-0.0285** | 9 domains |
| domain_bpb unweighted mean | 0.77209 | 0.76323 | **-0.00886** | 9 domains |
| mc_ceval Average | 23.1 | 27.7 | **+4.60** | C-Eval |
| minimal_pairs overall | 0.801444 | 0.801444 | **0.00000** | 277 pairs |
| minimal_pairs factual | 1.0000 | 0.8125 | **-0.1875** | n=16 |
| minimal_pairs function_word | 1.0000 | 0.8500 | **-0.1500** | n=20 |
| minimal_pairs numeric | 0.5800 | 0.6400 | +0.0600 | n=100 |
| minimal_pairs mean_margin | 2.5819 | 2.4643 | -0.1176 | 277 pairs |
| math_v2_like overall | 0.97145 | 0.97377 | +0.00232 | n=3012 |
| math_v2_like perfect_square | 1.0000 | 0.9000 | -0.1000 | n=10 |
| humaneval gold bpb (byte-wtd) | 0.49047 | 0.49199 | +0.00152 | 164 tasks |
| lambada_en acc | 0.24743 | 0.24704 | -0.00039 | 5,153 |
| lambada_zh open_acc5 | 0.4950 | 0.4710 | -0.0240 | 1,000 |
| l1_fewshot correct | 20 | 18 | -2 | 3 demos |

**Three readings of held-out likelihood, and they do not agree on which arm is better.** Final val
says N1 by 0.048. `domain_loss` unweighted mean says **N2** by 0.0285. `domain_bpb` unweighted mean
says **N2** by 0.0089. Same two checkpoints, same nine domains, opposite rankings. So the floor is
not a scalar to clear — on these aggregates the sign itself is not stable between two seeds, and
**no single-metric reading of R can rank it against N1.**

**The domain aggregate is 93% two domains.** Per-domain init noise spans a factor of 1,400:
`code_py_rp1t` 0.0001 and `code_py_starcoder` 0.0020 at one end, `chatml` **0.1394** and `chat_qa`
**0.0988** at the other. Those two are the smallest slices in the mix -- `mix_200m_4b_annealN.json`'s
`pool_rows_estimated`, the field both null arms read, is **9,043** for chatml and **8,854** for
chat_qa against 97,722 for code_py_rp1t and 2,139,719 for code_py_starcoder, 11x and 237x larger --
so their held-out splits are the smallest. Of the aggregate's 0.0285 movement,
(0.1394 + 0.0988) / 9 = 0.0265 is those two — **93%**. An arm compared on the unweighted mean is
being compared on chatml and chat_qa with seven domains along for the ride.

**`minimal_pairs.overall` is identical to sixteen digits while all five of its dimensions moved.**
Both arms scored exactly 222 of 277. N1: 39 + 16 + 20 + 58 + 89. N2: 38 + 13 + 17 + 64 + 90.
Same total, different 222. The aggregate cannot fail on a difference it does not represent, and
here it reported perfect agreement between two arms that disagree on 5 of 5 partitions.

**mc_ceval's floor is 4.6 points.** Every C-Eval comparison at this scale that quoted a gap under
4.6 points was inside init noise. This is the largest single number in the table and the one most
likely to have been read as a result before tonight.

**The small-n dimensions are unusable and should be reported as counts.** `factual` is 16 items
(16/16 vs 13/16), `function_word` 20 (20/20 vs 17/20), `perfect_square_pattern` 10 (10/10 vs 9/10).
A 3-item and a 1-item swing print as 18.75 and 10.00 percentage points. Nothing is wrong with the
measurement; the percentage is the wrong presentation for n=10.

## api_cloze scores every checkpoint against another program's row bounds

Both arms' `api_cloze.bounds` are byte-identical and name a run neither arm is:

```
mix: mix_200m_8b.json   seed: 42   world: 2   row_cursor: 80380 (as of step 3815)
```

The anneal arms ran `mix_200m_4b_annealN.json`, seed **1337 / 1338**, world **4**, 7,629 steps.
The bounds are the memory-layers program's (`prereg memory_layers_0905`, e1's 80,280-row
`data/probes/api_cloze.jsonl`), and the metric's own `gap_note` says so. Identical bounds across
two runs with different seeds confirms the split is a fixed reference, not derived from the
checkpoint being scored.

**So the "seen" region is rows these checkpoints never saw.** `within_region_gap` came out 0.0008
on N1 and exactly 0.0000 on N2 — the right answer for a partition with no meaning here, and the
reason nobody noticed. The bounds ARE stamped, which is what let this be found at all; what is
missing is a refusal when the stamped bounds do not describe the checkpoint being scored. Same
family as `vocab_id` and `.srcfp`: the fingerprint exists, nothing checks it at the read.

Not a claim that api_cloze is broken — inside the memory program it is measuring what it says.
It is a claim that the default score-matrix profile runs it on checkpoints where its partition is
arbitrary, and reports a number rather than a SKIP.

## Pre-registration — written before R, honest about N1 and N2

`runs/prereg.jsonl#anneal_reweight_noise_floor_0908`, registered 2026-09-09T01:15Z by fb.

**It does not claim to pre-register N1 and N2.** Both had finished when it was written; the row
says so in `registered_before`. What genuinely predates every arm is the *reading criterion*,
committed verbatim in `runs/anneal_arms.sh` at **`1e91d8da`, 2026-09-08T14:10Z — 39 minutes
before N1's first launch attempt at 14:49Z**:

> READ N1 vs N2 BEFORE LOOKING AT R. |N1 - N2| is the noise floor; if |R - N1| falls inside it,
> the reweight had no measurable effect at this budget, which is a result and not a failed run.

The row moves that criterion into the ledger where `prereg_citations_current` can see it, and
fixes R's decision rule while R's number does not yet exist. The three arms had been running with
no prereg row at all — the criterion was real and dated, but it lived only in a shell script's
header comment, where no check reads it.

## Cards

| | |
|---|---|
| aupai | **2, 4, 5, 7** — granted by the user 2026-09-08, machine fields set (`launch_block_granted=true`, `block_cards="2,4,5,7"`, `lane_card=""`) |
| tileRL | 0, 1, 3, 6 — 0 and 6 by the STANDING order of 2026-09-06 |
| now | 0 at 32.8 GiB / 100% (tileRL level-5 eval, claim `tilerl-l5eval.0.json`); **2 running N2's score matrix**, claim taken; 1, 3, 4, 5, 6, 7 at 0 MiB |

**N2 released its cards cleanly.** All four went to 0 MiB and `runs/claims/anneal_n2_0908.2-4-5-7.json`
is gone — no orphan, no reparented grandchild holding memory.

## Running now — arm R, cards 2,4,5,7

| | |
|---|---|
| run | R, the anneal reweight, `runs/anneal_r_0909.log`, exp row `anneal_r_0909` |
| launched | 2026-09-09 01:48Z by fb |
| cfg verified | `mix data/mix_200m_4b_annealR.json seed 1337 sample_seed 42 (pinned) anneal_frac 0.1`, batch 16 accum 2, world 4 — N1's recipe with the reweighted mix and nothing else |
| progress | step 1290 / 7629, 17%, 77K tok/s/gpu, s/step 1.707, ETA ~05:2xZ |
| val so far | 2.578 (500), 2.347 (1000) against N1's 2.577 and 2.348 |

**Two reads, both within 0.001 of N1.** R shares seed 1337 with N1, so `|R - N1|` is the mix effect
with init held fixed, and it is an order of magnitude below the 0.048 init floor. This is the
20-batch estimator and not the read point — see §286 for why that distinction is the whole game —
but the trajectory is tracking N1 far more tightly than N2 does.

**The chained scoring will fail on R too, and that is not a surprise to absorb quietly.** A
four-card grant with no lane card guarantees every arm's own scoring step deadlocks 30 minutes and
exits nonzero. N1 and N2 both did. The checkpoint is unaffected; the scoring is done by hand in the
gap. Two fixes exist and neither is chosen: score in the gap by hand (~10 min of three idle cards
per arm), or let the chain use one of the four cards it just released — the second is correct and
needs a one-line change in `run_ddp.sh`, which is frozen. Logged rather than worked around silently.

## Next gate — R

R's epoch-end val at ~05:2xZ, read **per metric** against the floor table above, never against a
single aggregate — §285 is the reason. `|R - N1| <= 0.048` on val is a bound and a result, not a
failed run; the pre-registered rule is `runs/prereg.jsonl#anneal_reweight_noise_floor_0908`.
Score by hand on a freed card after the chained pass exits nonzero.

## Queue — 8 open PRs, one mergeable, and the reviewer step is what holds

Read at 02:5xZ by checking each PR's reviews AND comments for a qualifying `artifact:` / `case:`
body, never by counting comments containing the token.

| PR | branch | qualifying review | state |
|---|---|---|---|
| #100 | fact-repro-table (98) | **yes** — de: "Approved. artifact: facts/corpus_supply.json#cs.reproducibility_table_0908" | **mergeable; unmerged 7h+** |
| #23 | tilerl-cache-sidecar | yes, but a **changes-requested** body from 3b | correctly blocked |
| #109 #106 #105 #103 #102 #92 | e1 / 44 / b0 / 3b / b0 / 98 | none | waiting on reviewers |

**#117 landed.** 44 merged it at `96c9b6f4` and pushed the pod in the same step, which is the
09-07 ruling working exactly as written.

**Why #100 is not merged by me.** Merging an approved CI-green PR as a third party satisfies the
rule's purpose — the author does not merge their own work, and whoever merges pushes the pod — and
fails its letter. The cost is not this PR: it would establish that the reviewer step is skippable
whenever a reviewer sleeps, and that step is what makes approval mean anything. The clean route
needs no exception, because nothing says a PR has one reviewer: a second roster reviewer who reads
the artifact and writes their own row may merge it. Proposed to 44 as their call. If #100 is still
open at 8h+, the choice goes to the user as a process question — who may merge is the user's, not
the controller's.

**The trap this table exists to avoid is one I fell into two ticks ago.** I told tilerl #23 was
approved and ready. It was changes-requested by 3b 14 hours earlier. My proxy counted bodies
containing `artifact:` — and a changes-requested comment carries that token too, because a good
rejection names the artifact it read. The token says a reader opened something; only the state
says what they concluded.

## Landed this tick — three defects, all found by reading rather than by a check

**de-85 (`fc6fd165`, amended `74fb76e7`) — the shared-file claim is broken in two dimensions.**
Measured, not inferred. VISIBILITY: fb held AGENTS.md in `../aupai-fb`, and
`claim-file acquire --path AGENTS.md --owner testprobe` from the integration tree returned
`claimed AGENTS.md for testprobe` rc=0 with no warning; each tree's `claim-file list` showed only
its own. Probe released immediately. LIFETIME: `merge_main.sh fb` printed
`released 1 claim(s): AGENTS.md` while PR #117 — the branch that actually edits AGENTS.md — was
open and unmerged, so the file sat unclaimed with an outstanding edit. Cause of both is two lines:
`file_claim.py:32-33` builds `CLAIM_DIR` from the tree the script lives in, and `.gitignore:48`
ignores `runs/claims/`. **The rule this implements exists to stop three between-session collisions
and cannot stop any of them** — the hook enforces that the author declared it in their own tree,
which is a record, not mutual exclusion. 44 ruled it one task, two dimensions: fixing visibility
alone leaves merge_main releasing early, fixing lifetime alone leaves two sessions blind.

**de-86 (`33c379da`) — a closed row's status is unreachable by every supported writer.**
`exp.py done` refuses a closed row, `note` refuses a closed row, `amend` takes only
`--reading_artifact` / `--finding` / `--decision`. Each refusal is correct alone; together they
leave no path. Observable consequence: the two null arms of one experiment carry different statuses
for the same event, and N2 cannot be fixed. 44's design constraint is in `--reading` as a
constraint on the fix — the ledger unions across branches and folds last-row-wins, so the
correction must be an **appended** status-correction event, never a rewrite; a rewrite grows a
duplicate id, which is the 2026-08-31 t39/t40 failure. The negative case is the one that keeps it
honest: the same command must still refuse a close that is merely being re-run, or "correct a wrong
status" becomes "overwrite any close".

**The pod/local contradiction is ruled (`daded540`).** `pod_push` reported "1 row(s) where both
sides state a different non-empty value" without naming it; `python3 scripts/pod_pull_ledgers.py`
with no flags names it and prints the differing fields — pod_push reports the count, pod_pull_ledgers
reports the row. It is `anneal_n1_0908 @ 2026-09-08 16:59`. Pod side is run_ddp.sh's chained close,
true about the COMMAND and silent about the run; local side carries the same 1.823 with its basis
and its reading. Ruled to local. Same shape as the `b0_p5_ctrl_bf16` ruling.

## §284 §285 §286 landed — PR #117, merged 96c9b6f4

Written by fb from 44's candidates, reviewed by 44 twice. 44's two findings on the first round were
both real and the second was worse than they read it: **chatml 7,974 / chat_qa 7,838 were the R
ARM's row counts**, read from `runs/anneal_r_0909.log` under the *reweighted* mix — a claim about
the null pair sized from the arm the null pair exists to be compared against. Corrected everywhere
to `mix_200m_4b_annealN.json`'s `pool_rows_estimated`, 9,043 and 8,854 against 97,722 and
2,139,719. The other finding was R10's own shape: §286 cited two logs that exist only on the pod,
now extracted and committed as `runs/anneal_null_val_series_0908.tsv`. Recount from that file:
fifteen reads, min 0.067, max 0.088, mean 0.0757.

## The peers are all asleep

`peer_stalled`: 6 members with an open task and nothing in the repo for 2h+ — 3b 544m, 44 284m,
b0 554m, de 285m, e1 651m, fb 184m. `owner_queue_depth`: tilerl idle with no open unblocked task.
It is 09:1x local. Nothing is being dispatched into that; the queue above is the whole ask, and
#100 is the one item where a single click by de unblocks another session's landed work.

## Harness — 0 FAIL, 14 WARN

`no_ghost_close, ckpt_facts_sources_present, pod_stamp_is_main, pod_ledger_rows_home,
keep_claim_reasons_live, owner_queue_depth, peer_stalled, one_deliverable_per_owner,
review_present, entrypoints_ran, prereg_citations_current, score_matrix_present,
selftest_counts_computed, tasks_stale`. 83 repo checks, 27 pod checks. CI green on main and on
all 8 PR heads.

`one_deliverable_per_owner` names the real shape of the stall: b0 holds 9 open tasks, de holds 9,
e1 holds 3, 3b holds 2. Nine open tasks is not a queue, it is a list nobody is working from.

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
