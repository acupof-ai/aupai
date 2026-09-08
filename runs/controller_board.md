# Controller board (fb) — 2026-09-08

Cards: all eight are tileRL's, granted by the user directly. aupai runs no GPU job. The cleanup is the whole programme until it is signed off.

## User orders in force

| order | state |
|---|---|
| Clean the whole repository, not one redundant word | seven tracks running, below |
| Any file readable by someone who has never seen the repo, self-contained, no unexplained shorthand | added to every track, no separate renaming round |
| Worktrees cut to the live working set | 32 → 15; 8 keep, the rest are their owners' to remove |
| At most two open tasks per person | b0 11 → 2 done; de 6, tilerl 9, e1 3 still over |
| Next: corpus reproducible byte-for-byte from zero | 98 leads, e1 supplies the fingerprint facts |
| Next: distillation pipeline, teacher Qwen3.8-27B | 44 designs, tileRL supplies the serving numbers |
| Never delete without a named target | every removal names its files and runs each first |

## Seven tracks

| track | owner | state | next gate |
|---|---|---|---|
| scripts and entry points | b0 | 14 candidates, 2 removable; 12 kept, 8 of them reached by edges the tool cannot see | teach reachability three edge types, and print a purpose line per file with a NO PURPOSE LINE count as the readability debt |
| AGENTS.md | de | PR #85 merged, 74,280 → 72,392 chars, rule set byte-identical | the shared-config guard first, then readability |
| docs | 44 | PR #83 landed three markers; 86 documents, zero duplicate questions | distillation design takes priority |
| facts | e1 | 489 facts, `retracted_value` gate covered under half its population, four self-negating boundaries restated | claims on bare numeric values; the `filters_fp` tier label |
| ledgers | 3b | 43 scripts inventoried, none dead; five hook-executed files my criterion would have deleted | ledger identity: five of thirteen union-merged ledgers have no key present in every row |
| eval, filters, probes | fb | 71 files, 31 selftests all pass; the divisor defect is isolated to `domain_bpb` | 16 metrics have no known-answer case |
| datagen and mathbank | 98 | PR #84 merged; the frozen math corpus was built by 4 modules against a 28-module registry | corpus reproducibility, below |

## Corpus reproducibility — the next main line

Goal, stated so it can be judged: a corpus build is a pure function of source bytes, pipeline version and seed, and a re-run produces byte-identical output.

| finding | consequence |
|---|---|
| `filters_fp` hashes exactly three files: `filters/pass{1,2,3}_garbage.py` | 15 of 50 domains can say the garbage filters were identical. **Zero of 50 are demonstrated byte-reproducible from the stamp.** The tier label "reproducible" is what made me issue a wrong instruction. The count was 14/49 here until 2026-09-08: the fact's `value` predated `rp1t_arxiv_papers`, and e1 recorded the drift as `config.count_drifted` rather than rewriting `value` |
| 2,010 shard files have link count above one | Domains are not disjoint. Disk holds 248.93 GB while per-domain sizes sum to 348.30 GB. A per-domain rebuild double-counts, and rebuilding drops the hardlinks |
| One frozen batch excludes inputs that no longer exist | A build whose inputs are gone is unreproducible by definition — a fourth answer in the table, not a special case of "no" |
| A frozen batch has 40,000 rows against a program cap of 1,200 | **The recorded command is not the command that ran.** This outranks reproducibility: it makes every recorded command unreliable as evidence |

## Distillation — the other next main line

The blocking constraint, to be verified rather than accepted: our vocabulary is 32,773 entries and the teacher's is much larger, so there is no token-level probability alignment and KL on logits is not a tuning problem. Three routes follow, and the deliverable is which one and why: sequence-level distillation (teacher generates text, we train on it, vocabulary-independent, available today); rebuilding our vocabulary to match the teacher (voids every existing checkpoint, and the vocabulary is frozen under three unfreeze conditions); cross-tokenizer logit distillation over aligned spans (research, not engineering).

In the first route the seed prompt set is the whole design — what the teacher generates is exactly what the student can learn. Teacher throughput must be measured, not estimated: whether the route is viable is decided by how many samples can be generated, not by the algorithm.

## Closed and not reopening

| item | state |
|---|---|
| The 30B leg | closed at step 34,000 of 38,146 by user ruling, recorded as an incomplete schedule and not claimed as an advance |
| What annealing was worth | −6.89% on the unweighted mean, same run and same held-out rows, all nine domains down. Per token it is 19× a constant-rate token |
| `domain_bpb` divisor | real, about 2×, fix merged. Known-answer test: true 8.000, reported 5.460 |
| `answer_present` at three demos | retired as a primary readout: 0.1147–0.5433 within one recipe, standard deviation 9.2× the binomial floor |
| Held-out row drift | latent, no observed instance. All nine caches stamped before both scorings |
| SFT packs | 21 packs, 17 with no holdout stamp, 4 stamped against two different superseded holdout sets, zero current. Stale is a reporting defect, not a training hazard — both cases refuse today |

## Corrections I owe, made 2026-09-08

| what I said | what is true | who caught it |
|---|---|---|
| ".venv is 94% of what a scan walks" — stated about the repository | It is a property of the integration tree only. `/Users/bytedance/code/aupai` holds 13,665 `.py`/`.sh` of which 537 are the repo's own; every session worktree has no `.venv` and is 537 files. The same scanner meets two worlds 24x apart with no code change and no report of which one it walked | b0 |
| "the shared-config guard window is the 55 s hook" | The window is `pre-commit:2430` to `:2444` and wraps one selftest subprocess. `harness check`'s 55.48 s is entirely outside it. Zero selftests is zero window, which is why e1's five-file facts commit landed in 42 s with no refusal | b0's measurement, my misreading of it |
| "the facts commit went through the `_ledger_only` exemption" | No facts file is in `.gitattributes`, so `_ledger_only` was False. It went through the earlier condition: `_clash = _main_touched_staged(...)` was empty. **Being behind main does not matter; whether main touched the files you stage does.** That is checkable with one command, `git log HEAD..main -- <file>`, and nobody runs it because `git status` shows the other number for free | e1 |

## Open, unowned

**What in `harness check` has anything to say about a union ledger row.** Every commit pays 55.48 s of it, including one that appends a single line to `runs/*.jsonl`. The selftest half is already proven to have nothing to say — the exemption path runs zero of them and the row is still correct. The check half has never been examined. Method: for a commit staging only one `runs/*.jsonl` line, which of the 71 checks have an input set that intersects that line. My expectation is single digits, and an expectation is not a measurement. Unassigned on purpose — nobody is under the WIP cap tonight.

## Open user decisions

1. Corpus composition for the next full run.
2. Pod disk at 95%.
