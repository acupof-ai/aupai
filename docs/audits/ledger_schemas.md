---
question: For each ledger under runs/, what is every field, which values are legal, and what is the row's primary key?
status: measured
source: read of the 14 multi-commit runs/*.jsonl on main, 2026-09-08; population from git, field presence and value domains computed per file, not transcribed
---

# The ledgers under `runs/`

Written for someone who has not seen this repository. Each ledger is one JSON object per line, appended never rewritten. A reader opening one row should be able to tell which fields a machine reads, which are prose for a person, and what makes the row a distinct row.

**No field is renamed by this document.** A rename breaks every reader; the readability is carried by the description. Where two names mean the same thing, both are listed and the one to write from now on is marked.

## Which files this covers, and by what criterion

`runs/` holds **32 tracked top-level `.jsonl`** (`git ls-tree --name-only HEAD runs/`; a `git ls-files 'runs/*.jsonl'` pathspec answers 54 because git's `*` crosses directories and a shell glob does not — the extra 22 are under `runs/audit_0904/`).

The criterion is **appended by more than one commit**, taken from git rather than from a remembered list: **14 files**. The other 18 are single-commit dumps written by one run and never touched again — `b0_final_he_EC.preds.jsonl`, `self_repeat_dump.jsonl` and the like. They are per-experiment data, not coordination ledgers, and a merge cannot damage a file nobody appends to twice.

Two criteria that look equivalent and are not. **`.gitattributes` has a merge driver for 14 paths, and it is a different 14**: `artifact_refs.jsonl` and `policy_metrics.jsonl` are appended by several commits and have no driver at all, so the opening claim "merged by union" was false for 2 of the files this document covers. `novel_ops_4way.jsonl`, `memory_diag.jsonl` and `moe_diag.jsonl` have `merge=union` while only the last is multi-commit. The git-derived criterion is the one used below because it names what actually happens to the file.

## Why the primary key matters more than it looks

Union merge keeps every line either side has and drops exact duplicates. It has no idea what a row *is*. So for a ledger with no stable key, two branches can each append a row describing the same event with one field different, and the merge keeps both, silently, with nothing reporting it.

**Two of the fourteen have no field present in every row: `friction.jsonl` and `milestones.jsonl`.** Under the weaker reading — some field *of the key* is missing somewhere — it is four: those two plus `tasks.jsonl` and `review.jsonl`. Both numbers are below; they answer different questions and only the first is a statement about the file as a whole.

The key is also what an assertion can be written against. `runs/experiments.jsonl` is the case that proved it: `exp.py` refuses to close an already-closed run, but a person can append the row by hand and commit it, and the refusal never runs. A predicate over the *file* catches that; a refusal in the writer cannot.

## The table

| ledger | rows | primary key | key holds today? |
|---|---|---|---|
| `experiments.jsonl` | 462 | `(name, started)` | present in all; 313 keys, 93 repeat, **149 extra rows** — by design, see below |
| `tasks.jsonl` | 632 | `(id, state)` | both in all; 53 pairs repeat, **63 extra rows** |
| `msg_log.jsonl` | 808 | `(ts, from)` | present in all; 74 repeat |
| `moe_diag.jsonl` | 472 | `(name, step, ts)` | all 10 fields in every row; unique |
| `review.jsonl` | 291 | `(ts, reviewer)` | **`ts` missing in 20 rows**; 26 keys repeat |
| `friction.jsonl` | 283 | `(when, who)` | **`when` missing in 8, `who` in 3** |
| `board.jsonl` | 104 | `(ts, who, topic)` | present in all; 15 repeat |
| `score_matrix.jsonl` | 90 | `(ckpt, profile)` | unique |
| `artifact_refs.jsonl` | 35 | `path` | unique |
| `milestones.jsonl` | 29 | `(ckpt, measured)` | **`ckpt` missing in 2, `measured` in 7** |
| `ledger_resolutions.jsonl` | 27 | `(ledger, key)` | present in all; 2 repeat |
| `prereg.jsonl` | 8 | `id` | unique |
| `retro.jsonl` | 8 | `(owner, date)` | present in all; 2 repeat |
| `policy_metrics.jsonl` | 4 | `date` | present in all; 1 repeats |

### `experiments.jsonl` — one run, one or more events

`(name, started)` identifies a **run**; a run has several rows, one per event, and the reader folds them. That is why 93 of 313 keys repeat and 149 rows are extra: a `running` row and its terminal row share the key. Folding is terminal-wins, and a human's close beats the monitor's.

| field | who reads it | legal values |
|---|---|---|
| `name` | machine | free text, the run's name |
| `started` | machine | `YYYY-MM-DD HH:MM`, **UTC** (`exp.py:139` uses `time.gmtime()`) |
| `status` | machine | see below — **21 values in the file** |
| `cmd` | machine | the command; **empty means the row was fabricated or hand-written**, see below |
| `hypothesis` | person | written before the run starts |
| `result` | person | the number |
| `finding` | person | what the number means, not the number |
| `decision` | person | what changes because of it |
| `commit` | machine | HEAD's sha when the row was written; `git_commit()` contracts `Never ""` |
| `writer` | machine | absent, or `monitor` when the monitor closed it |
| `class` | machine | `confirmatory` `incremental` `infra-verification` |
| `ended` `notes` `cards` `reading_artifact` | mixed | optional |
| `retracted_at` `retracted_reason` `retracted_result` `superseded_by` | person | present only on `retracted` |
| `reclassifies` `reclassify_reason` | machine | present only when a human overrode a monitor close |

**`status` has no enforced domain and 21 distinct values are in the file.** In frequency order: `fail` 127, `ok` 125, `running` 90, `killed` 63, `retracted` 10, `stopped` 7, `error` 7, `failed` 5, `done` 5, `flat` 4, `dropped` 4, `stop` 4, `rejected` 2, `null` 2, and one each of `skip` `probe` `cancelled` `provisional` `null_provisional` `refused` `partial`. Three synonym pairs are visible in that list — `stop`/`stopped`, `fail`/`failed`, and `ok`/`done` — and no reader collapses them.

**Empty `cmd` on a key with no `running` event is the defect signature.** 31 such keys exist. They come from `exp.py`'s fabrication branch — closing a run it cannot find makes a row with `started=now()`, `cmd=''`, `hypothesis=''` — plus, in at least one case, a row appended to the file by hand. Both produce the same shape, and one predicate over the file catches both.

### `friction.jsonl` — 39 field names, 12 used once

The sprawl is real and this is where the spec earns its keep. The eight fields that are actually the schema:

| field | in | write this one |
|---|---|---|
| `kind` | 282/283 | yes — domain is `FRICTION_KINDS`, `harness.py:11255`, see below |
| `who` | 280/283 | yes |
| `when` | 275/283 | yes — prefer over `ts` (8 rows) |
| `cause` | 274/283 | yes |
| `blocked_what` | 270/283 | yes — prefer over `what` (5) and `blocked` (2) |
| `sha` | 267/283 | yes |
| `minutes_lost` | 266/283 | yes — prefer over `cost_min` (8) and `cost` (4) |
| `fix_applied` | 262/283 | yes — prefer over `fix` (11) and `fix_idea` (7) |

`kind`'s legal values are the `FRICTION_KINDS` tuple, and the file uses 17 of them: `override` 108, `merge` 54, `check` 27, `defect` 19, `gate` 15, `hook` 13, `pod` 9, `launch` 8, `near_miss` 8, `process_failure` 7, `resolution` 6, `correctness` 2, `attribution-correction` 2, `coordination` 2, `dependency` 1, `blocked` 1, and one row with no `kind`. `misroute` is in the tuple and unused.

**The three synonym pairs are the finding, and `friction_minutes_required` shows both halves of the cost.** The check requires `minutes_lost` on `near_miss`/`process_failure`/`hook` rows and reports a baseline of 6 offenders. Reading those 6:

- **Three of them did report a cost** — lines 42, 43 and 44 carry it in `cost` (`"~4 min, zero artifacts"` and two longer ones). The check reads only `minutes_lost`, so half its baseline is a naming mismatch rather than a missing measurement.
- **Three of them print `?` where the description should be** — lines 26, 72 and 167 are `hook` rows carrying `blocked_what`, and the refusal message is built from `r.get("what", "?")`. The same predicate has one synonym that makes a row invisible and another that makes it unreadable.

So the coverage of every check over this ledger is an unknown quantity until its field names are checked against the file. For this one the number is exact: **baseline 6, of which 3 are false positives.**

### `tasks.jsonl`

`id` `owner` `state` `task` `why` `opened` are in every row. **`when` is in 7 of 632** and is not part of the key. `state` ∈ `open` 306, `done` 233, `dropped` 80, `parked: experiment` 8, `closed` 5. `owner`/`reviewer` ∈ the roster names.

A repeated `id` is a state transition by design: 253 of 316 ids repeat. **53 `(id, state)` pairs repeat, accounting for 63 extra rows** — those are not transitions and are what a key assertion would catch.

### `review.jsonl`

Only `verdict` and `reviewer` are in every row; **20 rows lack `ts`** (the same 20 under either "key absent" or "key falsy"). It carries the widest prose vocabulary in the tree (`what_i_ran`, `my_own_error_in_this_review`, `boundary`, `residual`, `not_reviewed`).

**`review_present` does not read `artifact` or `case`.** The function (`harness.py:8376`) reads `task`, `reviewer`, `verdict`, `state`, `closed`, `id`. The only reader of `artifact` in the tree is `scripts/review_row_lookup.py:77`, which is not a check. **141 of the 291 rows carry neither `artifact` nor `case`, and nothing has ever reported one.** The rule that a review must name what the reviewer opened is real and stated in AGENTS.md; the gate named after it enforces that a row exists, not that it names anything.

### The rest

`score_matrix.jsonl` — the only ledger with a **replacing** writer: `eval.score_matrix.write_records` rewrites the whole file and raises on a duplicate `(ckpt, profile)`. `type` ∈ `base` `control` `sft`; `profile` ∈ `full` `control` `milestone`. `metrics` and `skipped` are nested objects, and a metric can hold `{"error": ...}` rather than a number — 15 of the 23 rows with a `domain_bpb` key hold an error, so counting keys and counting values give 23 and 8.

`prereg.jsonl` — 8 rows, `id` unique, its own `prereg-union` merge driver because plain union would leave two lines with one id. Amendments are numbered pairs `amended_N` (the timestamp) and `amendment_N` (the text), written only by `harness prereg amend`, which reads N from the row.

`moe_diag.jsonl` — 472 rows, the only per-step training diagnostic in this population: `name` `step` `usage_frac` `entropy_norm` `load_gini` `tokens` `window_steps` `ts` `n_routed` `top_k` in every row, `tok_s_gpu` in 16. `(name, step)` collides 11 times across re-runs of a name; `(name, step, ts)` is unique.

`milestones.jsonl` — 29 rows, no field in every one; `ckpt` is in 27 and `measured` in 22. Written by `harness milestone --pin` and by hand.

`board.jsonl` — `ts` `who` `topic` `kind` `text` `artifact` in every row. `kind` ∈ `block` `done` `find` `note` `open` `rule`; `topic` ∈ eight values.

`msg_log.jsonl` (808), `ledger_resolutions.jsonl` (27), `policy_metrics.jsonl` (4), `artifact_refs.jsonl` (35), `retro.jsonl` (8) — each has its key present in every row.

## What to enforce, and what not to

Enforce on **new rows only**. Backfilling 283 friction rows to a schema they were not written against buys nothing and rewrites other people's records. The grandfathering pattern already exists here twice: `no_ghost_close`'s dated ceiling and `tasks_well_formed`'s `drop_reason` list, both ratchets that shrink without a commit and grow only with one.

Three assertions are worth writing, in this order:

1. **`experiments.jsonl`: the 31 empty-`cmd` keys as a literal set**, with new keys required to be empty. A count alone is defeated by one row added and one removed — the aggregate hides what it averages — so the assertion is over identities.
2. **Every check that reads a ledger field must read the synonyms too, or the ledger must have one name.** `friction_minutes_required` is the worked example and its number is exact: 6 baseline rows, 3 of them false. This is a property of the predicate, not of the data, so it cannot be fixed by editing rows.
3. **A check whose population is a hand-written list of ledger names.** The population belongs in git or on the filesystem — the same correction this document had to make to its own frontmatter, which said "all 13 tracked `runs/*.jsonl`" when there are 32.
