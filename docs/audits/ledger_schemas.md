---
question: For each ledger under runs/, what is every field, which values are legal, and what is the row's primary key?
status: measured
source: read of all 13 tracked runs/*.jsonl on main at e77ca348, 2026-09-08; field presence and value domains computed per file, not transcribed
---

# The ledgers under `runs/`

Written for someone who has not seen this repository. Each ledger is one JSON object per line, appended never rewritten, merged by union (`.gitattributes`). A reader opening one row should be able to tell which fields a machine reads, which are prose for a person, and what makes the row a distinct row.

**No field is renamed by this document.** A rename breaks every reader; the readability is carried by the description. Where two names mean the same thing, both are listed and the one to write from now on is marked.

## Why the primary key matters more than it looks

Union merge keeps every line that either side has and drops exact duplicates. It has no idea what a row *is*. So for a ledger with no stable key, two branches can each append a row describing the same event with one field different, and the merge keeps both, silently, with nothing reporting it. **Five of the thirteen have no field present in every row.**

The key is also what an assertion can be written against. `runs/experiments.jsonl` is the case that proved it: `exp.py` refuses to close an already-closed run, but a person can append the row by hand and commit it, and the refusal never runs. A predicate over the *file* catches that; a refusal in the writer cannot.

## The table

| ledger | rows | primary key | key holds today? |
|---|---|---|---|
| `experiments.jsonl` | 462 | `(name, started)` | present in all, **149 duplicate pairs** — see below |
| `tasks.jsonl` | 632 | `(id, state, when)` | `id` in all, repeats 307× by design |
| `score_matrix.jsonl` | 90 | `(ckpt, profile)` | unique |
| `prereg.jsonl` | 8 | `id` | unique |
| `board.jsonl` | 104 | `(ts, who, topic)` | present in all |
| `msg_log.jsonl` | 808 | `(ts, from)` | present in all |
| `ledger_resolutions.jsonl` | 27 | `(ledger, key)` | present in all |
| `policy_metrics.jsonl` | 4 | `date` | present in all |
| `artifact_refs.jsonl` | 35 | `path` | unique |
| `retro.jsonl` | 8 | `(owner, date)` | present in all |
| `review.jsonl` | 291 | `(ts, reviewer)` | **`ts` missing in 26 rows** |
| `friction.jsonl` | 283 | `(when, who)` | **neither in every row** |
| `milestones.jsonl` | 29 | `(ckpt, measured)` | **neither in every row** |

### `experiments.jsonl` — one run, one or more events

`(name, started)` identifies a **run**; a run has several rows, one per event, and the reader folds them. That is why 149 pairs repeat: a `running` row and its terminal row share the key. Folding is terminal-wins, and a human's close beats the monitor's.

| field | who reads it | legal values |
|---|---|---|
| `name` | machine | free text, the run's name |
| `started` | machine | `YYYY-MM-DD HH:MM`, **UTC** (`exp.py:139` uses `time.gmtime()`) |
| `status` | machine | `running` `ok` `fail` `error` `stop` `killed` `dropped` `refused` `retracted` |
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

**Empty `cmd` on a row with no `running` event is the defect signature.** 31 such keys exist. They come from `exp.py`'s fabrication branch — closing a run it cannot find makes a row with `started=now()`, `cmd=''`, `hypothesis=''` — plus, in at least one case, a row appended to the file by hand. Both produce the same shape, and one predicate over the file catches both.

### `friction.jsonl` — 39 field names, 12 used once

The sprawl is real and this is where the spec earns its keep. The eight fields that are actually the schema:

| field | in | write this one |
|---|---|---|
| `kind` | 282/283 | yes — `gate` `merge` `tooling` `process` |
| `who` | 280/283 | yes |
| `when` | 275/283 | yes — prefer over `ts` (8 rows) |
| `cause` | 274/283 | yes |
| `blocked_what` | 270/283 | yes — prefer over `what` (5) and `blocked` (2) |
| `sha` | 267/283 | yes |
| `minutes_lost` | 266/283 | yes — prefer over `cost_min` (8) and `cost` (4) |
| `fix_applied` | 262/283 | yes — prefer over `fix` (11) and `fix_idea` (7) |

Three pairs of synonyms, each with a clear majority. Writing the minority name is not an error today, and `friction_minutes_required` only reads `minutes_lost` — a row using `cost_min` is invisible to it.

### The other eleven

`tasks.jsonl` — `id` `owner` `state` `task` `why` `opened` in every row. `state` ∈ `open` `done` `dropped` `closed` `parked: experiment`. `owner`/`reviewer` ∈ the eight roster names. A repeated `id` is a state transition by design (251 of 316 ids repeat); 8 repeat at the *same* state, which is not.

`review.jsonl` — only `verdict` and `reviewer` in every row; 26 rows lack `ts`. It carries the widest prose vocabulary in the tree (`what_i_ran`, `my_own_error_in_this_review`, `boundary`, `residual`, `not_reviewed`) and that is the point: a review that names no artifact is not a review, and `review_present` reads for `artifact:` or `case:`.

`score_matrix.jsonl` — the only ledger with a **replacing** writer: `eval.score_matrix.write_records` rewrites the whole file and raises on a duplicate `(ckpt, profile)`. `type` ∈ `base` `control` `sft`; `profile` ∈ `full` `control` `milestone`. `metrics` and `skipped` are nested objects, and a metric can hold `{"error": ...}` rather than a number — 15 of the 23 rows with a `domain_bpb` key hold an error, so counting keys and counting values give 23 and 8.

`prereg.jsonl` — 8 rows, `id` unique, its own `prereg-union` merge driver because union would leave two lines with one id. Amendments are numbered pairs `amended_N` (the timestamp) and `amendment_N` (the text), written only by `harness prereg amend`, which reads N from the row.

`milestones.jsonl` — 29 rows, no field in every one; `ckpt` is in 27 and `measured` in 22. Written by `harness milestone --pin` and by hand.

`board.jsonl` — `ts` `who` `topic` `kind` `text` `artifact` in every row. `kind` ∈ `block` `done` `find` `note` `open` `rule`; `topic` ∈ eight values.

`msg_log.jsonl` (808), `ledger_resolutions.jsonl` (27), `policy_metrics.jsonl` (4), `artifact_refs.jsonl` (35), `retro.jsonl` (8) — each has its key present in every row and needs no repair.

## What to enforce, and what not to

Enforce on **new rows only**. Backfilling 283 friction rows to a schema they were not written against buys nothing and rewrites other people's records. The grandfathering pattern already exists here twice: `no_ghost_close`'s dated ceiling and `tasks_well_formed`'s `drop_reason` list, both ratchets that shrink without a commit and grow only with one.

The one assertion worth writing now is on `experiments.jsonl`: the 31 empty-`cmd` keys as a literal set, with new keys required to be empty. A count alone is defeated by one row added and one removed — the aggregate hides what it averages — so the assertion is over identities.
