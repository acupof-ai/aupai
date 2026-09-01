---
question: "Does every component a reported number passes through satisfy P1-P8?"
status: recorded
source: "tilerl-7 (user order 2026-09-01 08:20); reviewed scripts/eval_artifacts.py, scripts/harness.py check_cited_artifacts_attested, eval/score_matrix.py, eval/readout_30b.py, datagen/build_corpus.py stamping, datagen/corpus_fingerprint.py — none of it mine"
---

# Harness review, tilerl-7: the artifact and stamping path

Scope was everything a reported number passes through, reviewed against P1–P8. Two
**principle-level** defects found, both demonstrated with a failing case rather than argued.
Both are in the two places the task row named as suspects, which is the review working: the
task predicted where to look and the prediction held.

## Findings

| id | file:line | principle | failing case that demonstrates it | severity | proposed fix |
|---|---|---|---|---|---|
| T7-1 | `scripts/harness.py:217,226-252` | **P2** (property, not proxy) | `/tmp/p5_test.py`: a fact cites `data/eval/preds_A.jsonl` with hash H; the writer attested `data/eval/preds_TOTALLY_DIFFERENT.jsonl` with the same H. Check returns **PASS**. | **principle** | `attested` must be a set of `(path, sha256)` pairs, and the citation must match both. The `path` variable is already extracted at :241 and then never used in the comparison. |
| T7-2 | `datagen/build_corpus.py:520` | **P5** (fingerprint over SETTLED bytes) | Tonight's `math_owm_stage2`: stamp `a67cde07d3b3f63d` written while shards were still being rewritten. Cache built from it was 10 min OLDER than the shards beside it (15:46 vs 15:56). No guard at the stamp site refuses either condition. | **principle** | Before writing the stamp, assert no shard mtime is newer than the stamp being written, and re-run the `_preflight` unique-writer check. `_preflight` runs at build START only, so a writer that begins mid-build is not caught. |
| T7-3 | `scripts/eval_artifacts.py:132-193` | **P6** (no vacuous selftest) | `_selftest()` covers open/force/version/append/seal/truncate but **never calls `attest`**. The docstring at :39-42 names the requested-vs-written path bug and the incident it caused; nothing tests that the written path is what gets attested. | **defect** | Add a case: `with open_artifact(p, run="r1") as f`, then `attest(f.name)` and assert the recorded path ends `.r1.jsonl`; assert `attest(p)` on the un-versioned path is refused or flagged. |
| T7-4 | `datagen/build_corpus.py:653-690` | P2 | The unique-writer gate is `pgrep`-shaped: it matches `"build_corpus.py" in args and (domain in args or out in args)`. A writer invoked by a different path or wrapper name is not matched, and my own session hit the inverse of this bug four times tonight — a pattern that matches the checker itself. | defect | Hold an flock on the output dir for the build's lifetime. Presence of a lock is the property; a process name is a proxy for it. |
| T7-5 | `scripts/harness.py:232` | P3 (fail closed) | `contract_from = "2026-09-01"` exempts every citation measured earlier. Sound as written — 18 legacy citations cannot grow a hash retroactively — but the exemption is a date string, so a fact **backdated** to 2026-08-31 skips the check silently. | nit | Compare against the fact's `measured` AND require that any fact whose file mtime is inside the contract window carries a hash regardless of its stated date. |

## What I verified as CORRECT, so the record is not one-sided

`eval/readout_30b.py:385-395` refuses closed on a missing score record, with the reasoning in
the comment: "a lookup miss wearing a verdict's clothes". This is P3 done right, and it is the
pattern T7-1 lacks.

`eval/score_matrix.py:436,447,563` keys dedup on `(ckpt, profile)` consistently at all three
sites, so a milestone record cannot replace a full one — P4 satisfied.

`datagen/corpus_fingerprint.py:94-119` has a real known-answer selftest including transfer
invariance (copy/rsync change mtime, the fp must not move) — P1 and P6 done right, and it is
why the fingerprint itself is trustworthy even though the STAMP that records it is not (T7-2).

## The pattern joining T7-1 and T7-2

Both are **a guard that records the right thing and compares the wrong thing**. T7-1 attests a
path and then matches only hashes. T7-2 computes a content-based fingerprint — deliberately not
mtime-based, per its own docstring — and then writes it at a moment when the content is still
moving. In each case the artifact carries enough information to catch the bug and the check
declines to use it.

That is the same shape as three defects in my own work tonight, which is why I recognised it:
a docstring stating the correct property while the code enforces something weaker. It is
invisible to the author because reading your own diff, the prose confirms the intent you already
hold.

## Limits of this review

I did not run the full harness selftest suite against every check I read, so P1's "broken()
mutates a REAL artifact and the selftest FAILs there" is verified only for the checks I
exercised. I read `count_tokens.py` only through its `CONVENTION` import at
`build_corpus.py:506` and did not audit its counting arithmetic. And T7-5 is reasoning about a
backdating path I did not construct a failing case for, which is why it is a nit and not a
defect — by the standard I applied to the other four, an unproven claim does not earn a higher
severity.
