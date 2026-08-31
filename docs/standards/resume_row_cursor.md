---
question: How does a stage-2 resume continue reading each domain where stage 1 stopped, rather than from row 0?
status: recorded
source: de-7; train.py:1563-1614 (build_mix) and :1968 (i0 seek), read 2026-08-31
---

# Resume continues the row cursor (de-7)

## What is actually broken

`build_mix` rebuilds `used[name] = 0` for every domain on every process
(`train.py:1567-1576`), then walks the plan assigning `torch.arange(used, used+want)`
per phase. Within **one** run that is correct and a resume is already correct:
the loop seeks with `i0 = step * Cfg.batch * Cfg.accum` (`train.py:1968`) into a
plan that is materialised in order, so restarting at step k reads row k·B·A onward.

The break is **across mixes**. Stage 2 builds a new plan from its own mix with
`used` at 0, so every domain restarts at row 0 of its pool while the tail stays
unread. Measured by b0: 26% of code_rp1t, 34% of en_c4, 92% of zh_web.

This narrows the fix. The cursor does not need to survive a within-run crash —
`--auto-resume` already handles that through the step seek. It needs to survive a
**mix change**, which is exactly when the plan cannot be replayed.

## What to persist

Per domain, the row count consumed, in the checkpoint next to `step`:

```
"row_cursor": {"code_rp1t": 1043418, "en_c4": 588102, ...}
"row_cursor_srcfp": {"code_rp1t": "d8b9b18b...", ...}
```

`used[name]` at save time is that number. It is a count, not an index: pools are
read modulo their length (`idx % len(pool)`), so a cursor past one epoch is
meaningful and must not be clamped.

The `.srcfp` per domain is what makes the cursor interpretable. A cursor of
1,043,418 rows means nothing against a different corpus; a changed `.srcfp`
invalidates that domain's cursor specifically, not the whole set.

## Seeding on resume

`used[name]` initialises from the checkpoint rather than 0, for every domain whose
`.srcfp` still matches. A domain absent from the cursor (new in stage 2) starts at
0, which is correct. A domain whose `.srcfp` changed starts at 0 **and says so** —
silently reusing a stale cursor is the failure this exists to prevent.

## The checkpoint that has no cursor

Stage 1's checkpoint predates the field. Two options, and the cheap one is wrong:

| approach | why |
|---|---|
| replay the stage-1 plan (mix + steps completed) and read `used` off it | the plan is a pure function of (mix, rows, seed, val split); replaying it reproduces the exact cursor, not an estimate |
| assume proportional consumption from the weights | wrong whenever a domain was epoch-capped, which is the case that matters — cot capped at 295,512 of 310,546 wanted rows |

Replay, with an assertion that per-domain totals equal the stage-1 plan's, and
record the corpus `.srcfp` **at reconstruction time** in the exp row (fb): a
corpus that changed between stage 1 and the replay invalidates the reconstruction,
and the fallback triggers.

The fallback exp row carries the per-domain exposure distribution — head, middle,
tail — never the mean. A mean hides the shape that decides whether the tail was
read at all.

## What the cursor records

**As of the step, not the plan.** A checkpoint at step k has read `k x batch x accum`
rows per rank; the plan-complete counts describe a run that finished. Seeding stage 2
from the plan-complete figure skips everything between k and the end, and
`--auto-resume 2` makes a mid-plan checkpoint the expected case. Measured on the real
shape: at step 8000 of 16281 the honest cursor is ~49% of the plan-complete count.

`build_mix` keeps the per-row domain index for the rank (int8, ~0.5MB at 523,158 rows);
`save_checkpoint` counts the prefix the step consumed. A run-end save has no step and
keeps the plan-complete counts, which is correct there.

### Measured truncation

Replaying stage 1 reconstructs 3,646,940 rows against 3,646,944 actually consumed --
**4 rows short across seven domains**, from `int()` truncation once per domain per
phase. Recorded rather than rounded away: it bounds what the reconstruction can claim,
and a future discrepancy larger than single-digit rows is a different fault, not this
one.

## The stage-1 cursor is reconstructed, not recorded

Verbatim in the rehearsal report and in the stage-2 launch exp row's `hypothesis`
(fb), because the first person debugging a stage-2 data anomaly must meet it before
anything else:

> Stage-2 data continuity rests on a RECONSTRUCTED cursor, not a recorded one.
> ckpt_pretrain_15b_s1.pt.step16000 predates the row_cursor field, so its cursor
> was replayed by scripts/replay_cursor.py from (mix, step, pool sizes) rather
> than read from an artifact stage 1 wrote. No record of stage-1's actual
> consumption exists; the run finished before the field did. Two independent
> cross-checks agree with the replay: cot capped at 295,512 rows (44's measured
> figure) and zh_web at 0.08 epochs (b0's 92%-unread finding), neither supplied
> to the tool. That is good evidence and it is not a recorded artifact. A
> stage-2 data anomaly should suspect this before anything else.

The distinction the sentence protects: a passing rehearsal proves a resume *continues
from these counts*. It cannot prove the counts are what stage 1 consumed — no artifact
of that exists, and none can be made now.

## Loss continuity cannot be gated at 0.002 nat

The pre-registered criterion — resumed loss within 0.002 nat of continuous — is
**retired as instrument-impossible for per-step training loss** (fb, 2026-09-01): the
threshold sits below the σ_nondet floor, since per-step sd is ~0.29 nat and the
same-seed non-determinism reading above is ±0.13. Retired with cause, not dropped.

Per-step loss on this run has sd ≈ 0.29 nat. Stage-1's own final 40 steps span
**1.094 to 2.521**, a 1.4 nat range between adjacent logged steps on one continuous
run with no resume anywhere near it. A 0.002 nat threshold is three orders of
magnitude below that noise; no 50-step window can resolve it, and a gate nobody can
evaluate is a gate that gets waved through.

| window | n | mean | sd |
|---|---|---|---|
| stage-1 final 100 steps | 11 | 1.7039 | 0.2904 |
| resumed 50 steps | 6 | 2.0273 | 0.2489 |

Mean delta +0.3234 nat, SEM 0.1341 — 2.4 SEM, which is why the twin below replaced
the threshold rather than the threshold being relaxed.

### What replaced it: the twin, with the verdict rule pre-declared

Same checkpoint, same 50 steps, cursor against no-cursor. The rule was fixed **before**
the numbers, so a result cannot be explained after the fact:

- the no-cursor arm re-reads rows the weights already saw, the cursor arm reads unseen
  rows, so the expected signature is no-cursor ≈ stage-1 tail and cursor **higher** by
  the fresh-data premium
- **PASS**: no-cursor within 1 SEM of the stage-1 tail mean, and cursor above no-cursor
- **FAIL**: cursor **below** no-cursor (the cursor would be serving seen data labelled
  fresh), or either arm's mean diverging more than 1 nat

The +0.32 at 2.4 SEM is consistent with that mechanism, not evidence against it — which
is exactly why it needed a test with its interpretation fixed in advance.

### Result, with each contrast's own error

| contrast | n | mean | SEM | ratio | df |
|---|---|---|---|---|---|
| no-cursor vs stage-1 tail (unpaired) | 5 vs 11 | +0.0927 | 0.1332 | 0.70 | ~14 |
| cursor vs no-cursor (**paired**, same steps) | 5 pairs | +0.2672 | 0.1931 | **1.38** | 4 |

Both pre-declared conditions hold: no-cursor within 1 SEM of the tail, cursor above
no-cursor, neither arm near 1 nat.

The paired figure is the one to quote for the cursor premium, and it is **1.38 SEM, not
2.0** — the 0.70 belongs to a different contrast and pairing the arms raises the error
rather than lowering it here, because the per-step deltas are large and inconsistent:
−0.017, +0.617, +0.085, +0.827, −0.176. **Two of five are negative.** The mean is in the
predicted direction and five steps cannot establish the premium's size; the twin
demonstrates the cursor does not serve seen data as fresh, which is what it was built to
test, and nothing finer.

### σ_nondet

The no-cursor arm re-runs the same seed over rows the weights already read, so
+0.0927 ± 0.1332 (df ≈ 14) is a same-seed non-determinism reading, not only a pass
condition. It belongs in the σ_nondet fact with that df.

## Rehearsal before stage 2

Two assertions, the second easy to forget:

1. **Continuity.** Resume at step k, run 50 steps, assert the row indices drawn are
   the ones the cursor points at, and that loss stays within 0.04 nat of the
   continuous run.
2. **Fresh-run identity.** The plan for a **non-resume** run is byte-identical
   before and after the change. A cursor feature that perturbs a fresh run has
   changed every future baseline.

## What this does not fix

`Cfg.seed` is not in the token-cache key, so a cache built at seed 42 is reused
under `--seed 7` with a different shuffle and nothing raises. The cursor is a row
count into a pool whose order that seed decides — so a seed change silently
reinterprets every cursor. Fixing the cache key (the `--fone` pattern: seed in the
name) is a prerequisite for the cursor meaning anything across a seed change, and
it ships in the same train.py window.
