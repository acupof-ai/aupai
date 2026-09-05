---
question: When is a fitted parameter a measurement, and what does it take to know the difference before citing it?
status: measured
source: measured 2026-09-05 on runs/b0_headmix_armA_val_trace.log and runs/b0_headmix_armB_val_trace.log with probes/fit_data_exponent.py ; fact facts/data_scaling.json#ds.b_unidentified_from_val_traces (commit e0c912a7); the CPU-forward boundary in facts/efficiency.json#eff.model_cannot_forward_on_cpu (commit 7c20e080)
---

# A fit returns a number whether or not the data contains one

4c asked a decision question with a threshold: is the data exponent `b` in `L(D) = a·D^-b + c`
below 0.2? Below it, the paired-BPB eval design is load-bearing and worth its complexity; above
it, the vertical gap a sample-efficiency difference produces is large enough that a simpler
unpaired design resolves the same effect.

I fitted the two b0 head-mix arms' val traces and got `b = 0.131 ± 0.007`, R² = 0.988 for armA
and `b = 0.114`, R² = 0.962 for armB. Both under 0.2. Clean, Chinchilla-consistent, and it
would have settled the question in one message.

Both numbers were artifacts. The design cannot see `b` at all.

## What the traces actually support

Each arm gives 7 val points over a 7.0× token range, 0.131B to 0.918B, and the trace's own
per-point noise is 0.0388 nat (armA) and 0.0423 nat (armB) — estimated from second differences,
which for independent errors of SD σ have variance 6σ². Gridding `(b, c)` and keeping every
pair whose RMSE is within that noise:

| arm | admissible `b` | width | best fit |
|---|---|---|---|
| armA | 0.114 – 1.000 | 8.7× | b=0.4975, c=1.851, RMSE 0.0111 |
| armB | 0.107 – 1.000 | 9.3× | b=0.8060, c=2.180, RMSE 0.0056 |

`b < 0.2` and `b > 0.2` are both consistent with the data. The best fits are grid points in a
flat valley, not estimates. The answer to the decision question is that these traces cannot
answer it.

## The two errors that produced 0.131

**Fitting in log space.** Taking logs of `L - c` to make the fit linear reweights the residuals:
the small-`(L - c)` end dominates, and on a curve with a real floor the optimiser answers by
pushing `c` toward 0, which turns the three-parameter curve into a pure power law whose exponent
is whatever the log-log slope happens to be. That is where 0.131 came from. The R² of 0.988 was
computed in log space too, so it measured how straight the log-log plot looked after `c` had
been set to the value that made it straightest.

Fitting in loss space instead — `a` is linear in `L` at fixed `(b, c)`, so it is closed-form and
no optimiser is needed — recovers a `c = 1.5` floor that the log-space fit drives to 0. Checked
on a synthetic curve with a known floor.

**Reading R² as identifiability.** R² measures fit at the argmin. A flat valley has a high R² at
every point along it, so R² is high exactly when the parameter is least determined. The question
"what is `b`" and the question "can these data see `b`" are different, and the second one is
answerable before you have an answer to the first.

## Proving the limit is the data, not the fitter

"My fit is unstable" and "this design cannot resolve `b`" have the same symptom and different
conclusions — the first is a bug to fix, the second is a fact to record. Three checks separate
them, and all three are selftest worlds in `probes/fit_data_exponent.py`:

1. **Zero-noise recovery.** On synthetic curves at three truths the fitter returns `b` to within
   0.01 and `c` to 0.03 — grid resolution; the truth itself fits to 1e-9. A fitter that cannot
   do this proves nothing about any design.
2. **Loss at real noise.** The same 7-point/7.0× design loses `b` at 0.039 nat whatever the
   truth: the admissible interval exceeds 3× for both `b_true = 0.13` and `b_true = 0.30`.
3. **Recovery on a longer lever.** Widening to a 128× token range at the *same* noise narrows
   the interval again. This is the load-bearing one — if the verdict were an artifact of the
   method, more decades of D would not help.

Together they say the limit is the lever length, so the fix is more decades of `D`, not more
points inside 7×. Adding points inside the existing range buys `1/√n` on the noise and nothing
on the degeneracy.

## Excluding the epoch-end point

`train.py` prints periodic val with `Cfg.val_batches = 20` and the epoch-end val with
`Cfg.val_batches_full = 100`. Different sample, different noise, and it sits at the end of the
range where a curve fit is most sensitive to a single point. `parse_log` drops it, with a
selftest world asserting a log containing both yields only the periodic points. Including it
would have improved every R² while making the fit worse.

## The mutant that survived

Five mutations of the fitter went red on the intended assertion against a green baseline.
One survived: removing the tokens-per-step scaling, so the fit runs against step index rather
than tokens. Every selftest stayed green, because each world builds its synthetic curve and
fits it in the same units, and `b` is invariant to rescaling the x-axis.

The mutation is only invisible because all the worlds share the error symmetrically. `a` is not
invariant — a step-indexed fit returns `a · tok^-b` — so selftest 7 fits the same points in both
units and requires `a` to differ by that factor. This is not a units nicety: two runs at
different world sizes have different tokens-per-step, so a step-indexed `a`, and the reducible
term `R = a·D^-b` that every design number is proportional to, would be read off the wrong axis
and would not be comparable between runs.

A full mutant sweep can still certify a defect when N worlds cancel the same error. When a
mutant survives, the question is what quantity it changes, and whether anything asserts on that
quantity rather than on the one already being checked.

## What this costs the design it was asked about

My earlier claim — a paired design resolves the sample-efficiency ratio `r` to ×1.07 at 8000
documents — was conditional on `b = 0.095` and inherits the whole `[0.11, 1.00]` range instead.
At `b = 0.114` the same SE gives `r` to about ×1.06; at `b = 1.0` it gives ×1.005. The design
gets *better* as `b` rises, so the pessimistic end is the one to plan against and the paired
estimator survives this finding. What does not survive is quoting a specific resolution.

Second constraint, found the same day and recorded separately: the correlation term that
pairing's benefit depends on cannot be measured without a GPU. `fla.ops.kda.chunk_kda` is
Triton-only with no CPU fallback, so this architecture cannot complete a forward pass on CPU at
all — flags do not avoid it, and 9 of 12 blocks are KDA at `attn_every = 4`. Any probe of this
model that needs a forward costs card time.

## Rules

1. **Report the admissible interval, not the argmin.** After fitting, sweep the parameter and
   report the range whose RMSE is within the data's own noise. `probes/fit_data_exponent.py`
   prints `NOT IDENTIFIED` with the interval width when it exceeds 2×, and refuses to present
   the best fit as a measurement.
2. **Estimate the noise from the data.** Second differences of a smooth curve, scaled by √6.
   This over-estimates when there is real curvature, which widens the interval — the safe
   direction for a claim about identifiability.
3. **Fit in the units of the quantity.** Not in a transformed space chosen to make the algebra
   linear, unless the reweighting is what you want.
4. **R² is not identifiability.** A flat valley is high-R² everywhere along it.
5. **Separate a broken fitter from an unidentifiable design** with the three checks above before
   recording either a value or a negative result.
6. **When a mutant survives, assert on the quantity it changes.** Symmetric errors across all
   test worlds are invisible to any assertion those worlds share.
