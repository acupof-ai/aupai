---
question: "What does the between-stages shipment window actually run, what does it cost, and can a 50-step arm read a 3% gate?"
status: recorded
source: "fb ruling 2026-09-01; facts/efficiency.json#eff.seam_dynamo_disable, #eff.pad_dynamic_shapes_twin_control, #eff.lm_head_is_compute_bound, #eff.cache_load_gates_startup; runs/pretrain_15b_s1.log"
---

# Shipment A/B: five runs, and a problem with the throughput gate

fb's design: one three-arm set (baseline, baseline-twin, candidate) × two candidates, sharing the
baseline pair, `--profile` + `TORCH_LOGS=recompiles` on every arm so seam ms, gap counts and val
all come from the runs that judge them. **Five runs.**

| # | arm | candidate | reads |
|---|---|---|---|
| 1 | baseline | — | val, tok/s, seam ms, gap count |
| 2 | baseline-twin | — | the in-config noise floor |
| 3 | fp8 head | fp8 | 3% tok/s, val vs floor, grad_logits absmax |
| 4 | seam disable | `_dynamo.disable` | seam ms drop, flash recompile count, val vs floor |
| 5 | spare / re-run | either | whichever candidate lands ambiguous |

## Wall clock, from measured numbers

7 cards, global batch 16×2×4096×7 = 917,504 tokens/step. At the live run's 77K tok/s/GPU that is
539K tok/s aggregate, so **1.70 s/step**.

| item | measured source | cost |
|---|---|---|
| 50 steps compute | 1.70 s/step | 1.4 min |
| startup, 149 GiB cache load | `eff.cache_load_gates_startup`, 386 s | 6.4 min |
| **per arm** | | **~7.9 min** |
| **five arms** | | **~39 min** |

**Startup dominates compute 4.5:1.** The window's budget is set by cache loading, not by the
steps, which means adding steps is nearly free and adding arms is not.

## The problem: 50 steps cannot read a 3% gate

The live 7-card run's own tok/s, binned by phase:

| phase | n | mean | sd | cv | range |
|---|---|---|---|---|---|
| steps 10–200 | 19 | 69.1K | 11.39K | **16.5%** | 24–76K |
| 200–600 | 40 | 75.2K | 6.51K | **8.7%** | 36–77K |
| 600–2000 | 140 | 76.2K | 3.12K | **4.1%** | 52–77K |
| 2000+ | 1050 | 75.8K | 4.46K | 5.9% | 29–77K |

A 50-step arm sits entirely inside the 16.5%-cv warmup band, and even a 200-step arm sits in an
8.7% band. **The gate is 3%.** Comparing two arms' mean tok/s over 50 steps cannot resolve 3%
against noise five times that size — the measurement would be dominated by where in the warmup
curve each arm happened to sample.

Two options, and the first is nearly free given the cost table above:

- **Extend each arm to ~600 steps.** At 1.70 s/step that is 17 min of compute per arm against
  6.4 min of startup, so five arms go from ~39 min to ~2.0 h. It buys the 4.1% band, which is
  still above 3% but comparable, and the paired twin arm then measures the residual directly.
- **Judge throughput on the paired difference, not the means.** Run the arms interleaved or accept
  that the twin arm's tok/s spread IS the resolution limit, and report the candidate's gain as a
  multiple of it exactly as we do for val against the 0.094 nat floor.

**Recommendation: both.** 600 steps and report tok/s as a multiple of the twin spread. The second
costs nothing and is the same discipline b0 already imposed on val; the first costs 1.3 h of a
window that has no competing use once stage 1 ends.

## Ship gates per candidate

| candidate | throughput | val | extra |
|---|---|---|---|
| fp8 head | ≥3%, or as a multiple of twin spread | inside the in-config floor; if floor > 0.04 nat, stop and send to b0 for the paired-step form | `grad_logits` per-chunk absmax logged from the vendored path; FoNE assertion refuses `Cfg.fone` |
| seam disable | none — prize is 3.28% max | inside the in-config floor | seam ms must drop and flash recompile count must be 0; a NEW gap elsewhere means a host sync, not a win |

The seam candidate is numerics-neutral-only by fb's ruling: it ships on the trace numbers, not on
tok/s, because a constant tax is invisible to steady throughput. Reading tok/s alone scored it as
doing nothing in the lane test — 81K in both arms while recompiles went 70 → 0.

## What the twin arm answers for free

`eff.seam_dynamo_disable`'s open question: the step-110 loss delta of 0.157 nat, 1.7× the
single-card twin floor. Run 2 gives the in-config floor, so the same delta measured in run 4 is
judged against a floor from the same configuration rather than against a smoke-mix number.

## What this plan does not establish

The 1.70 s/step assumes the shipment runs at the live run's throughput; a candidate that changes
step time changes its own arm's wall clock. The 386 s startup is one measurement and benefited
from warm page cache. And the cv figures come from one run's tok/s at 1K display resolution, so
they bound the noise coarsely — the twin arm is what measures it properly, which is the argument
for having one.
