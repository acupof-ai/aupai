---
question: How do we choose the v41 gate mix from the data's own structure, instead of from hand-set domain weights?
status: planned
source: user order 2026-09-24 via aupai-1e; method follows DoReMi (arXiv 2305.10429), RHO-LOSS (arXiv 2302.03169) and RegMix (arXiv 2407.01492)
---

# 按数据规律定配方 — executable plan

This is a plan, not a result. Nothing here has been run. Every card-hour figure is an
**estimate** and is labelled; the two measured numbers it rests on are cited with their facts.

Scope: replace `data/mix_v41_gate.json`'s hand-set domain weights with weights derived from the
data's own structure. The gate mix stays the unit of composition — this plan re-weights the six
`_dc` domains and varies within them; it does not add or drop domains.

## Step 0 — what already exists

Naming this first because a plan that cites scripts that do not exist is not executable.

| need | what exists | status |
|---|---|---|
| loss on a held-out code set | `eval/domain_loss.py`, `eval/domain_bpb.py`, `eval/humaneval_bpb.py` | exists |
| a HumanEval-distributed code validation set, decontaminated | `data/eval/code_holdout_v2_500.jsonl` (registry `datagen/holdout.py`, kind `heldout`, field `instruction`); contamination split recorded in `facts/contamination.json#cont.code_holdout_carved` | exists |
| small-model training at the real recipe | `./run_ddp.sh` + `runs/v41_gate_0911.sh` flags | exists |
| card allocation | `scripts/harness.py launch`, allocation per `runs/card_assignment.json` | exists |
| embedding + clustering of the six domains | **none** | 待写 |
| label extraction (has tests / docstring / algorithmic / imports / length) | **none** | 待写 |
| reference-model loss pass (RHO-LOSS term) | **none** | 待写 |
| proxy sweep driver (N proxy runs, fit ratio→loss) | **none** | 待写 |
| constrained optimizer over the fitted surface | **none** | 待写 |

The four 待写 items are the whole engineering cost. Everything else is wiring.

## Step 1 — target metric

**Metric.** Loss on a decontaminated code validation set that is distributed like HumanEval.
Not pass@1: pass@1 at this scale is a noisy integer over 164 problems, and the plan's whole point
is to fit a smooth surface. Loss is smooth and cheap; pass@1 is the gate at the end.

**Set.** `data/eval/code_holdout_v2_500.jsonl`. It is already registry-tracked, already
carved with its contamination split recorded, and its rows carry `instruction`, so
`eval/domain_loss.py` reads it without a new loader.

**Why not HumanEval itself.** Fitting weights against HumanEval turns the gate into a training
signal. The validation set must be disjoint from the gate, and the correlation between them is
itself something this plan measures (Step 5) rather than assumes.

**Acceptance.**
- The set is registry-pinned: `REGISTRY_SHA1["code_holdout_v2_500"]` matches the file on disk.
- Zero 13-gram overlap with HumanEval/MBPP under the current gate (`cont.decontam_base_byte_rebuild_0921`
  established the gate's own bytes; `scripts/verify_dc_residual.py --root <root> --selftest` is the
  instrument's own known-answer, and the same scanner run over this set is 待写, ~10 lines).
- Loss is stable: two runs of `eval/domain_loss.py` on the same checkpoint differ by <0.001 nats.

**Commands.**
```bash
python3 eval/domain_loss.py --ckpt <ckpt>                 # existing read path
python3 datagen/holdout.py --selftest                      # registry + REGISTRY_SHA1 pins
```
The pin has no standalone checker: `REGISTRY_SHA1` lives in `datagen/holdout.py:330` and is
exercised by that module's own selftest, which builds its worlds by mutating `REGISTRY` and
`REGISTRY_SHA1` rather than touching files.
**Cost.** CPU only, minutes per checkpoint. **Estimate**: ~5 min/checkpoint.

## Step 2 — grouping

Two independent groupings; the plan keeps both, because they answer different questions and
disagreeing is informative.

**(a) Embedding clustering, k=64.** Embed a stratified sample of each domain, cluster, and keep
the cluster id as the data's own partition. This is the DoReMi/RHO-LOSS input.

**(b) Labels.** For each row, extract: `has_tests`, `has_docstring`, `is_algorithmic`,
`imports` (set), `n_lines` (bucketed). These are cheap, deterministic, and interpretable — they
are what makes a resulting mixture explainable to a human, which a cluster id alone is not.

**待写**: `scripts/domain_embed_cluster.py` — sample per domain (see sampling note), embed, k-means
k=64, write `runs/domain_clusters/k64.parquet` with `(domain, cluster, label_*)` per row.
**待写**: `scripts/domain_label_extract.py` — the AST pass for the five labels.

**Sampling note, load-bearing.** Embedding all of it is not affordable and not necessary:
`code_ultra_l2_dc` alone is 39,018,564 rows. Sample a fixed number per domain (propose 200k,
stratified by row length) and record the sampling parameters in the output, because **every
cluster proportion is a proportion of the sample**, not of the domain. A cluster's share of the
mix must be scaled back through the domain's share.

**Embedding model**: 待写 decision. It must be recorded in the output's config either way,
because a cluster id means nothing without the embedder that produced it.

**Acceptance.**
- k-means is reproducible: same seed, same input → byte-identical assignment.
- Labels are exact: `has_tests` agrees with an independent AST parse on 500 hand-read rows.
- Cluster count and per-cluster row counts are printed, with the sample parameters beside them.

**Cost. CPU only. Estimate**: 200k rows × 6 domains = 1.2M rows. Embedding at ~1k rows/s on CPU
(estimate; no measured number exists for this embedder in this repo) ≈ 20 min. Clustering k=64 on
1.2M × d is minutes. **Estimate: <1 h total, no GPU.**

## Step 3 — how much each group can learn

The quantity that matters is not a group's loss but its **reducible** loss: what a small model
still has left to learn there, relative to a reference.

```
learnable(g) = loss_small(g) − loss_ref(g)          # DoReMi / RHO-LOSS shape
transfer(g)  = Δloss_target when g is up-weighted    # measured, not assumed
```

`loss_ref` is a reference model's loss on the same group. RHO-LOSS uses a model trained on the
target; DoReMi uses a small proxy. **待写 decision**, and it is the plan's most consequential
one: a reference model too close to the target makes every group look unlearnable, one too far
makes every group look equally learnable.

**The transfer term is what this repo's history says not to skip.** The 0830v1 ablation that
produced 36%-vs-5% was mis-attributed because the freed 31% went to `web` — the variable and its
consequence moved together, and only naming what else changed with the variable separated them
(`docs/lessons/kept_methods.md`, and the rules list's "Before a two-arm test, name what else
changed with the variable"). A group can be highly learnable and still transfer nothing.

**待写**: `scripts/domain_group_loss.py` — per-group loss for a given checkpoint, over the
cluster file from Step 2, reusing `eval/domain_loss.py`'s read path.

**Acceptance.**
- Per-group loss sum, weighted by group size, equals the whole-domain loss to <1e-3 nats.
  (A partition whose parts do not sum to the whole is a partition bug, and it is silent.)
- Every group reports n; a group under a floor (propose n < 200) is reported as **unmeasured**,
  not as a low loss.

**Cost.** CPU per checkpoint once token caches exist; the token cache build is the real cost.
Tokenizing the six domains is a pod-CPU job: `eff.pretokenize_throughput` is the measured basis,
**estimate** ~4-6 h wall for the remaining domains at 180 cores with `RAYON_NUM_THREADS` set.

## Step 4 — fit the mixture

RegMix's shape: train N small proxy models on randomly-sampled ratios, regress ratio → target
loss, optimize the fitted surface under supply constraints.

**Design.**
- N proxy runs. Each is the **real recipe at reduced scale**, not a different architecture —
  otherwise the fitted surface describes a different model.
- Ratios sampled from a Dirichlet over the 8 gate domains (over the cluster×domain cells once
  Step 2 exists), with the supply constraint as a hard bound, not a penalty.
- Regression: target loss as a function of the ratio vector. RegMix uses gradient-boosted trees;
  the choice is 待写, and it must be recorded — a fitted surface whose form is unstated cannot be
  checked for extrapolation.
- **Repeats: each cell at most ~4 epochs** (user's constraint). This is a constraint on the
  optimizer, not just the sampler: the optimum must sit inside the supply-feasible region, and
  a 4-epoch cap makes that region smaller than the unconstrained argmin usually wants.

**Cost — the dominant line, estimated from measured numbers.** The gate recipe runs 786,432
tok/step; `eff.p500m_20b_throughput_and_dips` measures **11.87K tok/s/gpu median** at the 500M
shape on 8 cards, so 94,960 tok/s across 8 cards (and 786,432/94,960 = 8.3 s/step, vs the gate's
own 38,147 steps for 30B). A proxy at 1/10 the gate's tokens (3B) on 8 cards:

```
3e9 / 94,960 ≈ 31,600 s ≈ 8.8 h per proxy run
N=32 proxies: 32 × 8.8 h ≈ 281 h wall on 8 cards ≈ 2,247 card-hours ≈ 11.7 days
```
**Estimate: ~8.8 h per proxy at 3B tokens; ~281 h wall (11.7 days) for N=32 on 8 cards.**
N is 待写 and should be chosen from the fit's own stability, not fixed in advance. The throughput
figure is measured at the 500M shape on a different run; a proxy at a different shape has a
different rate, so this is an estimate that must be re-measured before the sweep is budgeted.

The reference-model pass (Step 3) and the large-scale validation (Step 5) are **on top of** this.

## Step 5 — validate the ordering at scale

Fit the surface on proxies, then check the **ordering** of the top candidates does not invert at a
larger scale. Not the losses — the ordering, because the deliverable is a ranking to pick from.

**Design.** Take the top ~3 fitted candidates plus the current hand-set gate mix as a control.
Train each at a scale between proxy and gate. Compare orderings by rank correlation.

**Cost.** At 30B tokens/candidate on 8 cards: `3e10 / 94,960 ≈ 315,900 s ≈ 87.8 h` per candidate.
**Estimate: ~88 h per candidate; 4 candidates ≈ 351 h ≈ 14.6 days wall on 8 cards** — that is the
gate's own full budget per candidate, which is why this step is last and why the candidate count
must be decided from Step 4's fitted ordering rather than fixed in advance.

## What this method cannot answer

Stated as boundaries, because a plan that only says what it will show is a plan that will be
believed past its evidence.

1. **Patterns a small proxy cannot learn.** Proxy models are smaller than the gate model. If a
   pattern only appears above some scale, the proxy's loss on it is uninformative and the fitted
   surface will treat that group as either useless or uniformly useful, depending on noise. The
   method has no internal test for this. **Partial mitigation**: Step 5's scale check can catch an
   inversion after the fact; it cannot prevent the fit from having been made on the wrong signal.
2. **Ordering may not survive scale.** RegMix's own validity is empirical. Step 5 is designed to
   test it for these candidates, and if it fails, the correct response is to discard the fitted
   ranking, not to report the large-scale result as if it confirmed the method.
3. **Groups are a partition of the observed sample, not of the domain.** Cluster proportions come
   from the sample (Step 2's note). A cluster rare in the sample and common in the domain will be
   mis-weighted, and nothing in this method detects that.
4. **Only the supplied domains.** Supply constraints cap what is reachable: a mixture the optimum
   wants but cannot be built from is not a finding.
5. **Loss is not pass@1.** The whole plan optimizes a proxy metric. The final claim must be the
   gate's own number on the final mixture.

## Sequencing and the first cheap check

Steps 1-2 are CPU-only and cheap; Step 3's cost is mostly tokenization; Steps 4-5 are the GPU
budget. **The first thing to build is Step 2's label extractor**, because it is cheap, exact,
already interpretable, and it can falsify the premise early: if `has_tests` and `is_algorithmic`
do not vary meaningfully across the six domains, then the grouping carries no signal and the
expensive steps should not be started.

## Open decisions (each must be recorded with its ruling, not defaulted)

| decision | why it matters |
|---|---|
| embedding model for Step 2 | a cluster id is meaningless without the embedder |
| reference model for Step 3 | too close or too far makes every group look the same |
| proxy scale (tokens) for Step 4 | sets the whole GPU budget |
| N proxies for Step 4 | governs fit stability; not fixed in advance |
| regression form for Step 4 | an unstated form cannot be checked for extrapolation |
| the epoch cap's interaction with supply | 4 epochs shrinks the feasible region |
