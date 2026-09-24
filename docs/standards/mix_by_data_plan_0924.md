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
| a loss reader for a file-path holdout | **none** | 待写 |
| proxy sweep driver (N proxy runs, fit ratio→loss) | **none** | 待写 |
| constrained optimizer over the fitted surface | **none** | 待写 |

The five 待写 items are the whole engineering cost. Everything else is wiring.

## Step 1 — target metric

**Metric.** Loss on a decontaminated code validation set that is distributed like HumanEval.
Not pass@1: pass@1 at this scale is a noisy integer over 164 problems, and the plan's whole point
is to fit a smooth surface. Loss is smooth and cheap; pass@1 is the gate at the end.

**Set.** `data/eval/code_holdout_v2_500.jsonl` — already registry-tracked (`REGISTRY_SHA1["code_holdout_v2_500"]`),
already carved with its contamination split recorded (`cont.code_holdout_carved`).

**Its loss reader is 待写, and this was wrong in an earlier draft of this plan.**
`eval/domain_loss.py`'s only input is `--mix` (`:627`, defaulting to the retired
`mix_scale_3.24b.json`); rows arrive through `val_seqs` → `train._domain_seqs`, i.e. as a **token
cache per mix domain name** under `data/corpus/<name>/` with the `vocab_id`/`.srcfp` guards. There
is no `--file`/`--jsonl`/`--holdout`. `code_holdout_v2_500.jsonl` is neither a mix domain nor
keyed `content` (its keys are `instruction / reference_code / expected_output / source / family /
sha1`), so the command an earlier draft gave would have scored **corpus head shards**, not the
holdout — and would have looked like it succeeded.

Two ways out, the second worth weighing:
- **(a)** add a file-path parameter to `domain_loss.py`. Its `head_texts()` already parses jsonl
  with a `content`→`text` fallback, so this is roughly a row-field parameter and ~15 lines.
- **(b)** write `scripts/holdout_loss.py` reusing `_ce`.

Either way a fifth 待写 item, and (b) is the honest default: the holdout's row shape differs from
a corpus shard's, and a reader built for mix domains should not be widened to hide that.

**Consequence for sequencing.** Scoring the holdout is a prerequisite for Step 3's reference-loss
term to mean anything, and it sits on the same no-GPU critical path as the label extractor. The
ordering section names both.

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
python3 datagen/holdout.py --selftest                      # registry + REGISTRY_SHA1 pins   [exists]
python3 scripts/holdout_loss.py --ckpt <ckpt> --set data/eval/code_holdout_v2_500.jsonl   # [待写, option (b)]
```
`eval/domain_loss.py --ckpt <ckpt>` is NOT this command — see above: it reads mix-domain token
caches and would score corpus head shards, appearing to succeed.
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
- N proxy runs. RegMix's proxies are **small models** — 1M to a few tens of M parameters, on the
  order of 1B tokens each — and N is large because each run is cheap. **Proxy shape: 20-50M
  parameters, ~1B tokens per run.** A proxy at the gate's own scale is not a proxy.
- Each runs the **real recipe at reduced width**, not a different architecture, so the fitted
  surface describes this model family.
- Ratios sampled from a Dirichlet over the 6 gate domains (over the cluster×domain cells once
  Step 2 exists), with the supply constraint as a hard bound, not a penalty.
- Regression: target loss as a function of the ratio vector. RegMix uses gradient-boosted trees;
  the choice is 待写, and it must be recorded — a fitted surface whose form is unstated cannot be
  checked for extrapolation.
- **Repeats: each cell at most ~4 epochs** (user's constraint). This bounds the optimizer, not
  just the sampler: a 4-epoch cap shrinks the supply-feasible region below where the
  unconstrained argmin usually sits.

**Cost — the proxy rate is NOT MEASURED at this shape. 待测.** No run in this repo has trained a
20-50M model, so the rate must be measured before the sweep is budgeted. The measured anchors
around it, all on 8×H20:

| shape | measured rate | fact |
|---|---|---|
| 200M dense, bf16 master | 62K tok/s/gpu | `eff.bf16_master_dense_200m_tps` |
| 200M-class, fp8 | 73K tok/s/gpu | `eff.fb_mfu` |
| gate shape, MoE-48 | 28-29K tok/s/gpu | `v41.gate_first_steps_0911` |

A 20-50M model should be **faster per token than the 200M anchor**, but small models are
latency- and memory-bound rather than FLOP-bound, so the gain is sublinear and cannot be
extrapolated reliably. **Order-of-magnitude estimate**, bounded by assuming the rate is between
1× and 4× the 200M anchor (62K–248K tok/s/gpu):

```
32 proxies × 1B tokens = 32e9 tokens total
  at 62K tok/s/gpu  : 143 card-hours
  at 124K tok/s/gpu :  72 card-hours
  at 248K tok/s/gpu :  36 card-hours
```

Several proxies can share a card. Wall-clock on 8 cards:

| | 62K | 124K | 248K |
|---|---|---|---|
| 1 proxy/card | 17.9 h | 9.0 h | 4.5 h |
| 2 proxies/card | 9.0 h | 4.5 h | 2.2 h |
| 4 proxies/card | 4.5 h | 2.2 h | 1.1 h |

**Estimate: 1.1-17.9 HOURS wall on 8 cards, depending on the unmeasured rate and the
concurrency** (143 / 72 / 36 card-hours divided by 8 cards). That range is
too wide to plan against, which is why measuring the proxy rate is a prerequisite rather than a
detail. Co-residency also has a floor: at 20-50M the model may not saturate a card, so the
concurrency column is where the real win is, and it needs its own measurement.

The reference-model pass (Step 3) and the large-scale validation (Step 5) are **on top of** this.

## Step 5 — validate the ordering at scale

Fit the surface on proxies, then check the **ordering** of the top candidates does not invert at a
larger scale. Not the losses — the ordering, because the deliverable is a ranking to pick from.

**Design.** Take the top fitted candidate and the **current hand-set gate mix** as the control —
two candidates, not a field. Train each at the gate shape for **3-5B tokens** and compare their
**ordering**. Not the loss values: the deliverable is a ranking, and the question is whether the
proxy's ranking survives at scale.

**Cost.** At the gate shape the rate IS measured: 28-29K tok/s/gpu (`v41.gate_first_steps_0911`,
world 6). Taking the conservative end, 28.5K:

```
8-card aggregate at this rate: 228,000 tok/s
  full 30B on 8 cards : 36.5 h   (the gate's own budget, for scale)
  3B on 8 cards       :  3.7 h
  5B on 8 cards       :  6.1 h
2 candidates on 4 cards EACH (parallel): 3B -> 7.3 h wall, 5B -> 12.2 h wall
```
**Estimate: 7.3-12.2 h wall** (2 candidates × 3-5B tokens, 4 cards each). The full 30B gate run at
this rate is 36.5 h on 8 cards, so this step is about a fifth of one gate run -- it checks whether
an ORDERING inverts, which 3-5B tokens can answer.

This is deliberately much smaller than a "run 4 candidates at 30B each" design: the question is
whether an ORDERING inverts, which 3-5B tokens can answer, and it does not need each candidate
trained to convergence.

## Minimal viable version

The smallest sequence that produces a defensible answer, for when the full plan is too expensive:

**label extractor → clustering → one reference-model loss pass → N=32 proxies at 20-50M/1B tokens
→ the 2-candidate scale check.**

That is Steps 2-5 with the proxy sweep at its cheapest useful size. Cost:

| line | estimate |
|---|---|
| Steps 1-3 (CPU: labels, embed, cluster, per-group loss) | < 1 day, no GPU (tokenization ~4-6 h) |
| 32 proxies, 1B tokens each | **4.5-17.9 h wall** on 8 cards at 1 proxy/card (the conservative column), or **1.1-4.5 h** if 4-per-card concurrency holds — both depend on the unmeasured rate |
| scale check, 2 candidates × 3-5B at gate shape | **7.3-12.2 h wall** |

**Estimate, at 1 proxy/card, summing each term's own range** (CPU 0.5-1.0 d + proxies 4.5-17.9 h
= 0.19-0.75 d + validation 7.3-12.2 h = 0.30-0.51 d): **1.0-2.3 days total wall**, taking each
term's lower bound for the lower end and its upper bound for the upper end. If 4-per-card
concurrency holds the proxy term becomes 1.1-4.5 h = 0.05-0.19 d, giving **0.9-1.7 days**. The
bounds are stated per term because a total over terms whose ranges are written as "< 1 d" cannot
be re-derived without knowing which end each contributed. The single most valuable
action before committing to this plan is to measure the 20-50M proxy's tok/s/gpu and its
co-residency behaviour — one short run, and it collapses the range that decides whether this plan
costs hours or tens of hours.

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

Steps 1-2 are CPU-only and cheap. Step 3 is mixed: its loss pass is CPU, but its real cost is
tokenizing the domains, a pod-CPU job of hours. Steps 4-5 are the GPU budget, and Step 4's cost is
dominated by a rate nobody has measured.

Two things go first, both cheap, both able to kill the plan before it is expensive:

1. **The label extractor** (CPU, no GPU). It is exact, interpretable, and it falsifies the premise
   directly: if `has_tests` and `is_algorithmic` do not vary meaningfully across the six domains,
   the grouping carries no signal and the expensive steps should not start.
2. **The proxy rate measurement** (one short GPU run). Train a 20-50M model at the real recipe for
   a few hundred steps and read tok/s/gpu and co-residency. This one number is the difference
   between a 3-day plan and a 3-week plan (see the estimate range in Step 4), and it is currently
   待测 — no run in this repo has trained anything this small.

Neither is a step in the method; both are prerequisites to knowing whether the method is
affordable here.

## Open decisions (each must be recorded with its ruling, not defaulted)

| decision | why it matters |
|---|---|
| embedding model for Step 2 | a cluster id is meaningless without the embedder |
| reference model for Step 3 | too close or too far makes every group look the same |
| proxy scale (tokens) for Step 4 | sets the whole GPU budget |
| N proxies for Step 4 | governs fit stability; not fixed in advance |
| regression form for Step 4 | an unstated form cannot be checked for extrapolation |
| the epoch cap's interaction with supply | 4 epochs shrinks the feasible region |
