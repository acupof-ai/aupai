# 0830v1 gates

The controller session (aupai-fb) drives these in order. A gate opens only when its
evidence exists as an artifact — not when someone reports it done. Update the status
column in place; do not add a second copy of this table anywhere.

## Objective changed 2026-08-30: a reasoning model, coding and math

**The scaling law is no longer the deliverable and does not need proving.** The target is
a reasoning model with coding and math capability, trained on ~30B tokens. The six-point
ladder finishes because p324 is the largest checkpoint we will have and nothing else needs
those cards; the curve will exist and is no longer what the round is judged on. The local
slope at 3.24b is still consumed by tilerl's tolerance derivation, so the fit is not
wasted.

What this changes, and what it does not:

| survives | superseded |
|---|---|
| σ̂ = 0.0516 and every MDE recomputed from it | the natural-vs-synthetic ratio as the organising data decision |
| the pre-registration practice — four experiments retired before spending a card | zh:en 84:16, set for a Chinese-LM target |
| the harness, its 25 checks, the four-layer frozen-config enforcement | the ~16B Chinese-web base of the 36B plan |
| the fetch/clean/dedup/score contract and the MinHash known answer | the scorer funnel's prevalence bands, priced against a general corpus |
| §8's logic — "does output move when discrimination saturates" | §8's metrics; lambada measures expression, not reasoning |

Three consequences worth stating because they invert assumptions this round was built on:

- **Measurability rises rather than falls.** Verifiable answers give a binomial label
  error — 6.3pt at N=500 — against val NLL's σ̂ = 0.0516, which made the KDA A/B
  unresolvable at any affordable n.
- **Synthetic stops being a defect.** The 10% cap on templated encyclopedic prose stands
  on its own reasoning. Verifiable synthetic math with worked solutions, and code with
  tests, are the mechanism rather than a compromise.
- **Code cannot currently be measured at all** — no discrimination instrument, no
  generation instrument, no dataset carrying execution signal. That is the blocking gap,
  ahead of any corpus decision.

Architecture is fixed for this round: KDA (9 layers) + full causal gated MLA (3 layers,
latent=d/4) + AttnRes Full (blocks=0), NoPE throughout. Commit b3cad87. Changing it
reopens corpus-ready.

## Run config for all six budget points

Settled 2026-08-30. Every point uses the same values; a change to any of them reopens the
whole ladder.

**This table is executable, not prose.** Its values live in `data/mix_scale_run_config.json`
— 18 fields, one file rather than six copies — and four checks enforce them, each against a
different failure mode:

| check | catches |
|---|---|
| `_strip_frozen` in `run point` | a launch flag disagreeing with the frozen value — refuses before the run |
| `frozen_keys_complete` | a `train.py` parser flag in neither the frozen set nor the allow-list, so the list cannot rot |
| `ladder_cfg_consistent` | a code edit *between* points — compares all 46 recorded `Cfg` fields across ladder checkpoints, excluding only `mix` |
| `ladder_config_frozen` | a code edit *before the first point* — compares each checkpoint against the recorded intent, including the five constants with no CLI flag |

The last one exists because the table and the JSON once disagreed: `chunk_size` was frozen
in this prose and absent from the machine-readable config, on the correct reasoning that a
field with no CLI flag cannot drift via a launch. It can still drift via a code edit, and
nothing would have noticed — `pod_drift` goes blind the moment the manifest is regenerated,
which the controller did mid-ladder on 2026-08-30 to clear an unrelated drift. **A rule
enforced only by someone remembering it had already failed twice that day.**

| setting | value | why |
|---|---|---|
| cards | 7 (GPU 1-7) | GPU 0 is the bench/scoring lane |
| batch / accum | 16 / 2 | effective 32. AttnRes Full OOMs at batch 32 on 96GB; accumulation keeps the optimizer recipe valid. Block AttnRes changes the architecture under test; grad_ckpt measured 2.4x slower at batch 72 |
| `vocab` | 32784 | 32773 is 2-byte aligned, so cuBLAS falls back to an SM75 align-1 kernel at 41% of bf16 peak. Padding to a multiple of 8 reaches 92%: +14-16% end to end, tokenizer untouched. 32784 is additionally 16-aligned, so `_fp8_ok` accepts the head — a free fp8 option. The 11 padding rows (vocab_real:vocab) are zero-initialized so they stay neutral in the FLCE softmax (`eff.vocab_padding_softmax_defect`); A/B max \|delta\| 0.0016 at LR 0.03 |
| `chunk_size` | 32 | fla KDA default 64; halved chunks measured 19.1% faster on the kernel, numerically neutral (max delta 0.0017 on loss ~8) |
| `bucket_cap_mb` | 50 | 100 leaves DDP communication unhidden. 50 and 25 tie at 75K tok/s/gpu on both 3 and 7 cards, so 50 wins on fewer allreduces |
| NCCL protocol | default | forcing `PROTO=Simple` adds 2K tok/s but depends on an env var that a launch can silently omit |
| warmup | fixed 20 steps | absolute steps, not a fraction: momentum/second-moment reliability needs a roughly constant count. The fraction varies 9.2% (0.2b) to 0.57% (3.24b) — a known confound that overestimates beta; the proportional alternative biased the same direction harder (measured 0.52 val at 0.2b, `eff.warmup_absolute_not_fractional`) |

## The six-point run plan

Sequential, seven cards each. Card count is part of the config, so the points cannot be
run side by side on fewer cards. At the measured 75K tok/s/gpu the whole ladder is about
3.5 hours of GPU time; tokenization is paid once, before the first point.

| point | tokens | steps/card | wall |
|---|---|---|---|
| 0.2b | 2.0e8 | 218 | ~6 min |
| 0.3b | 3.0e8 | 327 | ~10 min |
| 0.4b | 4.0e8 | 436 | ~13 min |
| 0.8b | 8.0e8 | 872 | ~25 min |
| 1.6b | 1.6e9 | 1744 | ~51 min |
| 3.24b | 3.24e9 | 3537 | ~103 min |

Each point gets an `exp.py start` row before it launches and an `exp.py done` row plus a
score-matrix record when it ends. A point that dies is rerun once at the same config;
a second failure escalates. Per the frozen fit protocol, a point is never dropped
because it fits badly.

**D comes from the log, not the mix. The 3.24b point loses 1.53% of its budget to the
validation split.** Exact row counts from the tokenizer (`harness.py run pretokenize`,
2026-08-30): 3.6285B tokens over 7 domains.

The mix is not weighted by corpus share, whatever the `_comment` says. It is built so that
six of seven domains are consumed **exactly once** and web_hq fills the remainder:

| domain | rows | supply | demand at 3.24b | delta |
|---|---|---|---|---|
| web_hq | 350,110 | 1.4344B | 1.0508B | +383.6M |
| textbook | 393,021 | 1.61021B | 1.61021B | −0.00M |
| wiki | 59,974 | 0.24571B | 0.24572B | −0.00M |
| en | 39,240 | 0.16077B | 0.16077B | −0.00M |
| math | 19,936 | 0.08168B | 0.08168B | −0.00M |
| code | 14,051 | 0.05757B | 0.05757B | −0.00M |
| chat | 9,320 | 0.03818B | 0.03819B | −0.00M |

Six domains land on their supply to the row. That is a deliberate design, not an accident —
but it leaves no margin, and `train.py:1348` takes the validation split off the top before
the pool exists: `n_val = min(max(1, int(len(seqs) * 0.05)), 5000)`, then
`pools[name] = seqs[n_val:]`. The epoch cap at line 1362 measures `len(pool)`, so every one
of the six is capped by exactly its own `n_val`:

| domain | n_val | pool | want | lost |
|---|---|---|---|---|
| textbook | 5,000 | 388,021 | 393,022 | 20.49M |
| wiki | 2,998 | 56,976 | 59,975 | 12.29M |
| en | 1,962 | 37,278 | 39,240 | 8.04M |
| math | 996 | 18,940 | 19,937 | 4.08M |
| code | 702 | 13,349 | 14,051 | 2.88M |
| chat | 466 | 8,854 | 9,321 | 1.91M |
| | | | **total** | **49.7M = 1.53%** |

web_hq's 383.6M surplus does not cover it: `want` is computed per domain from its own
weight and nothing redistributes.

Only the 3.24b point binds — checked at 0.2b/0.3b/0.4b/0.8b/1.6b, no domain caps at any of
them. So the loss lands entirely on the point with the most leverage over beta and does not
cancel across the curve.

`train.py:1363` handles it correctly and loudly: it caps, prints `wants N rows, epoch cap
leaves M -> capped`, then prints the true scheduled total at line 1384. Nothing repeats
silently. This is not a data defect and no weight changes. It is an x-axis defect, and the
fix is free: **fit-protocol v1.9 reads D for every point from that scheduled-tokens line,
never from the mix file**, and pre-registers a tolerance above which the gap is a declared
curve boundary rather than rounding.

Correction to the record: the controller first reported this as a 6.4% wiki corpus deficit,
derived from a sampled bytes/token estimate with a stated ±2% uncertainty. The exact counts
show wiki is not short of corpus at all. The sampled number was used to make a claim finer
than its own error bar — the same shape as two earlier errors this round. The finding
survives with a different cause and a larger magnitude, but it was luck, not method.

A retracted number does not stop travelling when it is retracted. fit-protocol v1.9
(`f77310b`) justifies its 10% tolerance as "far above the known measured value (6.4%)" —
positioning a threshold above a quantity that no longer exists. The threshold itself is
sound and independent: leverage < 0.01 nat, one fifth of the noise floor, from a ~59%
back-derivation that never references the observed gap. The citation is the defect, not the
number. Two consequences, both cheap: the 6.4% clause comes out, and the tolerance is
labelled as derived-and-written-after rather than pre-registered, because 1.53% was known
to its author before it was written. The call is identical either way — 1.53% is 6.5x
inside 10% — so the only thing at stake is whether a later reader can tell a derived
threshold from a fitted one. This repo has paid for that distinction once already: a
pre-registered null was written up as settled, and its amendment is labelled as written
afterwards.

**The `en` domain is 85% Chinese.** Per-shard Han census, 400 docs per shard:
`cosmopedia_extra_000..006` is 690.3MB at 82.9–84.1% Han (chinese-cosmopedia, the same
source as `textbook`); `en_textbook_000..001` is 122.9MB at 0.0%. At the frozen weights
English is 0.75% of training tokens, not the declared 4.95%, and chinese-cosmopedia is
53.8%, not the 49.6% the mix `_comment` names as the figure a reader is warned not to
over-read. The curve is unaffected — all six points share the composition — so the fix is
the six `_comment` strings, not any weight (`mlm.corpus.en_domain_is_mostly_chinese`).

The cause is not a mystery and does not need to be recorded as one. `scripts/build_domains.sh:32`
builds the domain from two sources in one command:

```
has en && "${BC[@]}" --domain en --filters light --target_tokens 1e9 --no_near_dedup \
  --source jsonl:data/cosmopedia_extra.jsonl \
  --source jsonl:data/en_textbook.jsonl
```

`scripts/rebuild_corpus.sh:37` repeats it. So `en` never meant "English" — it meant "the
domain holding these two files", and the larger of the two is Chinese. The name was read as
a language label by everyone downstream, including the mix `_comment`. Nothing was
misfiled; a directory name was trusted as a content claim. The general form is already a
rule here: metadata is a claim about provenance, not about content.

### What runs before the ladder

Tokenization is paid once. Then eight 0.2b runs, then the ladder. The eight are not a
detour: four of them are the ladder's own 0.2b point plus its seed replicates.

| runs | what | serves |
|---|---|---|
| 4 | current arch, `attn_every 4`, seeds 0-3 | ladder's 0.2b point (seed 0); the seed-variance measurement; lessons-e1's control arm; lessons-44's F arm |
| 4 | `attn_every 1`, seeds 0-3 | **VOID — the arm is not a model.** See below |

**`attn_every 1` is not a full-attention arm. It is a model with no position
information.** Measured 2026-08-30: `p02_a1_s2` val 4.765 against `p02_s2` val 3.679, same
seed, same mix, same config — a gap of 1.086 nat, **21× the measured seed σ̂ of 0.0516**. An
effect that large in an A/B is a symptom, not a result.

`train.py:272` states the cause: *"Gated MLA: latent KV compression + full causal attention
(NoPE, KDA handles position)."* The attention layers carry no position encoding at all — no
RoPE, no learned positions. `attn_every 1` removes all nine KDA layers and with them every
source of positional information in the model. The 1.086 nat prices *having any position
information*, which was never in question, and says nothing about KDA versus attention as
operators.

Everything built on this arm is void: σ_d, ρ, the paired design, the fit protocol's §6
amendment, the minimum-two-pairs rule, the 721 GPU-min costing, the falsifier. All of it
was elaborating the statistics of a comparison whose treatment arm is not a coherent model.
The defect was upstream of every question asked about it, and an hour of careful work by
three sessions went into the wrong layer. lessons-e1 verified the two hard unknowns — data
order determined by `Cfg.seed`, init comparability across shared and replaced layers — and
both answers hold. Nobody asked the prior question: **is the treatment arm a model that
works.** The controller allocated the cards and approved the design, so the omission is the
controller's first.

Same shape as the `en` directory. A name — `attn_every 1` reading as "all attention" —
trusted as a claim about what the thing is.

A valid arm needs RoPE on the MLA layers, which is a code change rather than a flag. Even
then the comparison becomes "KDA + NoPE" versus "attention + RoPE": two changes, not one.
There may be no clean single-variable form of this question while KDA is load-bearing for
position, and if so that should be stated rather than designed around
(`eff.attn_every_1_has_no_position_information`).

One number survives on its own: **612s vs 425s at the same token count — all-attention is
44% slower** over seq 4096, consistent with quadratic attention. A throughput measurement,
not an MDE judgement.

44's F arm left this table on 2026-08-30. The W/F quality-cut experiment is **killed by its
own pre-registered threshold**, not null: the union hit rate of the Step 0/1 filters on
web_hq is 0.326% ± 0.050% (163/50,000, reservoir sample over 1,366,324 docs, `text[:600]`,
the filters' exact window) against a 2% line written before the number existed. Two arms
99.67% identical have no contrast to measure. That is what the pre-registration was for —
it says the experiment cannot resolve anything, rather than reporting that we looked and
saw nothing. The surviving finding is more useful than the experiment: the old chain had
already cleaned web_hq, so a filter upgrade now has 0.33% to beat. The old-chain arm
returned 0/50,000 — a predicted zero that came back zero, with 163 = 163 + 0 − 0 closing —
which settles at the determinism level that 3b's build record was accurate. The weight
ablation (32.4% → 20%) proceeds as an independent experiment, queued behind the six points,
not as a replacement arm.

**Resolved 2026-08-30 (commit 101bf1b): `--seed` and `--attn_every` added to the whitelist.**
The block below is kept as the incident record. Line 1433 described the parser as
"any --flag below overrides Cfg.<flag>" and AGENTS.md repeated it as a general rule. It was
not: the parser is an explicit whitelist, and `seed` and `attn_every` were absent although
both are real `Cfg` fields. Without the flag all four runs would have trained the identical
model — four checkpoints, four scores, σ̂ = 0, nothing raising. It surfaced only because
argparse is strict. The doc claim was the defect that produced the launch. Same shape as the
`en` directory: a label read as a claim about what is underneath it. AGENTS.md now states the
whitelist explicitly; a `Cfg` field without a parser entry cannot be set from the CLI.

**σ̂ = 0.0516 — measured, 2026-08-30. The first seed-variance number this repo has had.**

| seed | val NLL |
|---|---|
| s0 | 3.691 |
| s1 | 3.762 |
| s2 | 3.679 |
| s3 | 3.638 |

mean 3.6925, range 0.1240, σ̂ 0.0516 with 3 df (`ds.seed_variance_0p2b`). That is
**1.47× the 0.035 that two experiments were designed against** — and 0.035 was a fit
residual borrowed as a seed variance, which carries model misspecification where seed
noise does not.

The branch was committed at `20e1b7a` while s2 was at step 170/217 and s3 had not started.
It fires:

| σ̂ | MDE at 4+4 = 1.98σ̂ | consequence |
|---|---|---|
| ≤ 0.040 | ≤ 0.079 | inside the pre-registered 0.08 gate — the second four runs happen |
| **> 0.040** | **0.1021** | **outside it — the KDA A/B moves to the 3.24b checkpoint** |

**Ruling: the second four runs (`attn_every 1`, seeds 0-3) do not happen.** At 0.2b a
4-vs-4 resolves 0.1021 nat (normal approximation) or 0.1223 (b0's frozen t-version, df=6),
and an experiment whose lack of resolution is computable beforehand should be declared
beforehand rather than spending 24 GPU-minutes to produce a null. The ruling is robust to
which formula is used; both exceed 0.08.

**Correction: "the question is re-asked for free on the 3.24b checkpoint" is false, and
the controller repeated it twice.** The ladder produces `attn_every 4` at 3.24b. Comparing
it to `attn_every 1` means *training a second 3.24b model*. lessons-e1 priced it and the
arithmetic is worse than the experiment just cancelled:

| design | MDE at σ̂ 0.0516 | vs 0.08 gate | cost |
|---|---|---|---|
| 0.2b 4+4 — cancelled | 0.1022 | MISS | ~48 GPU-min |
| 3.24b 1+1 | **0.2043** | MISS | ~206 GPU-min |
| 3.24b 2+2 | 0.1445 | MISS | ~412 GPU-min |
| 3.24b 4+4 | 0.1022 | MISS | ~824 GPU-min |

Moving to 3.24b makes the comparison **worse at four times the cost**, or equal at
seventeen times. Nothing about a larger D improves an independent-samples MDE; only σ or
n do. For 1+1 to reach the gate, σ at 3.24b would have to be 2.6× below 0.0516; for 2+2,
1.8× below. Both are possible — more steps may average seed effects down — but neither is
measured, and "free" was never true for the isolating form of the experiment.

What is genuinely free is the scaling-law residual: fit 0.2b–1.6b, predict 3.24b, compare.
That detects whether the architecture is the bottleneck but cannot attribute a gap to KDA
rather than anything else. It runs as a sanity check, not as the A/B.

**The fit gate moves and must be renegotiated before the ladder runs.** ACCEPT is
RMS ≤ 1.14σ̂ = **0.0588**, not the 0.05 quoted this round — that figure assumed σ = 0.035
(1.14 × 0.035 = 0.0399). Expected RMS of a 3-parameter fit through 6 points is
0.71σ̂ = 0.0366. Moving it now is renegotiation; moving it after a fit misses would be
fitting the threshold.

One caveat that travels with the number: 3 df makes the σ confidence interval roughly 0.6×
to 2.9× wide. 0.0516 is a point estimate and is treated as one. It does not license
reasoning that depends on σ being precisely 0.0516, only on it being materially larger
than 0.035 — which the data supports at any reading of that interval.

The first four are unconditional. The second four run only if the measured seed variance
says a 4-vs-4 comparison can resolve anything: lessons-b0 computes `s_pooled` from the
first four and pre-registers, before seeing it, the MDE range that licenses the second
four. If the MDE lands above that range, the A/B moves to the 3.24b checkpoint, where it
is free, rather than spending 24 minutes to produce a null that was predictable in
advance. An experiment whose lack of resolution is computable beforehand should be
declared beforehand.

Seed variance has never been measured in this repo, and two experiments had already
been designed against an assumed value — lessons-e1 assumed 0.035 as seed variance,
lessons-44 assumed the same 0.035 taken from a fit residual. Those are different
quantities: a residual carries model misspecification, seed variance does not. The four
replicate runs give the real number with 3 degrees of freedom, and every MDE in this
round is recomputed from it.

The KDA A/B runs before the ladder because only one of its outcomes is expensive to
learn late: if full attention wins, the ladder is measuring an architecture we are about
to change, and re-running it costs 3.5 hours. The A/B costs 48 minutes.

Reading rules for the A/B are pre-registered by lessons-b0 before any of it runs, not by
the session that designed it. A null does not delete KDA — a null at 0.2b says 0.2b has
no resolution, and the question is re-asked for free on the 3.24b checkpoint.

## Gates

Gates are named for what they gate. The old G0-G5 labels are gone: a reader had to hold
a lookup table in their head to follow a status line, and the labels carried no
information the name doesn't.

| gate | opens when | evidence | owner | status |
|---|---|---|---|---|
| **harness-green** | `harness.py check` and `--selftest` both exit 0, on the pod | both exit codes | aupai-de | **OPEN** 2026-08-30: pod `check` exit 0, `--selftest` exit 0 (21 checks each verified to FAIL on a broken world). Both verified by the controller, not reported |
| **metric-panel** | the 200M resolution panel is committed and frozen | `docs/lessons/base_eval_panel.md` + `facts/base_eval.json` | lessons-b0 | OPEN, frozen |
| **panel-runners** | every panel metric has a runner that passes its known-answer gate | `eval/` runners + `be.known_answer_panel_3_4` | lessons-b0 | 5 of 6; LAMBADA-zh needs the held-out slice from the rebuilt corpus. The generative metric SKIPs by rule |
| **step-time-profile** | step time split by source, summing to ~100 | `facts/efficiency.json` | lessons-e1 | OPEN; roofline table landed, kernel line closed |
| **fit-protocol** | fitting method, accept thresholds, and falsification shapes frozen before the first point | `docs/lessons/scaling_fit_protocol.md` + `scripts/fit_scaling.py` | lessons-b0 | OPEN, frozen at v1.9.2 (`ef3760c`): D from `train.py:1384`, 10% tolerance derived from the noise floor alone, labelled post-hoc. The retracted 6.4% survives only in the changelog that retracts it |
| **corpus-ready** | `web_hq` rebuilt, holdout excluded, fingerprint stamped | PROVENANCE fetch/build/result block + `corpus_fp_matches` green | aupai-3b | **OPEN** 2026-08-30: fp `30838d423348b2e5`, 1,366,324 docs, 5.91GB, 1.434B tokens; `corpus_fp_matches` 7/7 on the pod |
| **warmup-stable** | 2-step warmup does not destabilise the 0.2b point | smoke vs `warmup=20` control, same seed | aupai-de | OPEN; fixed warmup=20, `eff.warmup_absolute_not_fractional` |
| **six-points** | all six `mix_scale_*` points trained and scored | six score-matrix rows + six experiments rows | aupai-fb | waiting on harness-green |
| **scaling-curve** | the fit runs and its verdict is recorded | `fit_scaling.py` output with RMS and the beta profile interval | aupai-fb | waiting on six-points |

Kernel work (tilerl-bench-harness-plan) runs alongside and gates on step-time-profile for target
selection, not on harness-green. It lands only through the five correctness gates in its brief:
fp64 reference no worse than the kernel it replaces, bit-exact reruns, 200-step
step-by-step loss comparison, `test_arch_compat` coverage, and a working fallback.

## Who does what

| session | owns | does not do |
|---|---|---|
| aupai-fb | controller: GPU allocation, gate rulings, launching runs, the experiment record, and the reasoning about what each result means | writes no code |
| aupai-de | harness, CI gates, doc deletion, code cleanup, pod hygiene | training runs, research |
| aupai-3b | corpus build, filters, quality head, eval-set construction | kernels, harness |
| tilerl-bench-harness-plan | performance measurement and verification: reads the real artifact (generated code, kernel names, ncu counters, autograd saved tensors) to confirm or refute a claim; owns serving infrastructure; writes kernels only when a measurement says one is worth writing | changing the architecture's math |
| lessons-b0 | research: what has resolution at 200M | writing repo code |
| lessons-e1 | research: where the step time goes | writing repo code |
| lessons-44 | research: filter transfer | writing repo code |

## GPU allocation

aupai-fb allocates all 8 cards. Nobody starts a GPU process without asking; kill only by
exact PID, never `pkill -f`.

All 8 cards are idle as of 2026-08-30 05:00 (the 27B service and the quality-head scoring
were both killed by exact PID). The lanes for the rest of this round:

- **GPU 1-7**: the six budget points, sequentially. Nothing else runs there while a point
  is training — card count is part of the run config, so a competing process invalidates
  the point it overlaps.
- **GPU 0**: benchmarks, `score_matrix`, probes. Ask before starting; a run there is
  minutes, not hours.

There is no quality-head scoring lane this round. The six points train on `web_hq` built
without the quality cut; that cut becomes the W arm of 44's W/F experiment instead of a
prerequisite for training.

## The 0.2b point, 2026-08-30: trained on the wrong corpus

`data/corpus/web_hq/` became eight symlinks into an in-progress CCI3 build at 22:24.
`/data00/tokens_web_hq.pt` was rebuilt from them at 22:25. The 0.2b run started at 22:41
and reused that cache, so 32.4% of its tokens came from the corpus this round had barred
(measured filter recall 1.7%, hand-read junk 30.3% ± 4.6pp). The original fineweb2 shards
are gone from `/work`, `/data00`, and `data/raw`; the domain is being re-downloaded.

The run is recorded `status=fail` and does not enter the scaling curve. What survives is
the mechanics: loss 7.871 to 3.204 with no NaN, `--batch 16 --accum 2` at 50.8GB/card,
497s on seven cards.

`corpus_fp` already stored all seven domain fingerprints in the checkpoint, and
`corpus_fp_matches` already existed. It reported `1 domain(s) match` and PASSed. A check
that covers one seventh and still reports PASS is worse than no check: it produces the
appearance of having been checked. Both guards below close that gap.

## Rules this round does not bend

- No GPU pretrain while `harness check` is red. A permanent red is the same as no signal.
- No number enters the repo without its measurement config.
- Data-side harness runs locally. Every check of the real runtime runs on the pod.
  `test_arch_compat.py` is the case that set this rule: green locally a dozen times,
  never once executed on the pod, where fp32 + flash-attn raises before the first
  assertion. Local green is not evidence about the machine that trains.
- A harness or kernel improvement that has landed and been reviewed is used by the next
  run. Improvements are not batched into a later version.
- `/work/aupai` on the pod is not a git repo. Code arrives by `podput` and drifts
  unmonitored; a stale pod copy already produced one wrong diagnosis this round.
- A mix domain directory holds real files. No symlinks, and nothing pointing at a build
  still being written. Corpora under construction stay in their own directory and enter a
  mix by name, never by repointing an existing domain.
- A partial check that reports PASS is a defect. State the coverage in the evidence
  string, and FAIL when coverage is incomplete.
- A check must not pass for a reason unrelated to what it checks. The restartability
  check was built because `train_quality_head.py` loses everything on interrupt, and its
  first version silenced that exact script: a substring heuristic saw `glob(` and
  `checkpoint` in the file and read them as evidence of resumability. Same shape as the
  27B service reporting healthy on `/v1/models` while its engine was dead. A heuristic
  over substrings is not evidence; the only exemption is an explicit marker next to the
  code, and the selftest carries a decoy that would re-silence the check if anyone adds
  the heuristic back.
- A number the controller derived rather than measured is labelled "hypothesis, do not
  schedule work against it" and carries its falsification test. On 2026-08-30 three
  unmeasured AttnRes attributions from this session were each overturned by measurement,
  and two sessions had already scheduled work against them. The controller's output is
  who measures what and which reading counts, not an estimate.
- No kernel proposal without a `torch.compile` baseline, **and the baseline must carry
  evidence that compilation took effect.** `compile=True` and a silently-eager model are
  indistinguishable from the outside: `profile_step.py` omitted `cache_size_limit`, dynamo
  fell back to eager past the 9th of AttnRes's 25 sources, the header printed
  `compile True`, and the profiled model ran 36% slow with every elementwise row inflated
  threefold. Evidence means zero recompile-limit warnings or a recorded limit. The
  original reason still holds: AttnRes measured 42.2ms eager and 3.85ms compiled for the
  same work, so the eager number would have made a zero-benefit kernel look like a 10x
  opportunity.
- **A benchmark's correctness assert runs before its timing is read.** An arm computing
  something different is not faster or slower — it is not a comparison. A short-conv bench
  failed its own `allclose` because PyTorch's conv is cross-correlation and the taps were
  reversed; without that assert it would have produced a clean set of numbers for two arms
  computing different things, with the wrong one winning.
- **A ranked optimisation states its recoverable fraction, not its block size.** Two
  estimates were revised down 3x in one afternoon by their own author for the same reason:
  short_conv 3.1% → 1.0% (51.19ms is the block; 1.46x on fwd+bwd is what is recoverable),
  and the fp8 amax hypothesis, where cost turned out to scale with bytes rather than
  launches so fusing buys nothing.
- **A throughput change is priced in two currencies that move in opposite directions as
  D grows.** Wall-clock: the same 1% of step time saves more absolute hours at a larger
  budget — 1% is ~20 minutes of a 34-hour ladder. Quality: the same 1% buys 1.12× the
  tokens, and those tokens buy *less* loss as the curve flattens — the exchange rate fell
  from 0.910 to 0.567 nat per e-fold between the 0.2b→0.3b and 0.3b→0.4b segments, taking
  the tolerance for a 10.7% saving from 0.103 to 0.064 in a single point. **So a
  quality-neutral change (short_conv, an fp8 recipe) becomes more worth doing at 10×,
  while a quality-costing change (Block AttnRes) becomes harder to justify** — the budget
  it must fit inside is collapsing while its wall-clock prize grows. Never price one with
  the other's rate. `eff.throughput_quality_exchange_rate` stores the rate keyed by
  segment rather than as a single number, so it cannot be cited without its range: a
  number that cannot be fetched cannot be fetched wrong, which beats a warning that can
  be skipped.
- **A discrimination instrument built from easy perturbations saturates and stops
  steering.** `math_v2_like` reads 94.69% at the 0.8b point, up from 76.69% at 0.2b — it
  moved 18 points across the ladder and then ran out of room. A metric near its ceiling
  cannot distinguish a good change from a great one, which is the whole job of a steering
  metric. Build the next one from *hard* perturbations and let it start low: a mutated
  algorithm or a missing edge case, not a flipped operator. Headroom is a design
  parameter, not something to discover afterwards.
- **Run the ground truth through the checker. A checker that cannot score its own gold
  answers is broken, and the test is exhaustive and free.** `algorithms/rlvr_reward.py`
  scores 217,932 of 217,953 gold answers correct — 99.9904% — and the 21 failures are two
  real bugs that a hand-built variant suite did not find. **`对` sits in the Chinese
  unit-stripping list** (from 一对, a pair), so `normalize_answer('对')` returns None while
  `'错'` returns `'错'`. In a filter that is a dead row. **In a reward function it is
  asymmetric: the model can be paid for answering 错 and never for answering 对, so RLVR
  pushes it toward one answer on every true/false question regardless of the truth.**
  Second bug: a *relative* 1e-4 tolerance scores `10000` against `10001` as correct, and
  the false-positive window grows with magnitude while math answers stay exact.
- **A bad checker in a filter gives a wrong survival rate. A bad checker in a reward
  function is what the model optimises against.** Reward hacking is not adversarial
  behaviour — it is the model finding the checker's blind spots faster than we do, because
  that is precisely what training pays it to do. Every checker that becomes a reward gets
  the ground-truth round trip before a single RL step.
- **"Verifiable" is a property of the answer, not of the checker. Name the checker before
  quoting a survival rate.** `scripts/eqcheck.py` reads 21.6% coverage on a Chinese
  synthetic math batch and its flagged bad steps are almost all false positives: Chinese
  units break the equation chain, so `1600元 × 3/10 = 480` parses as `3/10 = 480`. Its
  91.65% "survival rate" is instrument noise. Answer-level comparison against the gold
  field on the same batch reads 99.85% raw and 99.91–99.97% after hand review — the batch
  was clean and one checker could not see it. **The dangerous failure in a verification
  pipeline is not low survival; it is a checker that confidently measures the wrong
  thing** (`dq.verification.eqcheck_blind_spot`). Two requirements follow: a synthetic
  batch must retain its gold answer field, and free-text generation needs a pre-registered
  extraction contract — `\boxed{}` is the one already in use.
- **Where the answer is verifiable, the measurement is a different economics.** Execution
  gives a binary label with no judge and no seed variance entering it, so the error is
  binomial: δ = 1.4/√N, or 6.3 points at N=500. Against val NLL's σ̂ = 0.0516 — which
  made the KDA A/B unresolvable at any affordable n — the same comparison on pass@k may
  be affordable. **Pivoting to reasoning raises measurability rather than lowering it**,
  and that is a reason to prefer verifiable targets wherever a choice exists.
- **A generative metric reading zero on a base checkpoint measures two things at once:
  absent capability and absent format.** The panel's SKIP rule — "generative; a base
  checkpoint reads zero" — encodes the conflation. A base model continues text; it does
  not answer questions, so 0/500 on math-500 cannot separate "cannot solve" from "was
  never asked in a form it responds to". Few-shot continuation separates them for the
  price of a prompt.
- **A number that agrees for the wrong reason is harder to catch than one that
  disagrees** — disagreement starts an investigation, agreement ends one. Two instances
  today: `ds.mde_recomputed_from_measured_sigma` at 0.1021 (two-sided detection, normal
  approximation) sitting next to §7's δ_res at 0.104 (one-sided non-inferiority, t-form),
  which answer different questions and match by coincidence; and `profile_step.py`
  printing `compile True` over a half-eager model.
- **When the upper bound on the benefit is below the lower bound on the cost, the
  experiment is redundant — decline it in arithmetic, not on a card.** Three retirements
  in one day on this shape, in three unrelated domains: W/F on prevalence × Δ_marginal
  (0.326% against a 2% line), the KDA A/B on MDE against gate (0.1021 against 0.08), and
  torchao's `ROWWISE_WITH_GW_HP` on FLOPs against quantize overhead — the config's best
  case saves at most the whole 60.93ms quantize row while certainly adding 82ms of bf16
  GEMM, and it in fact disables only two of six casts so it cannot even reach that bound.
  Granting the proposal more than it can have and still refusing needs no measurement.
- **Until σ̂ is measured, every significant/not-significant call this round is
  provisional.** 0.035 came from a fit residual, which carries model misspecification;
  seed variance does not. Two experiments were designed against the conflation before
  anyone noticed they were different quantities. Anything already written that leans on
  0.035 is re-read when the real number lands, not grandfathered.
  **Scope, and it is narrow:** provisional means *judgments that passed through an MDE*.
  Direct readings did not — 75K tok/s/gpu, MFU 31-32%, 206.13M, the corpus counts, the
  census hit rates. Widening this rule to the whole round would be worse than the
  grandfathering it replaces: it would put a measurement and an inference under the same
  caveat, which is the distinction the rule exists to enforce.
- **Nothing lands on the pod between the first and last run of a series.** `train.py`
  changed at 06:30:37 on 2026-08-30, after `p02_s0` (06:15:07) and `p02_s1` (06:25:06)
  had loaded it and before s2 and s3 would — four runs that must differ only in seed,
  produced by two revisions. This instance was verified clean afterwards: the frozen
  values are exactly what s0 logged, and the change added a `Cfg` class attribute
  carrying the value argparse already supplied, so it moved the recorded metadata and
  not the computation. σ̂ stands. But "harmless" was established after the fact, and the
  controller allocates the cards, so the missing instruction was the controller's. A
  code change during a series is a change to the thing under measurement until someone
  proves otherwise.
- **The frozen run config is enforced by nothing.** The table above opens with "a change
  to any of them reopens the whole ladder", and the only thing keeping six points
  identical is that whoever types the launch remembers eight values. The controller got
  it wrong on the first attempt, on a run it had specified twice in writing: `Cfg.batch`
  defaults to 32 and `Cfg.accum` to 1, the frozen values are 16/2, and the launch OOMed
  in 45 seconds exactly as the table predicted. That was luck. The dangerous member of
  that table is any value wrong *and still runnable* — `warmup 30`, `bucket_cap_mb 100`,
  a different `chunk_size` — which yields a completed point, a checkpoint, a
  score-matrix row, and a curve with one point measured under a different recipe, with
  `score_matrix_present` green throughout. The fix belongs in the mix files, so the
  recipe travels with what it is frozen against, and a check compares each ladder
  checkpoint's stored `cfg` against that block.
- A per-shard sample of `data/corpus/web_hq` must span the shard range, not a prefix. The
  corpus is ordered by source and is not shuffled: shard 1 alone is 20.9% Traditional
  against a corpus figure of 17.715% ± 0.064% (all 62 shards, 242,048/1,366,324). The
  controller's first-three-shard scout read 15.20% and was wrong by 2.5pt for that reason
  alone — the same run's token count read all 62 shards and landed within 0.3%. The method
  was sound; the shard selection was not.
- **A container restart is a source change, and every derived artifact under it is stale
  with nothing raising.** 2026-08-30, the shared container was restarted to recover from a
  bind-mount incident. The writable layer went with it; the image layer did not. What that
  cost, measured rather than assumed: five packages (`liger_kernel`, `fla`, `flask`,
  `opencc`, `trackio`) — the rest of the stack lives in the image and never moved; the
  Triton autotune cache; and the entire token cache, because `/data00` was never a mount
  inside the container, only a directory on the writable layer. A 0.2b smoke run re-encodes
  `web_hq`'s 1,366,324 documents from scratch to discover this. This is the `vocab_id` /
  `.srcfp` / `filters_fp` failure class with site-packages as the source and every
  checkpoint as the derived artifact — and `ckpt_p324.pt` carries `model`, `cfg`,
  `vocab_id`, `corpus_fp` and **no environment field at all**, so nothing could have raised.
- **The recovery path for a container incident is the exited container's own writable
  layer, and it is on a clock.** `crictl stop` leaves the previous attempt's container
  object and its overlayfs snapshot in place. Attempt 7 — created 2026-08-14, the container
  that actually ran the six-point ladder — was still readable at
  `/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots/<id>/fs`. **An upper
  layer contains, by definition, exactly what was hand-installed on top of the image**, so
  it *is* the list nobody remembered: 60 packages, read off rather than recalled, plus the
  binaries themselves. Same image sha, same Python, same CUDA, so the recovered `.so` is
  binary-exact — better than recompiling from a source tree whose commit is also unrecorded.
  The clock: one `crictl rmp` or one GC pass and the only surviving record of the working
  environment is gone with no second snapshot behind it. **Copy the whole upperdir out
  before anything else.**
- **A real difference is not a cause until it is shown to be on the path. Two sessions made
  this mistake on the same failure within an hour, in opposite directions.** The failure:
  `fla/ops/common/chunk_delta_h.py` → Triton autotune → `Triton Error [CUDA]: invalid
  argument`, on the first forward, on every rank, identically at the frozen `--batch 16` on
  one card.
  - **Hypothesis 1, fla version drift** (controller). `pyproject.toml` pins no version and
    the ladder's version is recorded nowhere, so the reinstall plausibly moved it.
    **Refuted by the attempt-7 snapshot**: `flash_linear_attention` 0.5.2, `fla_core` 0.5.2,
    `liger_kernel` 0.8.2 — identical on both sides; torch and triton live in the image layer
    and never moved. The reasoning was self-consistent and the trace was real; the trace was
    of the fallback path, not the intended one.
  - **Hypothesis 2, missing `flash_kda`** (tilerl). A genuine difference, found by reading
    the exited container rather than by recall: `flash_kda` 0.0.1+1ce47ea present in attempt
    7, absent in attempt 8, hand-built before the ladder. **Refuted by installing it**: the
    crash is byte-identical. `fla/ops/kda/backends/flash_kda.py` opens its verifier with
    `if torch.is_grad_enabled(): return False, "FlashKDA only supports inference mode"` —
    **it is an inference-only backend and was never on the training path**, so its absence
    could not be the cause. It does explain a real observation: b0's L1 probe runs fine on
    GPU 7 while training dies, because inference and training take different dispatch
    branches — which is also why "the environment works, imports succeed" was never evidence
    about training.

  The shared defect is not carelessness about the evidence; both differences were real and
  both traces were accurate. It is skipping the step that connects the difference to the
  failure. Reading the verifier that refuted hypothesis 2 took two minutes and was available
  before the claim was made.
- **Hypothesis 3, a cold Triton disk cache** (tilerl), was refuted too: attempt 7's 1.2 GB
  `/root/.triton` restored, crash unchanged. A cache does not change a grid.
- **The answer came from measuring the failure point, not from enumerating differences.**
  Wrapping `CudaLauncher.__call__` printed `grid=(2, 78936, 1)` against CUDA's `gridDim.Y`
  limit of 65535. `cuLaunchKernel` answers an over-limit grid with a bare `invalid
  argument` and no further detail, which is why it read as a broken environment for an
  hour. **In an environment where everything changed at once, enumerating differences is
  the most expensive path available — differences are always plentiful enough to build a
  self-consistent story from.** Three such stories, three refutations; the fourth approach
  landed in one step. (tilerl, 2026-08-30.)
- **A result verified in one configuration is not a result in another, and "it passed" is
  the easiest place to forget that.** The grid finding came with a falsifiable prediction —
  smaller batch should pass — and `--batch 8` did pass, on one card. The controller
  launched seven ranks on it. Rank 2 died on the same `invalid argument`, with ranks 3 and
  4 reporting NVLink peer-memory errors behind it (secondary: a rank dying inside a
  collective). Ranks hold different rows, the grid is data-dependent, so a single-card pass
  says nothing about the unluckiest rank of seven.
- **The kernel bug was real and we were the ones triggering it: `<eos>` padding became 489
  documents per row.** `data/sft/sft_all.pt` pads to `seq` with `<eos>` rather than packing
  — mean 489.4 per 4097-token row, max 3721, p99 2417. `doc_cu_seqlens` opened a document
  after *every* `<eos>`, so each pad token became its own length-1 document; fla's varlen
  grid is per-document, and 16 × 489 is how 78936 happens. Fixed at `5e643cb`: a run of
  `<eos>` opens one document, taking 8 padded rows from 32768 documents to 8, and SFT then
  ran at the frozen `--batch 16` with no run-config deviation at all. Two things this
  changes about the reading: the grid arithmetic tilerl measured was exact and the fla
  gridY-folding bug is real, but it is no longer on our path and drops to lowest priority;
  and the reason the six ladder points never hit it is that pretrain rows are packed, not
  padded. **A fix in the first version dropped the unconditional row-start boundary, which
  would have let one document span two rows of the batch and KDA state flow between them —
  a silent correctness bug traded for a loud crash.** Row starts are unconditional; the
  known-answer case moved `[0,2,4,5,6,8]` → `[0,2,4,6,8]` and three cases were added.
- **30% of every SFT step trains on padding, and the loss mask is what hides it.** 251,155
  of 255,968 `<eos>` positions carry `-100`, so the objective is correct and the model is
  not taught to emit `<eos>` forever — which is exactly why nothing complained. Only 70.2%
  of tokens are supervised. Packing several examples per row instead of padding recovers
  ~30% of SFT compute at no quality cost, which is 30x `short_conv`'s 1.0%. A correct loss
  mask makes wasted compute invisible; it does not make it cheap.
- **"Imports succeed" is not "the runtime works", and a check that asserts the first while
  the second is broken is worse than no check.** After the restart the environment was
  reported repaired on the evidence that `import liger_kernel, fla` succeeds. It does. SFT
  then dies in the *first forward*, inside Triton's autotune benchmark of
  `fla/ops/common/chunk_delta_h.py`, with `Triton Error [CUDA]: invalid argument` — on all
  seven ranks, and identically at the frozen `--batch 16` on one card, so it is neither
  shape nor scale. `FLA_CI_ENV=1` (which shrinks the autotune config space) and
  `FLA_USE_TMA=0` both fail unchanged. An `env_importable` gate reads green through all of
  it. **And it would have stayed green after a correct repair too, because it checks a
  hand-written list and no such list contains `flash_kda` — nobody remembered installing
  it.** That is the general defect: a list written from memory omits exactly the packages
  whose installation was never written down, which is the same set that a restart destroys.
  The gate that catches this compares a *fingerprint* of the live environment against the
  one stored in the checkpoint, and covers what takes effect rather than the nominal version
  — `flash_kda` is a locally built wheel whose `.so` can change while `0.0.1+1ce47ea` does
  not. Corollary: two sessions each measured half of this and neither half was the answer —
  the import check and the kernel launch are different claims about the same word
  "environment".
- **`uv sync` cannot rebuild this environment, and the blocker is one unresolvable
  source.** uv 0.11.21 is installed, `pyproject.toml` and a 231KB `uv.lock` are both
  present — and `/work/aupai/.venv` does not exist, so every package lives in system
  `dist-packages` and the lock describes an environment that has never trained anything.
  `flash_kda` appears in neither file, and its `direct_url.json` records
  `{"dir_info": {}, "url": "file:///tmp/flash-kda"}` — installed from a source tree in
  `/tmp` that is gone, with no index, no git URL, and a commit suffix (`1ce47ea`) whose
  repository is unrecorded. There is nothing for uv to resolve. What makes it resolvable:
  the recovered files *are* a wheel minus the zip (`RECORD`, `WHEEL` with
  `Tag: cp312-cp312-linux_x86_64`, `METADATA` all present), so re-packing them into a real
  `.whl` under a durable project root and pointing `[tool.uv.sources]` at it converts a
  package nobody remembered into a pinned artifact. **Do not let a uv venv inherit the
  image layer's torch to save the download** — that only moves the unrecorded state from
  the writable layer to the image layer, and the same incident returns when the image is
  rebuilt. The split that holds: `pyproject.toml` + `uv.lock` in git, vendored wheels on
  durable storage inside the project root, `.venv` disposable and rebuilt by one command.
- **A refit at equal budget is the decision; a comparison against a larger vocabulary is
  not.** Our 32K vocabulary, fitted on Chinese web before the objective changed, reads
  3.7073 code chars/token against Qwen2.5-Coder's 4.3806 — an 18% gap that was never a
  decision anyone faced, because that vocabulary is 151,665 slots, 4.6x the embedding
  parameters. The comparison that decides is our 32K against a *32K refitted with code*:
  3.7073 -> 3.8765, **4.6% of the code token budget recovered at zero parameter cost**
  (`facts/tokenizer.json`, aupai-3b). Against unfreeze condition 2 (the corpus distribution
  changed materially) the arithmetic is: 4.6% of code, which is a minority of a 30B mix,
  against invalidating every checkpoint including `ckpt_p324` and the six-point ladder.
  **Declined.** The 151K number stays as the ceiling it is: reaching it is a parameter-budget
  decision about the embedding table, not a tokenizer decision.
- **A pass with 0.2% of margin and a pass with 97% look identical in the log.** tilerl's
  gridY sweep, run across the `doc_cu_seqlens` fix, gives the cleanest measurement of the
  day — same card, same batch, one commit apart: batch 13 reads `grid=(2, 65400, 1)` before
  and `(2, 1718, 8)` after, a 38x drop, and gridZ moves off 1 because fla picks a normal
  layout once the document count falls. The part worth keeping is the *before*: 65400
  against a limit of 65535 is 135 of headroom. Anyone who had tried batch 13 first would
  have seen it work and written it into the run config, and the next batch of data would
  have killed it. **A near-limit success reports the same thing as a comfortable one, so a
  ceiling has to be measured rather than inferred from a green run.**
- **"I said where I measured it" is not "I said where I did not."** The grid finding came
  with a correct falsifiable prediction and a correct report — `--batch 8` passes, on GPU 2.
  The controller read that as a result and launched seven ranks; rank 2 died on the same
  error. The sentence that was missing is the one the reader needs: *not verified under
  DDP*. Naming the configuration you tested leaves the reader to notice the ones you did
  not, and under time pressure they will not. (tilerl, 2026-08-30.)
- **The controller made the recoverable-fraction error inside an hour of writing the rule
  against it.** "30% of every SFT step is padding" is the size of the unsupervised block,
  not what packing recovers: 29.9% of tokens carry `-100`, but only **11.8%** is trailing
  `<eos>` padding — the other ~18% is prompt, masked on purpose and needed as context.
  Same shape as short_conv 3.1% → 1.0%, one rule and one afternoon later. The ranking is
  unchanged (11.8% is still 12x short_conv, at no quality cost); the number was wrong.
- **The round's first positive generative result was contamination, and the guard that
  should have caught it ran and filtered nothing.** SFT zero-shot code-500 read
  83/500 = 16.6% against a 2δ instrument threshold of 12.5%, with base few-shot at 0.0% —
  a clean story about a format bridge. Then a direct check of the pack: **19 of the first
  20 `code_holdout_500` questions appear verbatim as exact token subsequences inside
  `data/sft/sft_all.pt`**, and the holdout was carved from `data/synthetic/code_python_zh.jsonl`,
  which is `SOURCES[7]` in `prepare_sft.py`. Questions and solutions both. The reading is void.
  The mechanism is three timestamps: `code_holdout_500.jsonl` carved 11:41:25,
  `sft_all.pt` packed 11:46:57, `holdout_hashes.txt` regenerated 13:18:44 — an hour and a
  half late. `prepare_sft.py:75` does call `is_holdout(q)` and `holdout.py:16` does list the
  file in `EVAL_FILES`; the filter excluded nothing because **the hash file it reads is
  itself a derived artifact and was stale at pack time.** The math holdouts were already in
  the set and this same probe finds them clean (0/15 each), which is what makes the
  diagnosis exact rather than a suspicion. Same failure class as `vocab_id`, `.srcfp` and
  `filters_fp` — a derived artifact staying valid-looking after its source changed — now
  with the eval holdout as the derived artifact. Two rules follow. **`is_holdout()` must
  raise on a stale fingerprint, never return False**: today "there is no guard" and "the
  guard says this question is clean" are indistinguishable to the caller. And **a
  contamination scan of the source files is not a scan of the pack** — the source scans
  that read 0 hit were correct about the sources, and the leak entered between them and
  the artifact that was trained on.
- **A jump that clean is a symptom, not a result.** 0.0% to 16.6% in one step, straight
  through the instrument threshold, was the shape of the evidence before anyone checked the
  pack. A base model reading exactly zero and a fine-tune clearing the bar on the first try
  is the signature of memorisation, and it deserved suspicion in proportion to how much it
  was wanted.
- **A check that no run's behaviour can turn green is not a gate, it is a fact.** The
  practical harm is not the red itself: a suite that is normally red gets `--force`d as
  routine, and that is exactly how a real red gets ignored. `env_fp_present` demonstrated
  it the hour it landed — permanently red against every pre-existing checkpoint, and it
  blocked a launch immediately. Rulings: `root_durable` becomes a WARN, because no run can
  fix `/work` being an emptyDir — with a revival condition in the comment, that it returns
  to FAIL once durable storage is mounted in the container and a root under `/work` is
  again a fixable violation. `ladder_config_frozen` gets an exemption list, because
  `ckpt_p02_a1_s2.pt` is the KDA-vs-attention A/B arm and is *supposed* to deviate; a check
  that reports a deliberate experiment arm as a violation teaches people to ignore it.
  **Every exemption carries `{checkpoint, deviating_flag, experiment_name}`, never a bare
  name** — an exemption that cannot state why it exists is worth the same as no check.
  Deriving it from the run's own `experiments.jsonl` row beats hand-writing it, for the
  same reason a hand-written package list omits exactly what nobody wrote down. And a new
  gate that must FAIL against existing artifacts ships **with** its ratchet, rather than
  landing red and being patched afterwards. (tilerl, 2026-08-30.)
- **A baseline about artifacts must be generated on the machine that holds them.** The
  first `env_fp` grandfather list was written from the local checkout, which has 3
  checkpoints; the pod has 28. Local PASS, pod entirely FAIL. This is the same rule as
  "broken worlds mutate a real artifact, never a hand-written one", seen from the other
  side: both are **an assertion about one world, written from inside another**.
- **The eighth instance happened to the person enforcing the rule, and it is the most
  instructive one.** The controller stated explicitly that the clean SFT rebuild would not
  be coupled to tilerl's unproven lookahead packing — one variable at a time. The rebuild
  then used the lookahead packer anyway: the new pack is 46,938 rows against the predicted
  46,898, and `prepare_sft.py` had been pushed to the pod between the launch and the run.
  Nobody edited anything wrongly; the artifact was simply built against a source that
  changed underneath it, with nothing to say so. Consequences were small — the old reading
  was void, so there was no difference to attribute, and the paired A/B had by then shown
  lookahead costs nothing measurable — but the mechanism was live. **An SFT pack carries
  only `vocab_id`. It should carry a fingerprint of the packer, its `SOURCES`, and the
  holdout hash file's own fingerprint**, exactly as shards carry `filters_fp` and scores
  now carry `input_fp`. Then "which packer built this" and "was the holdout guard live"
  are properties of the artifact rather than of who remembers.
- **`cards` in the frozen run config packs two different things into one string.**
  `"1,2,3,4,5,6,7"` is the allocation, and `NGPU` is `len(split(","))` — so someone dodging
  one busy card by dropping to six silently changes the *recipe*, and nothing raises. Card
  count changes the effective batch and the gradient noise; card indices change nothing,
  since the eight H20s are identical and any seven of them span the same NVSwitch domains.
  Ruling: **count is frozen, indices are allocation.** The fix is to split the field —
  `world` (the number, in `_FROZEN_KEYS`) and `cards` (the allocation, not frozen but
  asserted `len(cards) == world` before launch) — rather than to add an assertion on top
  of a field that carries two meanings. One field with two meanings is where the silence
  lives. The lane convention survives separately as an operational rule: GPU 0 is the
  bench/scoring lane, so training there puts rank 0 on the card most likely to be contended,
  and DDP is synchronous.
- **A commit message is a derived artifact, and it can go stale against its own diff.**
  `5e643cb` ("an `<eos>` run opens one document") also introduced a rewrite of
  `DeltaRecurrence.forward` from `nn.Conv1d` to four shifted multiply-adds: the controller
  committed another session's uncommitted working-tree change alongside its own fix and
  described only its own half. `131f420` ("fp32 master weights behind `--fp32_master`")
  reverted that rewrite 42 minutes later and described only *its* own half. Net effect
  zero, which is why nothing raised and why the owning session still believed its patch
  was an unapplied file in `scratchpad/` — a path that exists on neither machine. Two
  consequences, and the second is the one that costs: the patch is recoverable only from
  `git show 5e643cb -- train.py`, and **for 42 minutes the repo trained on a different
  code path than the ladder did**. `ckpt_sft_p324_v1.pt` (13:02) is inside that window;
  `_v2.pt` (13:47) is outside it. The six ladder points are all at 07:xx and therefore
  homogeneous, but that had to be *checked*, not assumed. Rule: **`git add -A` on a shared
  file is how someone else's change enters your commit unnamed.** Stage by path, and when
  a commit touches a file another session owns, name every hunk in the message or split
  the commit. The failing question is not "does the tree work" — it does — but "does the
  message describe the diff".
- **An idle card is not a free card.** The controller read `nvidia-smi`, saw GPU 0-5 at 0%,
  and released them to another session for a 7-card A/B. Those cards were the step gap
  inside a running `eval_all.sh`: math-hard had finished its seven shards and math-500 had
  not yet launched its own. 55 seconds later all seven were occupied again. A card's owner
  is the script still running, not the instantaneous utilisation row — the same confusion
  as the orphan process holding a card at 100% with no one able to account for it, read
  from the other side. Rule: **allocate on a pipeline's exit, not on a sample of its
  utilisation.** `eval_all.sh` reclaims its cards between every step, so the only safe
  release point is after the whole script exits.
- **The pre-registration goes in the ledger row, not in the conversation.** `harness run
  point` hard-coded the ledger `hypothesis` to the string "0830v1 budget point", so five
  runs measuring fp32 master weights would have recorded only which mix they used and
  nothing about what they tested. Conversations get compacted; `runs/experiments.jsonl`
  does not. A reading rule that lives only in a session's context is a reading rule that
  will be gone before the number is read. Fixed by `--hypothesis` (675e834), and the
  launch script now carries the full decision rule — the 0.24 nat threshold and the
  control-arm precondition — into the row itself.
- **A null is readable only if the control arm shows the defect it controls for.** The
  bf16 A/B measures whether fp32 master weights recover discarded updates; the discard
  rate was measured at ~91% on 3.24b. If the 0.2b control reads far below that, the point
  does not discard many updates in the first place and Δ≈0 means "not measured", not "no
  effect" — the same shape as a permanently-red CI being the same as no signal. Report the
  two frozen fractions (model bf16, optimizer) alongside the val NLL, never Δ alone.
- **"It only adds an output flag, so it cannot change the number" is a claim to verify,
  not an assumption.** `eval/l1_fewshot.py` existed in three versions at once: the process
  that produced L1 = 1/497 (started 12:56, wrote `preds_l1.jsonl`), the pod's copy, and
  HEAD. The number survived — the demo split, prompt construction, and `score()` are
  byte-identical across all three, and only `--out` and its JSON write differ — but that
  was established by diffing, after a peer's drift gate flagged the mismatch. Nobody
  noticed on their own, and the session that wrote the flag believed it had verified an
  integration whose code had never reached the machine it ran on. The drift gate that
  caught it walked the same rule from the other side: it compared `sha256`, found a
  mismatch, and reported "the pod is running something different" — true of the bytes,
  unestablished of the behaviour, since a lone `--out` flag and a rewritten prompt look
  identical to a hash. **A hash difference is not a behavioural difference, and a claim of
  zero net effect is zero only once someone has read the diff.** Both directions are the
  same rule as "read the source before trusting the benchmark": identity and behaviour are
  separate questions, and neither answers the other.
- **A ratio does not carry its denominator, so write the denominator into its name.**
  Three times on 2026-08-30 a ratio measured against one denominator was compared or
  multiplied against another. (1) Code fertility came back as 3.03 *characters per token*
  and was read against the 1.55 gate, which is *tokens per word* — opposite polarity, so
  "the tokenizer compresses code 2× worse" was exactly backwards, and it was one step from
  a rebuild that invalidates every checkpoint. (2) `0.329 tok/byte`, measured on decoded
  content, was multiplied by 260GB of *compressed parquet* to yield "85.7B tokens" — a
  product with no definition. (3) Earlier the same day, "30% of the SFT pack is padding"
  was the block size, not the recoverable fraction (11.8%). None of these is carelessness:
  a ratio written as a bare number has already lost the thing that makes it checkable.
  Rule: name ratios `tok_per_decoded_byte`, `chars_per_token`, `tokens_per_word` — never
  `fertility`, never `ratio`. And where an exact count is affordable, take it: decoding one
  whole file and counting tokens has no denominator at all, and here it was cheaper than
  the estimate it replaces.

## Review: what to look at, and who looks

Every delivery is reviewed by a session that did not write it. Self-test is the author's
evidence, not the review — the four defects below all shipped with a green self-test.
Review is not reading the diff for style; it is running the four checks that have actually
caught something in this repo, in this order:

1. **Does the fix introduce the inverse failure?** The most common shape. `score_matrix.py`
   lost every scored record when one checkpoint threw; the fix caught per-checkpoint and
   continued — and now the script prints "wrote 0 record(s)" and **exits 0** when every
   checkpoint fails. A crash was replaced by a silent success. Verified 2026-08-30 by
   running it against a nonexistent checkpoint, not by reading it.
2. **Does the exit code distinguish success from total failure?** A loop that skips every
   item must not return the same status as one that processed every item. Same rule as
   "a partial check that reports PASS is a defect", one level down.
3. **Does `broken()` mutate a real artifact?** A hand-written broken world shares the
   check's own assumptions; three of six checks were once dead while the self-test passed.
4. **Is every ratio's denominator in its name?** See the ratio rule above — three
   cross-denominator errors in one day, one of them a step from a tokenizer rebuild.

Pairing, so the reviewer is never the author: de ↔ e1 (harness and eval plumbing),
3b ↔ 44 (corpus numbers and measurement discipline), b0 ↔ 44 (panel readings), and the
controller reviews tilerl's A/B, which is the one result that can invalidate the ladder.
A review reports what it *ran*, not what it read.
- **A check can only ask what its own environment can answer.** `check_lane_respected` first
  tried to name the process on each training card: `nvidia-smi` reports **host** PIDs, the
  harness runs **inside the container**, and `ps` there resolves none of them — so every GPU
  process was skipped as "another container's" and the check returned PASS no matter what was
  on the block. It shipped with a green self-test and its PASS was quoted as evidence the
  cards were clean. The self-test was green because `HARNESS_GPU_PROCS` injects `(gpu, cmdline)`
  pairs directly, bypassing the one step that was broken — a hand-written world again, in the
  shape of an injection hook. `--query-compute-apps=process_name` is closed too (`[Not Found]`
  from inside). The fix was not a better PID mapping but **a different question**: occupancy is
  visible in the container, process identity is not, so the check asks "is a block card busy
  with no training process present" and states that ceiling explicitly. Its first real run
  FAILed on the controller's own scoring job sitting on GPU 0 — authorised, in the author's
  head, as "the step that unblocks the block", which is the same "idle is not free" error one
  turn after writing that rule down. The deadlock it exposed is the shape to remember: **the
  run that clears one red check turned another one red.**
- **Pre-registration protects against a post-hoc rationale, not against a bad input.** The
  bf16 A/B's experiment arm never cleared the model's gradients: with `--fp32_master` the
  optimizers hold the fp32 masters, so `opt.zero_grad()` cleared `m.grad` while the bf16
  `p.grad` accumulated across every step (`2.0 → 4.0 → 6.0` on a three-step reproduction).
  The arm was not measuring fp32 masters; it was measuring a running sum against a normal
  gradient. It would have produced a large improving Δ, which is exactly what the
  pre-registered rule promotes to "the six ladder points were measured on a defective
  optimizer, reopen the fit" — a real defect wearing the shape of the result we had agreed
  to believe. **A defect that blocks you is cheap; a defect that produces a plausible
  positive is the expensive kind, and the reading rule is what it exploits.** The review
  pass, not the pre-registration, is what caught it.
- **Two places in one file answering the same question differently: one of them is wrong.**
  The same training loop cleared *both* the optimizer's and the model's gradients on the
  rollback path (`train.py:1853-1855`) and only the optimizer's on the normal path. That
  asymmetry is visible without understanding either path, and it located the bug faster than
  reading the code did. Look for it first.
- **A PASS must be able to say how many things it verified, and that number cannot be zero.**
  Both of 2026-08-30's dead guards were the same shape, and it is not the empty-world shape:
  `score_matrix_present` went green locally because the ledger held **zero** ok runs, so its
  filter produced nothing to check; `lane_respected` went green always because `nvidia-smi`
  returns host PIDs the container cannot resolve, so **zero** processes were examined. An
  audit of all 35 checks against an empty repo found 34 correctly FAIL or SKIP — the class is
  not "no repo", it is "nothing matched". Two more still carry it: `corpus_filters_fp` and
  `score_input_fresh` both report PASS at `0 domain(s)`, which is right on a dev checkout and
  would be silently wrong on the pod the day `data/corpus/*` vanishes — exactly how the token
  cache vanished. Zero items must SKIP (the inputs are not on this machine) or FAIL (they are,
  and none matched); today those two cases give the same answer. `--selftest` asserts every
  check FAILs on a broken world and never asserts that a PASS examined anything, so the
  mechanical form of this rule is its missing half: **no check may return PASS with every
  count in its evidence at zero.** This is one level below the banner the harness already
  prints — that one covers "could not run", this one covers "ran and looked at nothing".
- **An injection that feeds the real code path is a legitimate broken world; one that
  replaces it is a hand-written world wearing a hook.** Audit of all 32 `broken()` builders,
  2026-08-30: 16 copy a real artifact, 13 write a small file whose surroundings are real, and
  3 inject through an environment variable. Two of the three are sound — `HARNESS_REQUIRE_EXTRA`
  adds a package name to the requirement list and the real read-then-import still runs;
  `_broken_mix_supply` copies the real 0.2b mix and builds genuinely undersized caches. The
  third, `HARNESS_GPU_PROCS`, supplied `(gpu, cmdline)` pairs directly and thereby skipped the
  one step that was broken — the host-PID-to-container-`ps` resolution — which is how a check
  that could never fail shipped with a green self-test. **The test to apply: after the
  injection, does the step that can break still execute?** Note the trap survives the fix:
  the rewritten occupancy check still self-tests through the same hook, so it is verified
  against reality only by the one manual run a person did. A regression in its real
  `nvidia-smi` read would still pass `--selftest`.
