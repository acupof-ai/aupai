# 0830v1 gates

The controller session (aupai-fb) drives these in order. A gate opens only when its
evidence exists as an artifact — not when someone reports it done. Update the status
column in place; do not add a second copy of this table anywhere.

Architecture is fixed for this round: KDA (9 layers) + full causal gated MLA (3 layers,
latent=d/4) + AttnRes Full (blocks=0), NoPE throughout. Commit b3cad87. Changing it
reopens corpus-ready.

## Run config for all six budget points

Settled 2026-08-30. Every point uses the same values; a change to any of them reopens the
whole ladder.

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
