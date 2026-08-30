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

**D comes from the log, not the mix.** Corpus supply measured 2026-08-30 is 3.619B tokens
(`mlm.corpus.tokens_total`). At the frozen weights with `epochs: 1`, the 3.24b point asks
for more than four domains hold:

| domain | supply | demand at 3.24b | headroom |
|---|---|---|---|
| textbook | 1.608B | 1.610B | −0.1%, inside the ±2% sampling error |
| web_hq | 1.432B | 1.051B | +36% |
| wiki | 0.231B | 0.246B | **−6.4%, the only deficit outside the error bar** |
| en | 0.158B | 0.161B | −1.8%, inside the error |
| math | 0.082B | 0.082B | 0.0%, at the line |
| code | 0.062B | 0.058B | +8% |
| chat | 0.044B | 0.038B | +15% |

`train.py:1363` caps at the epoch limit and prints `wants N rows, epoch cap leaves M ->
capped`, then prints the scheduled token count at line 1384. Nothing repeats silently, so
this is not a data defect and no weight changes. It is an x-axis defect: taking D from
`mix_scale_3.24b.json`'s `total_tokens` puts the largest point ~0.6% right of where it
trained, and the largest point has the most leverage over beta. The five smaller points do
not bind, so the error does not cancel across the curve. fit-protocol v1.9 reads D from
each run's scheduled-tokens line and pre-registers a tolerance above which the gap is a
declared curve boundary rather than rounding.

**The `en` domain is 85% Chinese.** Per-shard Han census, 400 docs per shard:
`cosmopedia_extra_000..006` is 690.3MB at 82.9–84.1% Han (chinese-cosmopedia, the same
source as `textbook`); `en_textbook_000..001` is 122.9MB at 0.0%. At the frozen weights
English is 0.75% of training tokens, not the declared 4.95%, and chinese-cosmopedia is
53.8%, not the 49.6% the mix `_comment` names as the figure a reader is warned not to
over-read. The curve is unaffected — all six points share the composition — so the fix is
the six `_comment` strings, not any weight (`mlm.corpus.en_domain_is_mostly_chinese`).

### What runs before the ladder

Tokenization is paid once. Then eight 0.2b runs, then the ladder. The eight are not a
detour: four of them are the ladder's own 0.2b point plus its seed replicates.

| runs | what | serves |
|---|---|---|
| 4 | current arch, `attn_every 4`, seeds 0-3 | ladder's 0.2b point (seed 0); the seed-variance measurement; lessons-e1's control arm; lessons-44's F arm |
| 4 | `attn_every 1`, seeds 0-3 | lessons-e1's KDA-vs-full-attention arm — **conditional, see below** |

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
| **harness-green** | `harness.py check` and `--selftest` both exit 0, on the pod | both exit codes | aupai-de | RED: `restartability` flags `scripts/_audit_anchor.py` and `algorithms/rl.py`. Neither is on the training path |
| **metric-panel** | the 200M resolution panel is committed and frozen | `docs/lessons/base_eval_panel.md` + `facts/base_eval.json` | lessons-b0 | OPEN, frozen |
| **panel-runners** | every panel metric has a runner that passes its known-answer gate | `eval/` runners + `be.known_answer_panel_3_4` | lessons-b0 | 5 of 6; LAMBADA-zh needs the held-out slice from the rebuilt corpus. The generative metric SKIPs by rule |
| **step-time-profile** | step time split by source, summing to ~100 | `facts/efficiency.json` | lessons-e1 | OPEN; roofline table landed, kernel line closed |
| **fit-protocol** | fitting method, accept thresholds, and falsification shapes frozen before the first point | `docs/lessons/scaling_fit_protocol.md` + `scripts/fit_scaling.py` | lessons-b0 | OPEN, frozen at v1.8 |
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
- No kernel proposal without a `torch.compile` baseline. AttnRes measured 42.2ms eager and
  3.85ms compiled for the same work — using the eager number would have made a
  zero-benefit kernel look like a 10x opportunity.
