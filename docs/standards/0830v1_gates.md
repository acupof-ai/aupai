# 0830v1 gates

The controller session (aupai-fb) drives these in order. A gate opens only when its
evidence exists as an artifact — not when someone reports it done. Update the status
column in place; do not add a second copy of this table anywhere.

Architecture is fixed for this round: KDA (9 layers) + full causal gated MLA (3 layers,
latent=d/4) + AttnRes Full (blocks=0), NoPE throughout. Commit b3cad87. Changing it
reopens G3.

## Run config for all six budget points

Settled 2026-08-30. Every point uses the same values; a change to any of them reopens the
whole ladder.

| setting | value | why |
|---|---|---|
| cards | 7 (GPU 1-7) | GPU 0 is the bench/scoring lane |
| batch / accum | 16 / 2 | effective 32. AttnRes Full OOMs at batch 32 on 96GB; accumulation keeps the optimizer recipe valid. Block AttnRes changes the architecture under test; grad_ckpt measured 2.4x slower at batch 72 |
| `vocab` | 32776 | 32773 is 2-byte aligned, so cuBLAS falls back to an SM75 align-1 kernel at 41% of bf16 peak. Padding to a multiple of 8 reaches 92%: +14-16% end to end, tokenizer untouched |
| `chunk_size` | 32 | fla KDA default 64; halved chunks measured 19.1% faster on the kernel, numerically neutral (max delta 0.0017 on loss ~8) |
| `bucket_cap_mb` | 50 | 100 leaves DDP communication unhidden. 50 and 25 tie at 75K tok/s/gpu on both 3 and 7 cards, so 50 wins on fewer allreduces |
| NCCL protocol | default | forcing `PROTO=Simple` adds 2K tok/s but depends on an env var that a launch can silently omit |
| warmup | 1% of steps, floor 2 | `max(20, 1%)` left the 0.2b point at 9.2% and the 3.24b point at 1.0% — a 9.2x systematic difference that would read as monotonic drift in the fit residuals |

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

## Gates

| gate | opens when | evidence | owner | status |
|---|---|---|---|---|
| G0 harness green | `harness.py check` and `--selftest` both exit 0, on the pod | both exit codes | aupai-de | GREEN; pod 6/7, only `web_hq` red by design |
| G1 metric panel | the 200M resolution panel is committed and frozen | `docs/lessons/base_eval_panel.md` + `facts/base_eval.json` | lessons-b0 | GREEN, frozen |
| G1b panel runners | every panel metric has a runner that passes its known-answer gate | `eval/` runners + `be.known_answer_panel_3_4` | lessons-b0 | 5 of 6; #3 LAMBADA-zh needs the held-out slice from the rebuilt corpus. #6 generative SKIPs by rule |
| G2 profile | step time split by source, summing to ~100 | `facts/efficiency.json` | lessons-e1 | GREEN; roofline table landed, kernel line closed |
| G2b fit protocol | fitting method, accept thresholds, and falsification shapes frozen before the first point | `docs/lessons/scaling_fit_protocol.md` v1.1 + `scripts/fit_scaling.py` | lessons-b0 | GREEN, frozen |
| G3 corpus | `web_hq` rebuilt from fineweb2, holdout excluded, fingerprint stamped | PROVENANCE fetch/build/result block + `corpus_fp_matches` green | aupai-3b | in progress, 36-way build |
| G3b warmup | 2-step warmup does not destabilise the 0.2b point | smoke vs `warmup=20` control, same seed | aupai-de | running |
| G4 six points | all six `mix_scale_*` points trained and scored | six score-matrix rows + six experiments rows | aupai-fb | blocked on G3, G3b |
| G5 scaling curve | the fit runs and its verdict is recorded | `fit_scaling.py` output with RMS and the beta profile interval | aupai-fb | blocked on G4 |

Kernel work (tilerl-bench-harness-plan) runs alongside and gates on G2 for target
selection, not on G0. It lands only through the five correctness gates in its brief:
fp64 reference no worse than the kernel it replaces, bit-exact reruns, 200-step
step-by-step loss comparison, `test_arch_compat` coverage, and a working fallback.

## Who does what

| session | owns | does not do |
|---|---|---|
| aupai-fb | controller: GPU allocation, gate rulings, launching runs, the experiment record, and the reasoning about what each result means | writes no code |
| aupai-de | harness, CI gates, doc deletion, code cleanup, pod hygiene | training runs, research |
| aupai-3b | corpus build, filters, quality head, eval-set construction | kernels, harness |
| tilerl-bench-harness-plan | kernels and their benchmarks, written in this repo under `scripts/` | changing the architecture's math |
| lessons-b0 | research: what has resolution at 200M | writing repo code |
| lessons-e1 | research: where the step time goes | writing repo code |
| lessons-44 | research: filter transfer | writing repo code |

## GPU allocation

aupai-fb allocates all 8 cards. Nobody starts a GPU process without asking; kill only by
exact PID, never `pkill -f`. Current plan, in order:

- **Phase A (now)**: GPU 0-7 free. lessons-e1 profiles the real 7-card DDP config for G2;
  tilerl runs `attn_res_bench.py --full` and `bench_gated_mla.py --full` on GPU 0.
  Both are minutes, and the profile must precede the long run — a 5-hour run started
  before we know where its time goes cannot be re-decided afterwards.
- **Phase B**: GPU 1-7 run the 0.2b pretrain. GPU 0 goes to aupai-3b for quality-head
  scoring.

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
- A number the controller derived rather than measured is labelled "hypothesis, do not
  schedule work against it" and carries its falsification test. On 2026-08-30 three
  unmeasured AttnRes attributions from this session were each overturned by measurement,
  and two sessions had already scheduled work against them. The controller's output is
  who measures what and which reading counts, not an estimate.
- No kernel proposal without a `torch.compile` baseline. AttnRes measured 42.2ms eager and
  3.85ms compiled for the same work — using the eager number would have made a
  zero-benefit kernel look like a 10x opportunity.
