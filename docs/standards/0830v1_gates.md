# 0830v1 gates

The controller session (aupai-fb) drives these in order. A gate opens only when its
evidence exists as an artifact — not when someone reports it done. Update the status
column in place; do not add a second copy of this table anywhere.

Architecture is fixed for this round: KDA (9 layers) + full causal gated MLA (3 layers,
latent=d/4) + AttnRes Full (blocks=0), NoPE throughout. Commit b3cad87. Changing it
reopens G3.

Run config for all six budget points: `--batch 16 --accum 2`. AttnRes Full OOMs at
batch 32 on 96GB — the accumulation loop at `train.py:444-446` leaves one [B,T,D]
temporary per source per sublayer to the backward, 25x24 of them, which the old
architecture never had. Gradient accumulation keeps the effective batch at 32, so the
optimizer recipe tuned for batch 32 still applies and the six points stay comparable.
Block AttnRes and grad_ckpt were both rejected: the first changes the architecture under
test, the second measured 2.4x slower at batch 72 on 2026-08-27.

| gate | opens when | evidence | owner | status |
|---|---|---|---|---|
| G0 harness green | `scripts/harness.py check` exits 0 AND `--selftest` exits 0 | both exit codes | aupai-de | GREEN 2026-08-30 (73cef7b, 16/16 selftest) |
| G1 metric panel | the 200M resolution panel is committed and frozen | `docs/lessons/base_eval_at_200m.md` + 12 `facts/base_eval.json` entries | lessons-b0 | GREEN 2026-08-30, frozen; reporting the panel is now mandatory per run |
| G1b zh minimal pairs | the eval set built to `be.minimal_pair_rules`, n>=277 | the set + its build script | aupai-3b | open; needed before the panel's minimal-pair row can be read, not before G3 |
| G2 profile | step time split by source, percentages summing to ~100 | `facts/efficiency.json` | lessons-e1 | open work; also selects the kernel target |
| G3 first run | 0.2b budget point trained on the new arch | `ckpt_*`, one `experiments.jsonl` row, one `score_matrix.jsonl` row | aupai-fb | waits on the pod cleanup report and on G2 |
| G4 scaling curve | all six `mix_scale_*` points scored | six score-matrix rows, fitted E + B/D^beta with residuals | aupai-fb | waits on G3 |

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
