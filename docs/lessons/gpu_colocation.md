---
question: "can a second workload share a training card productively, on our hardware and at our occupancy"
status: open
source: "e1-5 2026-09-01; arithmetic over eff.steady_state_composition (t57) and eff.kda_occupancy_bound; literature per row"
span_ms: 1676.63  # denominator for every percentage here, t57's own span; scripts/doc_numbers_check.py reads it
---

> **Status: open.** The verdict below is written to closure on the evidence
> available, so it can be acted on without waiting. It is `open` rather than
> `recorded` because one measurement could overturn it and has not been taken —
> tilerl-9's on-box interference run. What that measurement must show to overturn
> it is stated explicitly at the end, before it is run rather than after.

# GPU co-location

Research, 2026-09-01 (e1-5). Whether a second workload can usefully share a card
that is already running training, on H20 with a 200M model.

## The answer is arithmetic on our own trace

The card can be shared two ways and they have very different sizes. My first
version of this section counted only the first and called it the answer; fb
caught it, and a reviewer reading 1.3% as the whole ceiling would have concluded
co-location is pointless for the wrong reason.

**Temporally** — a co-tenant running in the gaps between the primary's kernels.
**Spatially** — a co-tenant's blocks occupying SM slots that a *running*
low-occupancy kernel leaves empty. MPS and multi-stream both do the second.

From `eff.steady_state_composition` — tilerl's t57 steady-state trace, profiler
window steps 56–76, one card, stage-1 shapes:

| | ms/step |
|---|---|
| busy | 1600.25 |
| span | 1676.63 |
| **idle** | **76.38 (4.56%)** |

Of that 76 ms, **54.9 ms — 72% of all steady idle — is the `rms_norm → flash`
seam**, and the contents of those gaps are `compile_attempt_1` at 78–80 ms and
`build_guards` at 38–50 ms. That is *CPU* time. The GPU is idle there because the
host is busy compiling, not because SMs are available to anyone else. Those gaps
are the primary stalled on its own Python thread and about to resume; they are not
schedulable windows.

Subtract them and the temporal term is **21.5 ms/step, 1.28% of the step**.

### The spatial term is eight times larger

The KDA gated-delta window is 174 ms/step (10.4%) at **6–13% achieved
occupancy**, so 87–94% of SM slots sit empty *while the card is busy*. That is
fillable capacity a temporal accounting misses entirely:

| term | ms/step | % of step |
|---|---|---|
| temporal (gaps, net of compile stalls) | 21.5 | 1.3 |
| spatial (SM slots idle during the KDA window) | 151–164 | 9.0–9.8 |
| **ceiling, before bandwidth contention** | | **10.3–11.0** |

**That ceiling is arithmetic, and the one directly-relevant measurement says it is
unreachable.** (Recorded as `eff.low_occupancy_is_not_free_capacity`; our own
occupancy is `eff.kda_occupancy_bound`.) Elvinger et al., SoCC '25 ([arXiv:2501.16909](https://arxiv.org/abs/2501.16909),
H100, Nsight Compute instrumented) ran **two kernels at 6.25% achieved occupancy
each**, in separate streams, with every SM available — the regime our KDA sits in,
on Hopper-class silicon. Result: **1.73× latency increase each.** Naive occupancy
math predicts no interference at all.

Their diagnosis is that occupancy is the wrong predictor. The binding constraints
are IPC and warp-scheduler saturation, register and shared-memory capacity at the
block scheduler, and L2/HBM bandwidth — none of which appears in an occupancy
number. Three specifics that map onto us:

- A compute-bound kernel paired with a memory-bound one — the textbook
  "complementary" case — **doubled** the copy kernel's time. Complementary roofline
  profiles do not imply complementary hardware profiles.
- **Head-of-line blocking despite streams**: a decode kernel needing 64512
  registers/block found only 63288 free per SM and the block scheduler serialised
  it entirely. Our worst KDA kernel is at 254 registers/thread, near the 255
  hardware ceiling, which is precisely the footprint that triggers this.
- Even Green Contexts with strict SM isolation showed **up to 1.3× slowdown** from
  L2/HBM contention alone. Disjoint SMs do not buy isolation for bandwidth-bound
  work.

So the honest reading of the spatial term is: 9.0–9.8% is what the empty slots
would be worth if slots were the constraint, and the measured evidence says they
are not. tilerl's micro-probe is what settles the real figure.

Memory is a red herring throughout. The training job holds 52.7 of 96 GB, so 43 GB
is free, and none of it helps: the constraint is SM time and bandwidth, not capacity.

### "95.44% busy" does not mean the SMs are 95% used

Worth stating because the number invites the wrong reading, including from me.
Our busy figure is GPU kernel wall time over step span — a *residency* measure, the
`PROF_GR_ENGINE_ACTIVE` family, which NVIDIA's DCGM documentation defines as "the
fraction of time **any portion** of the graphics or compute engines were active".
It says a kernel was resident, not that the machine was working.

The gap between the two is measured and it is large. Acme
([arXiv:2403.07648](https://arxiv.org/abs/2403.07648), 4,704 A100s, six-month
production trace) reports, on the same clusters in adjacent sections: median
nvidia-smi GPU utilization **97–99%**, median DCGM `PROF_SM_ACTIVE` **~40%**. The
paper flags its own coarse metric — "'GPU utilization' may sometimes be a weak
utilization indicator" — citing nvidia-smi as the reference.

So a 95.44%-busy card is not a card with no room. It is a card whose engines are
always resident, which is exactly why the spatial term above is computed from KDA's
**6–13% achieved occupancy** and not from the busy figure. Two different questions,
and only the occupancy one bears on whether a co-tenant fits.

**A consequence worth stating before it surprises someone**, and it bears on the
temporal term only. `eff.seam_dynamo_disable` eliminates the flash recompiles,
70 → 0. It is not a proposal: verified in the live job — `train.py:143` on the pod
carries it as of 06:25:20, and the running stage-2 process started 08:02:22, so the
job measured today already has it. **The 76 ms idle figure therefore comes from a
t57 trace taken before the fix, and the live temporal term is smaller than the
table above.** With it, most of the 54.9 ms of
compile gap disappears and steady idle falls from 76 ms toward 21 ms — so the
temporal term shrinks and a *gap-based* interference measurement taken before the
fix would overstate that term by roughly 3.5×. Measure after the fix.

The spatial term is untouched by this: KDA's occupancy does not change because the
host stopped recompiling. That asymmetry is the reason both terms have to be
carried separately rather than summed into one "idle" number.

## Sharing the card versus fixing the kernel

The reason this question came up is the KDA gated-delta backward at 6–12%
occupancy. It is worth putting the two responses side by side, because they are
not close.

`eff.kda_occupancy_bound`: the four dominant KDA kernels run at 6–12% occupancy
and reach 15–29% of HBM peak, while **every kernel above 80% occupancy on the same
card runs at 85–93% of HBM peak**. Grid size is ample (8192–131072 blocks on 78
SMs); the binding constraint is registers and shared memory per block leaving ~4–8
resident warps with nothing to hide memory latency. Latency-bound, not
bandwidth-bound and not grid-bound.

| response | ceiling | who benefits |
|---|---|---|
| raise KDA to the 85% of HBM peak its healthy neighbours achieve | 174 ms → 59 ms, **saving 115 ms = 6.86% of the step** | the primary |
| fill the same window with a second workload | up to **9.0–9.8% of the step** of slot-time, minus whatever bandwidth contention removes | the co-tenant |

**The two options compete for the same 174 ms.** That is the sharp version of the
argument, and it is sharper than the one I first wrote. A retune does not just
return more than co-location collects — it *destroys the thing co-location would
have used*. A KDA kernel at 85% occupancy has no empty slots to rent. So these are
not two independent levers to weigh; they are mutually exclusive uses of one
window, and only one of them makes the primary faster.

The order follows: retune first, and only ask about co-location afterwards, against
whatever slack the retune leaves. Doing it the other way builds a dependency on the
inefficiency we are trying to remove.

A co-tenant also never makes the primary faster. It makes the primary slower by
whatever it consumes.

### H20's roofline inverts the intuition, and I had it backwards

I have been describing H20 as "bandwidth-rich, FLOP-poor" and drawing the wrong
inference from it. The specs are right; the conclusion was not.

| part | bf16 dense | HBM | **ridge point** |
|---|---|---|---|
| H20 | 148.0 TF | 4.00 TB/s | **37 FLOP/byte** |
| H100 SXM | 989.5 TF | 3.35 TB/s | 295 |
| H200 SXM | 989.5 TF | 4.80 TB/s | 206 |

The ridge point is the arithmetic intensity above which a kernel is compute-bound.
Dropping from 295 to 37 means **more** operations cross that threshold on H20, so
H20 is *more* compute-bound than H100, not less. The scarce resource here is the
tensor pipe; bandwidth is the one in surplus — 1.2× H100's bandwidth feeding 0.15×
its FLOPs.

That sharpens the verdict rather than softening it. The only tenant that could fit
is FLOP-light and bandwidth-hungry, and **a second training job is the worst
possible pairing** — it wants exactly the resource H20 has least of. Our own KDA
is consistent with the surplus: it runs at 1.16 TB/s of 4 TB/s, 29% of peak.

(Use dense-vs-dense figures. H100's 1979 TF is the sparsity number; comparing it
against H20's dense 148 doubles the apparent gap. NVIDIA publishes no H20
datasheet, so the 148/4.0/78-SM figures come from on-chip telemetry corroborated
by independent configs, and sparse bf16 and TDP are not reported.)

## Intra-process concurrency: the one lever that is real is not concurrency

The brief asked whether stream concurrency could fill KDA's idle SMs. It cannot,
and chasing it would miss a one-line change sitting underneath.

| path | what blocks it |
|---|---|
| CUDA streams | the register file is the constraint; a co-resident block cannot be allocated. Physical. |
| Green contexts | usable today (`torch.cuda.green_contexts`, UMD 12.8+) but Hopper's granularity is 8 SMs and **78 is not a multiple of 8** — any split strands SMs. NVIDIA's own guide says the purpose is latency, not throughput, and that "concurrent execution of independent GPU work is not guaranteed". Measured *regressions* exist on Jetson. |
| Programmatic Dependent Launch | reachable — Triton has `launch_pdl=True` since PR #6394 — but it hides a fixed per-boundary bubble. Measured end-to-end: +3.0% (TRT-LLM, B200), +2.2–2.9% (GB200), **0.8%** (SGLang), and noise at batch >16. Our grids are 26–840 waves, so the tail is 0.06–1.9%. `fla` has zero `gdc_wait` call sites. |

**The finding, verified on our own install rather than read from the survey:**

```python
# fla/ops/kda/chunk_bwd.py:27, on the pod
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]
```

`IS_NVIDIA_HOPPER` is `True` on our H20 (checked: device "NVIDIA H20", capability
(9,0)). So **the autotuner never tries 8 warps** on the kernels that are
occupancy-starved — a cap written for H100 that H20 inherits by identifying as
Hopper, on a part with a very different FLOP:bandwidth ratio.

That is the retune's first move: one line, directly testable, and the autotuner
does the rest.

**Why occupancy is the wrong target and bytes-in-flight is the right one.** Volkov
(GTC 2010) measured **87% of pin bandwidth at 8% occupancy** — higher than
`cudaMemcpy`'s 71% — by giving each thread 8–14 in-flight `float4`s. Same occupancy
as ours, three times our bandwidth. The difference is not warps, it is bytes in
flight, and ours are locked out by the register file at 254/255. That makes the
diagnosis *harder*, not softer: the machine demonstrably delivers ~87% at 8%
occupancy, so our 29% is a register-budget problem, not an occupancy law.

Which is why `maxnreg` is a poor second move and I have withdrawn a worse one:

| lever | verdict |
|---|---|
| unlock `num_warps=8` | first; one line; the cap is inherited, not chosen for H20 |
| move in-flight bytes out of registers (cp.async / TMA staging) | the right direction — it attacks load destinations that should never have been in registers, unlike `maxnreg` which attacks the fp32 state accumulators that must be |
| ~~step `num_stages` down~~ | **withdrawn.** `num_stages` is also the cp.async multi-buffer depth, so lowering it frees registers *and* reduces in-flight bytes — the exact quantity we are short of. A direct TLP-vs-ILP bet, not a free win |
| `maxnreg` to force occupancy | high downside. 254→232 does not buy a warp; reaching 2× needs ≤128, which is spill territory |

The measured record is one-sided on that last row. Forcing registers down to raise
occupancy has **no published win**: Triton layer-norm on B200 went 57–64 µs → 90 µs
on an occupancy cliff with *zero spills either side*; FA2 varlen measured 19.6×
slower at 32 registers versus 255-with-no-spill. The one clean gain from `maxnreg`
(Triton PR #9248, H200, 2290→2640 GB/s, +15%) succeeded specifically because it
did not induce spilling.

Three numbers to read before changing anything: `kernel.n_spills` at 254 (are we
already spilling?), whether the occupancy limiter is registers or shared memory,
and whether stalls are `long_scoreboard` (memory-latency starved, more warps would
help) or `lg_throttle`/`mio_throttle` (already saturated, they would not).

And a caution against treating the cap as an oversight: `fla` carries **no comment,
issue or commit explaining why Hopper is capped** — I grepped the installed package.
It may be an H100 measurement or a workaround for a compiler bug. Measure before
changing it, and keep the old value as the control.

One honest caveat on the concurrency question:
`chunk_gated_delta_rule_bwd_kernel_dhu` has grids as low as 32–128 blocks against
78 SMs, so it genuinely is under-filled. It is also the kernel holding fp32 state
accumulators and carrying the most register pressure — both properties come from
the same design choice, which is why the fix is the register budget and not a
second stream.

## A second PROCESS cannot backfill anything

This is the finding that settles the question, and it is architectural rather
than empirical. NVIDIA's own MPS documentation
([docs.nvidia.com/deploy/mps/architecture.html](https://docs.nvidia.com/deploy/mps/architecture.html),
verified directly, Background):

> The GPU also has a time sliced scheduler to schedule work from work queues
> belonging to different CUDA contexts. **Work launched to the compute engine from
> work queues belonging to different CUDA contexts cannot execute concurrently.**

Without MPS, a second process does not fill our idle — each process gets "a
serially scheduled time-slice on the whole GPU". So the temporal and spatial terms
above are both unavailable to a plain second process. They are only reachable
inside one CUDA context (streams) or through MPS, which merges clients into a
shared scheduling domain.

MPS carries two costs the same page documents. Fault isolation is partial: on
Volta+ the server recovers once faulting clients disconnect, but in shared-server
mode "a fatal fault from one client may bring down a different user's client that
shares any GPU with the faulting client". And the page documents **no memory
partitioning** — only per-client address spaces and "limited execution resource
provisioning for Quality of Service".

Measured co-location slowdowns, second workload on a busy GPU:

| source | primary degradation | context |
|---|---|---|
| TGS, NSDI '23 | **43%** | ResNet-50 + ShuffleNet, A100 |
| AntMan, OSDI '20 | **5.2×** | ESPnet + ResNet-50, V100, Alibaba production |
| Zico, ATC '21 | **8×** | MPS under memory pressure |
| MIG (measured in TGS) | **23%** | even hardware partitioning is not free |
| Gandiva, OSDI '18 | **−13%** | the best case, and only for an already 94%-utilized job |

### MIG is disqualified by one line, not by its overhead

MIG does give real isolation — the user guide describes "separate and isolated
paths through the entire memory system", L2 banks and DRAM busses assigned per
instance, which is exactly what MPS lacks. And H20 supports it: the supported-GPU
table lists H20 / GH100 / 96GB / 7 instances, and the driver source gives it the
same partition geometry as H100 96GB.

None of that matters here, because the same guide says:

> **NCCL is currently not supported with MIG.**

No NCCL means no DDP, no FSDP, no TP or PP. A 7-rank data-parallel job cannot run
across MIG instances at all. The partitioning overhead — measured at 6–12% on
Ampere, and *negative* on A30 where slicing recovered utilisation a single job
could not use — is beside the point.

Two operational notes if MIG is ever considered for single-card work: instances
cannot be destroyed while a process is using them (`NVML_ERROR_IN_USE`, and a
monitoring process counts), and on Hopper **MIG mode does not survive a reboot or
a driver reload**, unlike Ampere where it is sticky.

## What the frameworks actually do

Every major framework co-locates evaluation **in the same process, sequentially
between steps**, and none runs a second process on a training card:

| framework | mechanism |
|---|---|
| Megatron-Core | `--eval-interval` / `--eval-iters`, inline `evaluate_and_print_results`, training stalled |
| TorchTitan | `validator.enable` / `freq`, inline `validate()` |
| NeMo | `val_check_interval`, Lightning validation loop, blocking |
| DeepSpeed | no built-in loop; Megatron-DeepSpeed uses the Megatron flags |

For anything beyond loss, all of them recommend checkpoint-then-evaluate
elsewhere. TorchTitan says so explicitly, warning that installing `lm-eval` may
break the training environment. NVIDIA sets `limit_val_batches = 0` in its own
performance benchmarking scripts.

**The RLHF frameworks are the interesting case and they answer it unanimously.**
veRL, OpenRLHF, TRL, NeMo-RL and Megatron-LM's RL path all "colocate" generation
with training — by *time-sharing*, never concurrently. HybridFlow
([arXiv:2409.19256](https://arxiv.org/abs/2409.19256) §2.3) states the reason:
models on the same GPUs "are executed sequentially in a time-sharing manner, as
out-of-memory error may easily happen if colocated LLMs execute concurrently".
NeMo-RL's API names it: `blocks_training()` is true when generation shares GPUs,
and "engines on dedicated GPUs never block training".

They pay 0.2–3 s per swap to *avoid* co-residency. That cost is roughly fixed per
transition, so it amortises worst at short steps — ours are 1.68 s.

## Ranked verdict

**Do not co-locate a second workload on a training card.** Ranked by what each
option returns and to whom:

| # | option | returns | to whom | ship cost |
|---|---|---|---|---|
| 1 | **unlock `num_warps=8` for KDA on H20** — `fla/ops/kda/chunk_bwd.py:27` caps it at [2,4] for Hopper, and H20 inherits that | unknown until measured; it is the gate in front of the 6.8% | the primary | one line, autotuner does the rest, no numerics change |
| 2 | rest of the retune: `num_stages` down, then `maxnreg`, checking `n_spills` each step | up to 6.8% of the step | the primary | a tuning pass |
| 3 | **keep evals on the lane card** — what we already do | the full lane, uncontended | evals | none, it is the status quo |
| 4 | **in-process eval between steps**, if a metric ever needs the training weights live | ~1% at a sane interval on a 200M model | evals | small; this is what every framework does |
| — | second process on a training card, no MPS | **nothing** — different CUDA contexts time-slice the whole GPU | neither | — |
| — | second process under MPS | a slice of the same 174 ms window | the co-tenant, primary pays | partial fault isolation, no memory partitioning |
| — | MIG | **NCCL is not supported with MIG** — no DDP, no FSDP. Ends the option for a 7-rank job outright | neither | — |

The ranking is not close, and three independent lines of evidence agree:

1. **Architecture.** A second process cannot backfill idle SMs at all — different
   CUDA contexts time-slice. Reaching the idle requires MPS or one process.
2. **Measurement at our occupancy.** Two kernels at 6.25% occupancy each still slow
   each other 1.73×. Empty slots are not free capacity.
3. **Arithmetic on our own trace.** The retune and co-location consume the same
   174 ms, and only the retune makes the primary faster.

**The order matters more than the sizes.** A retune to 85% occupancy destroys the
capacity co-location would have used. So retune first and re-ask afterwards
against whatever slack remains — doing it the other way builds a dependency on the
inefficiency we are trying to remove, and a co-location plan that pays off is one
that quietly needs KDA to stay slow.

## What would overturn this, stated before the measurement

The verdict rests on two facts and one piece of arithmetic. Each has a specific
result that would break it, and they are written here so the reading rule exists
before tilerl-9's numbers do.

| the claim | what overturns it |
|---|---|
| **SoCC '25**: two kernels at 6.25% occupancy each slow one another **1.73×** | **the named threshold (3b, before the run): a co-tenant at ~6% occupancy slows the primary by less than 1.4×.** Below that, the published interference does not transfer to H20's roofline, the 9.0–9.8% spatial ceiling is partly real, and "strict retune-first" weakens to "measure, then decide". |
| **`eff.kda_occupancy_bound`**: KDA runs at **6–13% occupancy**, 15–29% of HBM peak, while kernels above 80% occupancy reach 85–93% | the retune returns **well under 1%** of step time. Then the 174 ms window is not recoverable by tuning, and renting it stops competing with fixing it. |
| **arithmetic**: retune and co-location consume the same 174 ms | a co-tenant that measurably does **not** slow the primary — which would mean it is running in genuinely dead time the trace cannot see, and the two are not competing. |

**Clearing 1.4× does not flip the verdict, it only softens row 1.** Rows 1 and 3
fall to different kinds of evidence and a reader stopping at the first will
overread it (tilerl-0a, before the run). Row 3 is arithmetic: the retune and a
co-tenant consume the same 174 ms window, so *whatever* the interference factor
turns out to be — 1.73×, 1.4×, even 1.2× — fixing KDA still destroys the capacity
co-location would rent. A measurement cannot refute that; only a different window
could. So the most a sub-1.4× result buys is "measure, then decide", never
"co-locate".

**The two constraints are not the same knob.** 3b's threshold sizes the co-tenant
by *occupancy* (~6%, matching SoCC's design so transferability is the question
being asked). My execution clause sizes it by *step-time share* (~9%, which is the
"is it worth renting" question). One guest cannot satisfy both: how much step time
a 6%-occupancy guest consumes is set by its own arithmetic intensity, not by a dial
tilerl-9 can turn. Resolution, theirs and correct: size by occupancy, and record
the step-time share it actually takes. One run then yields both the falsifiable
1.4× criterion and the exchange rate, and if the two land far apart both numbers go
in the doc rather than one being chosen.

**What the step-time share will mean, predicted before the run** (tilerl-0a).
The gap between the occupancy the guest is sized to and the step time it turns
out to consume is a finding in itself, and the two directions point opposite ways:

| the guest at ~6% occupancy eats | it is | and row 3 is |
|---|---|---|
| **~3% of step time** | FLOP-light, bandwidth-hungry | **strengthened** — KDA is itself bandwidth-bound at 15–29% of HBM peak, so guest and host contend for the same resource |
| **15%+ of step time** | FLOP-heavy | **challenged for the first time** — it wants something other than what KDA is starved of, so "the same 174 ms window" stops being obviously true |

Recording this now matters more than it looks. Without it, either outcome could
be narrated afterwards as consistent with the verdict, and the table would be
decoration. This is the one branch where a result genuinely threatens row 3, and
it is named before the number exists.

**The single measurement that decides it**: the primary's tok/s with and without a
co-tenant, same seed, same step window, on the current tree. Not the co-tenant's
throughput — that always looks like a free lunch, because whatever it achieves it
mostly took from the primary.

Two conditions on that run, both of which change the answer if ignored:

1. **Take it on the current tree.** The seam fix (`torch._dynamo.disable` on
   `flash_attn_varlen_func`) is live in the running job. A pre-fix measurement
   overstates the temporal room by roughly 3.5×.
2. **Measure the primary, and measure per unit of co-tenant work.** A co-tenant
   throttled to near-nothing will show near-nothing; the quantity that matters is
   the exchange rate.

**Read 1.4× as a test of the literature, not as a price worth paying.** At 1.4× the
primary keeps 71.4% of its throughput — it is giving up 28.6% of tok/s to host a
guest. The threshold asks whether interference on H20 is *below what SoCC
measured on H100*, which is a question about transferability. It does not ask
whether co-location is cheap, and a result of 1.4× would not by itself justify
doing it. For reference: 1.0× is no interference, 1.2× keeps 83.3%, 1.73× keeps
57.8%.

**Who decides what, agreed before the run.** tilerl-0a runs the measurement and
reports the row, the numbers and the conditions — not a conclusion, and not a
recommendation about which row to move. e1 decides what the doc says and names
the row that moved rather than editing the conclusion quietly. The split exists
because whoever holds both "what to measure" and "what the number means" makes
the pre-registered table above worthless.

If the measurement comes back and the verdict stands, this document becomes
`recorded` with the number in it. If it overturns any row above, the row says so
and the verdict changes — which is the point of writing the condition down first.

## Ceilings

- The 1.3% is arithmetic over one steady-state trace on one card, not an
  interference measurement. tilerl is running the real thing; where their numbers
  and this arithmetic disagree, theirs win.
- The compile-gap subtraction assumes no scheduler can use a gap the primary is
  about to reclaim. That is the right default for a 150 ms host stall, but it is an
  assumption, not a measurement.
- Everything here is the *training* card. The lane card is a separate question and
  is already answered: it is free and we use it.
- The 7-rank data-parallel case has a risk no single-GPU measurement shows: a
  co-tenant that slows one rank delays the collective and stalls all seven.
  MegaScale measured 0.5% of machines running ~10% slow and recovered 0.7% MFU by
  removing them. Joining that to co-location is an inference, not a measurement,
  and is labelled as one.
- No source I found measures the full chain "foreign co-tenant steals SMs → NCCL
  collective delayed → all DP ranks stall". That gap is real and worth knowing
  before anyone treats a single-card probe as settling the multi-rank question.
