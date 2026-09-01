---
question: "can a second workload share a training card productively, on our hardware and at our occupancy"
status: recorded
source: "e1-5 2026-09-01; arithmetic over eff.steady_state_composition (t57) and eff.kda_occupancy_bound; literature per row; tilerl's on-box interference measurement cited where it lands"
---

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
unreachable.** Elvinger et al., SoCC '25 ([arXiv:2501.16909](https://arxiv.org/abs/2501.16909),
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
| raise KDA to the 85% of HBM peak its healthy neighbours achieve | 174 ms → 59 ms, **saving 115 ms = 6.8% of the step** | the primary |
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
whatever it consumes — and on a bandwidth-rich, FLOP-poor part like H20, two
kernels sharing SMs contend for exactly the resource KDA is already latency-bound
against.

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
| 1 | **retune the KDA kernel** (registers, block dims, `num_warps`/`num_stages`) | up to 6.8% of the step | the primary | a tuning pass, no numerics change |
| 2 | **keep evals on the lane card** — what we already do | the full lane, uncontended | evals | none, it is the status quo |
| 3 | **in-process eval between steps**, if a metric ever needs the training weights live | ~1% at a sane interval on a 200M model | evals | small; this is what every framework does |
| — | second process on a training card, no MPS | **nothing** — different CUDA contexts time-slice the whole GPU | neither | — |
| — | second process under MPS | a slice of the same 174 ms window | the co-tenant, primary pays | partial fault isolation, no memory partitioning |
| — | MIG | 23% measured loss even with hardware partitioning; cannot be reconfigured while a job runs | neither | — |

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
