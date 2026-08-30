---
question: "这个架构在 H20 上的 MFU 可达上界是多少,每个架构选择各自花了多少、换来什么——六个预算点之后的架构迭代需要哪些数"
status: recorded
source: "project measurements 2026-08-30; facts in facts/efficiency.json; profile traces in bench_eff/"
---

# Architecture Efficiency Measurement Plan

**Goal**: Quantify the achievable MFU upper bound and per-architecture-choice cost/benefit, to guide architecture iteration after the six budget points.

**Current state**: MFU 31% (FP8 peak 296T) / 62.6% (bf16 peak 148T). GPU 98.5% busy. DDP overhead 6.25% (not identified, closed by timebox).

## 1. MFU Upper Bound (preliminary, 2026-08-30)

**Per-component time (4-GPU bucket50 profile, 5 active steps) and FLOPS (architecture-derived):**

> **⚠️ Profiler bias warning (2026-08-30)**: These numbers are from profiled traces. Profiler inflation is **non-uniform across kernels** — overall step 1.85×, LM head bf16 GEMM 2.1× (226ms profiled vs 105ms no-profiler, CUDA event). Component percentages below are **biased and should not be cited** until recalculated with no-profiler measurements. Total MFU (31%) is from wall-clock and is unaffected.

| Component | ms/step | Time % | FLOPS % | Efficiency |
|---|---|---|---|---|
| FP8 GEMM (`_scaled_mm`, `nvjet_qqtst_*`) | 533 | 29% | ~84% | **93% of FP8 peak** (tilerl, exact FLOP from shapes) |
| bf16 GEMM (LM head via Liger FLCE, `nvjet_tst_*`) | 226 | 13% | ~13-20% | memory-bound (vocab=32776) |
| FP8 quant fusions | 112 | 6% | — | — |
| KDA kernel | 240 | 13% | ~2% | **~3%** (latency-bound; ncu occupancy 6-12% confirms) |
| inductor fusions | 290 | 16% | — | — |
| elementwise | 118 | 7% | — | — |
| GatedMLA attention (flash) | 69 | 4% | ~1% | ~58% |
| AttnRes | 63 | 3.5% | ~0.1% | bandwidth-bound |
| FLCE (liger) | 45 | 2.5% | — | — |
| DDP NCCL (non-overlapped) | 62 | 3.5% | — | — |
| GPU idle | 28 | 1.5% | — | — |

**Key findings**:
- FP8 GEMMs at 93% of peak (no headroom — tilerl's exact measurement)
- bf16 226ms is **LM head** (Liger FLCE internal matmul), not attention — confirmed by kernel name prefix (`nvjet_tst_*` = bf16, `nvjet_qqtst_*` = FP8; tilerl GPU7 实测)
- **No-profiler LM head: 104.7 ms/step** (CUDA event, GPU0, 2026-08-30) — profiled 226ms was 2.1× inflation
- Flash attention confirmed in use (68.8ms/step, matches tilerl's 69ms)
- KDA kernel at ~3% of FP8 peak (240ms for ~2% of FLOPS)
- Non-GEMM time = 53% of step, contributes ~3% of FLOPS

**64K→75K throughput gain decomposition** (three confounded changes, not additive — ceiling at 75K):
- bucket_cap_mb 100→50: +14.1% (3-GPU isolated A/B)
- vocab padding 32773→32776: LM head saves 80ms/982ms = 8.1% (single-kernel isolated)
- chunk_size 64→32: ~2% (tilerl)
- Combined effect is bounded by the 75K ceiling; individual effects overlap and are not additive

**MFU scenarios** (corrected 2026-08-30 after GEMM efficiency reconciliation):
- Current: 31%
- GEMM → 93%: **0% (already there)**
- KDA 2× faster: +1.4%
- KDA 2× + generic overhead 50%: ~39%
- Absolute upper bound: **~39%**

**Biggest lever**: KDA (240ms, 2% FLOPS, 3% peak) and generic overhead (~256ms, was 478ms — 222ms LM head misattributed to generic, corrected 2026-08-30). GEMM efficiency is not a lever.

## 2. KDA vs More Attention A/B (approved 2026-08-30, runs before six points)

**Design**: `--attn_every 1` (12 GatedMLA, all-attention) vs `--attn_every 4` (9 KDA + 3 GatedMLA, current), same config, same corpus.

**Why not old-vs-new (b)**: Sliding window was deleted in b3cad87. It was never a valid control — `infer_local.py` never implemented the window (train/infer inconsistency). Re-adding it would reproduce a bug. tilerl already measured the +17.2% decomposition item-by-item. (b) is out, not queued.

**Config (both arms)**: vocab=32784, chunk_size=32, bucket_cap_mb=50, warmup=20, batch=32/accum1, same corpus shards, same mix.

**Param difference**: Current 206M vs all-attention 194M (-5.8%). KDA 5.26M/layer vs GatedMLA 3.93M/layer.

**Param matching**: Primary test does NOT match. If KDA wins ≥ MDE, MUST run `--ffn_hidden 3392` matched arm (3392 = 16×212, FP8-compatible; 3400 breaks `_fp8_ok` 16-alignment). If all-attention wins, no matched arm needed (fewer params and still wins = strong result).

**Long dependency**: Full causal attention covers 4096 positions. KDA's long-range rationale (carry beyond attention window) is already covered. KDA may still provide compression, different inductive bias, or chunk-level features (short_conv). This test answers whether KDA adds value at 4096 context.

**Metrics**: Primary = per-domain NLL. Secondary = CLiMP/LAMBADA/math v2 (may lack resolution at 0.2b, trend only). Throughput also reported.

**0.2b A/B cancelled (2026-08-30)**: Control arm (4 seeds, `attn_every 4`) ran as ladder's 0.2b point. Measured σ̂ = 0.0516 (3 df), 1.47× the 0.035 design assumption (which came from a fit residual, not seed variance — different quantities). MDE at 4+4 = 0.1021 > b0's 0.08 gate; even 8+8 = 0.0722 still misses. Treatment arm (`attn_every 1`) does not run at 0.2b. Question moves to 3.24b checkpoint comparison (free: control is the ladder's own checkpoint).

**σ̂ boundary**: measured at 218 steps. More steps may average seed effects down — 0.0516 may be an upper bound for larger points. Not assumed either way.

**Control arm data** (`ds.seed_variance_0p2b`, `ds.mde_recomputed_from_measured_sigma` in `facts/data_scaling.json`):
- s0=3.691, s1=3.762, s2=3.679, s3=3.638; mean=3.6925, range=0.1240, σ̂=0.0516
- Throughput: 75K tok/s/gpu flat across 217 steps, MFU 31-32% (prediction held)

**Memory probe** (tilerl, H20 GPU0, batch 16, seq 4096, bf16, no compile, no fp8, 3-step avg):
- `attn_every 4` (current): 66.3GB peak, 1537.8ms/step, 42616 tok/s
- `attn_every 1` (all-attention): 53.9GB peak, 1590.9ms/step, 41195 tok/s
- All-attention **saves 12.4GB (-18.8%)** but is **3.3% slower** (not +2% as initially predicted)
- Mechanism: KDA's `disable_recompute=True` trades memory for speed (+3GB, 8-15% faster, train.py:262). Replacing 9 KDA layers loses this trade-off — saved activation memory and lost throughput are two sides of the same coin.
- Both arms at batch 16 are memory-safe. The constrained arm is `attn_every 4` (66.3GB), not all-attention.
- 66.3GB is a lower bound (single-process, no compile, no fp8); not comparable to the 50.8GB measured with fp8+compile.

**Cost**: 0.2b A/B cancelled (MDE > gate). 3.24b comparison: ~103 GPU-minutes for 1 treatment seed (control is ladder's own checkpoint, free).

**Pre-registered decision rules (0.2b, superseded)**:
- ~~All-attention wins ≥ MDE → ladder switches to `attn_every 1`~~ — cancelled, MDE > gate
- ~~KDA wins ≥ MDE → unreadable until matched arm~~ — cancelled
- ~~|gap| < MDE → KDA stays~~ — cancelled, question moves to 3.24b

**3.24b checkpoint comparison design** (draft, b0 pre-registers reading rules):

At 3.24b, the control is the ladder's own checkpoint (`attn_every 4`, already paid for). The treatment requires a `attn_every 1` run at 3.24b tokens.

**Option A — Direct comparison (clean, costs one 3.24b run ~103 min)**:
- Run `attn_every 1` at 3.24b, 1 seed (add seeds if σ̂ at 3.24b requires)
- Evaluate both checkpoints on same val set (same val_batches, same val prefix)
- Compare per-domain NLL
- Reading: |gap| vs MDE (calculated from σ̂ at 3.24b, measured from ladder seeds if available)
- If all-attention wins ≥ MDE → switch. If KDA wins ≥ MDE → param-matched arm. If < MDE → no resolution at 3.24b either.

**Option A' — Paired comparison (fb proposal 2026-08-30, under evaluation)**:
- Same seed in both arms → data order cancels (train.py:1343 mix planner uses `Cfg.seed`, architecture-independent)
- Relevant quantity: σ of *paired difference* (σ_paired), not σ of independent runs
- If σ_paired < σ_independent, MDE improves dramatically:
  - σ_paired 2× below σ → 3.24b 1+1 reaches MDE 0.10
  - σ_paired 2.6× below σ → 3.24b 1+1 reaches 0.08 gate at 206 GPU-min
- **Probes (~18 GPU-min, running 2026-08-30)**:
  1. Same seed (s2), same arm (`attn_every 4`), re-run vs existing `ckpt_p02_s2` (val 3.679) → pure nondeterminism floor. If diff > ~0.01, CUDA nondeterminism contaminates all σ estimates, σ̂=0.0516 is an upper bound.
  2. Same seed (s2, s3), both arms (`attn_every 1` vs existing s2, s3 `attn_every 4`) → σ_paired first read.
  - Pairing against s2/s3 (not s0/s1): s0/s1 ran before a train.py edit; s2/s3 are on the same side as new runs.
- **Corrected bands** (b0's frozen paired t: `(t.975,n-1 + t.80,n-1)·s_d/√n`, not normal approximation):
  - n=2 pairs (df=1): factor 14.08, needs s_d ≤ 0.0080 (ρ≈0.99, effectively unreachable)
  - n=3 (df=2): factor 5.364, needs s_d ≤ 0.0258
  - n=4 (df=3): factor 4.160, needs s_d ≤ 0.0385 (ρ≈0.72)
  - n=6 (df=5): factor 3.491, needs s_d ≤ 0.0561
  - **1-pair design does not exist** (n=1, df=0, s_d not estimable). Minimum 2 pairs.
- **Real question**: σ_d = σ√(2(1−ρ)). Affordable form is 4 pairs at ~721 GPU-min (not 206 as initially quoted). Requires ρ ≈ 0.7.
- **Binding falsifier**: If σ_d ≈ σ√2 (ρ≈0), pairing bought nothing, independent already failed, and **KDA A/B is dead at 200M — no further design substitutions**. Probe is simultaneously the design's own falsifier.
- **Lower bound with teeth**: σ_d ≥ √2·σ_nondet. If probe 1 alone exceeds threshold, pairing dies without waiting for probe 2.
- Init comparability: shared layers (embedding, 3 GatedMLA, FFN, norms) are identical with same seed; replaced layers (9 KDA → 9 GatedMLA) differ but that's the treatment, not a confound

**Option B — Scaling law residual (free, indirect)**:
- Fit scaling law to ladder data (0.2b → 1.6b, `attn_every 4`)
- Predict 3.24b loss, compare with actual
- If actual >> predicted → architecture may be bottleneck (but doesn't distinguish KDA vs other causes)
- Does NOT directly compare `attn_every 1` vs `attn_every 4`

**Recommendation**: Option A. Option B is a free sanity check but can't answer the KDA question. The 3.24b run is expensive but it's the only clean comparison. If σ̂ at 3.24b is lower (fb: "may be an upper bound"), 1 seed may suffice.

**Key difference from 0.2b reading rules**: At 3.24b, the comparison is between checkpoints, not fresh runs. The reading rules must account for: (1) σ̂ at 3.24b may differ from 0.0516, (2) the treatment checkpoint starts from random init (not warm-started from KDA), (3) the val evaluation must be identical (same batches, same prefix, same scoring script).

**Prerequisite**: ~~tilerl memory probe~~ DONE (2026-08-30): both arms memory-safe at batch 16, all-attention saves 12.4GB but 3.3% slower.

**Scheduling**: 0.2b A/B cancelled (σ̂ too high). Question moves to 3.24b checkpoint comparison. Control arm (4 seeds at 0.2b) already ran as ladder's 0.2b point — σ̂ measurement was free.

## 3. Old-vs-New Six-Point Comparison (zero-cost, post six-points)

**What it can support**:
1. "Combined change" (architecture + config + corpus) shifted scaling curve E/B/β by X — valid but not architecture-attributable
2. E comparison is most reliable (if both converged), but still confounded by lr_scale
3. **β comparison is unreliable** — lr and corpus both affect data efficiency, cannot disentangle. **Write this at the top of the comparison report, not as a footnote.**

**What it cannot support**: Clean architecture attribution (lr_scale 0.5→1.0 is the fatal confound).

## 4. Scale Extrapolation (200M → 2B)

- 200M: launch-bound (elementwise, reductions, small kernels) ≈ 230ms (11% of step)
- 2B: GEMMs 10× larger, launch overhead constant → ~1.1% of step
- MFU improvement from scale alone: ~10%
- DDP 5K overhead: may be amortized at 2B (re-test — "not identified" not "intrinsic")

## 5. Post-Six-Points Architecture A/B (needs checkpoints)

| Choice | Cost | Benefit | Experiment |
|---|---|---|---|
| AttnRes | 13.4% e2e | never measured | `--no_attn_res` vs default, same tokens, compare val loss |

Note: KDA vs attention (section 2) moved to BEFORE six points (fb approved 2026-08-30).

## Priority

1. **KDA vs attention at 3.24b** — checkpoint comparison, design drafted above, b0 pre-registers reading rules
2. **Post-padding 7-GPU efficiency baseline** — when six points start (current baseline is stale: single-GPU, pre-compile, pre-fp8, pre-padding)
3. **Old-vs-new comparison** — zero-cost, post six-points
4. **AttnRes A/B** — post six-points, needs checkpoints
