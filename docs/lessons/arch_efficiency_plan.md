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

| Component | ms/step | Time % | FLOPS % | Efficiency |
|---|---|---|---|---|
| FP8 GEMM (`_scaled_mm`) | 533 | 29% | ~84% | **93% of FP8 peak** (tilerl, exact FLOP from shapes) |
| bf16 GEMM (attention QK^T etc.) | 222 | 12% | ~3% | ~58% (FA2 Hopper) |
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
- KDA kernel at ~3% of FP8 peak (240ms for ~2% of FLOPS)
- Non-GEMM time = 53% of step, contributes ~3% of FLOPS

**MFU scenarios** (corrected 2026-08-30 after GEMM efficiency reconciliation):
- Current: 31%
- GEMM → 93%: **0% (already there)**
- KDA 2× faster: +1.4%
- KDA 2× + generic overhead 50%: ~39%
- Absolute upper bound: **~39%**

**Biggest lever**: KDA (240ms, 2% FLOPS, 3% peak) and generic overhead (478ms). GEMM efficiency is not a lever.

## 2. Controlled Architecture A/B (old vs new)

**Design**: Old architecture (sliding window + blocks=4) vs new (full causal MLA + AttnRes Full), same config, same corpus.

**Confounds controlled**: vocab (32776), chunk_size (32), bucket_cap_mb (50), warmup (1% floor 2), lr_scale (1.0), batch (16/accum2), corpus (same shards).

**Confounds NOT controlled** (in the old-vs-new six-point comparison): lr_scale (0.5→1.0), warmup (150→2), corpus shards. The controlled A/B fixes these.

**Cost**: 0.2b point = 218 steps = ~6 min/run (75K tok/s/gpu, 7 GPUs). Tokenization shared, already paid.

**Seed requirement** (anchor: `ds.kaplan_noise`, σ=0.05 nat):
- 1 seed: MDE ≈ 0.05 — only detects large effects
- **4 seeds × 2 arms: ~50 min, MDE ≈ 0.035** ← minimum
- 8 seeds × 2 arms: ~100 min, MDE ≈ 0.025

**Rule**: Never interpret a single-seed difference < 0.05 as a conclusion. If result is at MDE boundary, add seeds.

**What it answers**: Architecture effect on loss at 0.2b tokens. If significant, validate at 0.3b/0.4b (interleave with six points, ~30 min total).

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
| KDA vs attention | 11.7% kernel | never measured | Replace KDA with standard attention, same params, compare loss + throughput |

## Priority

1. **Controlled A/B** (4 seed × 2 arm, ~50 min) — can interleave with six points
2. **MFU upper bound refinement** — per-GEMM-size efficiency microbenchmarks
3. **Old-vs-new comparison** — zero-cost, post six-points
4. **AttnRes/KDA A/B** — post six-points, needs checkpoints
