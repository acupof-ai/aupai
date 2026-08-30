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

**Metrics**: Primary = per-domain NLL (4 seed × 2 arm, MDE≈0.035). Secondary = CLiMP/LAMBADA/math v2 (may lack resolution at 0.2b, trend only). Throughput also reported (expected +2% for all-attention).

**Cost**: 4 seeds × 2 arms × ~6 min = ~48 min (0.2b, 7 GPUs). Tokenization shared with six points, paid once.

**Pre-registered decision rules** (b0 writes pre-registration doc):
- All-attention wins ≥ MDE → ladder switches to `attn_every 1`. Corpus unchanged, G3 not re-opened.
- KDA wins ≥ MDE → **unreadable** until `--ffn_hidden 3392` matched arm runs.
- |gap| < MDE → KDA stays, recorded as "0.2b no resolution", re-asked at 3.24b checkpoint (free, no extra run).

**Prerequisite**: tilerl measures `attn_every 1` memory at batch 32/accum1 (60-step probe). If arms need different batch sizes, that's a confound — must know before the 8 formal runs.

**Scheduling**: Runs BEFORE six points, right after tokenize. (c) first = +48min fixed; ladder first = one branch might need 3.5h re-run.

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

1. **KDA vs attention A/B** (4 seed × 2 arm, ~48 min) — runs BEFORE six points, after tilerl memory probe
2. **Post-padding 7-GPU efficiency baseline** — when six points start (current baseline is stale: single-GPU, pre-compile, pre-fp8, pre-padding)
3. **Old-vs-new comparison** — zero-cost, post six-points
4. **AttnRes A/B** — post six-points, needs checkpoints
