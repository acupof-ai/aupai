---
question: Which small encoder should the L2 quality-regression head use to census-scan 40-80M code+English(+some Chinese) docs, and what does the scan cost?
status: measured
source: scripts/probe_lengths.py, scripts/gpu_forward_bench_pod.py, scripts/probe_encoders.py; facts/data_quality.json#dq.l2_encoder_truncation_real_docs_0916
---

# L2 encoder base selection (2026-09-16)

**Recommendation: BAAI/bge-m3.** It keeps full document signal on real docs, covers code +
English + Chinese with one tokenizer, and a 40-80M-doc census scan is ~85-170 single-H20
hours (~11-21 h on 8 cards) at measured 131 docs/s forward.

## The three candidates, self-measured (no paper numbers)

| encoder | params | weights | hard ctx | pooling | output dim | languages |
|---|---|---|---|---|---|---|
| BAAI/bge-m3 | ~568M | 2.27 GB | 8192 | CLS | 1024 fixed | multilingual (XLM-R), zh strong |
| Qwen/Qwen3-Embedding-0.6B | ~0.6B | 1.2 GB shard | 32768 | last-token | 1024 fixed | multilingual |
| BAAI/bge-large-en-v1.5 | ~335M | 1.34 GB | 512 | CLS | 1024 fixed | English only |

All three: L2-normalized vectors, fixed 1024-d output, and bit-identical across two calls
on the same input (`deterministic=true`, fp32 CPU; fp16 GPU is deterministic run-to-run at
batch 1, tolerance 1e-3 — not asserted bitwise). Pooling follows each model's shipped
`1_Pooling/config.json`; we do not pick our own.

## Truncation on the real doc-length distribution is the deciding axis

4500 real docs (1500 each from en_c4_stage2_dc, code_ultra_l2_dc, code_py_starcoder_dc),
tokenized by each model's own tokenizer (facts/data_quality.json#dq.l2_encoder_truncation_real_docs_0916):

| | en_c4 P50/P95 | code-l2 P50/P95 | starcoder P50/P95 | fraction truncated |
|---|---|---|---|---|
| bge-large-en (512) | 241 / 1562 | 543 / 3255 | 592 / 4922 | **0.257 / 0.516 / 0.543** |
| bge-m3 (8192) | 262 / 1743 | 540 / 3414 | 598 / 4933 | 0 / 0 / **0.0013** |
| Qwen3-Emb (32768) | 243 / 1582 | 428 / 2577 | 464 / 3939 | 0 / 0 / 0 |

bge-large-en throws away the tail of **over half the code docs** at 512 tokens. A quality
head that never sees the latter 80% of a long file cannot score that file's correctness or
instructional value; the truncation is systematic against code, which is the majority class.
Its raw speed advantage is real but buys the wrong measurement. bge-m3 truncates 0.13% of
starcoder and nothing else; Qwen truncates nothing but most docs are <1k tokens, so its
32k window mostly adds attention cost it never uses.

## Throughput, self-measured

Pure GPU forward on one H20, fp16, pre-tokenized length-sorted token-budget packs
(tokenizer deliberately outside the timed region — a scanner pipelines CPU tokenize ahead),
n=1200, warmup 2 / 6 reps
(`facts/data_quality.json#dq.l2_encoder_gpu_forward_throughput_h20_0916`):

| encoder | docs/s @32k tok | docs/s @64k | tokens/s @32k | peak mem |
|---|---|---|---|---|
| bge-m3 | **131.1** | 123.7 | 116k | 3.91 GB |
| Qwen3-Emb-0.6B | 74.1 | 66.1 | 53k | 17.15 GB |
| bge-large-en | 513.8 | 488.7 | 175k | 3.14 GB |

Laptop CPU (10 threads, batch-1 bounded point; m3/qwen large-batch CPU is too slow/OOM to
be a census path): bge-m3 15.5, Qwen 9.5, bge-large 15.3 docs/s. CPU is a spot-check path
only; the census runs on H20.

## Census-scan cost (H20 card-hours)

From 131.1 docs/s forward (`facts/data_quality.json#dq.l2_census_scan_cardhours_h20_0916`):

| docs | bge-m3 1 card | bge-m3 8 cards | Qwen 1 card | bge-large 1 card |
|---|---|---|---|---|
| 40M | 84.8 h | 10.6 h | 149.9 h | 21.6 h |
| 80M | 169.5 h | 21.2 h | 299.9 h | 43.3 h |

Budget at a conservative derated 100 docs/s (padded IO + pipelined tokenize, no overlap):
40M = 111 h, 80M = 222 h single-card. Multi-card is independent shard scans, near-linear.

## Why bge-m3 over the faster bge-large and the longer Qwen

- vs bge-large-en: 3.9x slower but keeps >99.8% of code docs whole; bge-large is
  English-only and truncates half the code, disqualifying for a code-primary funnel.
- vs Qwen3-Emb-0.6B: 1.77x the docs/s, 4.4x less peak memory (3.9 vs 17.2 GB — four scans
  fit per card), equal 1024-d, and its 8192 ctx already truncates ~0%. Qwen's 32k headroom
  is unused on a P95-3.7k-token distribution.
- Chinese: bge-m3 is a first-class XLM-R multilingual model with documented strong zh;
  Qwen is also multilingual. Both clear the "some Chinese" requirement; bge-m3 clears it at
  lower cost.

The head and training script are encoder-agnostic to the pooled 1024-d contract, so if a
later domain is long-document-only the base can be switched by flag (`--encoder`) without
touching the head; this recommendation is for the stated code+English-primary mix.

## What is not measured

- End-to-end scan rate with real disk IO + CPU tokenize overlapped against forward; the
  card-hours are forward-only with a stated derate, not a pipeline benchmark.
- Downstream head quality per base (embedding separation on the 4 rubric dims). That needs
  66's labels; the base choice above is on signal preservation + throughput, and is
  revisitable once head val MSE/rank exists per encoder.
