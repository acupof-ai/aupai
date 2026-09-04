---
question: Do sparse memory layers buy loss at a 200M backbone for near-zero FLOPs, and is what they buy knowledge or reasoning?
status: open
source: this charter; runs/prereg.jsonl#memory_layers_0905; control ckpt_b0_headmix_armA.pt
---

# Memory layers at 200M (program opened 2026-09-04T16:44Z, user order)

## Why this and not another architecture knob

N2: at equal compute the params arm wins 0.0108 nat. N7 and Stage E: reusing weights buys
nothing. Head-hybrid: B loses 0.087 nat. Every knob that changes the shape of compute per FLOP
is at its margin; the variable that moved loss was the number of parameters a token can reach.
A product-key memory layer adds parameters at near-zero FLOPs, so it is the direct test of
that finding, and it splits capability into two measurable halves: what the table can store
and what the backbone can compose.

## Control

`ckpt_b0_headmix_armA.pt`: d1024 L12 h8 ffn3072 attn_every=4, mix_200m_8b, 3815 steps = 1B
tokens, seed 42, batch 16 accum 2 world 2, launched by the line in
`runs/experiments.jsonl` (b0_headmix_armA). Its doc_cu row is in `runs/score_matrix.jsonl`.
Every arm below reuses that launch line unchanged plus the memory flags, so the data order is
identical and block-paired scoring applies.

## Arms

| arm | memory values | value dim | params added | wall time vs control |
|---|---|---|---|---|
| M1 | 1,048,576 (1024 x 1024 product keys) | 1024 | 1.07B | measured lower bound +6.5% (tilerl, lookup alone, 2026-09-05); readout 5 is the constraint |
| M2 | 262,144 (512 x 512) | 1024 | 0.27B | unmeasured; readout 5 is the constraint |
| M3 | 2,096,704 (1448 x 1448) | 1024 | 2.15B | --peak-only before launch; 2048x2048 OOMs (48 GiB of table tensors + 47.6 GiB baseline > 95.2) |

Design fixed for both arms; b0 chooses the rest inside these bounds:
- one memory pool read by ONE layer, layer 6 (0-indexed), added in parallel to the FFN
  (amendment_10, 2026-09-05: three layers 3, 6, 9 measured 0.79 of control throughput and
  no gradient-path fix reaches 0.85; one layer measured 0.90),
  `h = h + mem(norm(h))`. The FFN is not replaced, so the dense parameter count equals the
  control's and the arm differs from it only by the memory.
- product-key lookup, top-k = 32, one query head, output gated (Meta memory+ style: value
  projection then silu gate). Memory values and keys are excluded from FP8 and from Muon;
  they train with Adagrad at their own lr. **The value table is outside FP8 by module
  type, not by the filter**: it is an `nn.Embedding`, and `convert_to_float8_training`
  only replaces `nn.Linear`, so torchao never reaches it. The three projections
  (`query`, `gate`, `out`) ARE reachable and are excluded by path — verified by listing
  what the real filter converts, 0 of 3, against 3 of 3 under a leaf-name test
  (`probes/mem_boundary_audit.py`, reviewed against `4d0319cf`). The reason to exclude
  them is not cost but that `query` feeds a top-k selection, so FP8 noise in the scores
  changes WHICH rows are read — a discrete effect the block-paired readout cannot
  separate from the memory's own (4c, 2026-09-05).
- gradient exchange across DDP is chosen **per arm, by measurement**. The original rule
  was "gather touched indices, never all-reduce the dense table"; it was struck on
  2026-09-05 because sparse loses at both launched shapes. Measured on cards 5+0:
  M1 touches 86.5% of its table per step (69.2% under a peaked draw) and M2 touches
  **99.97%** — smaller table, more saturated, since the same 2,097,152 draws hit 4x
  fewer values. Sparse costs 1.5–2.3x dense in time and 1.4–2.0x in bytes at every
  shape. **M1 and M2 run dense all-reduce** with `mem_sparse=False`; NCCL raises on a
  COO gradient, so there is no automatic sparse path either way. M3 is re-measured
  before its launch (2,097,152 draws into 4,194,304 values is ~39% distinct, the first
  shape where sparse could win). `scripts/memory_ddp_bench.py`.
- `test_arch_compat.py` gains: memory fwd/bwd on CPU, save/load round-trip, and a legacy
  checkpoint (no memory) still loads.

The `<= +3% FLOP` bound that stood in this table was struck 2026-09-05: the lookup is
memory-bound, not FLOP-bound, and its wall cost at M1 is 51.6 ms of a 799 ms step (6.5%) while
throughput stays at 93.9% of the control. Readout 5 (tok/s/gpu at step 30, stop below 70K) is
the only cost constraint. Sparse-vs-dense gradient exchange is chosen per arm by measured
bytes per step, not by rule: at M1 a uniform draw touches ~86% of the table per step.

## Pre-registered readouts (runs/prereg.jsonl#memory_layers_0905)

1. Primary: block-paired doc_cu val, arm minus control. Adopt if <= -0.010 nat (the size of
   the N2 params effect), null if |delta| < 0.003, in between is "measured, not adopted".
2. Split, a difference-in-differences: an API-name cloze (4-way, real names from the same
   module) drawn from two regions of the code_py_starcoder cache the arms read -- SEEN rows the
   arms train on and the never-read tail (74.6% of the pool) as UNSEEN. delta_seen minus
   delta_unseen, both arm-minus-control block-paired, above both SEs is "memory buys knowledge";
   delta_unseen alone is generalisation and readout 1 already has it. Region boundaries and
   seeds are pinned in the item file. No slice is carved, no registry entry is written. Fallback
   domain textbook_30b (Chinese). The reasoning probe is l1_fewshot answer-present, 3 demos.
3. Scaling: M2 vs M1 gives the slope of loss against memory size; two points plus the control
   are a line, not a law, and the doc says so.
4. Diagnostics, logged every 100 steps to `runs/memory_diag.jsonl`: fraction of values
   touched in the window, top-k weight entropy, key-usage Gini. A pool below 20% touched at
   step 1000 is a collapse and the arm is stopped and reported, not tuned in place.
5. Throughput: the arm's tok/s/gpu divided by the control's at the SAME step, read at step 30
   (re-read at 100 and 1000). Below 0.85 the arm is stopped: a memory that costs 15% of
   throughput is not near-zero FLOPs on this hardware. (Restated from a 70K absolute floor on
   2026-09-05 after M1's stop, verdict unchanged: the control itself reads 56K by step 200.)

## Cards (2026-09-04T16:44Z)

M1 on cards 1+2, M2 on cards 4+6, world 2 each; lane card 5 for the smoke test, milestone
scores and probes, one job at a time; card 0 unassigned spare (foreign-held twice on
2026-09-04); card 3 foreign; card 7 the user's. M2 launches after M1 has printed step 100
with no NaN and diagnostics within bounds.

## Owners

| role | owner | reviewer |
|---|---|---|
| model.py memory layer, sparse optimizer, arch_compat | b0 | 44 |
| kernel and throughput, sparse DDP gradient path | tilerl | b0 |
| fact probe + holdout registration, reasoning probe wiring | e1 | 3b |
| data: cache freshness, corpus fingerprint unchanged, holdout slice carved | 3b | e1 |
| harness launch, memory_diag ledger + schema, score_matrix_present #cu fix, monitors | de | 44 |
| prereg review within 15 min, shapes | 44 | -- |
| progress page section with the two curves | 98 | -- |
| controller: grants, rulings, stop decisions | fb | 44 |

## Memory cost per table parameter (measured 2026-09-05, b0, card 5)

The table is fp32 (nn.Embedding in the default dtype, nothing casts it), its dense gradient is
fp32, and Adagrad holds one fp32 moment: 12 bytes per parameter, not the 6 the arms table
assumed. M1 = 12.0 GiB of table tensors; measured peak 68.46 GiB reserved against a 59.64 GiB
prediction, so activations are 8.82 GiB. M3 at 2048x2048 = 48.0 GiB of table tensors on a
47.64 GiB memoryless baseline: does not fit before an activation is allocated, and died in
backward allocating the gradient. M3 is 1448x1448 (2.10M values, 2.15B params, 24 GiB of
tensors) so every tensor dtype is identical across the three arms; a bf16 table for one arm
would confound readout 3's slope with a precision change. Memory's own throughput ratio at
world 1: 30.6K / 38.1K = 0.803 (mem-off baseline in the identical config), above the 0.70 bar.

## Amendment 10 (2026-09-04T21:48Z): one pooled layer, not three

tilerl's five-cell decomposition at the arm shape (`runs/mem_decomp_0905.jsonl`; world 2, fp8,
compile, `expandable_segments`), tok/s/gpu and ratio to the memory-off control:

| cell | fwd ms | bwd ms | opt ms | step ms | tok/s/gpu | ratio |
|---|---|---|---|---|---|---|
| off | 642.4 | 911.8 | 37.2 | 1592.6 | 82,300 | 1.00 |
| m1 (3 layers, 1024², k32) | 732.1 | 1227.0 | 56.0 | 2016.9 | 64,987 | 0.79 |
| k16 | 725.1 | 1105.7 | 56.2 | 1888.9 | 69,391 | 0.84 |
| l1 (layer 6 only) | 672.4 | 1032.4 | 56.3 | 1763.3 | 74,334 | 0.90 |
| m2 (512²) | 729.9 | 1212.0 | 41.8 | 1984.4 | 66,051 | 0.81 |

Fit cost = 3a + b against l1 = a + b: per-layer 29.9 / 97.3 / -0.1 ms, table-fixed 0.1 / 23.3 /
19.2 ms (fwd / bwd / opt). 382 of the 424 ms excess is per-layer. Gradient bench at the M1 shape:
dense embedding backward 40.7 ms, index_add 91.9, sort+segment 98.3; a free scatter would give
0.841. Ruling: the arms relaunch with `--mem_layers 6`. The 0.85 gate is unchanged. M1's stop at
0.78 stands as the three-layer result. The table dtype paragraph above is superseded by
amendments 7-9 (bf16 table, table-owned fp32 master, 14 B/param steady, 16 at the in-step peak).

**The diagnostics block is not a cost and gating it would be a regression.** The `nobk` cell
(`runs/mem_nobk_0905.jsonl`) ran M1 with the `torch.is_grad_enabled()` block at `model.py:538`
stripped — the `index_put` over the 1,048,576-element bool buffer plus both `bincount`s — and
the step got **52 ms slower**, not faster: forward -2.8 ms (noise, against a predicted -38),
backward +57.7, ratio 0.770 against m1's 0.790. The hypothesis it was written to test was a
graph break splitting the backward; a break would move the step the other way, and
`dynamo.explain` reports graphs=1 breaks=0 ops=36 with the block live. Mechanism, offered as a
hypothesis rather than a measurement: the block reads `flat`, `sel`, `i0` and `i1` under
`no_grad`, keeping them resident across the region, and without it backward recomputes what it
used to find live. Two consequences that are measurements: the ~12 ms/layer of forward the fit
leaves unexplained is not the bookkeeping, and the step-gate held in reserve as b0's fix is
closed — gating that block makes the arm slower.
