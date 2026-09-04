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

| arm | memory values | value dim | params added | FLOP vs control |
|---|---|---|---|---|
| M1 | 1,048,576 (1024 x 1024 product keys) | 1024 | 1.07B | <= +3% |
| M2 | 262,144 (512 x 512) | 1024 | 0.27B | <= +3% |

Design fixed for both arms; b0 chooses the rest inside these bounds:
- one memory pool shared by layers 3, 6, 9 (0-indexed), added in parallel to the FFN,
  `h = h + mem(norm(h))`. The FFN is not replaced, so the dense parameter count equals the
  control's and the arm differs from it only by the memory.
- product-key lookup, top-k = 32, one query head, output gated (Meta memory+ style: value
  projection then silu gate). Memory values and keys are excluded from FP8 and from Muon;
  they train with a sparse optimizer (SparseAdam or Adagrad) at their own lr.
- sparse gradients across DDP: gather touched indices, never all-reduce the dense 1B table.
- `test_arch_compat.py` gains: memory fwd/bwd on CPU, save/load round-trip, and a legacy
  checkpoint (no memory) still loads.

## Pre-registered readouts (runs/prereg.jsonl#memory_layers_0905)

1. Primary: block-paired doc_cu val, arm minus control. Adopt if <= -0.010 nat (the size of
   the N2 params effect), null if |delta| < 0.003, in between is "measured, not adopted".
2. Split: a closed-book fact probe (cloze over a held-out slice of an English encyclopedic
   domain, registered in the holdout registry before launch) against a reasoning probe
   (l1_fewshot answer-present, 3 demos, existing). The claim "memory buys knowledge, not
   reasoning" is the fact delta exceeding the reasoning delta by more than both SEs.
3. Scaling: M2 vs M1 gives the slope of loss against memory size; two points plus the control
   are a line, not a law, and the doc says so.
4. Diagnostics, logged every 100 steps to `runs/memory_diag.jsonl`: fraction of values
   touched in the window, top-k weight entropy, key-usage Gini. A pool below 20% touched at
   step 1000 is a collapse and the arm is stopped and reported, not tuned in place.
5. Throughput: tok/s/gpu at step 30 against the control's 82K. Below 70K the arm is stopped:
   a memory that costs 15% of throughput is not near-zero FLOPs on this hardware.

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
