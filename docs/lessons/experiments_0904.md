---
question: What do the four 2026-09-04 experiments conclude, at what resolution, and what can each not say?
status: measured
source: facts/data_scaling.json#ds.n2_params_vs_data_matched_compute; facts/smelt_deeploop.json#repo.loop_from_scratch_stage_d, #repo.loop_not_adopted_equal_compute; facts/efficiency.json#eff.conv_doc_isolation_ab_200m; runs/experiments.jsonl e1_c11_doccu_rescore; runs/prereg.jsonl b0_head_hybrid_3to1; runs/b0_headmix_armA.log, runs/b0_headmix_armB.log
---

# Experiments of 2026-09-04

Four experiments. Three decisions (N7 not adopted, N8 enters the recipe, headmix B loses), one no-difference verdict (N2).

## N2: parameters vs data at matched compute

**Verdict: no measurable difference on the path that matches training.**

Two arms, compute matched to 0.005%: params leg 438.4M × 3.76B tokens, data leg 206.1M × 8.00B tokens (`ds.n2_params_vs_data_matched_compute`).

| eval path | mean Δ (nat/token) | SE | t | sign (up/down) | status |
|---|---|---|---|---|---|
| doc_cu (matches training) | −0.000920 | 0.001661 | −0.55 | 329/247, params worse | current |
| cu_none | −0.010770 | 0.001655 | −6.51 | 227/349, params better | **retracted** |

Source: `runs/experiments.jsonl` e1_c11_doccu_rescore; `runs/score_matrix.jsonl` four `#cu` rows (data_leg 1.8678, params_leg 1.8668, equalcompute 2.2126, n8_fixed 2.2545).

The mean and the sign test summarise the same 576 doc_cu blocks and disagree in direction. No direction is put forward. The cu_none −0.010770 was 90.7% eval-path artifact (`eff.eval_path_cu_artifact_ce`): the scorer ran full-row causal across document boundaries, and the two legs pack different numbers of documents per row, so the leak does not cancel.

Resolution: one more leg per arm at a different seed (task e1-36, card job).

What this cannot say:
- Whether the sign holds on any other instrument. Only domain_loss was re-scored; ppl and the four other cu-blind scorers were plumbed and none re-run.
- Whether a reseeded pair reproduces the sign. n=1 seed per arm; `ds.seed_variance_0p2b` is 0.0516 nat at 0.2B, ~56× the doc_cu delta, so the SE is over blocks for this pair and says nothing about reseeding.

## N7: middle-layer loop

**Verdict: not adopted. The loop wins per token and loses per FLOP.**

### Stage B (post-hoc loop, SFT-scale)

Same SFT pack, with and without the loop, ~500 steps each, same seed. Each arm scored in its own topology.

| ruler | unlooped | looped | Δ | SE | z | source |
|---|---|---|---|---|---|---|
| humaneval BPB | 0.4635 | 0.4658 | +0.0023 | 0.0006 | 3.6 | roadmap_0903.md N7 row; runs/n7b_*.json |
| domain_loss | 1.9858 | 1.9951 | +0.0093 | — | 9/9 | same |

500-step rerun (fresh arms, e1 2026-09-04): humaneval 0.4643 vs 0.4676 (+0.0033, SE 0.0006, z 5.7, 112/164); training loss identical at every checkpoint (final 1.057 both) while BPB diverged, so the cost grows with steps. Not adopted.

### Stage D (from-scratch)

Two arms from scratch, 122.3M, 3815 steps = 1.0001B tokens each, seed 42, differing only in `--loop 4 7` (`repo.loop_from_scratch_stage_d`).

Equal tokens (both 1.0001B):

| ruler | looped − unlooped | SE | t | sign | reading |
|---|---|---|---|---|---|
| corpus loss, doc_cu, per-block paired | −0.022325 nat | 0.000665 | −33.55 | 537/576, 9/9 domains | loop wins |
| humaneval gold BPB, per-task | +0.006759 | 0.004160 | +1.62 | 93/164 worse | not significant |
| train-path val | 2.173 vs 2.192 | — | — | — | corroborates corpus sign only |

Equal compute (third arm: unlooped, 4824 steps = 1.2646B tokens, 26% more, same active-parameter FLOPs; `repo.loop_not_adopted_equal_compute`):

| ruler | equalcompute − looped | SE | t | sign |
|---|---|---|---|---|
| corpus loss, doc_cu | −0.043905 nat | 0.001351 | −32.49 | 544/576 |
| humaneval gold BPB | −0.038320 | 0.005945 | −6.45 | 123/164 |
| train-path val | 2.135 vs 2.173 | — | — | — |

Three rulers, one sign: the equal-compute arm beats the looped arm. The loop captures about a third of what the plain token spend captures (equalcompute − unlooped = −0.066 nat; the loop bought −0.022 of it for the same FLOPs).

What this cannot say:
- Whether a reseeded pair reproduces the sign. n=1 seed; `ds.seed_variance_0p2b` is 0.0516 nat, ~2.3× the delta.
- Whether the loop wins at larger scale. 122M, 1B tokens; SMELT's from-scratch claim has its own 1e20 interval reaching 1%, and this does not confirm or refute it.
- Whether the loop wins under a latency or memory constraint. The comparison is equal FLOP by the 6N accounting, not equal wall clock (0.962×) or memory (looped peaked 47.78 GiB vs 37.02).

## N8: conv document isolation

**Verdict: enters the recipe for correctness. Corpus likelihood improves 9/9 domains; HumanEval worsens.**

Two arms, identical except `--conv_doc_isolated`. 122.3M, 3815 steps = 1.0001B tokens, seed 42. One arm trained: the current arm is `ckpt_b0_sd_unlooped.pt` reused, bitwise-verified (`scripts/b0_n8_reuse_gate.py`, 576/576 blocks, worst |diff| 0.0). Source: `eff.conv_doc_isolation_ab_200m`.

fixed − current, both on doc_cu:

| ruler | Δ | SE | t | sign |
|---|---|---|---|---|
| corpus loss, per-block paired | −0.024353 nat | 0.000666 | −36.57 | 552/576, 9/9 domains |
| humaneval gold BPB, per-task | +0.014674 | 0.004612 | +3.18 | 97/164 worse |
| train-path val | 2.172 vs 2.192 | — | — | — |
| steady throughput | 116 vs 117 K tok/s/gpu | — | — | ~0.9% slower |

Pre-registered reading (6e, before launch): below −0.005 BPB on humaneval AND block-paired = the leak was costing accuracy and the fix enters on evidence. Corpus clears the threshold 5×; humaneval is +0.0147, significantly the wrong way. The first clause is not met; the fix enters for correctness with the disagreement recorded.

What this cannot say:
- The mechanism. The corpus gain does not track documents per row (corr −0.56 vs −0.95 for the leak itself; zh_web at 2 eos/row gains as much as chatml at 18). The gain is not simply leak removal. Candidate: the masking also zeroes taps at the first 3 positions of every row, a row-start effect, untested.
- Whether the sign holds at 500M or 30B, where the recipe decision applies. n=1 seed.

## head-hybrid A/B

**Verdict: B LOSES. Per-layer 6 KDA + 2 MLA heads (head_mixed=3, latent 256) loses to layer-level 3:1 at d1024 L12, 1B tokens, seed 42, on doc_cu by 0.087 nat, 576/576 blocks. The layer-level form stays.**

Design (`runs/prereg.jsonl` b0_head_hybrid_3to1, registered 2026-09-04T08:29Z, amended 08:34Z):

- Arm A: layer-level hybrid, d1024 L12 h8 ffn3072, attn_every=4 (9 KDA blocks + 3 MLA blocks).
- Arm B: head_mixed=3 — both mixers in every block on a 3:1 KDA:MLA head split (KDA h=6 inner=768, MLA h=2 inner=256, latent 256). Both read the full residual; outputs summed (o(concat(a,b)) == o1(a)+o2(b), verified max|diff| 1.43e-06).
- Question: does putting attention in every layer at 1/4 width beat concentrating it in every fourth layer at full width, at equal depth and near-equal parameters.

Latent asymmetry (read from the running checkpoints' tensor shapes, 2026-09-04): kv_down is 1024→256 in both arms, but kv_up differs — Arm A (2048, 256) reconstructs k|v at the full residual width, Arm B (512, 256) at the MLA half's 256-wide inner width. Both cache 256 numbers per token per layer, but Arm B's latent is sized for a 1024-wide attention while running a 256-wide one: under-compressed relative to what it feeds (2:1 on the up side vs Arm A's 8:1). This is an asymmetry between the arms, not an equivalence; it spends parameters rather than starving the path, so it is the conservative direction.

Parameter counts (recounted from the checkpoints on disk, 2026-09-04, no tied weights in either): delta +2,423,808 = **+1.01% of total** (242,171,976 vs 239,748,168, head untied 33.6M) and **+1.18% of non-embedding** (208,552,008 vs 206,128,200). The prereg's +1.18% is the non-embedding figure and does not name its population. An earlier draft said −1.91%: computed for a layout where each mixer read only its own slice, and does not survive the change to full-residual projections. The sign flipped.

What +1.18% buys at 1B: unmeasured. A B advantage smaller than the unmeasured quantity is undecidable: it cannot be attributed to the topology rather than to the extra parameters.

Confound: attention count vs per-layer width. Arm B has 12 attention paths of width 256; Arm A has 3 of width 1024. Total attention width is 3072 vs 3072, equal by construction, but the count and the per-layer width both move together and this design cannot separate them. A result favouring B reads as "this topology", not as either mechanism. A third arm separating them is not scheduled.

Threshold: val BPB on doc_cu at the shared token budget (1.0001B tokens, 3815 steps, same seed, same data order, 2 cards per arm). B WINS if lower by more than 0.003; LOSES if higher by more than 0.003; between is NO DIFFERENCE and the layer-level form stays.

Status: both arms trained to completion (Arm A: 3815/3815 steps, train 1.768, val 2.117 on the train path, 6968s; Arm B: 3815/3815, train 1.849, val 2.200, 8548s). Both scored on doc_cu (`runs/score_matrix.jsonl`, 2026-09-04). Block-paired on main: armA − armB = −0.087380 nat, SE 0.001077, t −81.12, 0 up / 576 down, 9/9 domains. domain_loss unweighted mean: Arm A 2.195, Arm B 2.2824. HumanEval gold BPB: A 0.6828, B 0.7176 per-task. L1 fewshot: A 3.2%, B 1.6%. lambada_en: A 18.8%, B 18.5%. Throughput: B 7.3% slower, +2.2 GiB at the clean step-30 read; later tok/s polluted by foreign co-residency, not quoted. The domain_bpb rows are cu_none (the tool ignores --cu_path) — labeled as such, not the ruler. One seed, one budget point.

Periodic val (train path, cu_none — NOT the ruler; the ruler is doc_cu at step 3815):

| step | Arm A | Arm B |
|---|---|---|
| 500 | 2.917 | 2.932 |
| 1000 | 2.589 | 2.604 |
| 1500 | 2.464 | 2.502 |
| 2000 | 2.399 | 2.420 |
| 2500 | 2.343 | 2.384 |
| 3000 | 2.284 | 2.358 |
| 3500 | 2.241 | 2.338 |
| 3815 (final) | 2.117 | 2.200 |

Source: `runs/b0_headmix_armA.log`, `runs/b0_headmix_armB.log`. These are the train-path val (cu_none), a different quantity from the doc_cu val BPB the prereg names as the decision rule.

What this cannot say:
- Whether the result generalises. n=1 seed per arm; the delta is large enough that seed variance (0.0516 nat at 0.2B) is unlikely to flip it, but a reseeded pair was not run.
- Why B loses. The +1.18% parameter count and the count-vs-width confound both stand; the loss cannot be attributed to either mechanism.
