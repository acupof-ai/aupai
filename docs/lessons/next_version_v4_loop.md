---
question: "v2 architecture spec: SMELT loop + sparse MoE + CSA/HCA hybrid attention + partial RoPE, and the prereg that governs its first run"
status: recorded
source: "facts/smelt_deeploop.json, facts/deepseek_v4.json; runs/prereg.jsonl#moe48_30b_0907@amended_12; 4c's 2026-09-08 brief"
---

# v2: loop transformer + sparse MoE + CSA/HCA hybrid attention

## Architecture

v2 is one package: the SMELT loop (middle 50% of layers execute twice), sparse
MoE in every layer, CSA-with-SWA + HCA hybrid attention with Lightning-Indexer
selection, and partial RoPE. KDA is removed. The control arm is the current
1.5b-a0.2b-e48 recipe — unlooped, KDA/MLA, NoPE — at equal active params and
equal tokens.

### Loop

Blocks 3-8 (the middle 6 of 12) execute twice per forward, giving 18 executions
(1.5L). This is SMELT's shape: the loop-span sweep at 200M puts the optimum at
6 layers (50%), val 1.9257, against 1.9445 at 0 layers and 1.9322 at 12
(full-stack looping regresses)
(facts/smelt_deeploop.json#smelt.ablation_and_downstream). The loop count is
2x: 1.9257 at 2x against 1.9385 at 3x and 1.9360 at 4x
(facts/smelt_deeploop.json#smelt.ablation_and_downstream). The existing
`--loop` flag (train.py:2855) implements this: `--loop 3 8` patches blocks 3-8
to run twice, recorded as `Cfg.loop_blocks`.

SMELT measured 6.8-18.0% compute-matched CE gain at 1e20-1e21 FLOPs, growing
~8pp per 10x compute (facts/smelt_deeploop.json#smelt.ce_gain). The scaling
law puts the frontier exponent at 0.250 vs 0.237 baseline (5.5% higher)
(facts/smelt_deeploop.json#smelt.scaling_law).

### MoE

Keep 48 routed experts, top-3, 1 shared, expert_ffn=768, every layer
(`moe_layers="0-11"`). This is the current 1.5b-a0.2b-e48 recipe:
`(top_k + shared) * expert_ffn = (3+1)*768 = 3072 = ffn_hidden`, the
exact-active-parity constraint MoEFFN refuses to violate (model.py:862). Load
balancing: bias_gamma 0.001, balance_alpha 1e-4, pre-registered from
DeepSeek-V3 and not tuned after a curve (train.py:254-256).

V4's MoE details NOT adopted in v2: top-6 with 256/384 routed experts
(facts/deepseek_v4.json#dsv4.moe) — our 48/top-3 is the registered recipe, and
changing it confounds the loop reading. Sqrt(Softplus) affinity, Hash routing
in the first 3 layers, and the routing-target-node constraint removal are V4
refinements a future loop can test (facts/deepseek_v4.json#dsv4.moe).

### Attention: CSA+DSA+SWA (landed), HCA + partial RoPE (unbuilt, target architecture)

KDA is removed in the treatment arm. The first v2 arm runs **CSA+DSA+SWA with
NoPE** (4c sequencing ruling, 2026-09-09): the loop is the only FLOP-multiplying
component under test, the control is NoPE, and putting partial RoPE in the
treatment would confound the loop's effect with a positional-encoding change.
Whether the first arm may run without HCA and partial RoPE is a user question
put 2026-09-09; this spec proceeds on the assumption it may, and the arm is
reversible if the answer is no.

**CSA as merged** (b0-35, `e9195327`, behind `Cfg.csa`, off by default and off
means absent): three branches over the q/k/v GatedMLA already builds, summed by
a zero-init gate — coarse attention over KV pooled every `csa_compress`
positions, full-resolution attention over the top-k blocks those coarse scores
pick (the DSA top-k), and an exact sliding window of `csa_window` recent
tokens (SWA).

**This is an NSA/DSA-family attention, NOT DeepSeek-V4's CSA**
(`runs/prereg.jsonl#v2_loop_moe_csa_0908@amended_3`): the grouping is
non-overlapping where the reference overlaps 2m neighbours, and there is no
Lightning Indexer — selection is top-k over the same coarse scores, not a
separate low-dim indexer. **A null result here must be read as this operator,
and so must a win**: attributing either to the V4 package would be wrong. The
V4 reference (overlapped grouping m=4, Lightning Indexer top-k=512, MQA core,
SWA n_win=128; Pro needs 27% of V3.2's single-token FLOPs at 1M context,
facts/deepseek_v4.json#dsv4.hybrid_attention, dsv4.efficiency_1m) is the
target the family is named after, not the implementation under test.

**HCA and partial RoPE are UNBUILT, not dropped.** As of 2026-09-09 neither
appears in model.py or train.py; the existing attention modules are
DeltaRecurrence (KDA, model.py:120) and GatedMLA (MLA, model.py:238). Both
stay in the target architecture the user named on 2026-09-07 (loop + sparse
MoE + CSA(SWA) + HCA, KDA out): HCA is KV compression m'=128 with dense
attention, and partial RoPE is the last-64-dim encoding with the -i output
correction (facts/deepseek_v4.json#dsv4.partial_rope). Unbuilt and dropped are
different states; a session cannot lift a user order, and the first arm's
sequencing is not a descoping.

### Positional encoding: partial RoPE

Partial RoPE on the last 64 dimensions of queries, compressed KV entries, and
core-attention outputs, with a -i position correction on the outputs so the
result encodes relative position (facts/deepseek_v4.json#dsv4.partial_rope).

This breaks checkpoint compatibility. Our checkpoints are NoPE: GatedMLA
carries no positional encoding, position is handled by KDA/DeltaRecurrence
(model.py:227). V4 ships the attention upgrade and the PE change together, and
the -i correction exists because CSA's KV-as-values structure introduces
absolute position (facts/deepseek_v4.json#dsv4.nope_rope_break). v2 is a
from-scratch run: no NoPE checkpoint is converted or continued. The first arm
is NoPE (see Attention): the break is paid when partial RoPE is built, not
before, and paying it inside the loop comparison would confound the reading.

### Residual

Standard residual. V4's mHC (n_hc=4, doubly-stochastic B via Sinkhorn-Knopp,
t_max=20) is NOT adopted in v2 (facts/deepseek_v4.json#dsv4.mhc). SMELT
measured with a plain residual
(facts/smelt_deeploop.json#smelt.ablation_and_downstream); mHC changes the
residual stream the looped updates land in and confounds the loop reading. mHC
is a future loop.

### AttnRes

AttnRes stays ON (`Cfg.attn_res = True`, train.py:320). The AttnRes x looping
interaction is an INTERVAL, not a point estimate
(facts/smelt_deeploop.json#repo.attnres_x_loop_corrected). The source-read law
is `n(n+1)` with `n = 2L+1`: 650 at L12, and the loop's 1.5L executions move
the effective depth the AttnRes denominator sees
(facts/smelt_deeploop.json#repo.attnres_loop_cost_derived). The loop_wrapper
already implements the ruled semantics: the second visit merges into the first
visit's done entry, so the source count is unchanged and no downstream AttnRes
softmax denominator moves (eval/loop_wrapper.py, option 3).

## The loop_not_adopted blocker

The repo already ran the loop experiment and ruled NOT ADOPTED on equal
compute (facts/smelt_deeploop.json#repo.loop_not_adopted_equal_compute). Three
arms from scratch at 122.30M, seed 42, identical except `--loop 4 7`. Equal
tokens: loop WINS (-0.022 nat, t -33.6, 537/576 blocks). Equal FLOPs: loop
LOSES (-0.044 nat, t -32.5; -0.038 humaneval BPB, t -6.45).

Three things make this verdict non-transferable to v2:

1. **MoE changes the looped path's cost structure.** The old experiment looped
   DENSE FFN layers. v2 loops MoE FFN layers. MoE's all-to-all + expert
   dispatch is a different fraction of the step than dense FFN
   (facts/smelt_deeploop.json#repo.moe_a2a_cost_h20). The loop's 1.5x FLOP
   multiplier lands on a different cost base.

2. **KDA is removed from the looped path.** The old experiment looped layers
   containing KDA (DeltaRecurrence). KDA's recurrent state across loop visits
   is unstudied: SMELT does not discuss stateful/linear/recurrent-state
   attention, and "how a recurrent state behaves across loop visits is
   unanswered by the paper"
   (facts/smelt_deeploop.json#smelt.stateful_architectures). v2 removes KDA
   and uses CSA/HCA, which carry no recurrent state. The per-FLOP loss the old
   experiment measured might have been a loop-KDA interaction, not a loop
   property.

3. **The equal-FLOPs comparison used more tokens, not width.** The old
   experiment's equal-FLOPs arm gave the unlooped model 26% more tokens
   (4824 steps = 1.2646B against 3815 = 1.0001B) at the SAME width. SMELT's
   compute matching runs through WIDTH: the looped model is wider, and the
   equal-FLOPs control is a thinner unlooped model
   (facts/smelt_deeploop.json#repo.smelt_shape_correction). These are
   different comparisons.

### What would retire looping a second time

The v2 run is a PACKAGE comparison (loop + attention + PE together), not a
loop isolation. The retire condition:

- If the v2 package does NOT beat the unlooped control at equal tokens (the
  primary metric, paired val loss), the loop is retired a second time —
  because the loop is the only FLOP-multiplying component, and a package that
  spends 1.5x FLOPs without beating the control retires the loop. The
  attention change (KDA → CSA/HCA) is independently motivated by long-context
  efficiency (facts/deepseek_v4.json#dsv4.efficiency_1m) and survives
  regardless.

- If the v2 package DOES beat the control, the loop is NOT automatically
  vindicated: the gain could be from the attention change. A loop-only A/B
  (same attention, loop vs no-loop) is registered as a CONDITIONAL follow-up,
  not part of v2.

- Tie-break: if val nats and humaneval BPB disagree in sign (as they did in
  the old experiment: -0.022 nat per-token but -0.038 humaneval BPB
  per-FLOP), the decision is the controller's, not an automatic adoption.

## Compute matching

The primary comparison is equal ACTIVE PARAMS and equal TOKENS: both arms use
d=1024, L12, heads8, ffn3072, MoE 48/top-3/1-shared/expert-ffn-768. The looped
arm spends 1.5x FLOPs (18 vs 12 executions). This is the "equal tokens"
comparison the old experiment found loop WINS
(facts/smelt_deeploop.json#repo.loop_not_adopted_equal_compute).

The equal-FLOPs comparison is a THINNER unlooped model: width matched so that
unlooped FLOPs = looped FLOPs. This is SMELT's compute-matching method
(facts/smelt_deeploop.json#repo.smelt_shape_correction). The thinner control
is a conditional arm, launched only if the primary comparison shows a gain.

## The NoPE/RoPE break

v2 is from-scratch. The NoPE/RoPE break
(facts/deepseek_v4.json#dsv4.nope_rope_break) is paid as a re-pretrain, not a
conversion. The attention path (CSA/HCA + partial RoPE) and the FFN/MoE path
are trained together from the same init.

## Prereg

Registered in `runs/prereg.jsonl#v2_loop_moe_csa_0908@amended_3`.

### Val-set definition

- Stop-trigger val: 160 rows (20 batches × 8), fixed by randperm seed 42,
  scored every `val_every` steps. This is the same instrument the 30B run uses
  (train.py:3769 passes `Cfg.val_batches = 20`, train.py:389).
- Endpoint comparison: the full val set (35,941 rows) scored with
  `eval/score_matrix.py` on the doc_cu path, per-domain.

### Warmdown

Warmdown 0.3: the cosine anneal starts at 70% of total steps. Basis, corrected
after review (tilerl-0a, 2026-09-08): the 30B run used warmdown 0.1 (anneal at
90%, `runs/prereg.jsonl#moe48_30b_0907@amended_12`) and val rose twice. The
first rise sits at the resume join (~22,500 steps = 59.0% of the corrected
38,146 total; parent 21400-22400 falling -0.01757/1k against resume 22600-23600
rising +0.02971/1k, t=+6.179, `@amended_9`): 78.1% of it is cot+chatml+chat_qa,
the three domains whose resume-1 weights fell to 0.076-0.078x, on 19.4% of
scored rows, primary prediction falsified (`@amended_10`), and the three
flattened to 0.13x/0.17x/0.19x by 25-26k (`@amended_11`) — starved-domain
forgetting, self-decaying. The second rise (26.4-29.6k, peak 1.884 at 29600)
took all nine domains up (runs/b0_domain_loss_valrise.jsonl); its three
hypotheses — composition, epoch wrap, repetition — are all falsified
(runs/review.jsonl, 2026-09-08T04:1xZ rows), and it reversed at full LR before
warmdown: -0.01041/1k, t=-28.67 over 29800-34000 — a 3200-step excursion, not a
regime change. Neither rise settles "peak LR held too long" as its cause: the
first is mix-driven, the second self-corrected without any lr change. The
endpoint table agrees: across the nine domains, the correlation between log
resume-1/parent weight ratio and endpoint-vs-8B BPB delta is r=-0.945
(recomputed from runs/score_matrix.jsonl, cu_none, profile full, and the
amendment-9 weight ratios) — the domains whose weight fell are the domains
that got worse. The rise is a supply effect; the schedule is not the
instrument for it, so the schedule choice is argued on its own grounds. v2 has
no resume join and trains the resume-1 mix from scratch, so the 30B episode
transfers no schedule prescription. Warmdown 0.3 is earlier than the 30B's 0.1
and WSD-conventional; the val_rise stop rule with the amendment-10 per-domain
instrument is the registered net. Warmdown 0.4 (anneal at 60%) was considered
and rejected: it over-fits the first rise's confounded onset and would truncate
a self-correcting excursion.

### Stop rules

1. **nan_or_oom**: any NaN in loss or val, or CUDA OOM. Stop.
2. **routing_collapse**: any expert's routed-token share drops below 1% for
   200 consecutive steps. Stop.
3. **cursor_unchecked**: the data cursor is not checked at resume. Stop.
4. **schedule_moved**: `total_steps` or `warmdown_start` differs from the
   registered value. Stop.
5. **val_rise**: two consecutive vals both exceed the anchor + 3 ×
   step-to-step SD. Stop, per-domain re-read, controller decision.

### Falsifying conditions

- **Claim "v2 beats control at equal tokens"**: the paired val difference's
  95% CI includes zero at the endpoint. FALSIFIED.
- **Claim "the loop contributes to the gain"**: the conditional loop-only A/B
  shows no per-token gain. FALSIFIED. (Conditional on v2 beating control.)
- **Claim "a rise under warmdown 0.3 is the supply effect the 30B showed,
  not a schedule failure"**: two consecutive vals rise during the peak-LR
  phase (before 70% of steps) AND the per-domain re-read (amendment-10
  instrument) attributes the rise to domains whose weight did not fall.
  FALSIFIED — the rise is not supply-driven, so warmdown 0.3 was too late.

## Owners

- b0: CSA+DSA+SWA implementation (LANDED, e9195327, behind Cfg.csa); HCA + partial RoPE when the user rules the first arm's sequencing insufficient
- de: loop + schedule
- e1: data
- 3b: mix/deriver/compat
- 98: report
- model.py review: tilerl-0a's session exited 2026-09-09 (the tileRL team is working, roster state active-session-unknown); reviewer to be reassigned by the controller

## What is not yet answered

- Whether CSA's Lightning Indexer works as described under partial RoPE at
  our scale — V4 gives no small-scale ablation.
- Whether the loop-KDA interaction was the source of the old per-FLOP loss —
  v2 removes KDA, but the loop-only A/B (conditional follow-up) is what
  isolates it.
- The partial-RoPE dimension (64) is small relative to V4's head dims
  (d_c=512/1536). Whether a 64-dim RoPE carries enough position signal at our
  seq_len, or whether the SWA branch (n_win=128) is doing the real positional
  work, is not separated in the paper.
