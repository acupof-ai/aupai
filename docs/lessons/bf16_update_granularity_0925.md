---
question: Does bf16 in-place storage suppress parameter updates in pretraining?
status: measured
source: scripts/ckpt_update_granularity.py over adjacent v41_ced_0923 checkpoints (mmap, CPU, read-only, self-vs-self control); fact v41.bf16_no_master_copy_update_granularity_0925
---

# bf16 storage sets an update threshold, and the threshold is lr-dependent

The claim under test: "bf16 storage means parameters barely update." It comes from a single-step
probe that saw only **0.16% of elements move** under Muon at peak lr. That sits badly against a
run whose val fell 1.79 → 1.53 and whose HumanEval rose 10% → 17%, and against the theory that a
Muon element's relative update should be **1e-3 to 1e-2** — the probe reported **1.6e-5**, three
orders of magnitude low.

**Result: the storage does not stop updates, but it does set a per-element threshold, and at the
tail learning rate that threshold is severe.** At the tail lr, the larger half of the MoE expert
weights **did not move at all in 500 steps**. At the mid lr the same weights move freely.

## 1. Control first

Same checkpoint against itself: **0 of 3,255,530,048 elements changed, rel_l2 exactly
0.000e+00** in all 13 parameter groups. The control is a gate in
`scripts/ckpt_update_granularity.py --control` — it exits non-zero rather than printing a table,
because a comparison that cannot return zero on identical inputs is measuring itself.

## 2. Two windows, same 500 steps, different lr

| window | Muon lr | total moved | w13 rel_l2 | attn qg rel_l2 |
|---|---|---|---|---|
| 37000 → 37500 | 5.50e-4 → 5.16e-4 | **10.96%** | 6.73e-3 | 1.05e-2 |
| 37500 → 38000 | 5.16e-4 → 5.01e-4 | 10.39% | 6.37e-3 | 9.94e-3 |
| 38000 → 38146 (146 steps) | 5.01e-4 → — | 8.60% | 3.16e-3 | 4.91e-3 |
| **17500 → 18000** | **9.39e-3 → 9.20e-3** | **69.41%** | **4.05e-1** | **5.00e-1** |

The mid-window lr is **17.44x** the tail. The update is **60.1x** larger in rel_l2 — an exponent
of **1.43**, where a linear response would give 17.44x. A threshold that discards an increasing
fraction of steps at the low end produces exactly that superlinearity.

## 3. The discriminator: which elements move

Two explanations predicted different things, and the test was stated before it ran:

- a **signed random walk accumulating across steps** → moved% roughly **flat** across |w| deciles;
- **in-place round-to-nearest with no residual** → each step faces the element's own *relative*
  half-ulp, so only elements whose step-update exceeds half their ulp move, and since half-ulp
  grows with |w| the profile **collapses** as |w| grows.

I had proposed the first; 1e proposed the second and specified this test. **The data chose the
second, decisively.**

**MoE w13, moved% by |w| decile:**

| decile | tail lr (37000→37500) | mid lr (17500→18000) |
|---|---|---|
| 1 (smallest \|w\|) | **60.57%** | 70.38% |
| 2 | 26.07% | 69.68% |
| 3 | 7.80% | 69.06% |
| 4 | 1.12% | 68.64% |
| 5 | 0.011% | 67.49% |
| 6 | 0.005% | 67.58% |
| 7 | 0.001% | 66.41% |
| 8 | **0.000%** | 65.65% |
| 9 | **0.000%** | 65.47% |
| 10 (largest \|w\|) | **0.000%** | 63.00% |
| **total** | 9.56% | 67.34% |

**At the tail lr the profile collapses monotonically to exactly zero; at the mid lr it is nearly
flat.** The same collapse reproduces in attention qg at the tail (97.81% in decile 1 → exactly
0.000% in deciles 7–10). So the movement reported by the aggregate is **entirely concentrated in
the small-|w| tail**, and the large-|w| elements are stationary — in the tail window, for 500
steps, without exception.

## 4. Why the probe was not wrong, only misread

The single-step probe saw 0.16% of elements move. That is consistent with everything above: at
the tail lr a single step's update is a small fraction of the element's half-ulp for most
elements, so most single steps round to no change. **The error was reading a per-step figure as a
long-run rate.** My own first explanation — that the updates accumulate as a random walk — was
wrong for the mirror-image reason: bf16 stores in place with round-to-nearest and **keeps no
residual**, so a discarded step leaves nothing behind and the next step starts from the same
value. There is no accumulation to invoke.

## 5. Structure of the checkpoint

There is **no fp32 master copy anywhere**. The `model` state dict is 219 bf16 + 12 fp32 tensors,
and the 12 fp32 tensors are exactly `blocks.*.ffn.expert_bias` (48-dim each). Muon's only
optimizer state is `mb`, a **bf16 momentum buffer**; the three AdamW groups carry fp32
`exp_avg`/`exp_avg_sq`.

`expert_bias` is the one group that changes **100% of its elements in every window** with rel_l2
up to 2.9e-1 — because it is stored in fp32, so it faces no bf16 threshold at all. It is a useful
internal positive control: the group that is *not* bf16 is the group that always moves.

## 6. What this does not say

**"The second half of pretraining is ineffective for large-magnitude weights" is a hypothesis to
test, not a conclusion.** The measurement establishes the mechanism and its lr-dependence. It
does **not** establish that the stationary large-|w| elements *needed* to move, and it says
nothing about loss, val, or HumanEval. Answering it requires knowing what the update *would* have
been without bf16 — an fp32 shadow run — which has not been done.

Other limits: the windows are checkpoint-to-checkpoint, not per-step, so any per-step figure is a
√500 extrapolation, and the finding that steps are *discarded below a threshold* is itself
evidence that the independence that extrapolation assumes does not hold. The lr exponent
compares two windows that differ in more than lr (warmdown shape, optimizer state, position in
training), so superlinearity is consistent with the threshold mechanism without being identified
as its sole cause. Deciles are computed per tensor and pooled; the collapse is far larger than
that pooling could produce.

## 7. Reproduction

```
python3 scripts/ckpt_update_granularity.py --control <ckpt>          # must be all-zero
python3 scripts/ckpt_update_granularity.py --a <ckpt_a> --b <ckpt_b> --deciles
```

Read-only `torch.load(mmap=True, map_location="cpu")`; nothing is written and no checkpoint is
modified. Both sides are held at once (~26 GB per 13 GB pair), so run one pair at a time.
lr values are read from each checkpoint's own `opt[*].param_groups[*].lr`, not from a schedule
file. The mid-lr pair came from `/data01/aupai_backup/v41_ced_0923/`, sha256-verified against the
originals before use (`bba35a5099b9e9ef…`, `99180324e4ad626a…`) and deleted afterwards.
