---
question: 在 200M/500 步这个 ladder 点上，多花 +33.6M 参数在一张 padded_vocab×d 表上值不值
status: measured
source: b0-17 + A/B (4) 的两个 exp 行、三条 score_matrix .ep1 行、runs/b0_17_rows.json；草稿给 e1 搬进 facts/efficiency.json（b0 不改 facts/）
---

# 两个 fact 草稿（b0-17）

1e 裁定：一条写"+33.6M 的表两种用法都不划算"，一条写"tied head 推大 tok 被否证"。
两条分开，因为第一条是配方结论、第二条是机制结论，混在一起会让前者继承后者的
不确定性。

---

## 1. `eff.padded_vocab_table_no_pay_200m`

**value** —

TWO DIFFERENT WAYS OF SPENDING +33,619,968 PARAMS (+16.3%, 206,128,200 -> 239,748,168) ON ONE
padded_vocab x d TABLE BOTH FAIL TO PAY FOR THEMSELVES AT 200M/500 STEPS -- AND THEY FAIL
DIFFERENTLY, which is the part an aggregate word hides. Both arms: world 4, 500 steps,
mix_200m_4b, seed 42, scored against the same base checkpoint's eval row
(ckpt_ab_shapelr_base.pt.ep1, domain_loss unweighted_mean 2.8480).

  A/B (4) --value_embed (one shared token-indexed table added to V in every MLA layer, gated by
  3*sigmoid over 12 residual dims): unweighted_mean 2.8479, delta -0.0001 nat. Per domain the
  sign is SPLIT 3 of 9 worse, mean -0.0002, sd 0.0100, t -0.05. Indistinguishable from zero in
  both sign and magnitude -- this is noise, not a small effect.

  b0-17 --untie_head (the LM head gets its own weights instead of aliasing tok.weight):
  unweighted_mean 2.8962, delta +0.0482 nat, 20% of the 0.24 bar. Per domain 9 of 9 WORSE, mean
  +0.0482, sd 0.0342, t +4.23. Consistently worse, and the consistency is the signal even though
  the mean is under a bar built for seed variance.

  A third arm (--untie_head --head_lr 0.003464, IDENTICAL parameter count to the second) reads
  2.9139, +0.0659 vs base and +0.0176 vs the second arm, again 9/9 worse (t +4.55). It does NOT
  measure a low-lr steady state: its head reaches row-norm median 1.5375 = 2.4x its 0.64 init, so
  500 steps at 28.9x lower lr is not enough to train it, and most of that +0.0176 is an
  initialization debt rather than a property of the lr.

WHAT THE TWO ARMS SHARE IS ONLY THE NEGATIVE: at this ladder point neither use of that parameter
budget helps. They are NOT one reproduced pattern. Calling them "two independent arms of the same
shape" -- which b0's own first draft of the b0-17 decision did, and 1e's ruling repeated -- claims
an effect was replicated where one of the two arms has no effect at all. The phrase that permitted
it was "flat-to-worse": it spans both outcomes without distinguishing them, so it reads as one
finding while covering two.

**measured** — 2026-09-03

**source** — runs/experiments.jsonl rows `ab_value_embed` (started 2026-09-02 17:18, done) and
`ab_untie_head` (started 2026-09-02 21:47, flat), both with pre-registered readings. Eval rows in
runs/score_matrix.jsonl: ckpt_ab_shapelr_base.pt.ep1, ckpt_ab_valueembed_valueembed.pt.ep1,
ckpt_ab_untiehead_untiehead.pt.ep1 (that last one lands when b0-17's readout finishes; arm 3's
per-domain numbers are runs/b0_17_arm3_dl.log, run separately on card 4). THOSE FIVE ab_ EVAL ROWS
EXISTED ONLY ON THE POD UNTIL ce6ea53a -- three closed A/B decisions had no artifact reachable
from the repo, and nothing checks for that (pod_drift asserts the files it lists MATCH, never that
a pod-written ledger row was brought back; de-36 is opening the pull path). Base code identity
attested in runs/b0_17_base_code_attestation.txt: SKIP_BASE matches world/steps/mix/seed and says
nothing about which train.py trained the base, so the diff was read and classified by hand
(b0-20 makes it mechanical).

**config** — 206M, L12 d1024 heads8 ffn3072, 3 MLA + 9 KDA, AttnRes ON, seq 4096, fp8 off,
padded_vocab 32832, vocab 32784, vocab_real 32773; batch 16 accum 2, warmup 300 of 500 steps,
warmdown 0.1, lr_scale 1.0; domain_loss on 9 domains x 262,144 val tokens, unweighted mean.

**uncertainty** — One seed, one depth, one token budget, per arm. No seed replicate, so the
run-to-run spread of a 500-step arm's unweighted mean is unmeasured HERE; the 0.24 bar comes from
ds.seed_variance_0p2b, measured on 0.2B runs, and applying it to these arms assumes that spread
transfers. Both t values above are paired per-domain statistics WITHIN one arm pair, not against a
seed distribution -- they say the sign is consistent across domains, not that the effect would
survive a reseed. 500 steps is 0.26B tokens; a table that needs more tokens than that to earn its
parameters would read exactly like this. minimal_pairs is NOT part of this reading: the five
500-step A/B checkpoints all sit inside 4.33pt (sd 1.67pt) against an 11.5pt readable-move bar, so
it cannot resolve any of these arms -- that is unresolvable, not flat.

**boundary** — Says nothing about 3.24B or 20B budgets, nothing about other depths, and nothing
about other uses of +16.3% params (both arms here happen to spend it on the same SHAPE of tensor).
Does not say the untied head is harmful in general: +0.0482 nat is under the pre-registered bar,
and the correct statement is "does not pay for itself at this budget", not "hurts". The per-domain
spread is real and unexplained: absolute deltas span 12.5x (en_c4 +0.0078 to zh_web +0.0977) and
relative 15.4x (+0.221% to +3.403%), and the two orderings DISAGREE (zh_web is 1st absolute, 4th
relative; textbook_30b 1st relative) -- so the structure is not a shadow of loss magnitude
(corr(base loss, absolute delta) = +0.592), but no positive explanation is offered. b0 also
checked whether the domain ordering matches eff.l9_branch_split_p200m's rescale ordering and it
does NOT: Spearman rho +0.40, sharing only a top cluster (chat/zh_web) while en_c4 moves 4th ->
9th and textbook 5th -> 2nd. An earlier claim that they were "almost the same shape" was retracted
on that number.

---

## 2. `eff.tied_head_does_not_inflate_tok_200m`

**value** —

THE TIED LM HEAD IS NOT WHY tok.weight GROWS UNIFORMLY, AND THE REFUTATION IS IN THE WEIGHTS
RATHER THAN IN A LOSS. The candidate mechanism (1e's, for b0-10's uniform per-quantile embedding
growth) was that a tied head trains tok.weight through a SECOND gradient path at embed_lr 0.1, so
removing that path should reduce the growth. It INCREASES it. Two medians, and they must not be
mixed in one sentence -- the WHOLE-TABLE median over all 32832 rows is 16.9115 -> 18.3308 ->
19.8036, while the median over the input+head class [0,32773) alone is 16.9211 -> 18.3390 ->
19.8118. The class figure is the one runs/b0_17_rows.json stores and the one the table below
uses; the whole-table figure sits slightly lower because it includes the 48 no-path rows near
0.62 and the 11 pad rows. Ratios on the class median, at full precision: x1.083796 (arm 2) and
x1.170835 (arm 3), i.e. the LOWER the head's lr the MORE tok grows. (Dividing the 4-decimal
medians instead gives 1.0838 / 1.1708 -- e1's independent recompute caught that my first draft
quoted 1.0839 / 1.1710, which is rounding-then-dividing, not the ratio.)

THE 11 ALIGNMENT-PADDING ROWS SETTLE IT, because they are the path itself and not a proxy for it.
Rows [32773, 32784) of tok.weight are never an input id and never a target (train.py:164-174), so
they are reachable ONLY through the head:

    row class            base tok    arm2 tok    arm2 head    arm3 tok    arm3 head
    [0,32773)             16.9211     18.3390      16.5053     19.8118       1.5375
    [32773,32784)          6.6575      0.0000       7.1036      0.0000       0.7302
    [32784,32832)          0.6209      0.6221       0.6209      0.6221       0.6408

Both untied arms read EXACTLY 0.0000 in the head-only class: the path was removed COMPLETELY, not
weakened. Arm 2's HEAD pad rows read 7.1036 against the tied base's 6.6575, so the path itself did
not weaken -- it got 1.067x stronger. Named path fully removed, path not weakened, effect moved the
other way: refuted, not merely unsupported. The no-path class sits at 0.6209-0.6408 against an init
of 0.02*sqrt(1024) = 0.6400 (model.py:453), which is the noise floor that makes "small"
distinguishable from "untrained" -- and it is why arm 3's head at 1.5375 reads as 2.4x init, i.e.
barely trained, rather than merely small.

WHAT REPLACES THE MECHANISM IS A GUESS AND IS LABELLED ONE: a tied table receives input-path and
head-path gradients that PARTIALLY CANCEL, and untying removes the cancellation so the input path
alone pushes tok further. No gradient signs were measured. What is established is only the
exclusion.

**measured** — 2026-09-03

**source** — scripts/head_path_rows.py at de0adf25 (five checks, all verified red on their own
broken code), artifact runs/b0_17_rows.json at 0397391a, run on the pod from the merged script
rather than an ad-hoc snippet. Zero cards: mmap load plus tensor norms. Checkpoints
ckpt_ab_shapelr_base.pt.ep1, ckpt_ab_untiehead_untiehead.pt.ep1,
ckpt_ab_untieheadlr_untieheadlr.pt.ep1. Prior: b0-10's uniform embedding growth and its interval-4
reading (runs/review.jsonl row b0-10-interval4).

**config** — as fact 1 above. The head-only class is [vocab_real, vocab) = [32773, 32784), 11
rows; the no-path class is [vocab, padded_vocab) = [32784, 32832), 48 rows. Untied arms zero tok's
pad rows at init (model.py:455, guarded by `head.weight is not tok.weight`), which is why 0.0000
there is exactly zero rather than an init-scale number.

**uncertainty** — One run per arm, one seed, one depth, 500 steps. The 0.0000 is a fact about
reachability plus that init zeroing, so it is not a statistical claim; the 1.067x on the head's pad
rows IS a one-sample comparison with no replicate. Arm 3's head is barely trained (2.4x init), so
its tok figure of 19.8118 (input+head class) conflates the lr change with an untrained head. AND
"LOWER HEAD LR -> MORE TOK GROWTH" IS NOT A TREND, NOR EVEN MONOTONE: there are exactly TWO
untied points, and two points admit no monotonicity -- any two values are ordered. Calling it
monotone (as an earlier draft of this line did) borrows the word's implication of a direction
that would persist. One of the two is also not at steady state. The honest statement is a
single comparison: the arm with the lower head lr had the larger tok growth, and its head was
undertrained, which is a candidate explanation for that ordering rather than evidence against
it. A third head_lr would be needed before direction is a claim.

**boundary** — Excludes ONE mechanism for b0-10's uniform growth; does not explain it. Says
nothing about why untying makes tok grow more, beyond ruling out that the head path was doing the
growing. The cancellation story is a hypothesis needing gradient signs on the two paths, which is a
different measurement from any A/B. The pad rows being nonzero in every arm's HEAD (6.6575 / 7.1036
/ 0.7302) is eff.vocab_padding_softmax_defect and is NOT a confound between arms -- all three carry
it at the same order, so it cannot produce a between-arm difference; it was checked for exactly
that. The copy-init arm (untied head initialized from the embedding table) is the only arm that
could separate tok's reverse growth from initialization cost, and it is DEFERRED, not dropped:
it answers a mechanism question with no measured loss consequence and changes no recipe at this
ladder point.
