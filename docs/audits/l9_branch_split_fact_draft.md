---
question: layer 9 的 KDA 分支劈叉有没有可测后果，还是被 bar 界住？——b0-16 读数落 facts 的条目草稿
status: measured
source: 状态 measured 指数字已实测且已复核（下方每个数都从 runs/*.json 重读过，不是从我的摘要抄的）；条目本身待 e1 提交进 facts/。b0 测量（scripts/l9_branch_probe.py，afad3b18 称重 + f8d85e66 三臂），产物 runs/b0_16_l9.json (0158a9cd) 与 runs/b0_16_l9_rescale.json (efd171a2)，review 行 b0-16-rescale。1e 裁定进 facts/efficiency.json；b0 不编辑 facts/，由 e1 提交
---

DRAFT for facts/efficiency.json -- b0 measured it, e1 commits it (b0 does not edit facts/).
Complete: all three rescale arms landed (b0-16 closed at efd171a2, review row b0-16-rescale).

{
 "id": "eff.l9_branch_split_p200m",
 "value": "ONE KDA LAYER'S OUTPUT PROJECTION STOPPED GROWING WHILE ITS PEERS DID NOT, AND ONLY
   mixer.o IS INVOLVED. On p200m_4b_0902 (L12, 9 KDA + 3 MLA), layer 9's mixer.o grows 1.2037x
   over step832->step3500 against the other eight KDA layers' median 1.7334x (MAD sigma 0.0241,
   z -21.9), while the SAME layer's ffn.w2 grows 1.6440x against their 1.6417x (z +0.18). The
   branch ratio mixer.o/ffn.w2 therefore falls away from the 12-layer median: 1.002x at step832,
   0.993x at step1192, then 0.836x / 0.746x / 0.698x at step2500/3000/3500 while every other
   layer holds 0.867. mixer.o's growth arrests at step2500 (1.2098 -> 1.1981 -> 1.2037).
   THREE MECHANISMS EXCLUDED BY MEASUREMENT: not a dead branch (all twelve Muon momenta live,
   4.8e-04..2.1e-02); not weight decay (Muon's step is w -= lr*NS(m) + lr*wd*w*mask and
   ||NS(m)||_F ~ sqrt(1024) = 32 by construction, so the push is lr*32 = 0.32 per step against
   lr*wd*|w| = 0.0051, 63x weaker -- the decay-equilibrium model solves to |w| ~ 2900, far above
   both layer 9's 46.6 and its peers' 64, so it refutes itself); not interval choice, though the correct
   statement is narrower than an earlier draft of this line claimed: the ratio falls monotonically
   across step832/1192/2500/3000/3500, but those intervals are 360/1308/500/500 steps, so only the
   last two are equal-length. What IS matched-interval is the direction-consistency reading below
   (step3000->3500, n=500 for every layer, same lr and shape), and that is where the cross-layer
   comparison is made.
   WHAT DIFFERS IS DIRECTION CONSISTENCY: displacement over step3000->step3500 against the
   fully-aligned budget lr*32*n = 160 is 6.85% for layer 9 against 13.29% for the other eight KDA
   layers (MAD sigma 0.33pp, z -19.3). CONSEQUENCE: BOUNDED, NOT ZERO. Three arms on step3500, one tensor
   changed between them: unscaled 2.1551; layer 9's mixer.o x1.4330 (the factor that puts its
   branch ratio at the 12-layer median) 2.2038, delta +0.0487; the SAME factor on layer 6 (control,
   ratio 0.8719, nearest the median) 2.1616, delta +0.0065. Both are under the 0.24 nat bar, so on
   b0-16's pre-registered reading the split is BOUNDED. But the arms are not equal -- layer 9's
   delta is 7.5x the control's -- so the probe has resolution and layer 9 is the sensitive layer;
   it lacks only enough resolution for a bar built for seed variance. Direction: forcing the branch
   to the median makes loss WORSE, which means the trained value beats the median, NOT that the
   split is harmless.",
 "measured": "2026-09-03",
 "source": "scripts/l9_branch_probe.py at f8d85e66 (probe + control arm) and afad3b18 (the
   weights-only reading); table artifacts runs/b0_16_l9.json and runs/b0_16_table.log committed at
   0158a9cd, rescale artifacts runs/b0_16_l9_rescale.json and runs/b0_16_rescale.log at efd171a2;
   checkpoints ckpt_p200m_4b_0902.pt.interrupt.step832, .interrupt.step1192, .pt.step2500,
   .pt.step3000, .pt.step3500, all five KEEP-claimed at 6625aa68. Review row b0-10-interval4 in
   runs/review.jsonl is where the split was first seen; task b0-16.",
 "config": {
  "run": "p200m_4b_0902",
  "model": "206M, L12 d1024 heads8 ffn3072, 3 MLA + 9 KDA, AttnRes ON, seq 4096, fp8",
  "block_split": "KDA = blocks with mixer.A_log present; MLA = the rest; 9 KDA, 3 MLA",
  "criterion": "branch ratio = |mixer.o| / |ffn.w2| in fp32 per block, against the 12-layer
    median; growth = ratio of a tensor's fp32 norm between two checkpoints; direction consistency
    = ||w_b - w_a||_F / (lr * sqrt(1024) * n), i.e. realized displacement over the budget if every
    Muon step pushed the same way",
  "muon_group": "lr 0.01, wd 0.010881 at step3500 (initial_wd 0.1, decayed), momentum 0.95, ns_steps 5",
  "domain_loss_bar": "0.24 nat on the unweighted mean (ds.seed_variance_0p2b)"
 },
 "uncertainty": "n=8 on the peer side, one seed, one run, one depth. No seed replicate, so the
   run-to-run spread of a per-layer branch ratio is unmeasured -- z -21.9 is against the
   within-run cross-layer spread, NOT against a seed distribution, and those are different
   quantities. Why layer 9 and not another layer is unexplained; nothing here says the position
   is reproducible. The direction-consistency figure uses ||NS(m)||_F ~ 32 as the per-step push,
   which is Newton-Schulz's design property rather than a measurement of this run's updates.",
 "boundary": "BOUNDED ON A BAR THAT IS THE WRONG INSTRUMENT FOR THIS EFFECT, which is the
   honest form of the verdict. 0.24 nat is ds.seed_variance_0p2b -- the seed-to-seed spread of a
   whole run's unweighted mean. The rescale is a 0.0487 perturbation of ONE checkpoint measured
   against itself with no seed involved, so run-to-run seed noise is not the noise this measurement
   has; reading a 0.05 effect would need paired per-domain deltas on a fixed batch (where A-vs-A is
   0 by construction), around 0.024. That is a different instrument, not more arms, and it is NOT
   offered as a re-reading of this result: on the pre-registered bar the answer is bounded and no
   action follows at 200M. THE TABLE CANNOT UPGRADE IT EITHER:
   domain_loss falls monotonically because the model is training and the ratio falls monotonically
   too, so the two correlate whatever the ratio means -- the five-point table is confounded by
   construction and is recorded for provenance, not as evidence of consequence. Says nothing about
   val loss or any eval outside domain_loss. Says nothing about L32: the L32 analogue (MLA layer 31)
   is a different tensor in a different block kind. The wd-exclusion arithmetic above is what killed
   my own first hypothesis, so it is recorded as a positive exclusion rather than an unexamined
   assumption. NO CO-RESIDENCY CONFOUND, and an earlier draft of this line asserted one that
   does not exist. 1e reported an 11.4 GB external tenant on card 7 at 21:0xZ; the 11443 MiB I
   then read at 21:12Z was MY OWN domain_loss process, not that tenant -- card 7 reads 3 MiB
   idle, and nvidia-smi -i 7 --query-compute-apps names only my pid. Wall times are therefore
   comparable within this run, and no boundary is needed. I nearly recorded the confound into a
   fact: a teammate reporting a tenant, plus the card showing memory, is not the same fact as the
   tenant being on my card -- the join has to go through the pid.",
 "status": "measured"
}
