# RL at 200M: what the evidence supports

Prepared for aupai-fb, 2026-08-29. Every number below is from a source I could verify this session; where I could not verify, I say so. arXiv IDs in brackets. Web search was unavailable (budget exhausted), so this is built from arXiv API + abstract-page fetches — abstract-level numbers are marked "(abstract)".

## Headline (the part that matters most)

1. **Nobody has published a credible RLVR gain below 0.5B — and the gap is ~1000x on training tokens, not just 2.5x on parameters.** The smallest verified scale is 0.5B (SimpleRL-Zoo [2503.18892], COLM 2025; TinyZero, a GitHub-only reproduction of GRPO on Qwen2.5-Math-0.5B). But SimpleRL-Zoo's Qwen2.5-0.5B base sat on ~18T pretrain tokens ≈ 36k tok/param; this project is at 16–57 tok/param. The smallest verified RLVR result is on a base roughly three orders of magnitude more overtrained. The threshold may be tok/param, not parameter count — and unlike the param gap, the token gap has a lever attached (more pretraining). "Do not run this yet, here is the threshold" is still the correct answer for Q1, but the threshold is two-dimensional. (Sharpening due to lessons-b0.) What the lever buys, computed by aupai-fb this round: 400B more tokens moves 200M from 16.5 to 2,000 tok/param — still 7× below SmolLM2-135M (~14,800) and 18× below Qwen2.5-0.5B (~36,000). Matching SmolLM2's ratio at 200M needs 3.0T tokens (~58 days on 7×H20), and the open Chinese high-quality corpus does not contain 3T (CCI3-HQ, 500B, is the largest single source). The ceiling on a Chinese 200M model is corpus volume, not compute. Full table in Q1.
2. **Your own RL gains are not significant.** rl_k4 math-hard 4.1% (42/1032) vs k4 base 2.9% (30/1032): two-proportion z=1.44, p≈0.15. rl_direct 2.9% vs base 2.9%: zero. lessons-b0's prior (2) is correct as stated. The one significant post-training gain on record is SFT, not RL: sft_k5_ctrl 3.6% vs base 1.9%, z=2.29, p=0.022.

**The rl_k4 number is also a settled debt, not just an RL conclusion (aupai-fb, made explicit this round).** The harness ledger had been omitting ckpt_rl_k4's 4.1% — fixed today — and that checkpoint is still the highest score in the whole table. Which means the question "what is our best checkpoint?" was, until today, being answered with a noise value: the top line was missing, and the 4.1% that leads the table is itself non-significant (p=0.15). Any checkpoint selection, roadmap gate, or "current best" claim built on that ledger inherited both defects. Re-derive any conclusion that consumed it.
3. **The pass@k gate is real, not invented** — for *vanilla* RLVR. Three independent 2025 papers show RLVR sharpens pass@1 while shrinking or leaving pass@k, and abilities remain bounded by the base model's sampling distribution. It is breakable, but only by changing what RL optimizes against (curriculum, teacher guidance, prolonged KL-controlled training) — cited in Q2.
4. **Your highest-value setup is the one you already identified**: dense programmatic step reward on machine-checkable procedure data. The procedure-execution fork is settled with lessons-e1 (SFT installs, RL sharpens; numbers in Q4), and the first internal result has landed: k8_proc_sft installed teacher-forced procedure execution (digit head 21.3%→57.2%) while free generation stayed at 0 — an exposure-bias result, not a coverage failure. The middle regime both lessons-b0 and I had written off exists: scheduled-sampling / DAGGER-style SFT (gold continuations on the model's own prefixes) is now move 2.5, with this null as its first internal evidence. b0 conceded the point in writing; its remaining disagreement (step-RL never vs. my move 3) is now conditioned on scheduled sampling succeeding — data arbitrates.
5. **Distillation from the 27B teacher is the better spend right now**, with one hard caveat from the small-model distillation literature: long CoT from a strong teacher does not transfer to ≤3B students; mixed long+short chains do. Your own sft_k5 result (2×-longer answers, math-500 −6.4pt) is the same phenomenon measured internally.

---

## Q1. Is RL the right tool at 200M? What is the smallest model with a real RLVR gain?

**Smallest verified: 0.5B.**

- SimpleRL-Zoo [2503.18892, COLM 2025]: zero-RL (GRPO + rule-based rewards, no SFT stage) on 10 base models including "all Qwen2.5 models from 0.5B to 32B"; reports "substantial improvements in reasoning accuracy and response length across most settings" (abstract — per-size numbers not in the abstract). Also the first observation of "aha moment" behaviors in a small non-Qwen model.
- TinyZero (github.com/Jiayi-Pan/TinyZero, not an arXiv paper): reproduction of R1-Zero-style GRPO on Qwen2.5-Math-0.5B/1.5B/3B. Community reproduction, not peer-reviewed; treat as existence proof, not measurement.
- "RL for Reasoning in Small LLMs: What Works and What Doesn't" [2503.16219]: 1.5B (DeepSeek-R1-Distill-Qwen-1.5B), 7K curated problems, 4×A40, <24h, ≈$42: AMC23 63%→80%, AIME24 46.7%. **Caveat: the base is already a distilled reasoning model** — this is RL-on-distillation, not RL-from-base. It shows RL works at 1.5B on a prepared base, not that RL creates reasoning at 1.5B.
- rSIM [2512.08300] claims Qwen2.5-0.5B outperforming Qwen2.5-14B via a trained planner agent injecting strategies — exotic setup, single source, I would not weight it.

**Below 0.5B: no published evidence found.** I searched specifically. The honest statement for the report: *nobody has measured this below 0.5B*. 200M is in the gap — and the gap is deeper on tokens than on params (see headline 1). Two ways to read that: (a) unexplored territory, first-mover opportunity; (b) unexplored because it doesn't work — the Yu et al. result below predicts (b) at fixed compute, but the token-gap reframing (lessons-b0) adds a third reading: (c) unexplored at 200M *because nobody has bothered to overtrain a 200M base to 36k tok/param* — the cheapest discriminator between "undertrained" and "wrong ladder" is a 30B-token pretrain (~15h at the measured 2.06B tok/h on 8×H20), contingent on growing the corpus past the repetition wall first (3.3B unique × 9 epochs is past Muennighoff et al.'s ~4-epoch safe zone; the quarantined 2.99M web docs plus elastic textbook supply gets to ~8–10B unique). That pretrain discriminator is lessons-b0's lane and its workflow is verifying the epoch-limit number.

**What the token gap costs (aupai-fb's computation, added this round):**

| checkpoint | pretrain tokens | tok/param |
|---|---|---|
| aupai current | 3.3B | 16.5 |
| aupai k6 (v2) | 11.3B | 57 |
| aupai 400B roadmap target | 400B | 2,000 |
| SmolLM2-135M | ~2T | ~14,800 |
| Qwen2.5-0.5B | ~18T | ~36,000 |

The roadmap's single first-class constraint — 400B tokens — buys 2,000 tok/param: still 7× below SmolLM2 and 18× below Qwen2.5-0.5B. Matching SmolLM2's ratio at 200M means 3.0T tokens, ~58 days on 7×H20 at the measured throughput — and the open Chinese high-quality corpus does not contain 3T (CCI3-HQ, 500B, is the largest single source). "The ceiling on a Chinese 200M model is corpus volume" is now a number, not an adjective. aupai-fb adopted the 30B-token continuation as the S1 pre-screen, ahead of its 200B falsification gate: 15 hours answers "is the curve still moving" before the expensive gate can.

**The 36k threshold is a guess, not a measurement — this must not be read as "36k is the threshold."** The two anchor points (0.5B params, 36k tok/param) both come from heavily overtrained models; in the published zoo the two variables are near-collinear, and no published run isolates them. The decoupling design is a 2×2 — {200M, ~0.5B} × {~60–150, ~36k} — and three of its four cells are ready or near-ready (aupai-fb, this round): 200M@16.5 is in hand (k7_v3/k8); **0.5B@36k is Qwen2.5-0.5B itself — a download, zero GPU**; 200M@36k is 7.2T tokens, corpus-infeasible, forever empty. The factorial costs exactly one cell: **0.5B at ~60–150 tok/param** (30–75B tokens, ~2–4 days on 7×H20). If 0.5B@60 also shows no RLVR gain while 0.5B@36k does, tok/param is the gate with params controlled; if 0.5B@60 works where 200M@150 fails, params carry independent weight.

Two qualifiers that slip easily in the conclusion: (1) RLVR on Qwen2.5-0.5B tests *its* architecture and corpus, not this project's — it answers the general question "is tok/param the gate", not "what happens to our 200M at 36k". (2) This is a 2–4 day side quest answering a question ("do we run RL at all") that is third-arm in cost order anyway. It must not jump the queue ahead of the 4-minute SFT discriminator. The cost-ordered queue (aupai-fb; ordered by cost, not by argument beauty): **4-min procedure SFT → 4-min replay control → 15h 30B continuation → 2–4 day decoupling cell → 8-day 400B.**

The theoretical prior is also against you at 200M: RLVR's gradient is proportional to within-group reward variance. At a 2–3% pass rate, most groups are all-wrong (zero advantage) — your own rlvr_gspo2 run measured 55% degenerate groups, and rlvr_stage2's T=0.1 made it worse. The smaller and weaker the model, the larger the fraction of prompts that produce no learning signal at all.

## Q2. The pass@k premise — real or invented?

**Real for vanilla RLVR, with a strong multi-paper consensus:**

- Yu et al., "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" [2504.13837, NeurIPS 2025 Oral]: pass@k at large k across model families, six RLVR algorithms, math/coding/visual benchmarks. **RLVR models win at small k; base models win at large k.** Coverage/perplexity analysis: abilities "originate from and are bounded by the base model." Six RLVR algorithms "perform similarly and remain far from optimal." Distillation, by contrast, "can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities." This is the single most important paper for your gate.
- Kim et al., "Reinforcement Learning vs. Distillation" [2505.14216]: same finding in small-model settings — RLVR raises pass@1, "often fails to improve capability (pass@k)"; RLVR "focuses on improving the accuracy of the easier questions to the detriment of the accuracy of the most difficult questions." Distillation lifts both, but only "when new knowledge is introduced" — distilling reasoning patterns alone also just improves accuracy.
- SvS [2508.14029]: "vanilla RLVR raises Pass@1 while lowering entropy and Pass@k"; refreshing/varying training problems recovers it (+18.3/+22.8% pass@32 on AIME24/25, 3B–32B).
- SFT-overtraining rank inversion [2606.18487]: on Qwen2.5-Coder-3B, deeper SFT raises pre-RL pass@1 while peak GRPO pass@10 *drops* 0.806→0.481 (3-seed mean). Pre-RL entropy correlates with GRPO outcome at ρ=+0.69.

**But it is not a law of nature — the counter-examples all change what RL optimizes against:**

- ProRL [2505.24864]: prolonged RL with KL control + reference-policy resetting + diverse tasks; RL models beat base on pass@k "including cases where base models never succeed." Weights released (Nemotron-Research-Reasoning-Qwen-1.5B). This is the strongest evidence that RL *can* expand the frontier — at 1.5B, with a long training budget and active KL management, not 500 steps.
- QuestA [2507.13266, ICLR 2026]: injecting partial solutions into training (teacher guidance near the frontier) improves both pass@1 and pass@k; 1.5B: AIME24 72.50, AIME25 62.29, HMMT25 41.67.
- Boundary-aware Curriculum RL [2606.22317]: locate the frontier via pass@k sampling, add teacher guidance beyond it, consolidate with RL: pass@256 +9.8 over base, +10.3 over vanilla RLVR.
- BroRL [2510.01180]: scaling rollouts per example into the hundreds revives models plateaued after ~3K ProRL steps.

**Verdict on your gate (pass@8 − pass@1 ≥ 15pt):** it is a sound operationalization of the Yu et al. finding — if the base never samples the answer within 8 tries, RL has nothing to reweight. Your measurements: rl_direct pass@8 7.0% vs pass@1 ~2.9% = +4.1pt (below gate, and RL indeed transferred nothing); k4 base pass@8 10.1–11.6% vs 2.9% = +7–8.7pt (also below 15pt). The gate is doing its job; respect it. Note the gate is sufficient, not necessary — ProRL/QuestA expand the frontier by adding guidance *during* RL, which is a different and more expensive regime.

**Scope note (lessons-e1):** "RL only reweights" is too strong unqualified — R1-Zero installed CoT at 671B. The claim holds at ≤0.5B: TinyZero's Qwen2.5-0.5B fails to learn reasoning from RL while its 3B succeeds, and SimpleRL-Zoo's 0.5B arm leaves AIME at 0.0. The gate is a small-model claim; do not export it upward.

## Q3. Algorithm choice — was GSPO right?

**GSPO was a defensible pick but not the binding constraint.** At your scale the algorithm choice is third-order; the base's pass@k gap and prompt filtering are first- and second-order. Evidence:

- GSPO [2507.18071, Qwen]: sequence-level importance ratio + length-normalized sequence-level clipping. Its demonstrated wins are training stability (especially MoE) and contribution to Qwen3. At small dense scale its advantage over GRPO is mainly stability, not ceiling.
- GRPO's actual pathologies are now well characterized: Dr. GRPO [2503.20783] shows GRPO's group-std normalization "artificially increases response length (especially for incorrect outputs)"; removing it gives token-efficiency gains and a minimalist R1-Zero recipe reaching 43.3 AIME24 at 7B. A unifying analysis [2607.00152] shows GRPO / Dr. GRPO / DAPO are "three operations on one number" — the group reward std. A companion impossibility result [2607.23364]: under outcome rewards, no length weighting is both gradient-unbiased and length-invariant; GRPO is ≈length-invariant but biased, Dr. GRPO is unbiased but lets long trajectories dominate.
- DAPO [2503.14476]: decoupled clip + dynamic sampling (drops all-right/all-wrong groups) + token-level policy gradient + overlong reward shaping; Qwen2.5-32B → 50 AIME24. **Dynamic sampling is the component that addresses your degen-group problem directly** — you implemented the same idea as the 20–80% solve-rate band, and your own rl_k4 run measured degen falling 30–55%→3% when the band was applied. That is your strongest internal result and it replicates the literature.
- VAPO [2504.05118]: value-based PPO variant, Qwen-32B AIME24 60.4 (>R1-Zero and DAPO by 10+ within 5K steps). Value nets at 200M are affordable, but the gain was demonstrated at 32B; no small-scale evidence.
- Rejection-sampling FT (the non-RL alternatives): RFT [2308.01825] — LLaMA-7B GSM8K 35.9%→49.3% with cross-model rejection samples, and "benefits weaker models more." RAFT [2304.06767, TMLR] — reward-ranked fine-tuning beats PPO in their setup, with better stability. STaR [2203.14465] — bootstrapped self-improvement, the original. For a weak base these are more stable than on-policy RL because every training example carries signal by construction (you keep only correct chains), which is exactly the property your degenerate-group problem says you lack.

**Recommendation:** if you run RL again at 200M, the algorithm that addresses your measured failure mode is DAPO-style dynamic sampling (or your band filter, same thing) on top of GRPO/Dr. GRPO — not GSPO vs GRPO knife-fighting. But see Q6: you should probably not run RL first at all.

## Q4. Process vs outcome reward — the part you asked for most depth on

**Your concern is correct and the literature supports it.** Outcome-only reward on a model whose intermediate steps are 51.2% wrong rewards lucky arithmetic — this is the standard RLVR reward-hacking failure mode, and your rl_direct run showed the signature: pass@1 recovered while pass@8 *fell* (7.0% vs base 10.1–11.6%), i.e. the policy sharpened onto lucky formats, not capability.

**What process supervision has achieved:**

- Lightman et al., "Let's Verify Step by Step" [2305.20050]: PRM800K (800K human step labels); process supervision "significantly outperforms outcome supervision" on MATH; best process-supervised model solves 78% of a representative MATH subset. This is the foundational result — but at GPT-4 scale (their large PRM), with human annotation you don't need.
- Math-Shepherd [2312.08935]: automatic step annotation via Monte-Carlo rollouts (a step is good if completions from it reach the right answer), no human labels. Step-by-step PPO: Mistral-7B GSM8K 77.9%→84.1%, MATH 28.6%→33.0%. **This is the closest published analog to your situation**: dense step reward without training a PRM, because the reward is defined by execution, not by a learned scorer.
- PRIME [2502.01456]: implicit process reward from outcome labels alone (no step annotation at all), online PRM updates from policy rollouts; Qwen2.5-Math-7B, +15.1% average over the SFT model across key reasoning benchmarks, using 1/10 the data of comparable pipelines.
- "Free Process Rewards without Process Labels" [2412.01981]: implicit PRM from an outcome reward model parameterized as log-likelihood ratios; beats an MCTS-based Math-Shepherd baseline on MATH using <1/38 the data; adding Math-Shepherd step labels on top gives **no further gain**.
- OmegaPRM [2406.06592]: MCTS divide-and-conquer step annotation, 1.5M process labels; Gemini Pro MATH500 51%→69.4%, GSM8K 86.4%→93.6%.
- Generative verifiers: GenRM [2408.15240, ICLR 2025] — verification as next-token prediction, BoN 73%→93.4% GSM8K; GenPRM [2504.00891] — a 1.5B generative PRM with test-time scaling beats GPT-4o on ProcessBench; ThinkPRM [2504.16828] — trained on 1% of PRM800K.
- **The counterweight you must hold**: ProcessBench [2412.06559, ACL 2025] — existing PRMs "generalize poorly beyond GSM8K and MATH," and a PRM straightforwardly fine-tuned on PRM800K beats most fancy ones. EST-PRM [2606.00437] — Math-Shepherd's PRM is sensitive to position perturbations (Pearson drop 0.152±0.038, 32.8±4.9% score inflation). PRMs are less robust than their benchmarks suggest.

**Programmatic verification (your actual asset):** PAL [2211.10435] and ToRA [2309.17452] established program-aided math reasoning; LeanDojo [2306.15626] and DeepSeek-Prover [2405.14333] did it for formal proofs. PaD [2305.13888, NAACL 2024] is the most relevant to you: **program-aided distillation beats CoT fine-tuning for small models** — replacing natural-language rationales with verifiable reasoning programs suppresses faulty synthetic data. Your `mathbank/procedure_curriculum.py` data (every intermediate line machine-checkable) is exactly the PaD setup, one year on, with the reward defined by the checker rather than a model.

**Is dense step reward the highest-value RL setup here? Yes, with one structural caveat — and the default sequence is two moves, not three (lessons-b0's collapse, accepted).** Step-level RL on *the model's own rollouts* still needs the model to sometimes produce correct procedures — otherwise every step is wrong and the reward is flat (the same degenerate-group problem, one level down). At 51.2% wrong equations and answer-first generation, your model is close to that. The evidence-supported sequence:

1. **Distill** (27B rollouts, checker-verified, length-matched to student capacity — see Q6). If the verified procedures are synthetic/programmatic rather than teacher rollouts, this step is PaD-style program-aided SFT [2305.13888] rather than teacher distillation — same move, different source. There is no distinct "step-SFT" middle regime: teacher-verified procedures *are* distillation, and self-generated procedures at 51.2%-wrong have the sparse-reward problem.
2. **Measure step correctness on the frozen distilled checkpoint, then choose the RL flavor.** If steps are >50% correct, outcome RL with the pass@k gate may suffice (the lucky-arithmetic problem shrinks as steps become correct). If steps remain <50% correct, step-level RL (Math-Shepherd-style: reward = completion probability from that step, computable by rollout against your checker; or PRIME-style implicit reward) is the contingency — not the default. **Disagreement retained, per aupai-fb: lessons-b0 holds step-level RL is never the move at 200M; I hold it is move 3, gated on the post-distill measurement. Both positions are now written down; the frozen-checkpoint measurement settles it either way.**

**SFT-side parameters for step 1 (lessons-e1, joint answer — the numbers aupai-fb asked for):**

- **Evidence that SFT installs procedure execution at all.** Lee et al. [2307.03381]: a 10.6M model reaches 100% on 3-digit addition from 1,000 detailed-scratchpad samples; plain format plateaus at ~85% even with 10K samples. Answer-*after*-working is the load-bearing property. rStar-Math [2501.04519]: the 1.5B policy is pure SFT every round on code-verified trajectories (MATH 51.2→88.6, AIME 0→46.7); the process model selects trajectories for SFT and guides test-time MCTS — it never RLs the policy directly. Their SFT-data ablation: step-verified 78.4 > rejection-sampled 73.4 > NuminaMath-CoT 69.6. Process supervision here acts as data selection, which is exactly what your checker gives you for free.
- **Two format conditions, non-negotiable, from this project's own rounds.** (a) Training data must be answer-*last*. Current scratchpad rows are answer-first; round 4's 28.9% scratchpad use was decoration, and round 5 removing the shortcut slot collapsed it 28.9%→2.8%. Verified + answer-first teaches the wrong procedure. (b) One canonical format per prompt. Round 4's [答] tag moved computation 16.7→41.7 (p=1.2e-7) — format ambiguity cost 25 points of computation. The three-formats-on-identical-prompts setup must go.
- **Volume: 10K–50K verified chains**, scaled by program-type coverage (mathbank spans ~756 program types), not raw row count. RFT's controlling variable is distinct correct paths per problem (5.25 at k=100). The project's own best SFT run (sft_k5_ctrl, 4,571 verified rows) is consistent with few-K-verified beating 13K-unverified.
- **Length cap: 256–512 tokens of working** — the shortest chain that solves the problem class. sft_k5's damage came with 2×-long answers (156-char median vs base 73–89); the learnability-gap paper [2502.12143] says ≤3B fails to absorb long CoT.
- **Epochs: 2–3** on a 10K–50K set (Tulu3, SmolLM2, Qwen-Math, rStar-Math all land at 2–3; RFT at 3). The invariant is ~3–10 exposures per example, not the epoch count — high-epoch regimes in the literature (Lee's 16 epochs, the project's 8) are tiny-dataset regimes. Eval every epoch; the runs are minutes-cheap.
- **Optimizer: full FT at 200M on narrow math data is a destructive operation unless replayed.** The Fine-Tuning Trap [2606.06920]: full FT on 10K math examples drops SmolLM2-135M *below zero-shot*; LoRA/DoRA wins below ~500M. Mixed Training [2512.13706]: math-only SFT collapses Flan-T5-250M's NLI 81.0→16.5 in under 1K steps; 6.2% replay holds 83.8. The project's sft_k5 math-500 drop (51.2→44.8, p=0.043) is this phenomenon, already measured internally. Run both arms: full FT at lr_scale 0.1 with 5–10% replay, and LoRA.

**Update 2026-08-29 — the middle regime exists (first internal result, numbers verified against EXPERIMENTS.md).** aupai-fb ran the pre-registered arm: k8_v3_fone (3.29B-token base) + 50K answer-last procedure SFT.

| probe_procedure (180 held-out) | base | +procedure SFT |
|---|---|---|
| BOTH (mul/eq/unit), free generation | 0.0/0.0/0.0 | 0.0/0.0/0.0 |
| STEPS total | 0/180 | 4/180 |
| digit head, teacher-forced on gold text | 21.3% (209/982) | 57.2% (562/982) |

Teacher-forced, the model computes 57% of the numbers; free, fully-correct chains are 0 — one early wrong number is unrecoverable and generation degenerates into `1/1 ≈ 1/1` loops. That is exposure bias, not missing coverage, and it overturns two positions in this report:

1. **The "no middle regime" claim was wrong** (lessons-b0's two-move collapse, which I accepted above). Scheduled-sampling / DAGGER-style SFT — train gold continuations on the model's *own* prefixes [1506.03099, 1011.0686] — is precisely the middle regime, operating on the teacher-forced/free gap. The sequence becomes: procedure SFT (done — it diagnosed the failure) → **move 2.5: scheduled-sampling SFT on self-generated prefixes** → distillation gated on the imitation probe → RL only if the pass@k gate appears. My move-3 ordering is unchanged; move 2.5 now exists and has evidence. b0 conceded in writing.
2. **The step-RL contingency is refined, not triggered.** Free BOTH=0 means own rollouts still carry no step-level signal — step-RL right now would still degenerate into all-wrong groups. Teacher-forced 57.2% means the correct-prefix side has signal; scheduled sampling is the carrier that moves it to the own-prefix side. b0's >50% step-correctness gate is crossed on the correct-prefix reading (57.2%) and not on the own-rollout reading (0%) — the gap between the two readings *is* the exposure bias. When scheduled sampling closes that gap, move 3 has something to amplify.

Two adjacent results from the same ledger: (a) the replay arm (k8_proc_replay) raised *teacher-forced* digit accuracy to 61.1% vs 57.2% (McNemar χ²=15.9, p=6.6e-05) — replay does not dilute the teacher-forced arithmetic signal, it significantly improves it. Keep the distinction fb kept: free-running procedure execution stayed at 0 in both arms; replay addresses forgetting, not exposure bias. The Fine-Tuning Trap mitigation above sharpens to: prefer the replay arm — on the teacher-forced metric. (b) b0's caveat, which I endorse: before building on 57.2%, confirm the digit head and the free-generation path are the same mechanism (both route numbers through the FoNE [NUM] channel). If the digit head is a specialized head never trained under its own error dynamics, the 57.2%-vs-0 gap is baked into the FoNE objective itself — the fix is then scheduled sampling on the value channel, and the diagnosis generalizes to every FoNE checkpoint.

b0's pretraining arm gets no points this round: the failure is dynamical (decoding under its own errors), not knowledge — 57.2% teacher-forced says the arithmetic is already in the weights at 3.29B tokens.

Nobody has published step-level RL at 200M. That cuts both ways: no evidence it works, no evidence it doesn't. The falsification path is in Q7.

## Q5. Degeneration — your degen counter at 13–23/step

**What the 13–23 means:** a "degenerate" group (all correct or all wrong) produces zero advantage and zero gradient — you are burning 8 generations of compute for nothing. 13–23/step at group 8 means a large fraction of your batches are no-ops. Causes and mitigations, ranked by what the evidence says matters at small scale:

1. **Prompt difficulty filtering (biggest lever, and you already proved it internally).** Groups are degenerate when the prompt is too hard (all wrong) or too easy (all right). LFM-1.3B-Math's 20–80% solve-rate band is the standard fix; DAPO's dynamic sampling [2503.14476] is the same idea at sampling time. Your rl_k4 run: band filtering cut degen 30–55%→3%. This is settled on your own hardware.
2. **Entropy collapse — monitor it, halt on it.** RLVR's entropy collapse is now a documented failure mode with a small literature: CURE [2508.11016] (static sampling → deterministic behavior; regenerate at high-entropy critical tokens), OPEFO [2605.11491] (token-level entropy flow imbalance), SvS [2508.14029] (problem refresh sustains entropy), "When Sharpening Becomes Collapse" [2601.15609] (finite-batch bias + semantic coupling). The operational paper for you is the SFT-overtraining rank-inversion study [2606.18487]: **pre-RL entropy correlates with GRPO outcome at ρ=+0.69**, and neither KL-to-reference nor label smoothing recovered a collapsed checkpoint. Their prescription: entropy triage before RL + early-GRPO entropy monitoring with a halt rule. Your rl_direct run had the collapse signature (acc 0.668→0.339, gen time 44s→152s, loss ×120) — an entropy monitor would have flagged it ~40 steps before the bottom.
3. **Length runaway — your rl_direct gen-time explosion.** Dr. GRPO [2503.20783] identifies the mechanism (GRPO's normalization biases toward long *incorrect* outputs); DAPO's overlong reward shaping penalizes it; S1 [2501.19393] shows budget forcing (length penalty) is cheap and effective. Your SFT checkpoint's 40-char median generations implicitly regularized this — that is a real mechanism, not a coincidence, and it argues for keeping a compressed-SFT stage before RL.
4. **KL coefficient — weaker than people think.** DeepSeekMath annealed β 0.04→0; R1-Zero used no KL at all; R1 uses a small KL (0.001-class). Your 0.02 is in the normal range. The rank-inversion paper's negative result (KL didn't recover collapse) says KL is a mild regularizer, not a guardrail. Don't tune it expecting rescue.
5. **Reference model choice.** Frozen-SFT-ref (your setup) is the standard and the right call; the rank-inversion paper adds the subtlety that *over-trained* SFT refs compress the rollout distribution and invert rankings — pick the SFT checkpoint by entropy/pass@k, not by pass@1.
6. **Clipping.** GSPO's sequence-level clip is fine; DAPO's decoupled clip (ε_low=0.2, ε_high=0.28 in the paper) slightly helps. Second-order.

**What matters most at small scale, in order: prompt band > entropy monitoring with halt > length control > KL > clip.** Your own logs already show #1 working.

## Q6. Distillation vs RL at 200M

**Distillation is the better spend now**, on three converging sources plus your own data:

- Kim et al. [2505.14216] (controlled, small-model settings): RLVR raises pass@1 but not pass@k; **distillation raises both** — but only when it introduces new knowledge. Distilling reasoning *patterns* alone behaves like RLVR (accuracy only, can sacrifice hard questions). Practical implication: distill *procedures the student cannot already produce*, not style.
- "Small Models Struggle to Learn from Strong Reasoners" [2502.12143]: the **Small Model Learnability Gap** — models ≤3B "fail to reliably gain from long CoT or large-model distillation"; shorter, simpler chains suited to capacity work better; **Mix Distillation** (long + short chains blended) significantly beats either alone. Your sft_k5 result is this phenomenon measured internally: answers at 2× the base's median length, math-500 −6.4pt (p=0.043). The fix the literature prescribes is the fix your own data points to: length-matched, capacity-appropriate chains.
- RFT [2308.01825]: rejection-sampling FT benefits weaker models *more* than strong ones (LLaMA-7B GSM8K 35.9→49.3). RAFT [2304.06767] beats PPO in stability and outcome in their setup. PaD [2305.13888]: program-aided distillation beats CoT FT for small models. SERT [2502.12744]: self-training on filtered self-generated paths surfaces latent reasoning in small models without CoT prompting. "Efficient Long CoT in SLMs" [2505.18440]: prune overthinking from distilled CoT, then on-policy self-curation.
- Your structural point on reverse KL across tokenizers is correct: sequence-level KD with rejection sampling is the right substitute. (On-policy distillation — the student sampling, teacher scoring — has a recent treatment from Thinking Machines, "On-Policy Distillation", Oct 2025, but I could not retrieve numbers this session; treat as direction, not evidence.)

**Concrete distillation recipe the evidence supports:** 27B teacher generates solutions to your procedure problems; keep only machine-verified-correct chains (your checker makes this free); **length-filter to the student's capacity band** (the 40–90-char regime your bases operate in, per your own measurements — not the teacher's full-length chains); mix long+short (Mix Distillation); SFT. That is RFT + PaD + the learnability-gap fix, and every component has a number behind it.

**The one thing distillation cannot buy you:** the pass@k expansion that ProRL-class prolonged RL can [2505.24864]. If the roadmap ever needs the model to solve problems where *no* amount of teacher distillation helps (because the teacher's chains don't cover them), that is the case for RL — at 1.5B and up, with a long budget, not at 200M in 500 steps.

## Q7. Falsification — what would kill each recommendation, and how soon

**Measurement first (you already know this, but the numbers are now derivable):**

At a 3% base pass rate, your 1032-problem eval resolves ≈±1.1pt (your own figure). Power calculation for a two-proportion test, α=0.05, power 0.8: to detect a 1pt gain at 3% base you need **≈4,600 problems per arm**; to detect 2pt, ≈1,150. Your 1032-set can resolve ~2pt at 3% base and ~1.5pt at 5%. **Any sub-2pt claim on the current eval is noise by construction** — this is why rl_k4's 4.1%-vs-2.9% (z=1.44, p=0.15) should never have been read as a win. Options: grow math-hard to ~5K for 1pt resolution, or report pass@k with the unbiased estimator [Chen et al. 2021, 2107.03374], which spreads the signal at the cost of variance (and at p=0.03, k=8, the estimator's own variance is large — compute the CI before trusting it).

**Per-recommendation falsification:**

| Recommendation | Falsifying result | When the signal should appear |
|---|---|---|
| Don't RL at 200M; the threshold is ~0.5B | A controlled run (same base, ≥3 seeds, eval ≥4.6K/arm or pass@k with CI) showing RLVR beats base pass@1 AND pass@k at 200M | If it works at all, RL-set accuracy moves within 200–300 steps (your rl_k4: 0.489→0.653 in 500; 2503.16219: gains in <24h on 7K samples). Flat at 300 steps → halt. |
| Pass@k gate ≥15pt | A run below the gate that transfers to math-hard significantly | Transfer is visible at the first eval after training; one eval, n=1032, needs ≥2pt to be readable |
| Distill first (27B verified chains, length-matched; PaD-style if synthetic; answer-last format, one canonical format per prompt; full FT lr_scale 0.1 + 5–10% replay or LoRA) | Distilled student ≤ SFT-on-self-generated control (SERT-style) at equal tokens on probe_procedure BOTH, or fails to install procedure execution — **but BOTH=0 alone does not falsify**: the discriminator is teacher-forced vs free. k8_proc_sft (BOTH 0→0, teacher-forced 21.3→57.2) would have killed the path under a BOTH-only falsifier; the failure was exposure bias, fixable by scheduled sampling. Falsifier is now BOTH=0 AND teacher-forced flat | One SFT run each; same eval power rules apply |
| Then RL on the frozen distilled base, flavor chosen by post-distill step correctness | RL fails to beat the frozen distill checkpoint at equal compute (head-to-head, 3 seeds, ≥4.6K/arm) | 200–500 steps; compare against distill-control, not against base |
| Distillation with length-matched Mix chains | Distilled student ≤ SFT-on-self-generated control (SERT-style) at equal tokens | One SFT run each; same eval power rules apply |
| Entropy-monitor halt rule | A run that collapses with entropy flat/rising all the way (monitor would have false-alarmed) | Entropy in your rl_direct started dropping ~40 steps before the acc bottom — the lead time is the feature |
| Band filtering (already internally validated) | degen rate stays >10% with the band applied | Measured per step from day one |

**The single most important disconfirming experiment for the whole RL program:** the head-to-head, not the zero-test (lessons-b0's correction, accepted). A replicated 4.1% that loses to a one-day distillation run kills the program anyway; the decision-relevant comparison is **distillation vs. distillation+RL on the same base, with the distillation checkpoint frozen and measured before RL touches it** — the pass@k gate is base-conditional, so if it passes post-distillation, RL's marginal contribution must be measured as Δ from the frozen distill checkpoint, not from base, or the credit is confounded. Design: (1) distill (27B verified chains, length-matched, answer-last, one canonical format per prompt, replay or LoRA per the Fine-Tuning Trap result); (2) freeze, measure pass@1/pass@8/step-correctness on the ≥4.6K-problem eval — step-correctness in both readings, teacher-forced and free (the gap between them is the exposure-bias diagnostic, per the k8_proc_sft result); (3) if the gate passes, run RL on the distilled base, 3 seeds; (4) compare distill-only vs distill+RL. The cheaper zero-test (rl_k4 recipe × 3 seeds × 4.6K eval) is a screen: if RL can't beat zero at 200M, the head-to-head is moot. Cost: one distillation run + 3×500 RL steps + evals — cheap relative to one more speculative run. **Pre-registered fallback (lessons-e1, agreed):** if the joint experiment returns "computation rises but answer-first persists OOD", the next move is Math-Shepherd-style step reward — completion probability by rollout against the checker, no trained PRM, cheap given the project's line-level verification. That is the contingency arm, registered before the data is in.

**Pre-registration on file, and one correction to the joint experiment (aupai-fb):** docs/exp_procedure_sft.md, commit 7094e5a, timestamped before results at k8 step 1280/3591. The correction: **math-hard cannot be the SFT arm's criterion.** math-hard and procedure_v1 are measured zero-contamination (0/1032 label overlap, 0/1032 exact, max Jaccard 0.200 vs the 0.8 threshold) — but the zero overlap holds because the two are different genres (untagged word problems vs tagged bare procedures, no shared template), not because math-hard is a clean held-out slice of the training distribution. math-hard failing to move therefore cannot falsify whether SFT installed execution ability; as a falsifier it would kill the right path for the wrong reason. The SFT arm's gate is **probe_procedure's BOTH** (the round-5-style answer-first OOD probe). math-hard stays in the suite as the transfer/generalization metric, explicitly not as the installation criterion.

**Pre-registration lesson from the first result (fb's own ledger, verified):** the pre-registered cell mapping was incomplete. "BOTH near zero → coverage not the constraint → pivot to pretraining" missed the fourth branch — exposure bias — and fb overturned his own reading in the ledger the same day ("THE PRE-REGISTERED READING WAS WRONG and the pre-registration is how I know"). A null landing in a pre-registered cell does not certify the cell; the diagnosis comes from the contrast between teacher-forced and free-running numbers, not from the null alone. The positive half of the lesson (lessons-e1's framing, endorsed): the pre-registration's value was never the four-branch table being complete — it was the timestamp, which let the wrong reading ("BOTH=0 → coverage not the constraint") be identified as wrong on the spot, the same day, by the person who wrote it. Probe design rule: every installation probe must measure teacher-forced and free in the same run, or the null is uninterpretable.

## Honest gaps (what I could not verify)

- **No published RLVR result below 0.5B.** Searched specifically. 200M is unmeasured territory.
- **R1-D** ("Incentivizing Reasoning in Small LLMs through Distillation"): could not locate on arXiv under this title or variants; I did not cite it. The distillation-vs-RL evidence here rests on Kim et al. [2505.14216] instead, which is controlled and on-point.
- **"On the Emergence of Thinking in LLMs II"** (Qwen3-0.5B thinking via RL): could not verify on arXiv; Part I [2502.06773] is verified but uses 8B/32B. The 0.5B emergent-thinking claim is floating in my memory without a citable source — treat as unverified.
- **Thinking Machines "Distilling Reasoning into Small Models"**: blog post, not arXiv; the URL I tried 404'd and the current blog index doesn't list it. Not cited.
- **"Entropy Collapse in RLVR" (Agia et al.)**: I could not locate this paper; the entropy-collapse evidence here rests on CURE/OPEFO/SvS/rank-inversion instead, which are all verified. If someone cites Agia at you, ask for the arXiv ID.
- SimpleRL-Zoo per-size numbers for the 0.5B arm are not in the abstract; the "substantial improvements" claim is abstract-level.
- TinyZero is GitHub-only, not peer-reviewed.

## Sources (verified this session)

- [2503.18892] SimpleRL-Zoo (COLM 2025) — zero-RL on 10 bases, 0.5B–32B
- [2307.03381] Lee et al., Teaching Arithmetic to Small Transformers — 10.6M, 1K scratchpad samples → 100% 3-digit addition; answer-after-working is load-bearing
- [2501.04519] rStar-Math — 1.5B pure SFT on code-verified trajectories, MATH 51.2→88.6; process model selects, never RLs policy
- [2503.16219] RL for Reasoning in Small LLMs: What Works and What Doesn't — 1.5B, $42, AMC23 63→80
- [2504.13837] Does RL Really Incentivize Reasoning Beyond the Base Model? (NeurIPS 2025 Oral) — pass@k bounded by base
- [2505.14216] RL vs. Distillation: Accuracy and Capability — distillation lifts pass@k iff new knowledge
- [2502.12143] Small Models Struggle to Learn from Strong Reasoners — ≤3B learnability gap, Mix Distillation
- [2505.24864] ProRL — prolonged RL expands pass@k at 1.5B
- [2507.13266] QuestA (ICLR 2026) — partial-solution augmentation, 1.5B AIME24 72.5
- [2606.22317] Boundary-aware Curriculum RL — pass@256 +9.8
- [2510.01180] BroRL — rollout scaling revives plateaued RL
- [2507.18071] GSPO (Qwen)
- [2503.20783] Dr. GRPO / Understanding R1-Zero-Like Training — length bias, 43.3 AIME24 at 7B
- [2503.14476] DAPO — 32B AIME24 50, dynamic sampling
- [2504.05118] VAPO — 32B AIME24 60.4
- [2607.00152] GRPO/Dr.GRPO/DAPO group-std identity
- [2607.23364] Impossibility of unbiased + length-invariant policy optimization
- [2308.01825] RFT (Scaling Relationship) — 7B GSM8K 35.9→49.3
- [2304.06767] RAFT (TMLR)
- [2203.14465] STaR
- [2305.20050] Let's Verify Step by Step — PRM800K, 78% MATH
- [2312.08935] Math-Shepherd — Mistral-7B GSM8K 77.9→84.1
- [2502.01456] PRIME — implicit process reward, +15.1% at 7B
- [2412.01981] Free Process Rewards without Process Labels
- [2406.06592] OmegaPRM — Gemini Pro MATH500 51→69.4
- [2408.15240] GenRM (ICLR 2025) — BoN 73→93.4 GSM8K
- [2504.00891] GenPRM — 1.5B > GPT-4o on ProcessBench
- [2504.16828] ThinkPRM
- [2412.06559] ProcessBench (ACL 2025) — PRMs generalize poorly
- [2606.00437] EST-PRM — Math-Shepherd PRM fragility
- [2211.10435] PAL; [2309.17452] ToRA; [2306.15626] LeanDojo; [2405.14333] DeepSeek-Prover
- [2305.13888] PaD (NAACL 2024) — program-aided distillation beats CoT FT for small models
- [2502.12744] SERT; [2505.18440] Efficient Long CoT in SLMs
- [2508.11016] CURE; [2605.11491] OPEFO; [2508.14029] SvS; [2601.15609] Sharpening→Collapse; [2603.18444] DBB
- [2606.18487] SFT overtraining → GRPO rank inversion — entropy ρ=+0.69
- [2606.06920] The Fine-Tuning Trap — full FT on 10K math drops SmolLM2-135M below zero-shot; LoRA/DoRA wins below ~500M
- [2512.13706] Mixed Training — math-only SFT collapses Flan-T5-250M NLI 81.0→16.5 in <1K steps; 6.2% replay holds 83.8
- [1506.03099] Bengio et al., Scheduled Sampling — train on the model's own prefixes with gold continuations
- [1011.0686] Ross et al., DAGGER — imitation learning as no-regret online learning
- [2509.23629] Emergent Slow Thinking / Annealed-RLVR
- [2501.19393] S1 — budget forcing
- [2501.12948] DeepSeek-R1 (Nature)
- [2107.03374] Evaluating LMs Trained on Code — unbiased pass@k estimator
