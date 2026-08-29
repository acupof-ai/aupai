# Adversarial report: is SFT-then-RL the right ladder for a 200M math model?

Written 2026-08-29. Method: 5 parallel evidence sweeps (overtraining / reasoning floor / reachable targets / distillation economics / eval-harness audit), each adversarially verified by an independent skeptic (11 agents, 1.35M tokens), plus a completeness critic whose 25 gaps are integrated below. Every external number carries a primary source; every internal number was re-verified against the live repo (EXPERIMENTS.md now has 50 runs — the brief's "best post-SFT 3.6%" is stale).

## Corrections

**2026-08-29, after aupai-fb's measured rebuttal:** §5.1 as originally written was wrong. The audit found the dirty file `rlvr_math.jsonl` (515 verbatim math-500 questions) and inferred it was used for training. It was not. Every recorded RL run's `--data` flag (runs/experiments.jsonl) points at `rl_band.jsonl` (rl_k4, rl_direct) or `rlvr_clean.jsonl` (rlvr_gspo/gspo2) — both clean. The 515 is exactly 218,468 − 217,953: the count the holdout filter REMOVED. It is evidence the filter works, the opposite of the original conclusion. rl_k4's 44.6% and rl_direct's 52.4% stand; the −7.0pt math-500 regression (z=-2.2) is therefore a clean measured effect, not contamination. Error shape, worth keeping: a dirty file on disk was counted and assumed used — file existence is not training provenance; only the run's cmd answers what was trained. (The original audit's verifier marked the claim "OK" without checking the cmds — adversarial verification that doesn't check the primary record performs the failure in a second voice.)

## 0. Corrected scoreboard — the ladder's own numbers, tested

| comparison | numbers | test | verdict |
|---|---|---|---|
| best SFT vs best RL | 4.0% (sft_k5_v9, 41/1032) vs 4.1% (rl_k4, 42/1032) | z=0.11, p=0.91 | **TIED** |
| RL vs base | 4.1% vs 2.9% (30/1032) | z=1.44, p=0.15 | not significant |
| SFT vs its own base | 3.6% vs 1.9% (k5 line) | z=2.28, p=0.022 | **the only significant training gain on record** |
| RL's math-500 effect | 44.6% vs base 51.6% (223 vs 258/500) | z=-2.2, p=0.03 | **significantly NEGATIVE** |

Both "best" numbers are winner's-curse peaks: SFT scored 1.3 / 2.6 / 2.7 / 3.3 / 3.6 / 4.0 across six near-identical recipes at ±1.1pt resolution (σ≈0.53pt/run; expected max of 6 draws ≈ +0.9pt), and RL's 4.1% is the best of four RL runs (rl_direct 2.9%, two stopped, one collapsed to 11.2%). Peak-picked vs peak-picked, the gap is zero.

RL's own precondition is unmet and unquoted in the briefs: rlvr_gspo2's gate — "No RL until SFT clears ~60% on math-500 and pass@8−pass@1 ≥ 15pt" — against current 44–51% and ~7–8.7pt. And the pass@k number itself is suspect (§5.2).

**The ladder as run: SFT rung +1.7pt significant; RL rung +0.1pt vs SFT, −7.0pt on math-500 significant.** The attack on the ladder is not "neither rung works" — it is "the RL rung adds nothing and charges for it."

## 1. Is the binding constraint simply undertraining?

**Corrected token counts (the brief's numbers were wrong).** SmolLM2-360M saw **4T** tokens (11,100 tok/param), not 11T — 11T is the 1.7B. Verified: Qwen3-0.6B 36T (60,000 t/p), Qwen2.5-0.5B 18T (36,000 t/p), SmolLM2-135M 2T (14,800 t/p), MobileLLM-125M 1T (8,000 t/p), OpenELM-270M 1.5T (5,560 t/p), TinyLlama-1.1B 3T (2,730 t/p). This project: k7_v3 = 3.29B = **16.5 t/p — below Chinchilla-optimal (20 t/p ≈ 4B tokens)**; k5/k4 = 11.5B = 57 t/p. The gap to every capable small model is 2.5–3.5 orders of magnitude in tok/param.

**Case FOR more tokens.** Every capable small model is massively overtrained; the controlled comparison is SmolLM2-135M HellaSwag 42.1 vs Cerebras-111M 26.8 at the same ~125M scale — almost entirely an overtraining gap, not architecture. Qwen2→2.5 at 0.5B (7T→18T): MATH 11.2→19.5, MMLU 44.3→47.5 (confounded by mix). MobileLLM's controlled 4x-token ablation: +0.9–1.4pt commonsense — real but modest.

**Case AGAINST, and it is stronger for MATH specifically.**
- SmolLM2-1.7B's own stage table: math-category 3.21 (0–6T) → 3.7 (6–8T) → 7.27 (8–10T) → **22.07 only in the final math-heavy anneal**. Knowledge/reasoning rose smoothly over the same span. Math tracks the **mix**, not raw token count.
- Internal: math-hard is flat 1.7–2.9% across 2.26B→11.5B tokens (5x, confounded by corpus/arch — weak but directionally null).
- k5 (11.5B unfiltered, 57 t/p) 1.9% vs k7_v3 (3.29B filtered, 16.5 t/p) 2.2%: corpus quality dominates token count; the one clean-vs-unfiltered comparison at equal tokens (k5 1.9 vs k4 2.9, p=0.152) is n.s. and directionally negative for math.
- Muennighoff data-constrained laws (400+ runs): repetition safe to ~4 epochs, meaningful gains exhaust ~16 epochs. **30B on the 3.3B unique corpus = 9 epochs — past the safe zone.**
- TinyLlama plateaued downstream at ~1,800 t/p on their corpus.

**Verdict.** For general capability, the overtraining case is strong and k7_v3 is *under*-Chinchilla. For math-hard, the case is weak: the lever SmolLM2's own data identifies is the anneal/mix (executed-procedure density), not token count. The project's actual v3 plan (docs/data_recipe_v3.md: 4.5B, one epoch, filtered web 40% + cosmopedia 31%) is a **language-modelling bet** — val 2.020 vs 2.086 supports it — and should be defended on those terms, not as a math play. The 30B extension as briefed: not supported as a math intervention; as a capability play it is gated on growing unique supply past the repetition wall first (quarantined web re-filter + textbook; CPU/teacher-bound, parallelizes with everything else). Near-free EV: extending k7_v3 from 16.5 to ~20 t/p is ~0.3h of GPU.

## 2. Smallest model with a credible reasoning result

**The floor is ~0.6–1B for controlled hard math, and only via distillation from a teacher 8–390x larger:**
- MobileLLM-R1-950M: AIME'24 15.5, MATH-500 74.0 — 4.2T tokens pretrain + mid-training KD from Llama-3.1-8B + 6.2M reasoning traces. **RL was tested and REJECTED** (SFT beat it).
- Qwen3-0.6B: AIME'24 10.7 — 36T tokens + QwQ-32B long-CoT cold-start SFT.
- LFM-1.3B-Math: AIME'24 0.2→51.0 after 4.5M distilled traces (3.2M OpenMathReasoning) + GRPO.

**RLVR-alone floor is ~1.5B.** Open-Reasoner-Zero scaling table (vanilla PPO, binary reward): AIME'24 1.0 at 0.5B, 3.5 at 1.5B, 17.9 at 7B. The controlled 0.5B study (arXiv 2506.13404): RL is the best of bad options — MATH-500 31.4→32.4 (+1.0), GSM8K +8.5 — while SFT drops GSM8K 45.5→30.9 and KD/hybrid collapses.

**At 200M itself, the only published controlled results are synthetic arithmetic** (Lee et al., 10.6M–124M models) — which this project has already replicated and exceeded (FoNE 0→16.7% bare two-digit, tag 41.7%, matched controls p~1e-7). Nobody has published multi-step math at 200M. "Nobody has measured this below 1B — and below 0.6B not even with a teacher" is the correct answer.

**Implication.** The 1.5B results everyone quotes sit on bases with 18T+ tokens AND 700B–1T-token math corpora AND distillation. The floor is three-dimensional: **params AND tokens AND teacher**. A 200M model at 16.5 t/p chasing math-hard is 2.5x below the smallest param scale, ~1000x below the smallest token budget, and has no teacher channel installed. All three gaps have levers; the SFT/RL ladder as briefed addresses none of them directly.

## 3. Is math the right target?

**math-hard cannot resolve the decisions being made against it.** At 2–3% pass, n=1032, σ≈0.53pt: the best-ever gain (RL 4.1 vs SFT 4.0) is z=0.11. Detecting 1pt at power 0.8 needs ~4,600 problems/arm. Every sub-2pt claim on the current eval is noise by construction — including the claims the RL brief rests on.

**Reachable-and-measurable at 200M, by evidence strength:**
1. **Code-mediated math (TinyGSM recipe, arXiv 2312.09241): 125M → GSM8K 63.1 (68.9 with verifier)** on 12.3M synthetic Python-solution problems ($3.6k, GPT-3.5-generated). Directly reproducible here: local 27B teacher + elastic synthetic supply. This is the strongest reachable target — it converts "multi-step arithmetic" from a procedure-execution failure into a code-generation task, where 200M models have a slope (HumanEval 9.15 at 1.1B → 30.5 at overtrained 0.5B).
2. **Format-following**: IFEval slopes 29.9 (135M) → 41.0 (360M) → 56.7 (1.7B); the project's own tag intervention moved computation 16.7→41.7 and termination 25.6→62.8 (p~1e-7). The lever exists in *this* model, today.
3. **ARC-E / PIQA**: the only MC metrics with a Chinchilla-scale slope (ARC-E +8.4, PIQA +3.3 from 111M→590M). This project's ARC-E 33.1 (z=9.1) and PIQA 54.2 (z=3.6) are significantly above chance — the "everything at the chance line" framing is wrong for 3 of 5 MC benchmarks. English MC understates a Chinese-corpus model; the scorer also has three fixable biases (§5.3). Chinese-native equivalents (C3/CLUE) or a 1-day scorer fix unlock these as resolving metrics.
4. **Single/two-step arithmetic with explicit format** — already instrumented, fast iteration.

**Recommendation.** Keep math-hard as the north star; stop making decisions below its resolution. Adopt **GSM8K-zh/MGSM-zh** as the resolvable math metric (Qwen3-0.6B MGSM 30.99 shows it resolves at sub-1B). Scope note that bites: **switching the metric to GSM8K-zh flips the RL ranking** — at 0.5B, RL +8.5 GSM8K vs SFT −14.6. "RL last" is scoped to math-hard; on a resolvable metric RL is the only strategy that helped at 0.5B. A 200M Chinese model evaluated on a Chinese-native verifiable suite also fills an empty public cell — that is a publishable contribution, not just an internal tool.

## 4. Distillation vs SFT vs RL, costed on 8xH20

H20: 148 TFLOPS BF16 / 296 FP8, 96GB, 4TB/s — bandwidth-strong, compute-weak (15% of H100 BF16). 27B teacher generation: ~5–12k tok/s aggregate BF16 (central 8k), ~10–22k FP8 — roofline-derived, anchored to official Qwen H20 single-stream numbers (Qwen3-32B: 20.7 tok/s BF16, 46.2 FP8 at batch 1); **no direct H20 batched benchmark exists, measure before committing**.

**Ranking by expected math-hard gain per GPU-hour:**

| option | cost (8xH20) | evidence at ≤1.5B | verdict |
|---|---|---|---|
| **2. Distillation** (27B, sequence-level, rejection-sampled) | 50M correct CoT tokens: 2–4h BF16; 200M: 8–14h BF16 / 4–8h FP8; student SFT 6 min | R1-Distill-1.5B (800k traces, SFT-only: AIME 28.9, MATH-500 83.9); Qwen3-0.6B (QwQ-32B cold-start); DSS (220M T5 + rationales > label-only); GKD (77M students absorb distillation) | **#1 — only arm with strong sub-2B evidence; directly targets the diagnosed answer-first failure** |
| **3. Procedure SFT** (mathbank, answer-last, program-level split) | hours (data gen + minutes of training) | Lee et al. at 10.6M (narrow arithmetic); multi-step at 200M: **empty cell** | runs first by cost (converged with e1, §6) |
| **1. More pretraining** | 30B = 14.6h | internal: flat for math; SmolLM2: math tracks anneal not tokens; 9 epochs past Muennighoff wall | queue behind corpus growth; general-capability play, not math |
| **4. RLVR** | outcome-only 2–5h; process-reward w/ 27B judge: 17.5 min/step → **87h/300 steps** | zero significant gains at 200M; one significant regression; 0.5B: +1.0 MATH-500; PRM>ORM never validated below GPT-4 scale | **last — and only on a frozen distilled checkpoint if the pass@k gate appears** |

Cross-tokenizer premise confirmed: logit/KL distillation is undefined across tokenizers (ULD paper states it); sequence-level KD (Kim & Rush 2016) is the tokenizer-agnostic standard substitute; rejection-sampling (STaR/RFT) is the math standard. The plan is sound.

**Numbers that would change the ranking:**
- **Teacher yield on math-hard, unmeasured.** Budget 30–70%; 200M correct tokens costs 6.9h (yield 0.7) to 23h (yield 0.3). Which 27B (Gemma2-27B vs Qwen2.5-32B) matters a lot for Chinese competition math. **Measure first: 100 problems × 8 samples, ~1h.**
- **In-context imitation probe — the kill criterion.** 1-shot worked example from the teacher on math-hard, ~1h, no training. If the 200M cannot imitate working-first format even in-context, the capacity gate binds and distillation's EV collapses. Run before any distillation spend.
- **pass@8−pass@1 re-measured at T=0.8** (current number is degenerate, §5.2). If ≥15pt: verify-first (majority-vote / teacher-judge) extracts existing capability at zero training cost — TinyGSM gained +5.8pt at 125M with a verifier — and RL becomes viable.
- Metric switch to GSM8K-zh: RL-first flips (§3).

**Distillation pitfalls to engineer around (all documented):** RFT diversity — distinct correct paths saturate ~5.2/problem at k=100 and stronger teachers are *less* diverse (33B: 2.78); sample 4–8 at T=0.7–1.0, dedup, spend budget on more problems. Length bias — 200M cannot absorb 10k-token R1 chains; cap trace length, prefer concise correct traces (the project's own sft_k5 evidence: length compression was the active ingredient). Endpoint-only verification passes wrong-path traces — add cheap step checks. Student cannot exceed teacher. Small KD sets degrade (0.5B study: KD with small data 42.3 vs 45.5 baseline) — pilot at 50M+ correct tokens, not 5M.

## 5. What are we measuring wrong

11 of 13 audit findings survived adversarial re-verification (2 refuted). The load-bearing ones:

1. **The RL holdout filter works — and its effectiveness was unrecorded.** `rlvr_math.jsonl` (218,468 rows) contains 515 verbatim math-500 questions; `rlvr_clean.jsonl` (217,953) contains 0 — the filter removed exactly the 515. Every recorded RL run used the clean files (rl_k4/rl_direct: `rl_band.jsonl`, a 1,048-row subset of the clean pool; rlvr_gspo/gspo2: `rlvr_clean.jsonl`), so no post-RL math-500 number is contamination-inflated. Two residual gaps: (a) the clean files' provenance is recorded nowhere — the fact that a filter ran and what it removed is reconstructable only by subtraction; write it into the data path (a header or a sibling `.provenance` row) so the next audit doesn't have to infer it; (b) `rl_band.jsonl` (the actually-trained subset) is deleted, so the trained band itself cannot be re-audited — keep it next time. The raw `rlvr_math.jsonl` should be quarantined or deleted: a dirty file sitting next to the clean one is how this mistake happened.
2. **pass@k is degenerate by default and mislabeled.** `scripts/eval_hard.sh:10` defaults `TEMP=0`; with k>1 all samples are greedy-identical so pass@k == pass@1 silently. Additionally `eval/math_hard.py:148` computes `pk = any(oks)` over `[greedy] + k sampled` — so `--k 8` reports pass@9 under a pass@8 label (confirmed by aupai-fb line-by-line). The base's 10.1–11.6% pass@8, the 7.0% rl_direct number, and the 15pt gate all rest on runs whose temperature is unrecorded, so they are unreproducible as well as suspect. **Re-measure at T=0.8, record temperature in every result row, and fix the pass@(k+1) label before building any option on the gap.**
3. **The MC suite is not all at chance.** ARC-E 33.1 (z=9.1), HellaSwag 26.5 (z=3.5), PIQA 54.2 (z=3.6) are significantly above chance and can resolve checkpoint differences; MMLU 25.7 (z=1.9, borderline); C-Eval 25.7 (z=0.5, at chance). eval_all.sh's "tripwire only" framing discards real signal. The scorer has three structural biases — no length normalization, separate prompt/option encoding, off-distribution prompt format (no ChatML) — ~1 day of work fixes, after which ARC-E/PIQA become resolving metrics without building anything new.
4. **math-500's grader is broken on 67/500 rows.** 6 gold solutions have no `\boxed` → `score()` returns 0.0 unconditionally (forced wrong for every checkpoint); 61 golds normalize to non-numeric strings (prose answers, mixed numbers, units like 公里/份/颗 not in the strip list) → matchable only by exact string, so a model mimicking Belle's prose format scores higher independent of math ability. This can flip math-500 cross-checkpoint comparisons — the metric's only sanctioned use. math-hard is clean (verified: 0 non-numeric, 0 multi-answer golds).
5. **math-hard's contamination protection is luck, not design.** The eval carries no program_id (0/1032 rows); protection is post-hoc exact-text hashing that catches verbatim copies only. Training batches are drains of the same program bank (two same-bank batches share 86.4% templates, 294/1031 identical rows); the eval's measured 0.3% overlap was against v8 ONLY, never v6/v7; `split_bank.py` (a real program-level md5 split) exists but is explicitly NOT APPLIED; PROVENANCE.md:150: "that disjointness is luck rather than design." If the eval is ever regenerated from the bank, contamination is immediate. e1's procedure-SFT experiment inherits exactly this trap — the holdout must be on program_id (his sharpening, confirmed).
6. **RL-set accuracy — the headline "RL works" evidence — is in-distribution, sampled (T=0.9), best-checkpoint-selected, on a band pre-selected by the model's own solve rates.** Not comparable to greedy math-hard. The internal post-mortem already named the mechanism: the band sits inside the capability envelope, so RL only polishes pass@k→pass@1 — "the transfer ceiling for BOTH runs."
7. **Latent traps for the next runs:** the RLVR generator has no FoNE path and a hardcoded 1024-token context (`rlvr_generate.py:20`) — RL on a k6+ checkpoint would read every number as literal `[NUM]` and score garbage without raising; the digit-head metric is scored on the tail of the *training* cache, not a holdout (memorization-inflated, moving target during a pretrain); `rlvr_data.py`'s normalize drifts from `rlvr_reward.py`'s (latent today); `ppl.py`'s val slice correspondence breaks if the token cache rebuilds mid-run.

## 6. Where the three briefs converge and where they do not

**Converged (all three, after the exchange):**
- The only significant training gain on record is SFT vs its own base (p=0.022); RL has no significant positive result at 200M and one significant regression.
- The RL rung is move 3 at the earliest, gated on the project's own pass@k test, on a frozen checkpoint, with credit assigned before RL touches it.
- Distillation is the under-weighted arm; the 27B is already in production as a judge (serving/integration solved, math yield unmeasured).
- The project's own metric discipline — significance first, instrument second, hypothesis third — is still being violated (§5).

**With lessons-e1 (SFT brief):** I conceded the load-bearing point: **3.6% is a data ceiling, not an SFT ceiling** — no SFT run on record used data containing executed procedures. His procedure-SFT arm runs first by cost. Remaining disagreements: (a) a null at 200M with no published precedent adjudicates little; a *win* with a clean program_id-level split is strong — the asymmetry he accepted; (b) the pretrain extension queues regardless, because every arm's ceiling is the base — he agreed; (c) if his procedures come from the 27B, his arm IS the distillation arm — the clean cheap version is mathbank-generated with a program_id holdout.

**With lessons-44 (RL brief):** converged on the 2D threshold (0.5B AND ~36k tok/param — the smallest verified RLVR result sits on a base ~1000x more overtrained than ours), head-to-head falsification (distill vs distill+RL, frozen checkpoint, 3 seeds, ≥4.6K problems/arm), and that the pass@k gate is the right test and both RL runs failed it. Residual disagreement: he keeps step-level RL as a move-3 contingency and PaD-style synthetic verified procedures as a third data source; I hold that at 51.2% wrong equations, step reward on the model's own rollouts is sparse one level down — the same degenerate-group problem — so the middle regime collapses into "teacher rollouts (distillation) or nothing." Both agree the gate decides.

**My residual adversarial position, stated plainly:** the strongest hypothesis the evidence supports is not "SFT-then-RL" and not "more tokens" — it is **"the base has never seen executed procedures at any scale, and every intervention that installed one (FoNE, the format tag, the anneal) moved the needle."** The anneal bought format (math-500 51.6 pre-SFT); the tag bought computation (+25pt); FoNE bought arithmetic (0→16.7%). The common factor is procedure-bearing data in the distribution, not the training stage. That predicts: distillation (procedure density, highest) > procedure SFT (narrow) > anneal redesign (cheap, in-flight) > RL (amplifies an empty distribution) > raw tokens (procedure-free).

## 7. Recommended sequence, cost-ordered

1. **Teacher yield probe** (100 math-hard problems × 8 samples, ~1h) + **in-context imitation probe** (1-shot teacher worked example, ~1h, no training) — kill criteria for the whole distillation arm.
2. **Instrument fixes (hours, no GPU):** `eval_hard.sh` TEMP default → 0.8 for k>1 and record temperature in every result row; fix the pass@(k+1) label in `math_hard.py:148`; quarantine raw `rlvr_math.jsonl` and write provenance for the clean RL files (the filter works — make that visible); re-measure pass@8 at T=0.8; fix math-500's 67 broken grader rows; apply `split_bank.py` before any procedure SFT.
3. **mathbank procedure SFT, program_id-level holdout** (hours) — e1's arm; the win case is strong and the null case is informative.
4. **If probes pass: distillation pilot, 50M correct tokens** (2–4h), frozen checkpoint, head-to-head vs SFT on a 4.6K-problem eval.
5. **Corpus growth in parallel** (CPU/teacher-bound, zero GPU contention): re-filter quarantined web + textbook toward 8–10B unique, so the pretrain extension is one epoch, not nine.
6. **RL only if the pass@k gate appears post-distillation**, on the frozen checkpoint, 3 seeds, 4.6K eval.
7. **Metrics:** GSM8K-zh as the resolvable math metric; fix the MC scorer and read ARC-E/PIQA; keep math-hard as north star, stop deciding below ±1.1pt.

## 8. Post-script: the procedure-SFT null is an exposure-bias result (2026-08-29, after the report)

aupai-fb ran e1's arm pre-registered (docs/exp_procedure_sft.md): k8_v3_fone (corpus v3 + FoNE, first combination, 3.29B tok) + procedure_v1 SFT (50K rows, answer-last, per-line machine-verifiable, single format per prompt). Probe: probe_procedure held-out, not math-hard.

| probe_procedure (180 held-out) | base | +procedure SFT |
|---|---|---|
| BOTH (mul/eq/unit) | 0.0/0.0/0.0 | 0.0/0.0/0.0 |
| STEPS total | 0/180 | 4/180 |
| digit head, teacher-forced on gold process text | 209/982 = 21.3% | 562/982 = **57.2%** |

**Teacher-forced, the model computes 57% of the numbers; free generation, fully-correct chains are 0.** It learned the chain shape and the terminator; the numbers are wrong and generation degenerates into `1/1 ≈ 1/1` loops. This is exposure bias — one early wrong number is unrecoverable — not coverage insufficiency.

Implications, each updating a position in this report:
1. **My step-level gate (§6, to 44):** "step reward is sparse unless own-rollout step correctness >50%." On the correct-prefix reading (given a gold prefix, is the next number right), 57.2% crosses it; on the own-rollout reading (free generation BOTH=0), it does not. The gap between the two readings IS the exposure bias. Either reading points at the same intervention: train on the model's own prefixes with gold continuations — scheduled sampling / DAGGER-style SFT.
2. **My "no middle regime" claim to 44 was wrong.** I argued the step-level middle collapses into teacher-rollouts-or-nothing. Scheduled sampling is precisely the middle regime — it operates on the teacher-forced/free-running gap — and this null is the first internal evidence for it. The sequence becomes: procedure SFT (done — diagnosed the failure) → scheduled-sampling SFT on self-generated prefixes → distillation (gated on the imitation probe) → RL only if the pass@k gate appears.
3. **My arm (more pretraining) gets no points this round.** The failure is dynamical (decoding under its own errors), not knowledge — 57.2% teacher-forced says the arithmetic is already in the weights at 3.29B tokens. A coverage intervention does not target a dynamics failure. Arm stays queued, unadvanced.
4. **fb's pre-registered cell was wrong, and said so:** "BOTH near-zero → coverage not the constraint → pivot to pretraining" missed the fourth branch (exposure bias). A null landing in a pre-registered cell does not certify the cell — the diagnosis has to come from the contrast between the teacher-forced and free-running numbers, not from the null alone. Concretely (44's sharpening, endorsed): the distillation arm's falsifier must be "BOTH=0 AND teacher-forced flat" — BOTH=0 alone would have falsely killed k8_proc_sft, which moved teacher-forced 21.3→57.2%. A null on the free-running metric does not falsify when the diagnostic metric moved.
5. One check before building on 57.2%: the digit head and the free-generation path must be the same mechanism (both route numbers through the FoNE [NUM] channel). If the digit head is a specialized head never trained under its own error dynamics, the 57.2%-vs-0 gap is baked into the FoNE objective itself — which would make scheduled sampling on the value channel the fix, and would generalize the exposure-bias diagnosis to every FoNE checkpoint.

**Confirmed from code (aupai-fb, train.py:564/577/755/919):** the same `num_head` serves training and free generation; the 57.2%-vs-0 gap is not architectural. The mechanism is sharper than "exposure bias" in general: `train.py:577` feeds `num_vals` back into the input embedding (`emb = emb + num_proj(feat)`), and during training `num_vals` are always gold — **the digit head has never once been trained conditioned on a wrong value in its input.** Scheduled sampling must therefore be done on the *value channel* (replace gold prefix values with the model's own decoded values, or inject value noise); token-channel scheduled sampling would not touch the gap. The cheapest first cut is value-channel noise injection (no generation); the real intervention is on-policy: generate with the current checkpoint, keep prefixes up to the first error, SFT the gold continuation (DAGGER-shaped).

**ARM B (+7.4% replay, gold knowledge_qa_zh + code_python_zh — no model error prefixes):** BOTH still 0/180 across all three checkpoints; STEPS 5/180 vs ARM A's 4/180 (no difference); digit head teacher-forced 61.1% (600/982) vs 57.2% — McNemar p=6.6e-05, significant. Replay does not dilute (e1's worry, backwards) but does not touch the rollout failure either. The dissociation is itself diagnostic: replay improves teacher-forced arithmetic without moving free-running procedure at all — consistent with the failure being dynamical (recovery from own errors), not coverage. What replay improved (likely numerical-representation generality) is unmeasured; no inference made.

**Agreed sequence (all three briefs):** procedure SFT (done — diagnosed the failure) → self-generated-prefix SFT on the value channel → distillation (imitation probe gate) → RL only if the pass@k gate appears.

## Sources

External (primary unless noted): Hoffmann et al. 2203.15556; Muennighoff et al. 2305.16264; SmolLM2 2502.02737; Qwen3 2505.09388; Qwen2.5 2412.15115; Qwen2.5-Math 2409.12122; MobileLLM 2402.14905; OpenELM 2404.14619; TinyLlama 2401.02385; H2O-Danube 2401.16818; Cerebras-GPT 2304.03208; Pythia 2304.01373; DeepSeek-R1 2501.12948; LFM-1.3B-Math (Liquid AI); MobileLLM-R1 2509.24945; Open-Reasoner-Zero; 0.5B reasoning study 2506.13404; Oat-Zero; DeepScaleR/JustRL/Open-RS 2503.16219; LFM2 2511.23404; Lee et al. 2307.03381; Nye et al. 2112.00114; TinyGSM 2312.09241; Fu et al. 2301.12726; Rho-1 2404.07965; Math-Shepherd 2312.08935; Distilling Step-by-Step 2305.02301; GKD 2306.13649; ULD 2402.12030; RFT 2308.01825; s1 2501.19393; SimpleRL-Zoo 2503.18892; Yu et al. 2504.13837; Kim et al. 2505.14216; SvS 2508.14029; ProRL 2505.24864; Small Model Learnability Gap 2502.12143; PaD 2305.13888. Internal: repo AGENTS.md, EXPERIMENTS.md (50 runs), docs/data_recipe_v3.md, eval/ and scripts/ audit (file:line inline).
