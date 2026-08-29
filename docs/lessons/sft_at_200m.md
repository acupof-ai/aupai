---
question: What does the evidence say about SFT at 200M?
status: recorded
source: literature with numbers plus the k8_proc_sft internal run, 2026-08-29
---

# SFT at 200M: what the evidence says

Written for aupai-fb, 2026-08-29. Sources are papers/repos with numbers; where no one has measured at this scale, it says so.

## TL;DR

1. **SFT can install procedure execution — the evidence is at 10M–1.5B params, and it is strong.** Lee et al. (10.6M, ~1K scratchpad samples → 100% 3-digit addition), Nye et al. (scratchpad generalizes OOD where direct fails), rStar-Math (1.5B: MATH 51.2→88.6, AIME 0→46.7, on trajectories whose every step is code-verified). The load-bearing properties of that data: **answer comes AFTER the working, every intermediate line is machine-verified, one format per prompt.** Your current SFT data has none of the three (answer-first, 51.2% wrong equations, three formats on identical prompts). The 3.6% math-hard ceiling measures that data, not SFT. **Update (2026-08-29, first direct test at 200M, k8_proc_sft):** 50K answer-last verified chains installed teacher-forced execution (digit head 21.3%→57.2%) but free-form stayed 0 (probe BOTH 0→0). The failure is exposure bias — the model cannot recover from its own early errors — not coverage. SFT installs the per-step computation; free-form execution needs scheduled sampling next. See §5.
2. **Nobody has published multi-step math SFT at 200M.** The closest scale points: Fine-Tuning Trap (2026) — full FT on 10K math examples drops SmolLM2-135M *below zero-shot*; LoRA/DoRA wins below ~500M. Mixed Training (2025) — Flan-T5-250M math-only SFT collapses NLI 81.0→16.5 in <1K steps; 6.2% replay holds 83.8%. Both are directly applicable and both say: at 200M, full FT on narrow data is a destructive operation unless you replay and/or adapt lightly.
3. **The b0 question (10x pretrain vs SFT) is unresolved and the priors cut both ways.** FOR more pretrain: SmolLM2-135M trained 2T tokens (600x your 3.3B); Qwen2.5-Math-1.5B *base* is 49.8 MATH from a 700B-token math corpus; MiniCPM: HQ data in the decay stage beats SFT (GSM8K 27.7→42.3); your own k4 math anneal is your best math-500 ever (51.6, no SFT). AGAINST: math-hard never moved in any stage including RL (best 4.1%), and the model has the form (88% boxed, 91% equations) without the skill — a procedure-execution gap, not a token-count gap. **My call (ran 2026-08-29):** the SFT arm ran first as proposed. It did NOT produce the pre-registered clean null: free-form BOTH stayed 0, but teacher-forced digit head rose 21.3%→57.2% — crossing b0's own >50% step-correctness gate. The binding constraint is exposure bias, not coverage or token count, so b0's arm did not receive its positive-evidence null; scheduled sampling is the next move (§5). b0's pretrain arm stays queued, not vindicated. You are not choosing one arm forever; you are ordering two experiments by cost.
4. **Joint position with lessons-44 (we converged): SFT installs the behavior, step-level RL sharpens it.** Process supervision's proven form at 1.5B is *trajectory selection for SFT* (rStar-Math's PPM picks which trajectories to SFT on; the policy itself is SFT'd), not direct policy RL. Step RL is the fallback if SFT installs the form but not the dependency — discriminating experiment below.

---

## 1. SFT data: how much, of what, in what proportion

### Real recipes (numbers, not vibes)

| Recipe | Size | SFT data | Epochs | LR | Batch | Notes |
|---|---|---|---|---|---|---|
| SmolLM2-1.7B | 1.7B | SmolTalk 1.1M pairs | 2 | 3e-4 | 128 / seq 8192 | LR is WSD-legacy, an outlier |
| SmolLM2-135M/360M | 135M/360M | *filtered* SmolTalk (no hard tasks) | — | — | — | Single-stage training; "data curation has outsized influence for smaller models" |
| Tulu 3 | 8B | 939K mix (math 35.6%, general 12.4%, code 15.1%, safety 11.8%, multilingual 10.7%, knowledge 11.2%, IF 3.2%) | 2 | 5e-6 | 128 | Linear, warmup 0.03 |
| Qwen2.5-Math-1.5B | 1.5B | 2.5M CoT (2M EN + 0.5M ZH) + 395K TIR | 3 | 2e-5→7e-7 | 128 / seq 4096 | Base→Instruct: MATH 49.8→75.8 |
| MiniCPM-2.4B | 2.4B | ~6B tokens SFT | — | = end-of-anneal LR (WSD) | — | HQ data in decay > in SFT (GSM8K 27.7 vs 42.3) |
| OLMo 2 | 7B | 939K (Tulu-3 mix) | 2 | 1e-5 | — | Sum loss, not mean |
| RFT (Yuan 2023) | 7B | 47K distinct correct paths (k=100 samples over 7.4K GSM8K) | 3 | 2e-5 | 128 | warmup 3% |
| rStar-Math | 1.5B | top-2 verified trajectories × 747K problems | 2 | 7e-6 | 128 | Step-verified data is the best SFT data they tested |
| LIMA | 65B | 1K pairs | 15 | 1e-5→1e-6 | 32 | 16x more data: no gain |
| DEITA | 7B/13B | 6K selected (3K ≈ 300K) | — | — | — | Alignment only |
| s1 | 32B | 1K reasoning traces | 5 | 1e-5 | 16 | 26 min on 16 H100 |
| DeepSeek-R1 | 671B | 800K (600K reasoning + **200K non-reasoning**) | 2 | — | — | Non-reasoning data rides along to preserve general ability |

### Where is the knee?

There are **two different knees** and the literature is only clear about one:

- **Alignment/format knee: 1K–6K examples.** LIMA (1K), DEITA (6K ≈ 300K; 3K ≈ 300K on MT-Bench), s1 (1K). All three explicitly invoke the "superficial alignment" reading: the capability is in the base, the small set teaches format and style. **This does not apply to you** — round 5 proved the capability is NOT in your base (removing the shortcut slot collapsed scratchpad 28.9→2.8).
- **Capability knee: nobody has found it at <1B for math, and the curves at 1.5B–7B show no plateau.** Tulu 3: average score still rises at 939K (GSM8K rises sharply; TruthfulQA *falls* — more SFT data is not free). Demystifying-Long-CoT (2502.03373): long-CoT SFT on Qwen2.5-Math-7B "has yet to plateau even at 3.5B tokens." RFT: the controlling variable is **distinct correct reasoning paths per problem** (5.25 for LLaMA-7B at k=100), not raw volume — dedup by solution structure, keep one per distinct path.

### Diversity vs volume at <1B specifically

This is where the 7B+ recipes break and the small-scale evidence is consistent:

- SmolLM2: smaller models get a *filtered, single-stage* mix; "math and code capabilities typically emerge only after extensive training" (MMLU-MCF above random only after 6T tokens at 1.7B — their scale, not yours, but the direction matters).
- Fine-Tuning Trap (2026): 10K narrow math examples (OrcaMath) *harm* SmolLM2-135M/360M. Narrow + full FT + <500M = below zero-shot.
- RFT: "RFT brings more improvement for less performant LLMs" — the 7B gained +5.8pt where the 33B gained ~0. Weak bases have more headroom for SFT.
- rStar-Math's SFT-data ablation (MATH, 7B): step-verified 78.4 > rejection-sampled 73.4 > NuminaMath-CoT 69.6 > MetaMath 55.2. **Verification beats volume; source diversity beats repetition.**

**Verdict for your setup:** measure the data by *coverage of distinct procedure types* (your mathbank has 756 programs — that is the right shape), not by row count. A concrete starting point: 10K–50K verified chains spanning the program families math-hard L3/L4 tests, one canonical format, answer-last. Your own best run (sft_k5_ctrl, 4,571 rows, Belle removed, verified synthetic) is consistent with few-K-verified > 13K-unverified. Falsifier: if 50K ≈ 10K on the held-out probe, volume is not your constraint and you are in b0's world.

## 2. Hyperparameters: what is load-bearing, what is cargo cult

**Load-bearing (ablated somewhere):**
- **LR: 1e-5–3e-5 for full FT at 1.5B–8B** (Qwen-Math 2e-5, RFT 2e-5, OLMo2 1e-5, Tulu3 5e-6, rStar-Math 7e-6). Your `lr_scale 0.1` convention sits in this range. SmolLM2's 3e-4 is a WSD artifact (SFT LR = end-of-decay LR), not a recommendation.
- **Epochs: 2–3.** Every math recipe above uses 2–3. The high-epoch outliers (LIMA 15, s1 5) are 1K-sample runs where epochs substitute for volume. Your 2–20 range: 20 is unjustified by any published recipe; your own arith runs (8 epochs, 200K rows, 4 min) are in the "tiny dataset, many epochs" regime Lee et al. operate in (5K steps on 100K = ~16 epochs). Epochs scale inversely with dataset size; the invariant is *total exposures per example*, ~3–10.
- **Data ordering and format** (see §5) — bigger than any optimizer knob.
- **Sum loss over tokens, not mean** (Tulu3, OLMo2): padding/accumulation otherwise silently reweights examples.

**Cargo cult (reported everywhere, ablated nowhere in these papers):**
- Warmup ratio 0.03 — copied, never swept.
- Weight decay 0.0–0.1 — most recipes don't even report it.
- Linear vs cosine — both used, no controlled comparison at small scale.
- Batch 128 sequences — conventional, not justified; throughput is compute-bound at your size (your own perf_sweep showed this).

**Freezing:** nobody freezes for SFT. The small-scale finding is the opposite direction — Fine-Tuning Trap: below ~500M, **LoRA/DoRA beats full FT** (SmolLM2-360M: DoRA 15.2 vs zero-shot 11.5 vs full FT 9.2 on OrcaMath). Full FT at 200M overwrites priors. Your call: full FT at low LR + replay (§4), or LoRA. Your own sft_k5 result (math-500 51.2→44.8, p=0.043) is the in-project version of the same phenomenon.

## 3. Loss masking and packing

**Your setup (mask prompt, supervise completion + turn terminator, doc-mask across packed rows) is the standard.** Tulu/OpenInstruct, LLaMA-Factory, axolotl (`train_on_inputs: false`) all mask the prompt and supervise the completion including EOS. test_sft_pack.py is enforcing the right thing.

What the literature adds:
- **EOS supervision is load-bearing for termination.** Your termination problem (17–21% on arith) is not a masking defect — it is a data/format defect (round 4: three formats on identical prompts left the answer underdetermined; the tag fixed it, +37pt termination). Keep supervising EOS; fix the format ambiguity.
- **Packing with document-boundary attention masking is standard** (varlen FlashAttention); nobody reports a controlled packing-vs-no-packing ablation at small scale — thin evidence, but the doc_mask_ab you already ran (free, no recompiles) is the right engineering answer.
- **Multi-turn:** supervise every assistant turn, mask every user turn (Tulu3 convention). You have this.
- One real subtlety from Tulu3/OLMo2: **sum, not mean, token loss** — with packing + gradient accumulation, mean loss silently reweights short completions up. Worth checking in sft_math.py.

## 4. Catastrophic forgetting and replay

Your SFT runs are narrow (math), and the forgetting is already measured in-project: sft_k5 dropped math-500 51.2→44.8 (p=0.043); sft_k4 dropped it 51.6→39.2 (p<0.001). The literature says this is expected at your scale and gives the fix:

- **Mixed Training (2025), Flan-T5-Base 250M — the closest published model to yours:** math-only SFT collapses NLI 81.0→16.5 *within the first 1,000 steps*; math only rises 3.1→12.0. Replay ratios tested: 1:1, 3:1, 7:1, 15:1 (math:general). **Even 6.2% general replay (15:1) holds NLI at 83.8%.** Math accuracy is flat (11.7–12.0) across all ratios — replay does not cost you the target skill.
- **DeepSeek-R1:** 200K of 800K SFT samples (25%) are deliberately non-reasoning (writing, factual QA, translation) to preserve general ability.
- **Tulu3:** the diverse mix (12% general chat, 11% knowledge, 11% multilingual) is motivated as forgetting prevention; removing subsets costs ~0.5–1.2 avg points.
- **Fine-Tuning Trap:** the forgetting mechanism at <500M is full-FT overwriting priors; LoRA/DoRA is the other fix.

**Verdict:** mix 5–10% general/chat data into the math SFT (your pretrain chat domain, in ChatML, already exists). It costs nothing on math per the 250M result and it is the difference between 16.5 and 83.8 on the preserved side. Falsifier: run math-only vs math+10% replay, score math-500 + C-Eval; if neither moves, replay is unnecessary at your mix — but the 250M prior says they will.

## 5. The core question: can SFT teach procedure execution, or does it need process supervision?

### What the evidence says SFT can do

- **Lee et al. (2307.03381), 10.6M transformer:** 1,000 detailed-scratchpad samples → 100% on 3-digit addition; 2,000 for simplified scratchpad; plain format plateaus at ~85% *even with 10K samples*. The scratchpad format puts the answer AFTER the working (`<scratch>…</scratch> 4 9 5`), so every digit is conditioned on prior computed digits. Curriculum (k→k+1 digits) needs 1K–5K samples. Plain format *forgets* 1–3 digit when learning 4-digit; scratchpad does not. Caveat Lee themselves report: length generalization is poor — the model learns "a mapping function constrained to trained digit lengths," not a flexible algorithm. Train on the lengths you will test.
- **Nye et al. (2112.00114):** scratchpad fine-tuning beats direct on polynomial evaluation (31.8→50.7) and Python execution (20→41.5); on 9–10-digit OOD addition, no-scratchpad models "completely fail" while scratchpad models keep improving. 100K examples, 5K steps. Scratchpad pretraining does NOT transfer to direct output — the gain is test-time conditioning on the working, not extra supervision.
- **rStar-Math (2501.04519), 1.5B:** MATH 51.2→88.6, AIME 0→46.7. The policy is pure SFT each round on trajectories whose steps are verified by **code execution** (NL reasoning embedded as Python comments; only steps that execute survive). Their SFT-data ablation: step-verified 78.4 > rejection-sampled 73.4 > NuminaMath-CoT 69.6. The process reward model is used to *select trajectories for SFT* and for test-time MCTS — not to RL the policy.
- **Qwen2.5-Math-1.5B:** SFT on 2.5M verified CoT samples: MATH 49.8→75.8. TIR (code-interleaved) variant: 79.9.
- **TinyGSM (2312.09241):** 12.3M Python-verified synthetic problems → 1.3B model at 81.5 GSM8K, above its GPT-3.5 teacher.
- **GKD (2306.13649):** on-policy distillation (student trains on its own generations with teacher feedback) beats off-policy at 77M–800M; 5% on-policy data beats 100% supervised KD. Relevant if you distill from your 27B teacher.

### What the evidence says SFT cannot do

- **Install a behavior the data does not contain.** Quiet-STaR (2403.09629) is the explicit test: self-taught rationales work on Mistral-7B but the authors state it bootstraps from "pre-existing reasoning ability" and is untested from scratch. Your round 5 is the same finding from the other side: 55,888 scratchpad rows whose targets are intact did not install scratchpad *generation*, because a third of the training data was `a + b = <answer>` — the model learned the shortcut, not the procedure. **Verified data in the wrong order teaches the wrong procedure.**
- **RFT (2308.01825):** SFT on gold solutions plateaus; the next gain comes from the model's OWN correct generations (7B: 35.9→41.7; multi-model union →49.3). SFT installs, but on-policy correct samples sharpen — and RFT helps weak models most.
- **RL-only at small scale:** TinyZero — Qwen2.5-0.5B "fails to learn reasoning" from RL; the 3B succeeds. SimpleRL-Zoo — Qwen2.5-0.5B RL from base: GSM8K 36.7→49.5, MATH-500 15.8→34.4, but **AIME stays 0.0**. RL at 0.5B polishes what exists; it does not install hard reasoning. (R1-Zero installed CoT at 671B — the scale-invariance of that result is exactly what TinyZero/SimpleRL cast doubt on.)
- **Process supervision (Math-Shepherd 2312.08935; PRM800K 2305.20050):** PRM > ORM, gap largest on hard problems — but every published win is ≥7B, and structurally, step RL on rollouts that are 51% wrong has the degenerate-group problem one level down (all-wrong steps → flat reward → no gradient; your own rlvr runs saw 30–55% degenerate groups at the sequence level).

### Joint answer with lessons-44 (we converged)

**SFT on machine-verified, answer-last procedure chains installs the behavior; step-level process supervision is the second move, for sharpening or for the case where SFT installs the form but not the dependency.** We agree on the discriminating experiment:

1. Build the SFT set from mathbank: **answer-last** (Lee/Nye format — working, then the answer), **one canonical format per prompt** (round 4: format ambiguity cost 25pt computation), **every line machine-verified** (you already have this), chains capped at ~256–512 tokens (learnability-gap evidence at ≤3B: 2502.12143; your own sft_k5 damage came with 2x-long answers). 10K–50K chains, 2–3 epochs, full FT at `lr_scale 0.1` + 5–10% replay (or LoRA — Fine-Tuning Trap predicts full FT costs you general ability below 500M; run both arms, it is minutes each).
2. Probe: **probe_procedure BOTH (held-out procedure problems) is the primary readout** — scored for (a) computation % and (b) **working-before-answer on problems NOT in training** (the held-out split must be on the problem hash, as arith_curriculum.held_out does — your 77%-leak disaster is the reason). **math-hard is a separate TRANSFER question, recorded but not a gate** — fb measured (2026-08-29) that math-hard and procedure_v1 share no templates (0/1032 tagged rows, 0 exact overlap, max Jaccard 0.200 vs the 0.8 threshold); they are different genres (word problems vs bare procedures), so math-hard cannot adjudicate whether SFT installed procedure execution. See §9.
3. **If computation rises AND working-before-answer generalizes → SFT did it; RL is sharpening only.** If computation rises but answer-first persists out of distribution → SFT installed the form and process supervision is needed to install the dependency. At that point the cheap version is Math-Shepherd-style: step reward = completion probability from that step, computed by rollout against your existing checker — no PRM to train, and your dense programmatic verification makes this genuinely available.

Where 44 and I nuance each other: 44 cites "RL reweights an existing distribution" as absolute — R1-Zero and TinyZero-3B show RL can install absent behavior, but at 0.5B it does not (TinyZero) or only polishes (SimpleRL). At 200M the practical conclusion holds either way. I add to 44's design: the answer-last ordering and format unification are not optional details — they are the difference between Lee's 100% and your round-4 scratchpad 28.9% (which was answer-first decoration).

### Measured at 200M (2026-08-29): the joint experiment ran

k8_proc_sft (50K answer-last verified chains, 3 epochs, lr_scale 0.1, full FT; EXPERIMENTS.md:75): **free-form probe_procedure BOTH 0.0/0.0/0.0 → 0.0/0.0/0.0 (STEPS 0→4/180); teacher-forced digit head 21.3%→57.2% (209→562/982).** The model predicts the right number on a gold prefix more than half the time and learned the chain shape + terminator, but cannot survive its own rollout — one early wrong number degenerates into 1/1 loops. SFT loss fell to 0.144 with zero free-form generalization: it fit next-token-given-gold-prefix, not execution.

This splits the original question. "Can SFT teach procedure execution?" has two answers at 200M:

- **Per-step computation: yes, installed** (21.3→57.2 teacher-forced). Lee/Nye/rStar-Math's claim replicates at 200M on the teacher-forced reading.
- **Free-form execution: no, not by SFT alone** (BOTH 0→0). The gap is exposure bias — the model never trains on its own prefixes, so at inference the first deviation from the gold distribution is unrecoverable.

The missing regime is **scheduled sampling / DAGGER**: train gold continuations on the model's OWN prefixes. Sequence becomes: procedure SFT → scheduled sampling (move 2.5) → distillation → RL. The Math-Shepherd step-RL fallback moves AFTER scheduled sampling: with free BOTH=0, rollouts are still all-wrong and carry no step signal to amplify; scheduled sampling must first move signal from the gold-prefix side to the own-prefix side. b0 and 44 both concede their "no intermediate regime" two-step collapse was wrong — the intermediate regime is scheduled sampling, and it now has a direct measurement pointing at it.

Replay arm (k8_proc_replay, same ledger): teacher-forced 61.1% vs 57.2%, McNemar χ²=15.9, p=6.6e-05 — replay significantly improves even the installation readout. My earlier "run both arms" sharpens to **replay-first**.

FoNE caveat (b0, I endorse): the 57.2%-vs-0 dissociation is only interpretable if the teacher-forced digit head and the free-generation path both go through the [NUM] channel. If the digit head is a separate FoNE objective, the gap is baked into the architecture and says nothing about generation. The fix is value-channel scheduled sampling regardless.

## 6. Falsifiers (what would show each recommendation is wrong, and how soon)

| Recommendation | Falsifying result | When the signal appears |
|---|---|---|
| Verified answer-last SFT installs procedure execution | **PARTIALLY FALSIFIED 2026-08-29:** teacher-forced computation rose (21.3→57.2%) but free-form BOTH stayed 0 — SFT installs per-step computation, not free-form execution | k8_proc_sft, one run |
| 10K–50K chains is the right volume | 50K ≈ 10K on the probe (volume not binding) OR 1K ≈ 10K (alignment-knee, capability is elsewhere) | One sweep, 3 runs |
| 2–3 epochs | Epoch 6+ still gains on HELD-OUT probe without train/test gap | Per-epoch eval; memorization shows as train-probe gap |
| 5–10% replay prevents forgetting | math-500 + C-Eval unchanged with vs without replay | One A/B; math-500 n=500 resolves ~±4pt, C-Eval n=1050 ~±2.7pt |
| LoRA ≥ full FT at 200M | Full FT at lr_scale 0.1 does not drop general evals and beats LoRA on math-hard | One A/B |
| SFT is sufficient (no step RL needed) | **FALSIFIED 2026-08-29:** teacher-forced 57.2% vs free-form 0 — but the indicated next move is scheduled sampling, not step RL (rollouts carry no step signal while free BOTH=0) | k8_proc_sft |
| Step RL as the fallback | pass@8 − pass@1 < 15pt after SFT (your own gate) — RL has nothing to amplify | pass@k probe, k=8, T=0.8 |
| b0's arm (10x pretrain dominates) | **NOT ADJUDICATED 2026-08-29:** BOTH=0 but teacher-forced 57.2% crosses b0's own >50% step gate — the constraint is exposure bias, not coverage/tokens; b0's arm stays queued without its null | k8_proc_sft |

Pre-registration lesson (fb's ledger): BOTH=0 alone does not falsify installation — the teacher-forced digit-head readout is what distinguishes "coverage was not the constraint" from "SFT cannot install". Any future SFT-arm pre-registration must pre-register BOTH readouts.

Two measurement caveats from your own EXPERIMENTS.md that apply to everything above: math-hard resolves ±1.1pt at 2–3% — report pass@k or size the eval before explaining a gap; and six of one day's conclusions came from probe artifacts — validate the instrument (the round-4 reverse-parser 24pt artifact is the template) before believing a number.

## 7. Where the three briefs stand

- **Consensus (me + lessons-44):** SFT installs procedure execution; the data must be verified, answer-last, single-format; step RL is the fallback, gated on the joint experiment; process supervision's proven small-scale form is trajectory selection for SFT (rStar-Math), not policy RL.
- **Open disagreement (me vs lessons-b0):** b0 attacks the SFT-then-RL ladder itself, arguing 10x pretraining or 27B distillation dominates at a 3.3B-token budget. I think b0's arm has real prior support (SmolLM2-135M at 2T tokens; Qwen2.5-Math-1.5B *base* at 49.8 MATH; MiniCPM's decay-stage result; your k4 anneal as best-ever math-500) and I do NOT claim SFT dominates more pretraining. I claim: (a) the 3.6% ceiling is uninformative because no SFT run has used data containing executed procedures; (b) the SFT arm is ~100x cheaper than the pretrain arm and is the correct first experiment; (c) if probe_procedure BOTH shows no execution gain, that null is positive evidence for b0 and the pretrain extension is next. fb should read this as "run SFT first, but b0's arm is queued, not rejected."
- **Not covered by any brief:** distillation from the 27B teacher. GKD (on-policy distillation, 77M students, 5% on-policy data beating full supervised KD) makes it a real third arm, cheaper than pretraining, and it composes with the SFT arm (teacher-generated verified chains ARE your mathbank data, possibly better than program-generated). Flagging so fb can assign it.

## Sources

- Lee et al., *Teaching Arithmetic to Small Transformers*, arXiv 2307.03381
- Nye et al., *Show Your Work: Scratchpads for Intermediate Computation with Language Models*, arXiv 2112.00114
- Yuan et al., *Scaling Relationship on Learning Mathematical Reasoning with LLMs* (RFT), arXiv 2308.01825
- Zelikman et al., *Quiet-STaR*, arXiv 2403.09629
- Deng et al., *Implicit Chain of Thought Reasoning via Knowledge Distillation*, arXiv 2311.01460
- Agarwal et al., *On-Policy Distillation of Language Models* (GKD), arXiv 2306.13649
- Wang et al., *Math-Shepherd*, arXiv 2312.08935
- Lightman et al., *Let's Verify Step by Step* (PRM800K), arXiv 2305.20050
- rStar-Math, arXiv 2501.04519
- SimpleRL-Zoo, arXiv 2503.18892 / github hkust-nlp/simpleRL-reason
- TinyZero, github Jiayi-Pan/TinyZero
- agentica-org, DeepScaleR-1.5B-Preview model card
- DeepSeek-R1, arXiv 2501.12948
- Qwen2.5-Math, arXiv 2409.12122
- TinyGSM, arXiv 2312.09241
- Allal et al., SmolLM2, arXiv 2502.02737
- Lambert et al., Tulu 3, arXiv 2411.15124
- Hu et al., MiniCPM, arXiv 2404.06395
- Abdin et al., Phi-3, arXiv 2404.14219
- OLMo 2, arXiv 2501.00656
- Zhou et al., LIMA, arXiv 2305.11206
- Liu et al., DEITA, arXiv 2312.15685
- Muennighoff et al., s1, arXiv 2501.19393
- *The Fine-Tuning Trap* (sub-1B math SFT, LoRA vs full FT), arXiv 2606.06920
- *Mitigating Catastrophic Forgetting in Mathematical Reasoning Finetuning through Mixed Training*, arXiv 2512.13706
- *Demystifying Long Chain-of-Thought Reasoning in LLMs*, arXiv 2502.03373

## 8. Addendum after the b0 exchange (2026-08-29)

b0 conceded the 3.6%-is-uninformative point and raised one trap that is now a **required condition** on the cheap experiment, plus refinements I accept:

1. **Program-level contamination (b0's trap, accepted and sharpened).** math-hard is generated by the same mathbank generator family as the SFT data. A row-level or problem-level hash split is insufficient — same template, different numbers is exactly the math-500 memorization shape (51/500 near-duplicates at Jaccard ≥ 0.8) and the arith probe's 77% leak. Required: (a) hold out **entire program_ids** for the eval families — eval programs must be programs never seen in SFT, not the same programs with different seeds; (b) run `scan_contamination` against the new SFT data before training (Jaccard alone misses template reuse — add a program_id overlap check, exact and free); (c) report **per-program-family accuracy** — a gain concentrated in trained families is memorization, not execution. The tooling exists: `arith_curriculum.held_out` is the hash-split pattern; generalize it from problem to program_id.
2. **Adjudication is asymmetric (b0's point, accepted).** A null at 200M with no published precedent is weak evidence. A WIN with a clean program-level split adjudicates a lot: it identifies the constraint as data/procedure and installs the behavior every downstream arm needs (RL amplifies an existing distribution; it does not create one). Run the experiment for the win case, not the null.
3. **The arms are not in contention (b0's cost point, accepted).** If procedures come from the 27B, the SFT arm IS the distillation arm; if from mathbank, cheap but narrow. The 30B-token pretrain extension's corpus growth (quarantined web + textbook) is CPU/teacher-bound and runs in parallel without GPU contention. Ordering: (1) mathbank procedure SFT with program-level split — hours; (2) 27B distillation — days; (3) pretrain extension — ~15h GPU, queued regardless because every arm's ceiling is set by the base.
4. **GKD correction for b0's distillation-economics agent:** GKD's arithmetic result is real — GSM8K few-shot CoT, T5-base 250M, on-policy GKD "substantially outperforms" off-policy, 1.9x average gain on arithmetic reasoning tasks — but it is encoder-decoder, few-shot, not 200M-decoder SFT. Use the GSM8K cell, not the summarization headline (the 7000x-smaller-than-PaLM number is XSum).

## 9. Addendum after fb's contamination measurement (2026-08-29)

fb measured b0's contamination trap directly (math_hard_eval_1k 1,032 rows vs procedure_v1 50,000 rows): **zero contamination** — 0/1032 tagged rows, 0 exact stem overlap, max Jaccard 0.200 (project threshold 0.8). b0's trap does not fire on this pair. But the reason matters more than the verdict: the two are different genres. math-hard is untagged word problems ("今年老师的年龄比学生的3倍少3岁…"); procedure_v1 is tagged bare procedures ("[单位换算] 把 777 千克 换算成 毫克"). They share no templates.

Consequences, agreed with fb:

1. **math-hard is a TRANSFER test, not the gate.** Using "math-hard clears ~5%" as the SFT arm's null criterion (as §5/§6 originally did) would risk killing the arm for the wrong reason: bare procedures not transferring to word problems says nothing about whether SFT installed procedure execution. The primary readout is now **probe_procedure BOTH (held-out procedure problems), in-distribution**; math-hard is recorded separately as the transfer question. The falsifier table and TL;DR are updated accordingly.
2. **Correction to my b0 relay (fb caught it):** procedure_curriculum's split is `prob_key(fmt, 题干)` — problem-level, not template-level. 12,506 distinct problems come from 3 templates, so the held-out 10% is "same template, different numbers." The residual risk lands on the probe, not on math-hard. This is acceptable and worth stating: for "execute a procedure," running the same template on unseen numbers IS the skill (Lee et al.'s excluded-number robustness: models stay at 100% even excluding half the 3-digit numbers; their excluded-DIGIT robustness is the weaker axis). What the probe does NOT measure is template-level generalization — with 3 templates there is no held-out template to test on. If the probe succeeds, the next question is new-template learning, which needs new templates in the SFT data.
3. **If the in-distribution probe succeeds and math-hard does not move**, that is not an SFT failure — it is a data-coverage gap, and the next move is bridging data (word problems rendered as procedures, or procedures embedded in word-problem context). Do not conclude "SFT doesn't transfer" without that arm.
4. **Replay arm confirmed.** fb runs procedure_v1 SFT after k8 lands (~1h), with a 5–10% replay control arm — motivated by the project's own measured forgetting (math-500 51.2→44.8, p=0.043; 51.6→39.2, p<0.001) and the Flan-T5-250M prior (6.2% replay: NLI 16.5→83.8, math flat across all ratios). Score replay on math-500 + C-Eval + the procedure probe; the 250M result predicts math stays flat while general ability is preserved.
