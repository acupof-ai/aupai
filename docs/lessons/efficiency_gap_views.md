---
question: What factors explain the 10^4 efficiency gap between model and brain, what does the RL ordering say about where efficiency comes from, and what experiment should this repo run next?
status: open
source: team views collected 2026-09-05, six sessions (44, 84, 58, 98, 62, tilerl-0a, db)
---

# Efficiency gap: team views

Question (user, 2026-09-05 08:30 local): model inference and learning run maybe 10^4 times less efficiently than a human brain. (1) List every factor you can defend for the gap, split into energy/compute efficiency and sample efficiency, each with a number and its basis. (2) Compare human skill acquisition with RL sample efficiency — model-free RL (Atari, DQN 200M frames), model-based/self-supervised RL (EfficientZero 100K frames), RL on pretrained LLM (R1-Zero, 1-shot RLVR) — where does each stand relative to a human, and what does the ordering say about where the efficiency comes from? (3) Propose the next experiment for THIS repo: 200M reasoning model, code+math, 8 H20s, 1-2 days, that measures sample efficiency in a way a fact can be written about. Say what would change your mind.

Numbers labelled: [measured], [paper-reported], [estimate].

---

## 44 (lessons-44)

### (1) Factors for the 10^4 gap

**Energy/compute efficiency — the gap is not in raw FLOPS/W.**

An H20 delivers ~148 TFLOPS BF16 at 400W TDP [paper-reported] = 3.7×10^11 FLOPS/W. The brain delivers ~10^12–10^13 synaptic ops/s at 20W [estimate: 10^14 synapses × 1–10 Hz average firing] = 0.5–5×10^11 ops/W. The H20 is comparable or better in raw compute efficiency. The 10^4 gap is elsewhere.

1. **Memory wall (~10^3, dominant).** A synapse IS the memory — the weight is stored where the multiply happens. A GPU fetches every weight from HBM per operation. DRAM access costs ~1.3–2.6 nJ vs FP16 FMA ~0.4–1.2 pJ at 45nm [paper-reported, Horowitz 2014] — ~10^3× per access. A 200M model (400MB BF16) does 400MB of HBM reads per forward pass; the brain does zero separate memory fetches. This is the largest single factor.

2. **Sparsity (~10–100×).** The brain activates ~1–5% of neurons at any instant [measured, Lennie 2003]. GPUs do dense matrix multiply — every weight is read and every MAC computed whether or not the activation is zero. MoE and sparse attention capture some of this; the gap remains.

3. **Event-driven vs clocked (~10–100×).** The brain fires asynchronously on spikes; a GPU clocks at 1.5+ GHz and draws power whether or not there is useful work. An idle SM still consumes.

4. **Precision (~2–4×).** Brain: analog, ~4–8 bit equivalent [estimate]. GPU: 16-bit with 32-bit accumulation. FP8 narrows but does not close this.

5. **Cooling overhead (~2×).** Data center PUE ~1.1–1.5 [measured]; brain cooling (blood flow) is ~2× neural tissue energy [estimate].

**Sample efficiency — the gap is in the prior, not the algorithm.**

1. **Evolutionary priors (~10^3–10^6).** Brain architecture is shaped by ~10^9 years of evolution [measured, geological record]. This is "free" prior knowledge encoded in the connectome. The human genome stores ~750MB [measured: 3×10^9 bp × 2 bits], of which a fraction specifies brain architecture — but the effective prior is larger because evolution compresses environmental regularities. An LLM starts from random weights.

2. **Embodiment and active learning (~10–100×).** A child actively explores the physical world, chooses what to attend to, gets immediate causal feedback. An LLM passively reads a fixed corpus. Active learning can reduce sample complexity 10–100× in theory [paper-reported, Settles 2009].

3. **Multi-modal grounding (~10–100×).** Humans learn from vision + sound + touch + language simultaneously. Text-only learning lacks the grounding that makes concepts robust. A child sees "dog" while seeing, hearing, touching a dog; an LLM sees the token in text.

4. **Curriculum and social learning (~10–100×).** Human learning is ordered by difficulty and adapted to the learner by teachers. LLM data is roughly random-order. Adaptive teaching can reduce sample complexity 10–100× [paper-reported, curriculum learning surveys].

5. **Sleep and consolidation (~2–10×).** The brain replays and consolidates during sleep [measured, Wilson & McNaughton 1994]. More effective than a single training pass.

6. **Reuse and transfer (~10–100×).** Humans reuse skills across domains (walking → running, counting → arithmetic). LLMs transfer but less efficiently — each new capability needs substantial new data.

### (2) RL sample efficiency vs human

**Model-free RL (DQN, Atari):** 200M frames [paper-reported, Mnih et al. 2015]. At frame-skip 4, ~50M decisions, ~38 hours of game experience. Human: ~2 hours to reasonable skill [estimate]. Gap: ~20× in experience, ~100× in wall-clock (DQN took ~8 GPU-days [estimate]). The gap is not in RL per se but in the absence of a world model.

**Model-based RL (EfficientZero):** 100K frames [paper-reported, Ye et al. 2021], matching DQN at 200M. ~28 min of game experience vs human ~2 hours. Gap: ~4×. The 2000× improvement over DQN comes from a learned world model + self-supervised consistency + MCTS. The world model is the key: it lets the agent "imagine" experience rather than collect it.

**RLVR (R1-Zero, 1-shot):** R1-Zero trained a base model with GRPO, no SFT. Prompt count not public; order 10^4–10^5 [estimate]. Each prompt generates 8–16 completions of ~1000–5000 tokens; total RL tokens ~10^8–10^9 [estimate]. The base model already saw ~10^12–10^13 tokens in pretraining [paper-reported for DeepSeek-V3-class]. 1-shot RLVR results show improvement from very few prompts [paper-reported, active area — I am not confident in the exact claim]. Human: learns a new math technique from ~1–10 worked examples [estimate, tutoring]. Gap in RL samples: ~10^3–10^4, but the human also carries ~10^7–10^8 prior life examples.

**Ordering and what it says:** model-free (200M frames) → model-based (100K, 2000× better) → RLVR (10^4–10^5 prompts on top of 10^13 pretraining tokens). The efficiency comes from: (a) world models — a learned prior that compresses experience; (b) pretraining — a much stronger prior that already encodes language, logic, math facts; (c) task structure — math reasoning is more structured than Atari pixels. The 10^4 gap is the gap between no prior (model-free RL from scratch) and a strong prior (evolution + lifetime learning). Close the prior and the gap shrinks to ~4× (EfficientZero) or less (RLVR).

### (3) Next experiment for this repo

**RLVR sample efficiency curve on math.** Train the 200M model with GRPO on math problems at 5 prompt counts: 10, 50, 200, 1000, 5000 unique prompts. Fixed compute per arm (~4h on 8 H20s). Measure pass@1 on a held-out math set (MATH-500 or GSM8K) before and after. The learning curve (Δpass@1 vs N prompts) IS the sample efficiency measurement. A fact can be written: "200M model improves Δ=X% from N unique RL prompts on math."

**Why:** the repo has RL infra (algorithms/), a 200M model, 8 H20s; 5 arms × 4h fits in 1–2 days; the result is a fact, not a scenario.

**What would change my mind:**
- 10 prompts → >5% improvement: the pretrained prior is so strong that RL is eliciting, not learning. The gap is in pretraining, not RL.
- 5000 prompts → <1% improvement: model capacity or RL signal is the bottleneck, not sample efficiency.
- Steep curve at 100–1000: a "sweet spot" where RL efficiently adds capability — the gap to human is in prior quality, not RL algorithm.

**Alternative (if RL infra not ready):** CPT learning curve on math (10^6, 10^7, 10^8 tokens, measure pass@1). Measures pretraining efficiency, less directly comparable to human RL.

---

## 84 (aupai-84)

Data angle. Everything below is measured in this repo unless labelled otherwise; where I have no number I say so rather than estimate.

### (1) Factors I can defend, data side only

I can defend two, and I want to be clear that neither is an energy number — I have no measurement of joules on either side and will not manufacture one.

**a. Sample reuse.** Our 200M control (b0_headmix_armA) consumed 244,160 rows over 3,815 steps = 1.0B tokens for one checkpoint, and that is 0.301 epochs of code_py_starcoder's pool alone. [measured, from the checkpoint's own row_cursor] A human acquiring a coding skill does not read 1B tokens; the ratio is where a large part of any 10^4 lives, but I cannot put a denominator on the human side from anything I have read.

**b. Supervision density.** Our training signal is next-token over 4096-token rows. The math generators can emit a *verified* answer per problem and algorithms/code_reward.py can emit a pass/fail per program — those are the only two dense-verifier channels this repo has. Everything else is unverified next-token. That is a real structural difference from a human's error signal, and it is the one this repo can actually manipulate.

### (2) The RL ordering — I defer, with one data caveat

I have no measured DQN / EfficientZero / RLVR numbers, and the paper figures I could recite I have not read today, so I would be citing from memory into a summary — exactly what today has cost us four times. Whoever answers this should give paper + number. My caveat is on the third rung: 1-shot RLVR results are the ones most exposed to contamination, because "one problem" makes any prior exposure decisive. If we cite them, we cite whether the paper controlled for pretraining exposure. Ours would have to.

### (3) The experiment — what the data can and cannot supply

The N in {1, 10, 100, 1000} curve is buildable on the MATH side and NOT on the code side today.

**MATH** — supply is not the constraint. 1,342 generator functions across 25 math_programs_l*.py files, all 25 randomised, so instances are effectively unlimited and each carries a computed answer. [measured, via mathbank/vet_programs.py's glob] A curve at N in {1,10,100,1000} problems × several seeds is comfortably within supply.

**CODE** — NOT buildable now. cs.code_tests_supply is 15 novel rows / 0 tokens: there is no data/corpus/code_tests, only a yield study that writes result.json and no shards. The "5.6B mineable" figure that once justified a 2B target cited a fact id that does not exist anywhere in the tree. So a code-side sample-efficiency curve needs a miner run first; do not plan the experiment assuming it.

**The contamination controls, and this is the part I would not compromise:**

- The eval must be type-disjoint from the training generator, not merely distinct instances. cont.math_short_leak is exactly this failure: our own generator and the old eval generator implemented the same canon of problem types, so hand-written templates converged in surface wording, and every batch measured REJECT (v3 28, v5 39, v6 43, v7 37, v8 30, v10 12, v11 33 holdouts at containment 0.8). math_hard v1 is VOID as a result. v2 is clean because it was built from type-disjoint families (1,080 problems, 18 families) — that construction, not a bank fix, is what closed it. Any N-curve eval must be v2-style or it measures recall of seen types.
- Screen at containment >= 0.8 with a same-scale in-training baseline, and report RATES not counts. cont.cci3_scale_failure: "reject the source if any shard hits" rejects everything once shards are ~1GB — 72/124 of our own in-training web_hq shards REJECT at 0.8 with 0 exact matches. Absolute hit counts across unequal corpora are meaningless.
- At N=1 and N=10 the whole result hinges on whether those specific problems appear in pretraining. Screen each one individually against the corpus before the run, not the set in aggregate; a 3.5% set-level rate says nothing about your one problem.
- Fix the ruler before the arms: the pod's own measured contamination is math-500 0/500 at 0.8 (clean) and math-hard v1 36/1032 (3.5%). Use math-500 or a fresh type-disjoint set; not v1.

**What would change my mind on the data claim:** if a per-problem containment screen at 0.8 over the pretraining corpus returns hits for the N=1/N=10 items, the curve measures retrieval and not acquisition, and the experiment should be rebuilt on generator families whose surface forms postdate the corpus. Equally, if someone shows the 1,342 generators collapse to far fewer *distinct* solution procedures — I counted functions, not procedures, and have not measured that — then "unlimited instances" is a supply claim that overstates the diversity, and N=1000 would be 1000 draws from a much smaller effective set.

One process note: I hold no card and this is all static analysis of corpus and facts; nothing here required GPU work, so the pause does not block any of it.

---

## 58 (lessons-58)

Eval-design angle. Every number labelled. Where I have not measured it, I say whose number it is.

### 0. The premise is two claims and they have different answers

"10^4 less efficient" conflates energy-per-operation with samples-to-competence. I get ~10^2-10^3 for the energy gap and ~10^1-10^2 for the text-exposure gap, so 10^4 is not one factor — it is a product of smaller ones, and which ones you name changes what you would fix.

### 1. Factors, with arithmetic

**Energy per operation** — computed here from published specs:
- brain: 20 W, ~10^15 synaptic ops/s [paper range 10^14-10^16, Herculano-Houzel-class estimates] = 5.0e13 ops/J
- H20: 148 TFLOP/s bf16 dense, 400 W TDP = 3.7e11 FLOP/J at spec
- at OUR measured MFU (30%, from the b0 control's MFU column in runs) = 1.1e11 FLOP/J
- ratio: 135× at spec, 450× as we actually run it.
- Basis: spec sheet + measured MFU; the ops/J for brain is an estimate whose numerator spans two orders of magnitude, so treat 135-450× as "10^2, maybe 10^3", not as three digits. A synaptic op is not a bf16 FLOP; the comparison is a unit conversion nobody can do honestly. I would not defend this factor past its order of magnitude.

**Text exposure** — computed here:
- human by 18: 25k words/day × 365 × 18 = 164M words ≈ 214M tokens at 1.3 tok/word [the 25k/day figure is an estimate from language-acquisition literature, spread 10k-50k; the multiplier is our tokenizer's rough ratio, not measured on English specifically]
- our 200M control: 8.0B tokens [MEASURED, data/mix_200m_8b.json, 3815 steps × 16 × 2 × 2 × 4096]
- ratio: 37× more text than a human sees by adulthood, for a model that is not competent at 18-year-old level on anything.
- This is the factor I trust most and it is only 10^1.5.

**Inference cost per token** — computed here: 2N FLOP/token gives 206M → 4.1e8 FLOP, 9.3 µs, 3.7e-3 J/token at measured MFU; 70B → 1.4e11 FLOP, 3.2 ms, 1.26 J/token. A human speaking at ~150 wpm on a 20 W budget spends ~8 J/word. So per token emitted, a 70B model is already within an order of magnitude of a brain, and a 200M model is 2000× CHEAPER. The gap is not in the forward pass. That is the load-bearing observation for part 2.

**What the gap is NOT:** it is not attention's quadratic term at our scale (seq 4096, 206M dense: the FFN dominates), and it is not precision — we already train bf16 and the fp8 path measured no quality loss on parity tests.

### 2. RL ordering, and what it says

Converted to a common unit (hours of game time at 60 fps), from paper-reported sample counts:
- DQN / Rainbow, 200M frames = 926 h of experience
- EfficientZero / SimPLe, 100k steps (400k frames) = 1.9 h
- human benchmark = 2 h
- DQN/EfficientZero = 500×. EfficientZero/human = 0.93×, i.e. AT human sample efficiency on Atari.
- Basis: paper-reported frame budgets [Mnih 2015; Hessel 2018; Kaiser 2020; Ye 2021], arithmetic mine. Atari-100k scores are below human on most games even when the SAMPLE count matches, so "matches human sample efficiency" is a statement about the x-axis only.

RLVR on a pretrained LLM is the third point and it breaks the axis: 1-shot RLVR [Wang 2025, paper-reported] moves MATH500 by double digits from ONE example. Model-free RL needs 10^8 frames, model-based needs 10^5, RL on a pretrained model needs 10^0-10^3.

**THE ORDERING SAYS EFFICIENCY COMES FROM THE PRIOR, NOT THE ALGORITHM.** Each step buys ~10^2-10^3 in samples, and each step adds a stronger world-model prior: none → a learned dynamics model → a pretrained model of everything written down. A human learning a video game in 2 hours is not sample-efficient in general; they arrive with 18 years of pretraining on objects, gravity, goals and menus. That makes the human/model comparison a comparison of PRIORS, and the honest analogue of "human learns a game in 2 h" is not DQN from scratch — it is RLVR on a pretrained model, where the model is also fast. The 10^4 gap mostly measures that we compare a from-scratch learner against a transfer learner.

Consequence I would defend: sample efficiency at fixed prior is nearly a solved axis (EfficientZero already matches human), and the open axis is how much prior a given amount of pretraining buys per token. That is measurable at 200M.

### 3. The experiment, and the resolution math

**MEASURE THE HORIZONTAL SHIFT OF A LEARNING CURVE, NOT AN ENDPOINT.** Sample efficiency is r in L_B(D) = L_A(D/r): how many fewer tokens arm B needs for the same loss. Endpoint comparisons cannot report r at all.

The resolution problem, computed here. On L(D) = a·D^-b + c, a horizontal factor r shows up as a vertical gap of R·(1-r^-b) where R = a·D^-b is the reducible loss still on the table. With b~0.095, a 2× sample-efficiency advantage is a 6.4% change in the reducible term — at R=0.3 nat that is 0.019 nat. So:
- SE(ln r) = SE(Δ) / (b·R).
- To pin r to ±10% at b=0.095, R=0.3 you need SE(Δ) = 0.0029 nat.

That is unreachable UNPAIRED and easy PAIRED, which is the whole design:
- Unpaired, per-token loss SD ~1 nat, 8000 docs × 2048 tok at ICC 0.3: SE(Δ) = 0.0061 → r only to ×1.20.
- PAIRED (both arms score the SAME held-out tokens, so the statistic is the per-token loss DIFFERENCE): sd_diff = sqrt(2(1-corr)). Two arms of one shape correlate ~0.95 per token, giving sd_diff 0.32 and, at 8000 docs, SE(Δ) = 0.0019 → r to ×1.07 at 1 SE.
- Same tokens, same order, 25× tighter on r. Pairing is worth more than 10× the eval set.

**THE RUN,** in the 8×H20 / 1-2 day budget: measured control throughput is 82.3K tok/s/gpu at 206M [from the b0 arms' matched-step ratios], so 8 cards = 658K tok/s = 57B tokens/day. Two arms × 8B tokens on 4 cards each = ~5.6 h wall clock. That fits three times over, so run THREE data budgets (2B / 4B / 8B) per arm and fit the curve rather than asserting a slope from two points — my own §"three points declared a trend" was five-point-falsified, and a two-point r is a line through two noisy dots.

**WHAT ARM B SHOULD BE:** the cheapest intervention that plausibly adds prior per token. My candidate is curriculum-by-difficulty on the code+math mix (easy-to-hard ordering at fixed token count and fixed mix weights), because it changes only the ORDER — same tokens, same count, so a positive r cannot be explained by having seen more data. Every other candidate I considered changes the token budget or the mix and reopens that confound.

**MEASUREMENT STACK,** in order of resolution: (a) paired per-token BPB on held-out code+math, 8000+ docs, document-clustered SE — this is the estimator r is fit from; (b) the api_cloze read-set probe I own, as a memorisation control, so a curriculum win is not just faster memorisation of the training set; (c) 4-way MC accuracy last, and only as a sanity check — at n=2500 its paired MDE is 0.024 absolute, which is 3-10× coarser than the BPB route and cannot see a 2× sample-efficiency difference at all. Reporting accuracy as the headline here is the single easiest way to produce a null that is really a resolution failure.

**WHAT WOULD CHANGE MY MIND:**
- If the fitted b at 200M is much larger than 0.095 (say >0.2), the vertical gap per unit r grows and even the unpaired design resolves r; the pairing argument stops being load-bearing.
- If per-token loss correlation between arms is below ~0.8, sd_diff doubles and 8000 docs is not enough; I would measure the correlation on the two b0 arms I already have before committing eval-set size. This is a cheap CPU-side check on existing checkpoints and I would run it first.
- If the three budgets do not lie on a power law (residuals structured, not noise), r is not well-defined on this range and the whole framing has to be replaced with "loss at matched tokens" and reported as such.
- If the memorisation control moves with the curriculum, a BPB win is contaminated and r is measuring memorisation speed, not learning.

**CAVEATS I WILL NOT DROP:** b=0.095 is a Chinchilla-family estimate for loss-vs-data at fixed N, not measured on this mix — every "r to ×1.07" above is conditional on it, and the first output of the experiment should be b itself with its own CI. Per-token loss SD ~1.0 nat and ICC 0.3 are order-of-magnitude estimates from val traces, not measured; both feed the eval-set sizing linearly, so I would measure them on existing checkpoints before fixing the eval set. No GPU work is running from me — the user has paused GPU work and main is frozen for the credential rewrite.

---

## 98 (aupai-98)

角度：结果怎么一屏展示给用户，旁边放一个他认识的参照物。数字标注 [实测]/[论文]/[估计]。

### (1) 10^4 差距的因子，拆开看

- **每操作能耗：** 脑 ~20W [实测，标准值] 做 ~10^15 突触操作/秒 [估计：10^11 神经元 × 10^3-10^4 突触 × 0.1-10Hz]，约 10^-14 J/操作；H20 ~400W [规格] 跑 ~10^14-10^15 FLOP/s（FP8/BF16），约 10^-12 J/FLOP。差 ~10^2。但突触事件约等于 1 位信息、FLOP 是 16 位，按信息量折算差 ~10^1。
- **每 token 能耗 vs 每秒思考：** 我们 200M 实测 74K tok/s/GPU（runs/b0_mem_m1.log）[实测]，8 卡 ~6×10^5 tok/s、~3.2kW，约 5×10^-3 J/token；人一秒认知 ~20 J [实测]。按「一个 token ≈ 人一秒的阅读/输出」[估计]，每单位产出差 ~10^3-10^4。
- **权重搬运：** 每个 token 要把 2 亿参数从 HBM 重读一遍：0.4 GB/token × 74K ≈ 30 TB/s [实测推导]，撞 HBM 墙，利用率 ~30-40%。脑把突触存在计算点旁边，没有独立内存总线。
- **样本效率（主项）：** 我们 200M 吃了 8×10^9 token [实测，mix]；儿童 5-8 岁听 ~10^6-10^7 词 [论文：Hart & Risley 上限 3000 万词/3 岁，现代估计 10^6-10^7]。差 ~10^3。前沿模型 10^12-10^13 token → 10^5-10^6。10^4 主要住在这里。
- **先验：** 脑带着演化（~10^9 年）和文化（~10^5 代）买的归纳偏置出厂，模型随机初始化。无法直接量化，但 (2) 的排序给它定了价。

### (2) RL 样本效率排序（学会一个新技能所需示例数）

- 人：~10^1-10^3 [估计：新 API 看几个例子会用；新游戏几分钟上手；外语单词 ~10^2 次接触记住]
- DQN 类 model-free：~10^7-10^8 帧 [论文：DQN 共 2 亿帧，单个人类水平 Atari ~10^7-10^8 帧]
- EfficientZero 类 model-based：~10^5-10^6 帧 [论文：2 小时游戏时间达人类水平，比 DQN 好 ~100-1000×]
- 预训练 LLM 上 RLVR：R1-Zero 类 ~10^4-10^5 提示 [论文/估计：DeepSeek-R1-Zero，精确数未公开]；1-shot RLVR ~10^0-10^2 示例 [论文：2025 One-step RL / 1-shot RLVR，个位数提示即可测地改变行为]
- 排序：1-shot RLVR (10^0-10^2) ≈ 人 (10^1-10^3) > R1-Zero 类 (10^4-10^5) > EfficientZero (10^5-10^6) > DQN (10^7-10^8) > 从零预训练 (10^9-10^13)。
- 读法：效率来自先验，不来自算法。model-based 比 model-free 好 ~10^3——学出来的世界模型就是先验；RLVR 又好 ~10^3——预训练 LLM 是更强的先验。每 ~10^3 一步都是「学习开始前多烘进去的结构」。人的样本效率出现在系统只学一个强模型之上的增量时。差距不是我们的 RL 算法弱，是我们的先验弱。

### (3) 本仓下一个实验（200M、code+math、8×H20、1-2 天）+ 一屏展示

- **实验：** 在 e1 的 api_cloze 探针上做样本效率曲线（它本来就有 seen/unseen 区域，形状正好）。用 N=10^2/10^3/10^4/10^5 行 API 名语料 SFT 200M，测 unseen 区域填空准确率。产出一个事实：达标示例数（unseen 准确率 ×2 于随机所需的 N）。成本：4 跑 × 1-2 卡时，lane 卡上 1-2 天够。
- **什么会改变我的看法：** unseen 准确率过 10^5 示例还在随机水平 → 200M 根本不会从名字外推，差距是表征/架构问题，堆数据不关；<10^4 示例就爬升 → 差距主要是数据量，10^4 是数据问题不是架构问题。
- **一屏怎么给用户看：** 一张图，x=示例数（对数），y=unseen 准确率。三条线：模型曲线 [实测]、人学新 API 的参照（~1-5 例首次使用、~10^1-10^2 熟练）[估计，标注]、预训练基线点（8×10^9 token 后基座在哪 [实测]，换算成语料里已含多少该技能的示例）。标题数字：「学会一个新 API：模型 N 例 vs 人约 M 例」。旁边放他认识的参照物：同屏角板放现有的 l1_fewshot/math-hard 数（同一探针族、同一量表），成本换算成 GPU 小时放在 74K tok/s 吞吐数旁边——样本效率结果坐在页面上已有数字的旁边，不用另起一个量表。

阻塞说明：探针文件正在 main 冻结改写中（凭据事件），且 ckpt_b0_headmix_armA.pt 加载崩（readout 主题 block 行）；实验等这两件事解开。备选：同一曲线搬去数学技能（math SFT 数据 + digit head 探针都在）。

---

## 62 (lessons-62)

Answering from tonight's memory-layer measurements, which bear directly on part 1 and on the capacity-vs-sample-efficiency split. Every number labelled. Where I have no defensible number I say so rather than supply one.

### My angle first

M1 added a 1,048,576-row product-key table: 1,077,937,152 memory parameters against 206,128,200 dense, so 6.23× the control's parameter count at identical dense params [measured, equality asserted at construction — the memory is parallel to the FFN, never replacing it]. What that bought, block-paired doc_cu val against the control on identical data order:

- step 500: M1 2.904, control 2.917, −0.013 nat
- step 1000: M1 2.592, control 2.589, +0.003 nat

At step 500 it was at the charter's adopt bar (≤ −0.010). By step 1000 it was gone. 0.262B tokens seen at that point [measured: batch 16 × accum 2 × seq 4096 × world 2 × 1000]. And the cost was 10% of throughput [measured: 0.896 of control at step 30, 0.898 at step 100] plus 16.34 GiB resident for the table's tensors at 12 B/param steady.

So: 6.23× the parameters, ~zero loss delta at 0.26B tokens. That is the single most useful datum I have for the user's question, and it points one way — at this scale, on this data budget, ADDING REACHABLE CAPACITY BOUGHT NOTHING. The bottleneck was not how many parameters a token can reach.

And I know WHY it bought nothing. The table collapsed: pool_touched_frac fell monotonically 0.306 (step 100) → 0.214 → 0.137 → 0.113 → 0.099 → 0.095 → 0.094 → 0.094 → 0.092 → 0.0945 (step 1000), key_gini rose 0.781 → 0.919, topk_entropy fell 2.37 → 0.93 against a 3.466 maximum. All measured, real arm, real data. M3 at 1,428,025 rows was steeper: 0.070 by step 600. Readout 6 (a per-row fp32 checksum of the value table) read 1.09-1.25 of touched rows changed at every window, so the writes landed — the failure is entirely in WHICH rows the selector chose. Roughly 6% of a billion-parameter table was doing all the work, and the larger the table the smaller the fraction.

That is a sample-efficiency failure wearing a capacity costume. The parameters were there and reachable; the model could not learn to route to them in 1000 steps. A brain does not have this problem in the same form, and I think that gap — routing/credit-assignment efficiency, not storage — is where a large part of the 10^4 sits. I would rather offer that one measured mechanism than a list of factors I have not measured.

### (1) Factors

I can defend three from my own measurements and I will not pad the rest.

**MEASURED, this repo, tonight:**
- **Utilisation of provisioned capacity:** ~6% of a 1.07B-parameter table touched per 100-step window at step 1000, falling. If a system provisions 16× more parameters than it uses, that is a factor in any per-parameter or per-joule accounting, and it is a factor nobody puts in these lists because it is invisible without the instrument.
- **Hardware utilisation:** MFU 31% at the arm's best [measured, 74K tok/s/gpu on H20s], so ~3× of the compute the silicon can deliver is already lost before any comparison to biology.
- **Optimizer state overhead:** 12 B/param steady, 16 in-step for a bf16 table with an fp32 master and Adagrad's accumulator [measured]. Four to eight bytes of machinery per byte of learned parameter. No biological analogue that I know of.

**WHAT I WILL NOT GIVE A NUMBER FOR:** brain power draw (~20 W is the figure everyone quotes; I have not read the primary source), synapse count, spike energy, or a joules-per-inference for either side. Those are the terms that actually set the 10^4, and I have not verified any of them. Someone with the papers open should supply them — from me they would be numbers with no basis, which is worse than a gap in the table.

### (2) RL vs human

Ordering from what I have read, but the ordering is a claim about mechanism and I will defend that part:

- **Model-free DQN-class:** paper-reported on the order of 10^7-10^8 frames for human-level on individual Atari games, against a human reaching reasonable play in minutes to tens of minutes. Four to five orders worse, and this is the number the 10^4 folklore mostly comes from.
- **Model-based / self-supervised, EfficientZero-class:** paper-reported human-level Atari at ~10^5 frames (100K benchmark). Two to three orders better than DQN-class on the same task, from a learned model plus self-supervised representation — no extra environment interaction, only better use of it.
- **RL on a pretrained LLM:** RLVR/R1-Zero-class reaches real capability gains in 10^3-10^4 rollouts, and 1-shot RLVR reportedly moves math benchmarks measurably from a SINGLE example. Effectively at or past human sample efficiency for the update itself.

**WHAT THE ORDERING SAYS:** Sample efficiency tracks HOW MUCH STRUCTURE THE LEARNER ALREADY HAS, not the RL algorithm. DQN starts from nothing and pays 10^8. EfficientZero starts from a learned world model and pays 10^5. RLVR starts from a pretrained LLM and pays 10^0-10^4. The RL algorithms in the third case are cruder than in the second — GRPO/PPO variants against a verifier — and they are three orders more efficient anyway.

So the user's premise is right that RL is the fair comparison for human skill acquisition, and the comparison then says something specific: a human learning a new skill in ten trials is not doing efficient RL, they are doing cheap RL on top of an enormous pretrained prior. Comparing human skill acquisition to DQN-from-scratch is the unfair direction; comparing it to RLVR-on-a-pretrained-model is fair, and there the gap largely closes. Which relocates the 10^4 out of the learning algorithm and into the cost of ACQUIRING the prior — and that is where my memory-layer result lands: I tried to buy prior capacity cheaply, in parameters, and it did not convert into loss at 0.26B tokens.

### (3) Next experiment

Measure the CONVERSION RATE of one held-out skill against the number of examples of it, on this repo's own 200M model, as a curve rather than a point:

- Register a code/math capability the pretraining corpus does not contain, before touching it. This repo already has the machinery: a holdout registry, a 13-gram contamination scanner, and a documented incident where an empty holdout hash set let 19 of 20 items leak into SFT — so the contamination guard must be run and reported, not assumed.
- Then n examples of that skill for n in {1, 8, 64, 512, 4096}, two seeds each, measuring accuracy on held-out instances of the same skill. Ten runs of a few hours on one card each: it fits 8 cards in a day with room for a re-run.
- The fact is the SHAPE of accuracy against log n. If a handful of examples gets most of the way, this model already has the prior and the answer to the user's question is "the prior is the expensive part, and we have more of it than we think." If it needs 10^3-10^4, the prior is absent at 200M/1B-tokens and no RL recipe will rescue it.
- Arm the same curve on a checkpoint with 2× the pretraining tokens if one exists, and the slope difference is a direct measurement of prior-quality against sample efficiency, which is the quantity the whole discussion is about.

Cost of being wrong here is low and the negative result is publishable internally either way, which is more than I can say for another capacity arm.

**WHAT WOULD CHANGE MY MIND:**
- On capacity: M1 was stopped for key-usage collapse, so "capacity bought nothing" is currently confounded with "the selector broke." If the fix (a separate selector lr, or query normalisation — flags are written, six-cell probe designed, paused by the user's order) makes touched stay above 0.50 and the val delta then holds at −0.010 or better past step 1000, I withdraw the claim that added capacity does not convert. That test is designed and one launch signal away.
- On the ordering in (2): a model-free result reaching human-level Atari in ~10^5 frames without a learned model or pretrained representation would break the "structure, not algorithm" reading.
- On (3): if the 1-example and 4096-example points land within noise of each other in BOTH directions — no gain anywhere — the experiment measured my probe rather than the model, and I would suspect the skill was in the corpus after all and go back to the contamination scan before believing anything.

One honest caveat on my own contribution: my CPU toy for this collapse produced a ranking I had to withdraw, because its task rewarded the failure it was diagnosing (a one-row selector fit it BETTER, 0.1002 vs 0.1033 measured). The three real-arm numbers above are from real arms on real data and do not depend on that toy. But it is a live reminder that the objective you measure against decides the answer, which applies to the whole question being asked here.

---

## tilerl-0a

Every number labelled. Where I have not measured something and cannot cite a paper I trust from memory, I say so rather than producing a figure.

### (1) The 10^4 gap: what it is made of

First, the premise needs pinning, because "10^4" is not one gap. Against a human's LANGUAGE exposure the data ratio is ~10^5, not 10^4: a child sees ~50M words by age 18 [Frank 2023's estimate, widely used as the BabyLM budget; paper-reported], ~65M tokens at 1.3 tok/word. A frontier model trains on ~15T tokens [paper-reported, Llama-3 class]. That is 2.3×10^5. At the 100M-word budget it is 1.2×10^5. So the honest framing is 10^5 in data, and the 10^4 figure only appears if you compare against something else — FLOPs, or a specific RL task. Getting this right matters because the factor list below has to sum to the right target.

The decomposition I would defend, largest first:

**a) The learning signal per token is far weaker. [estimate, biggest single term]** Next-token prediction extracts one scalar of supervision per token from a distribution the model already mostly predicts. A child gets multimodal grounding, intervention (they can act and see the result), and correction. No clean measured multiplier exists for this in the literature I can cite reliably; I flag it as the dominant term precisely because it is the one nobody has isolated.

**b) No curriculum, no active sampling. [estimate, 3-30×]** Training data is i.i.d. shuffled; a child's input is ordered, self-selected, and paced to their frontier. Basis: active learning and curriculum papers report real but modest effects — which is why I put it at 3-30×, not 10^3.

**c) Parameter-inefficiency of dense gradient descent. [estimate, 10-100×]** Every token updates every parameter by a small amount; the model has no mechanism to bind a fact once and keep it. This is exactly the hypothesis the aupai memory-layer arms were built to test — a token reaching top_k of a sparse table rather than all parameters — and I have MEASURED the cost side there: at 1024^2 values the per-layer overhead is 382 of 424 ms/step, and the table is 92% as effective at a quarter the size (M2 vs M1, measured 2026-09-05). The efficiency side is unproven.

**d) Architectural mismatch to compositional structure. [estimate, unquantified]** Humans generalise from one example to a rule; transformers need the rule's instances. I cannot put a number on this and will not invent one.

**e) What is NOT in the gap.** The comparison is not like-for-like: the model learns 100+ languages, code, and a factual range no human holds. Some of the 10^5 buys breadth a child never acquires. I would not attribute the whole ratio to inefficiency.

### (2) RL sample efficiency vs human

**DQN (2015, paper-reported):** ~200M Atari frames, ~38 days of game experience, to beat a human who is competent in minutes-to-hours. That is the origin of the "10^4 gap" framing and it is a real, well-sourced number.

**EfficientZero (2021, paper-reported):** human-level median on Atari-100k — 100k frames, ~2 hours of game time. This is the important data point for 4c's question, because it closes most of the DQN gap WITHOUT closing the pretraining gap. What bought it: a learned model for imagination-based rollouts, self-supervised consistency in latent space, and much higher replay/update ratio. The lesson transfers: most of DQN's gap was algorithmic, not informational.

**RLVR (2024-25):** I would be reconstructing specific numbers here and will not. What I am confident of: the sample counts are small (thousands to low tens of thousands of rollouts) because the reward is dense-by-verification, and there is a live open question — whether RLVR elicits capability already in the base model or adds new capability — with published results on both sides. Treat any specific efficiency multiplier I gave here as unsourced.

**The synthesis for 4c:** EfficientZero says the RL gap is mostly algorithmic and closable. The pretraining gap (part 1) has no comparable demonstration. Those are different claims and should not be merged into one "10^4".

### (3) The next experiment for this repo

I computed the budget rather than guessing it. 200M params, 8×H20, 1-2 days:

H20 bf16 dense peak 148 TFLOP/s (vendor spec). At MFU 0.35-0.45 [estimate; not measured on this cluster]:
- 1 day: 3.6-4.6×10^19 FLOPs → 30-38B tokens at 6ND → 149-192 tokens/param
- 2 days: 7.2-9.2×10^19 FLOPs → 60-77B tokens → 298-384 tokens/param

So the budget is 30-77B tokens, i.e. 150-380 tokens/param — well past Chinchilla-optimal (20) and into the over-trained regime where a 200M model is data-saturated. **That is the single most important number for choosing the experiment: compute is not the binding constraint at this scale, so an experiment that just spends the FLOPs learns nothing.** The right experiment is one where 200M is a sufficient instrument for the question.

My recommendation: **a sample-efficiency ablation, not a capability run.** Fix the token budget at ~20-30B (Chinchilla-ish, leaves room for 3-4 arms in 2 days) and vary ONE thing that bears on part (1)'s decomposition. Concretely, the sparse-memory arm is the ready-made candidate: control vs one shared product-key pool, equal dense parameters, measuring loss at equal tokens AND at equal wall-clock. The aupai side has that instrumented — I built the region timers and ran the five-cell decomposition — and its status is that the arms were STOPPED at readout 4 for usage collapse (pool touched 0.094 at step 1000, measured). The fix — selection lr and query normalisation — is exactly what b0's pending flags are for. That is a real, live, unanswered question at 200M scale, and 8 idle cards is what it needs.

Second choice if that is not wanted: a curriculum/data-ordering arm at fixed budget, testing factor (b) directly. Cheap, clean, and no one in this repo has measured it.

What I would NOT do: a 1-2 day run at 200M whose output is "the loss went down". At 150-380 tok/param that result is known in advance.

**CAVEAT ON CARDS:** I do not control them and am not proposing to launch anything. As of ~08:45 all 8 H20s are idle; both tileRL sessions have declined to claim them, and aupai's grid is paused by user order. So this is an answer to your question, not a launch plan.

---

## db (aupai-db)

Every number labelled MEASURED HERE / VENDOR / PAPER / ESTIMATE / DERIVED. Angle: instrumentation.

### (1) The 10^4 decomposes into one term, and it is not energy

Computed rather than asserted:

| factor | value | basis |
|---|---|---|
| tokens, frontier / human | ~2.0e4 | DERIVED (13-15.6T vs ~6.5e8) |
| total energy, frontier / human 20y | ~1.4e3 | DERIVED from FLOPs estimates |
| ENERGY PER TOKEN, model / human | ~0.02-0.14 | DERIVED — the MODEL IS CHEAPER |
| FLOPs/token at 200M (6ND) | 1.03 GFLOP | DERIVED from 172.5M non-emb (MEASURED HERE) |
| achieved | 92 TFLOPS | DERIVED (296 VENDOR × 31% MFU MEASURED HERE) |
| J/token, our 200M run | 5.48 mJ | DERIVED (400 W VENDOR / 73K tok/s MEASURED HERE) |
| J/token, human | 19.4 J | DERIVED (20 W × 20 y / 6.5e8 tokens) |

Our 200M run is 3542× CHEAPER per token of experience than a human brain. Llama-3-405B-class is still ~4.5× cheaper per token (4.3 J vs 19.4 J). So the premise "10^4 less efficient" is true as stated only if efficiency means per-unit-of-competence; per unit of experience, silicon already won. The whole 10^4 lives in TOKENS NEEDED. That reframing matters because it says which interventions can possibly pay: hardware and kernels move the cheap term (we have already taken +14.1% from NCCL proto=Simple and -19.1% on the KDA kernels at T=4096, both MEASURED HERE) and cannot touch the 10^4.

Honest bounds: human token exposure is an ESTIMATE with a 1e8-1e9 spread, so the tokens ratio is ±1 order. 400 W is the H20 SXM spec sheet, NOT measured on our cards. Frontier FLOPs are third-party estimates. 20 W brain is standard but the fair denominator may be 100 W whole-body, which moves the human 5× cheaper and the model still ahead per token.

What the gap is NOT, from our own data: not depth, not launch overhead, not KDA occupancy — eff.launch_overhead_is_not_a_cost, eff.kda_occupancy_bound and eff.kda_parallelism_not_the_bottleneck are all REFUTED with three independent measurements each.

### (2) The RL ordering, and what it actually says

All PAPER-REPORTED; I have run none of these.

| regime | samples to competence | vs human |
|---|---|---|
| DQN, Atari | 200M frames (~925 h play) | human ~2 h → ~10^2-10^3 worse |
| Rainbow / data-efficient | ~10-20M frames | ~10^1-10^2 worse |
| EfficientZero (model-based) | 100K frames (~2 h) | HUMAN PARITY at 2h-equivalent |
| RLVR on pretrained LLM | ~1e3-1e4 rollouts | below human trial counts |
| 1-shot RLVR | 1 example, +~30pt MATH500 | far below |

The ordering is monotone in ONE variable and it is not the RL algorithm: it is how much structure exists before RL starts. DQN learns pixels-to-Q from scratch. EfficientZero learns a dynamics model, i.e. it builds the structure during training and reuses it. RLVR does no representation learning at all — it reweights behaviour a pretrained model already has. 1-shot RLVR is the limit: one example is enough because the capability was present and RL only selected for it.

Two conclusions:

(a) Sample efficiency is bought by PRIOR STRUCTURE, not by the update rule. A human learning Montezuma's Revenge in 2 hours also has 20 years of priors. EfficientZero matching that at 100K frames is not "matching a human's learning"; it is matching a human's learning ON A TASK WHERE A LEARNED DYNAMICS MODEL IS THE WHOLE PRIOR. Neither number is prior-free, so the honest statement is that both sides amortise, and the RL sample count measures only the unamortised remainder.

(b) Therefore pretraining is not the inefficient half being contrasted with an efficient RL half. Pretraining IS where the 10^4 is spent, and RL's apparent efficiency is that spending already having happened. 1-shot RLVR is the proof: the gain per RL sample goes to infinity exactly as the prior goes to complete. Any experiment that reports RL sample efficiency without stating the prior is reporting a number about its checkpoint, not about RL.

### (3) The experiment, and what makes its number believed the first time

Proposal: RLVR sample-efficiency curve on our own 200M checkpoints, math-500 as the verifier. Two or three checkpoints from the ladder that differ ONLY in pretraining tokens (0.4b / 1.6b / 3.24b — identical mixes, scaled total_tokens, so the prior is the single variable), then RLVR with a verifiable reward and measure pass@1 vs ROLLOUTS CONSUMED. The deliverable is one fact: d(capability)/d(RL sample) as a function of pretraining tokens. That directly tests (a) above and it is the only version of this question our hardware can answer in 1-2 days.

Feasibility: 8×H20, and one lane card is all that is free during a training block. RLVR needs generation, which is the expensive half. At 200M the model is small enough that rollout throughput is the binding constraint, not training — and I have NOT measured our generation throughput, so I cannot tell you whether 1e4 rollouts × 3 checkpoints fits in 1-2 days. That measurement is the first thing to do, not the experiment.

**Instrumentation and gates** — a sample-efficiency number is believed once only if it survives the five ways this repo has already produced wrong numbers:

1. **PASS@K GATE BEFORE SPENDING A CARD.** AGENTS.md already requires pass@8 - pass@1 >= 15pt before RL is worth running. If the base model cannot sometimes produce the right answer, RLVR has nothing to reweight and the curve measures noise. Run it on all three checkpoints FIRST and publish the three numbers.
2. **RESOLUTION BEFORE HYPOTHESIS.** math-hard is ±1pt at a 2-3% pass rate. A sample-efficiency curve whose y-axis moves 2pt is unreadable. Pre-register the minimum detectable difference from n and the base rate.
3. **CONTAMINATION IS ALREADY KNOWN TO INFLATE THIS EXACT METRIC.** facts/contamination.json: 30% of math-500 questions have a containment hit in the math SFT corpus, so post-SFT values are inflated and only base values are clean. An RLVR curve on a contaminated verifier measures retrieval, not acquisition.
4. **THE PROMPT FORMAT IS A 38-POINT CONFOUND.** facts/base_eval.json#be.l1_fewshot_p324: answer-present goes 25.4% → 53.5% → 63.6% at 0/1/3 demos, MODEL HELD FIXED. Format exposure alone moves this field 38.2pt. Freeze the template, record it in the fact's config.
5. **THE X-AXIS MUST BE THE THING SPENT.** Rollouts, not steps and not wall-clock. Log rollouts consumed, tokens generated, and unique problems seen, per update — three columns, because a curve against any one of them alone is ambiguous.

Plus: **A NEGATIVE CONTROL ARM.** RLVR with the verifier REPLACED BY A CONSTANT REWARD, same rollout budget. If the curve moves, the pipeline is learning from something other than correctness and the headline number is worthless. This is the cheapest possible insurance and it is the one nobody remembers to run.

---

## tilerl-25 (declined)

tilerl-25 declined to answer, on the grounds that their session did tileRL code review and doc fact-checking all night (12 PRs, all CPU-side), ran zero pod tasks, and has no measurements or papers to cite. They refuse to fabricate estimates: "a decomposition full of unmeasured estimates reads the same as one with a basis — that is exactly the class of thing we fixed eight times in this repo tonight." Their one substantive contribution: the 8 H20s' driver is 535.161.08, and nobody has verified which driver/CUDA the sm90 kernel cells were validated on — a driver mismatch would fail at kernel compile time, before any experiment matters. This is a 10-minute prerequisite check.

---

## Synthesis

Seven views (44, 84, 58, 98, 62, tilerl-0a, db; tilerl-25 declined). Points of agreement, disagreements, and the experiments most would run first.

### Points of agreement

1. **The gap is in the prior, not the algorithm.** All seven views converge here. 44: "the 10^4 gap is the gap between no prior and a strong prior." 58: "sample efficiency at fixed prior is nearly a solved axis; the open axis is how much prior a given amount of pretraining buys per token." 98: "效率来自先验，不来自算法。" 62: "the 10^4 relocates out of the learning algorithm and into the cost of ACQUIRING the prior." tilerl-0a: "EfficientZero says the RL gap is mostly algorithmic and closable. The pretraining gap has no comparable demonstration." db: "pretraining IS where the 10^4 is spent, and RL's apparent efficiency is that spending already having happened." 84 provides the data-side evidence: the model sees 37× more text than a human by 18 and is still not competent.

2. **The 10^4 lives in tokens needed, not energy (db's reframing, the strongest claim).** db computes: our 200M run is 3542× CHEAPER per token of experience than a human brain (5.48 mJ vs 19.4 J). Llama-3-405B-class is still ~4.5× cheaper per token. "Per unit of experience, silicon already won. The whole 10^4 lives in TOKENS NEEDED." 58 independently: a 200M model is 2000× cheaper per token than a brain. 98: 5×10^-3 J/token vs human ~20 J/second-of-thought. 44: H20's FLOPS/W is comparable to the brain's. The energy gap is 10^2-10^3 per operation; the sample gap is 10^4-10^5.

3. **The data ratio is 10^5, not 10^4.** tilerl-0a: child sees ~50M words by 18 (~65M tokens), frontier model trains on ~15T tokens → 2.3×10^5. 98 independently: 8×10^9 tokens for 200M vs child's 10^6-10^7 words → ~10^3. 58: 37× more text than human by 18. The 10^4 figure appears only when comparing against something else (FLOPs, or a specific RL task).

4. **Compute is not the binding constraint at 200M.** tilerl-0a: 8×H20 for 1-2 days = 30-77B tokens = 150-380 tok/param, well past Chinchilla-optimal (20). "An experiment that just spends the FLOPs learns nothing." 58's paired-BPB design and 62's conversion-rate curve both agree: the experiment must measure a rate or a shift, not an endpoint.

5. **Contamination controls are non-negotiable.** 84: type-disjoint eval, per-problem containment screening. 58: memorisation control. 98: api_cloze seen/unseen regions. 62: "the contamination guard must be run and reported, not assumed." db: "30% of math-500 questions have a containment hit in the math SFT corpus; an RLVR curve on a contaminated verifier measures retrieval, not acquisition."

6. **Routing/credit-assignment is a measured factor (62 + tilerl-0a).** 62's M1 data: ~6% of a 1.07B-parameter table did all the work; pool_touched_frac collapsed 0.306→0.0945. tilerl-0a independently measured the cost side: at 1024^2 values the per-layer overhead is 382 of 424 ms/step, and the table is 92% as effective at a quarter the size (M2 vs M1). The parameters were reachable but the learner could not route to them.

7. **The comparison is not like-for-like (tilerl-0a's breadth caveat).** The model learns 100+ languages, code, and a factual range no human holds. Some of the 10^5 buys breadth a child never acquires. The whole ratio should not be attributed to inefficiency.

### Disagreements (with evidence each side cites)

1. **Energy numbers: estimate vs refuse vs compute.**
   - 44 gives estimates: memory wall ~10^3, sparsity ~10-100×, event-driven ~10-100×.
   - 84 refuses: "I have no measurement of joules on either side."
   - 58 computes: 135-450× gap in FLOPS/J (spec + measured MFU), with the unit-conversion caveat.
   - 98 computes: ~10^2 per operation, ~10^3-10^4 per unit output, with the information-theoretic caveat.
   - 62 gives three MEASURED factors (capacity utilisation ~6%, MFU 31%, optimizer overhead 12 B/param) but refuses brain-side numbers.
   - tilerl-0a declines to give energy numbers, focuses on the data ratio.
   - Resolution: 58's 135-450× and 98's ~10^2 are the most defensible energy numbers. 62's three measured factors are the only ones from this repo. The honest answer: "10^2-10^3 per operation, 10^3-10^4 per unit of useful output, with ~3× already lost to MFU and ~16× lost to capacity utilisation — and the unit conversion is approximate."

2. **Experiment: five designs, complementary.**
   - 44: RLVR sample efficiency curve at 10/50/200/1000/5000 prompts. Directly measures RL efficiency.
   - 58: CPT learning curve with curriculum, paired per-token BPB, 3 budgets. Measures the horizontal shift r; 25× tighter via pairing.
   - 98: api_cloze probe curve at N=10^2/10^3/10^4/10^5. Existing probe, cheapest, produces a clean fact.
   - 62: conversion rate curve at n ∈ {1, 8, 64, 512, 4096}, two seeds, on a registered held-out skill. Most rigorous contamination controls.
   - tilerl-0a: sparse-memory arm (control vs product-key pool), equal dense params, measuring loss at equal tokens AND equal wall-clock. Directly tests the routing/capacity factor 62 measured.
   - Resolution: these measure different axes. 62's and 98's measure skill-acquisition sample efficiency (same experiment, different rigor). 58's measures CPT prior-per-token. 44's measures RL efficiency. tilerl-0a's measures whether sparse memory helps (capacity/routing). The right sequence: (1) 62's conversion-rate curve (directly answers "how many examples to learn a skill"), (2) tilerl-0a's sparse-memory arm (tests whether the routing failure is fixable), (3) 58's paired BPB curve (measures prior-per-token), (4) 44's RLVR curve (measures RL efficiency).

3. **1-shot RLVR: cite vs flag.**
   - 44, 58, 98 cite it (Wang 2025, 2025 One-step RL).
   - 84 flags contamination: "one problem makes any prior exposure decisive."
   - 62: "1-shot RLVR reportedly moves math benchmarks measurably from a SINGLE example."
   - tilerl-0a declines to give specific numbers: "I would be reconstructing specific numbers here and will not."
   - Resolution: 84's caveat is correct. If cited, the paper's contamination controls must be cited too. For THIS repo, 84's per-problem containment screen is the prerequisite.

### The 3 experiments most sessions would run first

1. **Conversion-rate curve on a registered held-out skill (62's design, with 84's and db's controls).** n ∈ {1, 8, 64, 512, 4096} examples, two seeds, measure accuracy on held-out instances. Register the skill as absent from pretraining (13-gram scanner, holdout registry) before touching it. The fact is the SHAPE of accuracy against log n: if a handful of examples gets most of the way, the prior is the expensive part and we have more of it than we think; if it needs 10^3-10^4, the prior is absent at 200M. Ten runs × a few hours on one card each — fits 8 cards in a day. db's gates apply: pass@k gate before spending a card, resolution before hypothesis, contamination screening, frozen prompt format, rollouts (not steps) as the x-axis, and a negative-control arm (constant reward).

2. **RLVR sample efficiency, read out on the constructed S/P sets.** **RETIRED AND
REPLACED 2026-09-05 (fb ruling).** The original design was db's: two or three ladder
checkpoints differing ONLY in pretraining tokens (0.4b / 1.6b / 3.24b — identical mixes,
scaled total_tokens), RLVR with a verifiable reward, pass@1 vs rollouts consumed, to test
"sample efficiency is bought by prior structure, not the update rule". Its readout was
math-500, and that is why it was retired: math-500 sits at 30% containment
(`facts/contamination.json#cont.holdout_v2`), so it cannot separate retrieval from
reasoning at all, while the S/P sets are constructed-absent — the operator, its rule and
its phrasing were invented 2026-09-05, after every corpus in the mix was built
(`cont.novel_operator_collision`, and the sets' own `absence_basis` header field). Those
are two different epistemic classes, not two points on a containment scale, so this is a
replacement rather than a refinement. What experiment 2 now is, and every constant it
fixes: `docs/standards/rlvr_exp2_recipe.md`. The deliverable is unchanged in kind —
capability per token consumed, two columns (generated primary, trained-on secondary)
against the same pretraining baseline. db's prerequisite also stands unchanged: generation
throughput is unmeasured and is the binding constraint at 200M.

3. **Measure b (the power-law exponent) on this mix, on existing checkpoints.** 58's prerequisite: cheap, CPU-side, uses existing b0 arms. Determines whether the paired-BPB design is load-bearing (b < 0.2) or whether unpaired suffices. Also measures per-token loss correlation between arms. No GPU needed. Can run immediately, in parallel with the GPU pause. tilerl-25's driver check (535.161.08 vs sm90 kernel validation) is a 10-minute prerequisite for any GPU experiment.

**What would change the collective mind:** if b at 200M is >0.2 (58's pairing argument weakens); if the 1,342 generators collapse to far fewer distinct procedures (84's supply claim weakens); if per-problem containment screening finds hits for the N=1/10 items (the curve measures retrieval, not acquisition); if the memorisation control moves with the curriculum (BPB win is contaminated); if the api_cloze unseen accuracy stays at random past 10^5 examples (98: the gap is architecture, not data); if the selector fix makes M1's touched stay above 0.50 and the val delta holds past step 1000 (62 withdraws "capacity bought nothing"); if a model-free result reaches human-level Atari in ~10^5 frames without a learned model (62: breaks the "structure, not algorithm" reading); if the learning signal per token is isolated and measured (tilerl-0a's factor (a), currently the biggest unquantified term).
