---
question: 什么训练方案被实测能改善轨迹校准(teacher-forced 0.727/0.6875 vs free-running LCS ~0.23,误差即时非渐衰)——按"能否进今天的 pretraining 启动"和"是否测过轨迹级统计"两条线分级
status: recorded
source: fb P0 tasking 2026-09-01(砍单后唯一保留问题);all numbers fetched 2026-09-01, URLs inline; arXiv IDs title-verified; 3 wrong IDs caught and flagged
---

# 轨迹校准训练方案调研(2026-09-01)

**三条头条:**

1. **阻塞线:只有 scheduled sampling 和 unlikelihood 能进今天启动的 pretraining。** 其余一切(DAgger、自蒸馏、rationale SFT、RL)按构造就是 post-pretraining——需要一个已训练模型才能开跑。scheduled sampling 是其中唯一有正面证据的;unlikelihood 治的是退化/重复,不是一致性,只算机制相邻。
2. **0/14:没有一篇干预论文在干预前后测过 free-running 轨迹一致性(LCS/编辑距离/teacher-forced-vs-free-running gap)。** 本调研里每个"proven"标签都建立在下游基准分上。我们的 0.727→0.25 仪器是未发表领地——想要轨迹级证据只能自己造。
3. **深度对轨迹校准的已发表测量:不存在,两个方向都没有。** MobileLLM 及其所有后续只报 perplexity/基准。深度赌注是 unsupported,不是 contradicted。
4. **"误差即时非渐衰"有外部佐证:** He et al. 2019(arXiv 1905.10617)用前缀交换把 gold 前缀喂回去,发现失真 "limited... not... incremental during the generation"——与我们的 bin-1 观察一致、与渐衰累积故事矛盾,是完全不同仪器上的跨六年印证;但他们测的是开放式文本、幅度远小于我们的 0.727→0.23,code/math 轨迹可能比任何已发表对象都严格( caveat 与结论同句,不做脚注)。

## 1. 阻塞分类(启动决策)——先读这张表

| 方案 | 进 pretraining? | 原因 |
|---|---|---|
| Scheduled sampling | **是** | 训练循环 target-mixing 改动;BSS 变体是离线数据交错,连循环都不用改。LM 复现存在(TMLR 2024, arXiv 2410.14655)。 |
| Unlikelihood training | **是** | 辅助损失;token 级直接加,sequence 级需要 rollout 但可在循环内。 |
| 序列级目标(Edunov MRT/margin) | **原则上是,每步昂贵** | "requires generating and scoring multiple candidate output sequences for each input sequence during training"(arXiv:1711.04956)。pretraining 规模下是大成本乘数;只有 NMT 时代证据。 |
| DAgger | **否,按构造 post-pretraining** | 需要已训练策略 rollout + oracle 重标注访问过的状态;是绕训练模型的迭代外环。 |
| 自蒸馏(Born-Again, SDFT) | **否** | BAN 需要收敛的教师;SDFT 是微调方法。 |
| Rationale-distillation SFT(Orca 等) | **否** | 按定义是大学教师 trace 上的 SFT。 |
| RL 推理(R1-Zero 式) | **否** | 在 pretrained base 上做 RL;R1-Zero "on the DeepSeek-V3 base"。 |

## 2. 判别交付:轨迹级证据 vs 基准分

**14 篇候选论文里 0 篇测过干预前后的 free-running 轨迹一致性。** 所有干预论文报的是下游基准(BLEU/F1/PPL/AIME/GSM8K)或退化指标(重复/沉闷)。测过 gap 的只有测量类论文(无干预):He et al. 2019(1905.10617)、Xu et al. 2019(1910.11235)、Chiang & Chen 2021(2109.08705),外加 MiniLLM(2306.08543)摘要声称 "lower exposure bias" 但摘要页未给度量。**移动了基准但没测轨迹质量的方法不回答我们的问题——它们全都没测。**

## 3. Scheduled sampling(Bengio et al. 2015, 1506.03099,已核验)

机制:teacher forcing 训练时喂 gold 前一个 token、推理时喂模型自己的,失配导致 "errors that can accumulate quickly along the generated sequence"。Scheduled sampling 以概率 ε 把解码器输入换成模型自己生成的 token,ε 从 0 向 1 爬坡。

证据(全是 RNN/LSTM 时代,2015):
- 图像描述(MSCOCO,1 层 512 LSTM):BLEU-4 28.8→30.6,CIDEr 89.5→92.1;集成 30.7→32.3;MSCOCO 2015 榜首。
- 成分句法(WSJ):F1 86.54→88.08,与 dropout 可加(87.0→88.68)。
- 语音(TIMIT,2×250 LSTM):帧错误率 46.0→34.5。
- "Always Sampling"(ε=1 起步)表现差(BLEU-4 11.2)——课程是承重的。

自回归 LM 复现:有,少且新,全是基准分:
- **Batch-Scheduled Sampling**,arXiv 2410.14655(TMLR):离线把模型生成 token 交错进上下文窗口;摘要/QA/数学 QA,"overall improvement"。
- **RAD**,arXiv 2309.02823:GPT-2 对话(Persona-Chat, DailyDialog),"outperforms the baselines on most automatic and manual metrics"。
- 相邻:Professor Forcing(1610.09038,NIPS 2016)——对抗对齐 teacher-forced 与 free-running 动态,是 SS 的显式替代,动机正是 SS 的有偏梯度批评。

**分级:RNN 时代 3 任务一致收益(proven-small);LM 规模证据薄(2 篇,只基准)。无论文测过 teacher-forced/free-running gap 本身。**

## 4. DAgger(Ross, Gordon & Bagnell 2011, 1011.0686,已核验)

机制:迭代地 rollout 当前策略、让专家重标注访问状态、聚合进训练集——no-regret 在线学习归约,保证在 "the distribution of observations it induces" 下表现好。直接打我们的协变量偏移。

**在自回归语言模型上的应用:基本没有——已确认。** arXiv 搜 DAgger + "language model" 15 条命中,全是:(a) 用 DAgger 训练带 LLM 骨干的 VLA/机器人/导航策略;(b) LangProp(2401.10314)——代码优化外环,最接近但不是 LM 训练;(c) 负面结果(游戏 agent 上 "unhelpful, too slow, or too heavy";LLM 世界模型综合失败)。**没有任何已发表工作把 DAgger 用于文本生成 LM 训练。分级:对 LM 是 folklore(在模仿学习/机器人里 proven)。**

## 5. 序列级目标(Edunov et al., NAACL 2018, 1711.04956,已核验)

机制:不在 token 级做 MLE,而是 "minimize the negative log likelihood of an entire sequence rather than individual tokens",需要 "generating and scoring multiple candidate output sequences for each input sequence during training"。研究的损失:SeqNLL、Expected Risk(Risk)、Max-Margin、Multi-Margin、Softmax-Margin(没有叫 "max-loss" 的)。

证据(仅 NMT 时代):IWSLT'14 De-En 和 Gigaword 摘要——"new state of the art results on both";WMT'14 En-Fr——"41.5 BLEU which is on par with the state of the art"。卷积 seq2seq,摘要页无参数规模。**调研未发现 LM 规模后续;直系后裔是 RL-for-LM(R1 的 GRPO 是策略梯度/期望奖励)。分级:NMT-era-only;每步 k 候选成本是阻塞点。**

## 6. 自蒸馏

**Born-Again Networks(Furlanello et al., ICML 2018, 1805.04770,已核验):** 同架构同规模学生做 KD,"surprisingly outperform their teachers on both computer vision and language modeling tasks"。LM 证据:PTB,LSTM 52M 测试 PPL 71.87→68.56(BAN+L),CNN-LSTM 19M 80.05→76.97;只在教师输出+标签损失组合下有效。**RNN 时代,只 perplexity,无轨迹测量。按构造 post-pretraining。**

**SDFT(2402.13669,ACL 2024,已核验):** 标题实为 "Self-Distillation Bridges **Distribution Gap** in Language Model Fine-Tuning"——弥合的是微调中任务数据与模型输出的分布差,不是 teacher-forced/free-running。Llama-2-chat 7B/13B/70B;缓解灾难性遗忘,基准可比或更优。**≥7B,基准分,post-pretraining。**

**<1B 现代自蒸馏证据:未发现。**

**Rationale-distillation SFT 是另一个东西。** Orca(2306.02707,已核验)、Orca 2(2311.11045,已核验)、Shridhar et al.(2212.00193,已核验:GPT-2 large 超 GPT-3 6B,GSM8K/StrategyQA/SVAMP 提升 >70%)、Magister et al.(2212.08410:T5-XXL 学 PaLM-540B rationale,GSM8K 8.11%→21.99%)——蒸馏的是更大教师的推理,全是 post-pretraining,全是基准分。

**"SmallLLM, arXiv 2407.10932" 是错的 ID**——2407.10932 是 Brunn-Minkowski 数学论文(两次抓取确认);arXiv 全文搜 "SmallLLM" 零结果。可能指 Shridhar 或 Zhu et al.(2401.11864),无法确认。

**MiniLLM(2306.08543,已核验)** 是唯一声称轨迹属性的蒸馏论文:reverse-KL + on-policy,120M–13B,"lower exposure bias, better calibration"——但摘要页未给暴露偏差的度量方法,声称 plausible 但度量级未核验。post-pretraining,教师蒸馏非自蒸馏。

## 7. Unlikelihood training(Welleck et al. 2019, 1908.04319,已核验)

机制:MLE "leads to dull and repetitive outputs",病根在 "the likelihood objective itself";unlikelihood "forces unlikely generations to be assigned lower probability by the model",token 级(罚重复/负候选 token)和 sequence 级(罚退化续写)。

证据:"both token and sequence level unlikelihood training give less repetitive, less dull text while maintaining perplexity",人评超 nucleus sampling/beam blocking。

**治不治 teacher-forced/free-running gap?不治——治退化(重复/沉闷),度量是重复统计、PPL、人评。** Sequence 级轻度 on-policy(给模型自己的续写打分),机制相邻,但无论文测 own-prefix 上的 token 级一致性。分级:退化上 proven-at-scale;一致性 gap 上未证。

## 8. RL 推理(小规模)

**TinyZero 不是 arXiv 论文**——GitHub 复现项目(Jiayi-Pan/TinyZero),Qwen2.5-3B 上复现 R1-Zero(countdown + 乘法),"aha moment"= 自验证涌现,<$30 可复现;**关键:"Qwen2.5-0.5B reportedly fail[s] to learn reasoning" 而 3B 成功**。证据是 W&B log + X 帖,非同行评审。**sub-1B 结果是负面的:0.5B 失败。**

**SimpleRL-Zoo(2503.18892,COLM 2025,已核验):** 10 个 base 的 zero RL,含 Qwen2.5 全系列 0.5B–32B。"substantial improvements in both reasoning accuracy and response length across most settings";首次在非 Qwen 系小模型观察到 aha moment。**最强的已发表 sub-1B RL 证据(0.5B/1.5B 在扫列里)**——但产出是基准准确率+回复长度,不是轨迹一致性。

**MobileLLM-R1(2509.24945,ICLR 2026):** 950M,AIME 15.5(OLMo-2-1.48B 0.6,SmolLM-2-1.7B 0.3)。只基准。

**规模对照——DeepSeek-R1(2501.12948,已核验;brief 里的 2501.12968 是错的,那是黑洞物理论文):** V3 671B 总参/37B 激活;GRPO;R1-Zero 纯 RL 无 SFT,10,400 步 = 1.6 epoch。蒸馏小到 1.5B,但头条 RL 跑在 671B。

**分级:R1 proven-at-scale;基准收益 proven-small(SimpleRL-Zoo 0.5B–3B,MobileLLM-R1);TinyZero 社区级。全 post-pretraining,无一测轨迹一致性。**

## 9. 暴露偏差测量文献(LM 规模)

我们的精确度量(gold vs own prefix 上的逐 token top-1,位置-bin 曲线)**在 <1B 规模没有已发表的直接对应物**。存在的:

- **He et al., "Exposure Bias versus Self-Recovery"(1905.10617,EMNLP 2021)——最接近的已发表对应物。** 前缀交换:把 gold 前缀喂回去让模型续写,"the performance should become much better because the training-generation discrepancy in the prefix is removed"。两个直接相关的发现:(a) "the distortion induced by the prefix discrepancy is limited, and does not seem to be incremental during the generation"——**与我们"误差即时、非渐衰"的 bin-1 观察一致,与经典渐衰累积故事矛盾**;(b) LM 有 "self-recovery ability"。警告:他们测的是开放式生成质量/多样性/一致性,不是逐 token top-1,且幅度 "limited"——我们 0.727→0.23 的崩塌远大于他们报告的任何东西,可能因为 code/math 轨迹比开放式文本严格。
- **Xu et al., "Rethinking Exposure Bias In Language Modeling"(1910.11235):** 提出 "road exam" 度量 + RL/GAN 干预(multi-range reinforcing, multi-entropy sampling),"improvement over competing models with regards to BLEU scores and road exam"。摘要无数字。
- **Chiang & Chen, "Relating Neural Text Degeneration to Exposure Bias"(2109.08705):** GPT-2;隐状态检查定位退化前的错误;"text degeneration is likely to be partly caused by exposure bias",自增强机制解释放大。
- **Schmidt, "Generalization in Generation"(1910.00292):** 定义暴露偏差但主张泛化才是根本。
- **MiniLLM(2306.08543):** 唯一声称该度量的干预论文(见 §6)。

**报告了缩小 gap 且带测量的干预:只有 Xu et al.(road exam,摘要级)和 MiniLLM(声称,度量未说明)。测量文献小、早于 2021、RNN/GPT-2 时代,在 <1B 推理 LM 规模是空的。**

## 10. 深度问题(controller 追加):深度改不改 free-running 一致性?

**没有已发表测量存在——已确认。** MobileLLM(2402.14905,已核验)确立了 sub-billion 的 deep-and-thin(125M/350M 超前 SOTA 2.7%/4.3%),但只评 perplexity 和基准。后续(MobileLLM-Pro 2511.06719、MobileLLM-Flash 2603.15954、MobileLLM-R1 2509.24945)同样只报基准。定向搜 "depth" + "exposure bias" 7 篇命中,无一关于架构深度对轨迹校准(命中的 depth 是扩散深度图/句法树深度/循环迭代数)。**深度-宽度文献从未测过 free-running 一致性或轨迹校准——只测 perplexity 和基准。我们的 0.727→0.25 仪器若跨深度应用,会是首个此类测量。**

## 11. 底线表

| 方案 | 机制 | 证据强度 | 尝试成本 | 对 gold-prefix→own-prefix gap 买什么 |
|---|---|---|---|---|
| Scheduled sampling | 课程式混入模型自己的前一个 token | **RNN 时代 proven(3 任务);LM 复现薄(2 篇,只基准)** | **一次训练跑**——小循环改动;BSS 变体离线交错 | 直接打失配;唯一能阻塞启动且有正面证据的。从未在 gap 本身上测过。 |
| Unlikelihood | 辅助损失压低退化 token/续写 | **退化上 proven-at-scale;一致性 gap 未证** | **一次训练跑**——损失插件;seq 级要 rollout | 治退化不治一致性;机制相邻。 |
| Edunov 序列级损失 | 全序列损失,采样候选上优化 | **仅 NMT 时代**(41.5 BLEU) | **一次训练跑,但每步 k× 成本** | 按构造 on-policy;无 LM 证据,无轨迹测量。 |
| DAgger | rollout + 专家重标注 + 聚合 | **对 LM 是 folklore**(零文本 LM 论文;机器人/IL proven) | post-pretraining;需 oracle 重标注 | 协变量偏移的理论正解;无 LM 证据。 |
| Born-Again 自蒸馏 | 同规模学生蒸馏教师 | **RNN 时代 proven-small**(PTB 71.87→68.56 @52M);现代 <1B 无 | post-pretraining;2× 训练 | 间接;只 PPL 证据。 |
| SDFT | 用模型自己的蒸馏数据微调 | 7B–70B proven(遗忘/基准) | post-pretraining;reroll 数据+微调 | 治任务数据/模型 gap,不治 teacher-forced/free-running。 |
| Rationale SFT(Orca/Shridhar/Magister) | 大学教师 CoT 上 SFT | proven-small(GPT-2-large 超 GPT-3-6B) | post-pretraining;需教师+reroll 数据 | 另一个问题(模仿质量),只基准。 |
| MiniLLM on-policy reverse-KL | on-policy 蒸馏 + reverse KL | 120M–13B proven;声称低暴露偏差(度量未核验) | post-pretraining;教师+on-policy rollout | 唯一声称 gap 度量的;值得核验度量。 |
| RL 推理(SimpleRL-Zoo/R1 式) | 可验证奖励上的 GRPO | 基准 proven-small(0.5B–3B;TinyZero 0.5B 失败) | post-pretraining;**RL 基础设施** | 优化轨迹产出(奖励)但无人测一致性。 |

**<1B 文献的空白:(1) 无论文在任何干预前后测过 teacher-forced vs free-running 逐 token 一致性——我们的仪器是未发表领地;唯一的测量论文(He et al.)在开放式文本上发现失真"有限、非渐衰",既兼容我们的即时误差发现,也警告 code/math 轨迹可能不同于任何已发表对象。(2) 深度对轨迹校准的测量在整个深度/宽度文献里不存在。(3) DAgger 从未被用于文本生成 LM 训练。所有方案里只有 scheduled sampling 和 unlikelihood 碰得到今天的启动;其余全是 post-pretraining,且每个 "proven" 标签都建立在基准分上,不是轨迹质量。**

## 12. 判决(fb,2026-09-01)

**启动不带 trajectory 干预。Scheduled sampling 排队为 post-pretraining 第一干预,且先试 BSS 变体(离线数据交错,数据侧臂,非训练循环改动)。**

三条理由(按权重):
1. **无人有证据它做我们要的事。** 0/14 干预论文测过轨迹一致性;"proven" 建立在 RNN 时代的 BLEU/F1/FER 上,LM 复现只有 2 篇基准论文。机制匹配诊断,证据不碰诊断。
2. **一个跑里放两个变量。** 500M 跑已改深度 2.7×、参数 2.4×;再加未证实的训练循环改动,跑差了无法归因,跑好了不知道哪一半起作用——正是 36%-vs-5% 消融已付过学费的失败模式。
3. **课程是承重的且在我们规模上未测。** ε=1 起步惨败(BLEU 11.2),schedule 是超参搜索,deadline 前盲启;有偏梯度批评在 transformer 规模两个方向都未测——是在赌一个已知理论问题不出现,不是有证据它不出现。

**怎么跑才对(把发现 #1 从文献空白转成优势):** 启动 → 在 500M checkpoint 上测 free-running 一致性,对 200M 的 0.727/0.25 基线 → scheduled sampling 作为**对已测轨迹基线**的受控干预跑,不是对基准。我们有 14 篇论文都没有的仪器,就在唯一能读出结果的设置里跑这个干预。

深度发现按原话上呈用户:深度赌注 unsupported, not contradicted。

## Sources

已核验(摘要/全文):
- https://arxiv.org/abs/1506.03099, https://arxiv.org/html/1506.03099v3(Scheduled Sampling;效应量)
- https://arxiv.org/abs/1011.0686(DAgger)
- https://arxiv.org/abs/1805.04770, https://arxiv.org/html/1805.04770v2(Born-Again;PTB 数字)
- https://arxiv.org/abs/2402.13669(SDFT)
- https://arxiv.org/abs/2306.02707(Orca)· https://arxiv.org/abs/2311.11045(Orca 2)
- https://arxiv.org/abs/1908.04319(Unlikelihood)
- https://arxiv.org/abs/2501.12948, https://arxiv.org/html/2501.12948v2(DeepSeek-R1;671B/37B,GRPO,10400 步)
- https://arxiv.org/abs/1711.04956, https://arxiv.org/html/1711.04956v5(Edunov;损失清单,41.5 BLEU)
- https://arxiv.org/abs/2402.14905(MobileLLM)· https://arxiv.org/abs/2509.24945(MobileLLM-R1)
- https://arxiv.org/abs/2503.18892(SimpleRL-Zoo)
- https://arxiv.org/abs/1905.10617(He et al.,前缀交换测量)· https://arxiv.org/abs/1910.11235(Xu et al.,road exam)· https://arxiv.org/abs/2109.08705(Chiang & Chen,GPT-2 退化)· https://arxiv.org/abs/1910.00292(Schmidt)
- https://arxiv.org/abs/2306.08543(MiniLLM)· https://arxiv.org/abs/2410.14655(BSS)· https://arxiv.org/abs/2309.02823(RAD)· https://arxiv.org/abs/1610.09038(Professor Forcing)· https://arxiv.org/abs/2212.00193(Shridhar et al.)· https://arxiv.org/abs/2212.08410(Magister et al.)
- https://raw.githubusercontent.com/Jiayi-Pan/TinyZero/main/README.md(TinyZero——GitHub 项目,非 arXiv)

错 ID 存档(已抓取,勿引):
- https://arxiv.org/abs/2407.10932 = Brunn-Minkowski 数学论文(brief 称 "SmallLLM")——未核验
- https://arxiv.org/abs/2501.12968 = 强宇宙监督物理论文(brief 称 DeepSeek-R1;正确:2501.12948)
- https://arxiv.org/abs/2502.14758 = 宇宙学论文(TinyZero ID 猜测——TinyZero 无 arXiv 论文)
