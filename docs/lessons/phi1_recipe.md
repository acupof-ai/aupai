---
question: Phi-1 的真实训练配方(可照抄的细节级)——过滤器机制、合成数据、token 拆分、训练日程;因为 3b 的语料重建正被它 steering,且一个"按可执行性过滤"的错误说法已被传播
status: recorded
source: fb tasking 2026-09-01; verbatim extraction from arXiv:2306.11644v2 + 2309.05463v1 full text, fetched 2026-09-01; arXiv IDs title-verified before extraction
---

# Phi-1 训练配方(论文原文级提取)

**纠错存档(fb 已传播给 3b 的语料重建):Phi-1 没有按可执行性(executability / run-yield)过滤代码。它的过滤器是一个随机森林分类器——在 ~100K 个 GPT-4 质量标注上训练,标签轴是"教育价值",特征是 codegen 模型的输出嵌入。管道里没有任何代码执行。基于"可执行性"前提测出的 1.09% run-yield 测的是 Phi-1 过滤器根本不具备的属性,不能作为 Phi-1 配方的复现或对照。**

**引用警告:** arXiv:2306.11644 的 HTML 摘要有渲染 bug——数字重复显示("66B tokens"、"88 A100s"、"44 days")。正文一致是 ~6B/~7B tokens、8 A100。引用以正文为准。

## 1. "textbook quality" web 代码过滤器——分类器,不是可执行性

- **机制:** "We then use this annotated dataset to train a random forest classifier that predicts the quality of a file/sample using its output embedding from a pretrained codegen model as features."(§2.1)
- **标注者与样本量:** "We annotate the quality of a small subset of these files (about 100k samples) using GPT-4"——**~100K 样本(不是 39K),GPT-4 标注(不是 GPT-3.5,不是人工)**。
- **标签定义:** "given a code snippet, the model is prompted to determine its educational value for a student whose goal is to learn basic coding concepts"——标签轴是**教育价值**,不是可执行性。
- **使用边界:** "we use GPT-4 minimally only for annotations on the quality of a small subset of The Stack and StackOverflow samples"。
- **输入→输出:** 输入 "the Python subset of the deduplicated version of The Stack and the StackOverflow which together contain over 35 million files/samples, totalling over 35B tokens";输出 "about 6B tokens"。**留存率论文没写**(17% 是按论文数字的算术推导,不是论文声明)。
- **论文未声明:** 分类器自身的准确率/精确率-召回率。

## 2. 合成教科书(CodeTextbook)

- "This dataset consists of less that 1B tokens of GPT-3.5 generated Python textbooks"(§2.2)——**GPT-3.5 生成,<1B tokens**。
- 多样性机制:"diversity is obtained by providing constraints on topics and target audience of the generated textbook"。
- **主题数、prompt 模板、种子清单:论文未声明。**(20K 主题是 Phi-1.5 的,不是 Phi-1。)

## 3. CodeExercises

- "less than 180M tokens of Python exercises and solutions"(§2.2)——**<180M tokens,GPT-3.5 生成**。
- 格式:"Each exercise is a docstring of a function that needs to be completed."
- 多样性:"the main means of eliciting diversity is by constraining the function names."
- 数量:"879.5K total problems in CodeExercises"(§5.2)。

## 4. Token 预算拆分(§2)

| 成分 | Tokens |
|---|---|
| 过滤代码(The Stack Python + StackOverflow) | ~6B |
| 合成教科书 CodeTextbook | <1B |
| CodeExercises | <180M |
| **合计** | **"less than 7B tokens"** |

## 5. 训练日程(§2.3)

**硬件/时长:** 8×A100 + DeepSpeed;phi-1-base 训练 "under 4 days";phi-1 微调 "an additional 7 hours"。

**预训练(phi-1-base):**
- fp16,AdamW,linear-warmup-linear-decay,attention + residual dropout 0.1
- seq len 2048,effective batch 1024
- max LR **1e-3**,warmup 750 步,weight decay 0.1,共 36,000 步
- 取 24,000 步 checkpoint 为 phi-1-base
- "equivalent to ∼8 epochs on our CodeTextbook dataset for a total of little over 50B total training tokens"

**微调(phi-1):**
- effective batch 256,max LR **1e-4**,warmup 50 步,weight decay 0.01
- 6,000 步,每 1000 步存 checkpoint,取最佳

**论文未声明:** 三成分的混合权重/数据顺序;分类器精度;GPU 显存/并行细节(只写了 deepspeed)。

## 6. 论文自己的机制主张

- **质量是因果杠杆:** "We hypothesize that such high quality data dramatically improves the learning efficiency of language models for code"(§6);坏代码 "reduce[s] the quality and quantity of the signal that maps natural language to code"(§2)。
- **多样性是第二杠杆:** 暴露于不同表达/解法、降低过拟合特定模式的风险、提升对未见任务的泛化(§2)。
- **小练习微调为何泛化:** "our finetuning process might have helped the model in reorganizing and consolidating the knowledge acquired during pretraining"(§6)。

## Phi-1.5 的增量(arXiv:2309.05463)

1. **合成数据:20K 种子主题,生成器未署名。** "We carefully selected 20K topics to seed the generation of this new synthetic data."(§2.2)多样性技巧:"In our generation prompts, we use samples from web datasets for diversity." 主题从 Python 转向 "common sense reasoning and general knowledge"。**论文没写生成器是哪个模型**,没有 prompt 模板。
2. **80/20 拆分,5× 训练 token:** "train for 150B tokens, with 80% from the newly created synthetic data and 20% from phi-1's training data."(§2.3)新合成 NLP ≈ 20B tokens;总数据 30B(Table 1)→ 150B 训练 token ≈ 5 遍(**epoch 数是推导,论文未声明**)。
3. **日程:常数 LR,无 warmup,从零训练:** "constant learning rate 2e-4 (no warm up), weight decay 0.1",Adam(momentum 0.9, 0.98),fp16 + DeepSpeed ZeRO-2,batch 2048,context 2048。Table 1 记 1.5K GPU-hours / 8 GPU。
4. **web 过滤:复用 Phi-1 的方法,未重新描述**("following the filtering technique in [GZA+23]")。来源:88B tokens 过滤自 Falcon RefinedWeb + 7B 代码 tokens。
5. **没有独立练习集**——练习嵌在合成教科书里,无 token 计数。

## Sources

- https://arxiv.org/abs/2306.11644 — Phi-1 摘要页(ID/标题核验)
- https://arxiv.org/html/2306.11644v2 — Phi-1 全文(所有 §1–6 引文;过滤器机制经第二次定向抓取复核)
- https://arxiv.org/abs/2309.05463 — Phi-1.5 摘要页(ID/标题核验)
- https://arxiv.org/html/2309.05463v1 — Phi-1.5 全文(所有增量引文;v2 URL 404)
