---
question: 200M 参数 / 22B token 预算下,什么数据组成能出能力——文献里 100-500M 级有能力的模型各用了什么数据、能做什么
status: recorded
source: fb P0 tasking 2026-09-01; all numbers fetched 2026-09-01 from primary sources (papers / HF model cards), URLs inline
---

# 小模型数据组成证据(100M–500M 级)

**结论先行:文献里不存在 200M 参数、~22B token 训练出可测推理能力的模型。这不是配比问题,是预算问题——所有 ≤500M 有能力的模型都在 600B–18T token 上训练(我们的 27–800×),且每一个都用教育过滤/合成教科书为主的数据,没有一个用 89% 原始抓取。在我们的 token/参数比(~110×)下唯一有能力的点是 Phi-1.5(115× seen,80% 合成教科书,但 1.3B 参数)。所以:(a) 22B 预算下"零能力"与文献一致,不意外;(b) 若目标是 200M 出能力,下一跑需要 ~600B+ token 且教育/合成占多数,且按 SmolLM2-135M 的证据,知识可出、数学推理出不来(GSM8K 1.4 @ 2T);(c) 若目标是 ~22B token 出能力,唯一证据指向合成教科书为主 + 参数涨到 1.3B 级。**

## 1. 有能力的小模型:参数、token、数据组成

| 模型 | 参数 | 训练 token | 数据组成 | 原始 vs 精选 |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.5B | 18T | 未公开分数;重过滤 + 数学/代码(Coder 线 5.5T 代码) | 精选为主 |
| SmolLM2-135M | 135M | 2T | FineWeb-Edu + DCLM + The Stack + 新过滤集(未公开分数) | 教育/精选 |
| SmolLM-135M/360M | 135M/360M | 600B | FineWeb-Edu 220B(87%)+ Cosmopedia v2 28B(11%,Mixtral 合成教科书)+ Python-Edu 4B(2%) | 98% 教育过滤+合成 |
| Phi-1.5 | 1.3B | 30B(150B seen,~5 epoch) | 80% 合成教科书(20K 主题)+ 20% phi-1 数据(6B 过滤代码 web + 1B 合成) | ~80% 合成,~20% 过滤 web |
| Phi-1 | 1.3B | 7B | 6B web(The Stack Python + StackOverflow,35B→6B 教育过滤,留 17%)+ <1B 合成教科书 + 180M 合成练习 | 86% 教育过滤 web,14% 合成 |
| MobileLLM-125M/350M | 125M/350M | 1T | 论文未披露 | — |
| TinyStories | <10M | ~1–2B | 100% 合成(GPT-3.5/4 生成儿童故事,~2M 篇) | 100% 合成 |

fb 记忆核验:**SmolLM-135M @ 600B token、FineWeb-Edu + Cosmopedia——token 数和来源对,但 Cosmopedia 只占语料 11%,不是并列主力;FineWeb-Edu 占 87%。**

## 2. 它们各自能做什么(实测数)

| 模型 | 知识 | 推理 | 代码 |
|---|---|---|---|
| Qwen2.5-0.5B (18T) | MMLU 47.5, HellaSwag 52.1 | ARC-C 35.6, GSM8K 41.6, MATH 19.5, BBH 20.3 | HumanEval 30.5, MBPP 39.3 |
| SmolLM2-135M (2T) | MMLU(cloze) 31.5, ARC 均 43.9 | GSM8K(5-shot) **1.4**(=零) | — |
| SmolLM-135M (600B) | MMLU-Pro 11.22 | — | — |
| SmolLM-360M (600B) | MMLU-Pro 10.95 | — | — |
| Phi-1.5 (1.3B, 30B) | MMLU(2-shot) 37.6, ARC-E 75.6, ARC-C 44.4 | GSM8K 40.2(via coding) | HumanEval 34.1, MBPP 37.7 |
| Phi-1 (1.3B, 7B) | — | — | HumanEval 50.6, MBPP 55.5 |
| MobileLLM-125M (1T) | ARC-E 43.9, HellaSwag 38.9 | ARC-C **27.1(=随机)** | — |
| MobileLLM-350M (1T) | ARC-E 53.8, HellaSwag 49.6 | ARC-C 33.5 | — |
| TinyStories (<10M) | 无标准基准;GPT-4 评分的流畅连贯多段落故事 | — | — |

"这个规模的能力"具体指什么,按仪器分:
- **知识(MMLU/ARC-E)**:135M @ 2T 教育数据 → 31.5;0.5B @ 18T → 47.5。可测,随 token 和过滤质量涨。
- **数学推理(GSM8K)**:135M @ 2T → 1.4(零);0.5B @ 18T → 41.6。≤500M 出数学推理需要 18T 量级,或 Phi-1.5 的合成教科书路线(1.3B/30B → 40.2,但参数大 6.5×)。
- **代码(HumanEval)**:Phi-1 用 7B 教育过滤代码 + 合成练习在 1.3B 上到 50.6——代码是小预算下最容易出能力的域,因为过滤信号强(可执行)。

## 3. 关键对照:我们的 200M @ 22B

- **token/参数比**:我们 110×(22B/200M)。Phi-1.5 115× seen(150B/1.3B)→ 有能力;SmolLM 4,400×(600B/135M)→ 有能力;Qwen 36,000× → 有能力;MobileLLM 8,000×(1T/125M,数据未披露)→ ARC-C 随机。**我们的比值不离谱,离谱的是组成**:Phi-1.5 在同量级比值下用 80% 合成教科书,我们用 89% 原始抓取。
- **22B 预算下唯一有能力的点是 Phi-1.5**(30B,1.4× 我们),但它 1.3B 参数(6.5× 我们)且 80% 合成。参数效应未量化——135M 到 1.3B 之间的文献是空的,我们正坐在这个空档里。
- **MobileLLM 是反面参照**:1T token(45× 我们)在 125M 上 ARC-C 仍是随机——光堆 token、数据不精选,在小参数上不出推理。
- **Chinchilla 最优对 200M 是 ~4B token,我们在 5.5× Chinchilla**;但小模型出能力的配方不是 Chinchilla 最优,是 3,000–36,000× 参数的过度训练 + 精选数据。

## 4. 对下一跑组成的含义(按证据强度排)

1. **原始抓取占比 89% 与所有有能力的配方相反**。每个有能力的点都做了重过滤(教育分类器)或合成。DCLM(7B 级)的结论也是 model-based filtering 是关键。这是方向性结论,不是量级结论。
2. **若预算锁死 22B**:唯一证据是 Phi-1.5 路线——合成教科书为主(它 80%),且参数要涨到 1.3B 级。200M/22B 出推理在文献里没有先例,目标应改为"连贯 + 浅层知识",仪器用 MMLU/ARC-E 不用 GSM8K。
3. **若参数锁死 200M**:SmolLM 配方(FineWeb-Edu 式教育过滤 + 合成教科书 + 教育代码)@ 600B+ token 是唯一有先例的路径,且按 SmolLM2-135M 的证据,预期知识可测、数学推理不可测。
4. **代码是小预算最容易出能力的域**(Phi-1:7B token → HumanEval 50.6 @ 1.3B),因为可执行性提供了其他域没有的过滤信号。我们 code_rp1t 是原始 RedPajama GitHub,没做教育过滤——Phi-1 从 35B 过滤到 6B(留 17%)。

## 5. 证据薄处(诚实标注)

- SmolLM-135M/360M 的完整基准表在博客图片里,抓不到;只验证了 MMLU-Pro(11.22/10.95)。
- MobileLLM 数据组成论文未披露——"web 为主"是从其沉默 + Meta 惯例的推断,不是引文。
- Qwen2.5 组成无分数,只有"重过滤 + 数学/代码"的定性陈述。
- 135M–1.3B 之间(我们坐的位置)没有任何有能力的文献点。
- Phi-1 的"50B token seen"vs 7B 数据集是 ~7 epoch;Phi-1.5 是 30B 数据集 5 epoch。epoch 重复在小预算下是未量化的变量。

## 6. e1 仪器的文献参考带(BPB / 条件 NLL)

fb 2026-09-01:val-slice 缺陷后(`eval/domain_loss.py:47` 的 scored 集是训练池均匀随机样本,不是 held-out——tilerl 实测 0.625% 落 val vs 0.587% 随机期望),唯一不被解码病理和 val-slice 缺陷混杂的仪器是 e1 在建的 bits-per-byte 或 gold-answer 条件 NLL。文献里 100-500M 级有没有参考带?

**有,但发表的形式是 The Pile 上的 nats/token(或 ppl),不是 BPB。**

| 模型 | 参数 | 训练 token | Pile nats/token | 来源 |
|---|---|---|---|---|
| Cerebras-GPT-111M | 111M | 2.2B(Chinchilla 最优) | 2.608 | Cerebras-GPT 论文 Table 3 |
| Mamba-130M | 130M | 300B | 2.357(ppl 10.56) | Mamba 论文 Table 1 |
| Pythia-160M | 160M | 300B | 3.389(ppl 29.64) | Mamba 论文 Table 1 基线 |
| Cerebras-GPT-256M | 256M | 5.1B | 2.349 | Cerebras-GPT |
| Mamba-370M | 370M | 300B | 2.114(ppl 8.28) | Mamba |
| Pythia-410M | 410M | 300B | 2.298(ppl 9.95) | Mamba 基线 |
| Cerebras-GPT-590M | 590M | 11.8B | 2.181 | Cerebras-GPT |
| Mamba-790M | 790M | 300B | 1.992(ppl 7.33) | Mamba |

**带(sanity band, not a comparison——单位、语料、文本类型都不同,见下):** 100-150M **2.36-2.61 nats/token ≈ 0.85-0.94 BPB**;250-400M **2.11-2.35 ≈ 0.76-0.85 BPB**;600-800M **1.99-2.18 ≈ 0.72-0.79 BPB**。

**BPB 是显式换算的,不是发表的:** BPB = nats/token × 1.4427 ÷ bytes/token。上表用 **4.0 bytes/token**(50K 级 BPE 在英文为主文本上的经验值;这些 tokenizer 在 The Pile 上的 bytes/token 没有发表,敏感性:3.5 → BPB 高 14%,4.5 → 低 11%)。**这个敏感性(±11-14%)大于相邻规模档之间的带距(~0.09-0.11 BPB)——单是 tokenizer 假设就能把一个模型移过一整个规模档,这本身就是"只能做 sanity check"的理由,不只是免责声明。**我们的数是 **gold answer 字符串上的 bits/UTF-8 byte(code 0.918,math 0.590)**——不同单位、不同语料、不同文本类型。**不要把 0.918 和 0.85 并排读成"可比 Mamba-130M":math 0.590 低于带内任何值,反映的是答案字符串的公式化低熵,不是优越性。这个带只能做水平 sanity check。**

**条件 NLL of gold answers:没有发表(这本身是个发现)。** 这个规模的论文都报 accuracy/EM,不报 gold-answer NLL 的水平值。**所以 e1 的仪器没有外部参照,只能按趋势读——正如它预注册的那样。** 不用再找了。

**Pythia-160M 离群值(3.39 vs 同 token 数 Mamba-130M 2.36):** 保留。Pythia 自己的论文提到最小规模配置表现不佳,但我查过的来源没有确立确切原因(可能与全规模统一的大 batch 配置有关)——不知道就是不知道,不编。

**val-slice 缺陷的精确范围(44 分析,fb 2026-09-01 全量接受):** 杀掉的是跨实验水平比较(我们的 ppl vs 发表值)和任何"held-out 泛化"读法;**不杀**同实验同 scored 文本上的 Δ vs 同仪器 σ̂ floor(16B 判决作为"仪器相对"存活,30B prereg 同范围)。e1 的仪器是正解。

**跨角色过拟合混杂(与缺陷同级,必须写进事实):** scored 集是训练池的随机样本,所以**一个被读 6 次的域在自己的训练文档上损失天然低于被读 0.08 次的域,这读起来就是"学得更多"**——cot 6 epoch vs wiki_chat 0.077 epoch。我们已发表的任何跨角色排序都带着这个方向已知的偏差(多 epoch 域被系统性高估)。16B 的 nat/B 表和 §3 的成本表都在此列;角色内跨时间比较不受此混杂影响。

## Sources

- SmolLM 博客(语料 220B/28B/4B,600B token): https://huggingface.co/blog/smollm
- SmolLM-135M 卡(600B,MMLU-Pro 11.22): https://huggingface.co/HuggingFaceTB/SmolLM-135M
- SmolLM-360M 卡(600B,MMLU-Pro 10.95): https://huggingface.co/HuggingFaceTB/SmolLM-360M
- SmolLM2-135M 卡(2T,MMLU 31.5,GSM8K 1.4,配方): https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- Phi-1 论文(1.3B,7B,6B web+1B 合成,HumanEval 50.6/MBPP 55.5): https://arxiv.org/abs/2306.11644
- Phi-1.5 论文(1.3B,30B/150B seen,80% 合成,MMLU 37.6,GSM8K 40.2,HumanEval 34.1): https://arxiv.org/abs/2309.05463
- MobileLLM 论文(125M/350M,1T,ARC-C 27.1/33.5): https://arxiv.org/abs/2402.14905
- Qwen2.5 论文(18T,0.5B:MMLU 47.5,GSM8K 41.6,HumanEval 30.5): https://arxiv.org/abs/2412.15115
- TinyStories 论文(<10M,合成,连贯性): https://arxiv.org/abs/2305.07759
- DCLM 论文(过滤是关键;7B @ 2.6T → MMLU 64%): https://arxiv.org/abs/2406.11794
- Pythia 论文(300B The Pile,与同参 OPT 持平——原始基线): https://arxiv.org/abs/2304.01373
- Cerebras-GPT 论文(Pile nats/token 表,§6 参考带): https://arxiv.org/abs/2304.03208
- Mamba 论文(Pile ppl 表 + Pythia 基线,§6 参考带): https://arxiv.org/abs/2312.00752
