---
question: "3.6B -> 36B 预训练语料：源组合、跨源去重、register 迁移的打分——语料半的设计（不是把当前权重放 10x）"
status: recorded
source: "aupai-fb hypothesis (discrimination/generation gap is composition not volume); multilingual.json supply; measured 2026-08-30"
---

# 30B code+math 推理语料计划（语料半）

目标改为：**推理模型（coding + math 能力）**，~30B（用户方向 2026-08-30 变更）。scaling law 非交付、不需证。前面的通用中文组合（natural 68%/synthetic 32%、zh:en 84:16、~16B 中文 web）为此目标**作废**。新组合 by 能力角色，`run score` 对 math/code 域 = 验证器（答案查得过 / 测试过不过），非质量头。

## 组合假设（阶梯裁决，密度换表达在减速）

阶梯四点的判读（b0 逐点读能力面板，非 val NLL）：D / two-way / open@1 / GAP =
0.2b 90.4/16/74.4；0.3b 91.65/21.9/69.75；0.4b 92.28/25.1/67.18；0.8b 94.78/30.9/63.88。
open@1 每 e-fold +8.4pt（仍在升，高于 4.8 噪声），**生成在真改进**；但 gap 64pt 宽，
所有投影在整段 10× 只闭合 3-18pt，**乐观端点仍剩 ~46pt**。
**裁决：体积买表达但在减速；10× 在任何实测投影下不闭合表达 gap，**不应以表达为由作 10×。cap 的独立理由不受影响。
**cap 的理由：54% 合成从不是被选择的数**（= 49.6% textbook + 85% 一个大家都以为是英文的域，today 才发现的意外）；~10% 是把意外校正到标准做法（SmolLM2 用 Cosmopedia ~11%）。**不在任一方向写 "cap 会改善表达能力"**——体积杠杆真但不足，且阶梯不归因组合（什么都未 implico 组合）。44 的 premise 测（模板在 cosmopedia vs web_hq 的浓度、top-100 句框覆盖、n-gram 多样性）是唯一能证我解释的那条，并行跑。

**同组合对照臂（10× 必须有）**：b0 的不对称条款——阶梯可弱化假设但不能定 36B 会发生什么，所以 10× 构建需**同组合对照臂**（一个在 36B 跑当前权重复制的对照组，一个跑自然前移组合的交臂），否则 36B 上表达是否移动不可证。这是设计进 build、不是事后补。

## 词表解冻条件 2（material，需自主裁，非默认）

词表冻结的三个解冻条件之一是「语料分布 material 变化」。0.75%→16% en、54%→10% 合成，material。**在 fetch 提交前**：对提议组合取样本跑 `scripts/tokenizer_eval.py`，验冻结词表仍过闸（round-trip 无损 / 256 字节 / hanzi≥0.95 / en fertility≤1.55 / never-used≤0.01）。过→保持冻结并记录「条件已查非忽略」；不过→重建决策（会作废所有旧 checkpoint）→归属用户。方向有利：`tok.en_share_sweep` 量 14/33/50% 的 fertility，14% 是我们拟合的值；语料现仅 0.75% en = 词表已与语料很不匹配——16% en 反而比 0.75% 更贴合 14% 拟合的词表，是不重建的论据。

## 1. 源组合（code+math 推理 @30B，by 能力角色）

**derive ratio 重推**（旧 84:16 zh:en 死于中文 LM 目标）：code/math/papers/CoT 英文压倒性；粗估 en 主导 ~60:40（依 code/math 源落地 token 再定）。**general-Chinese-web 从 ~16B 缩到 ~4B（~13%）**——它是支撑读题的角色，非基底。

### 30B 组成表（2026-08-30 裁定，t08；供给已实测，非估计）

| 角色 | 30B 权重 | 量 | 现有 | 倍数 | 源 | 供给 |
|---|---|---|---|---|---|---|
| **code 原始** | 27% | 8.0B | 0.058B | **×138** | RedPajama-1T github | **73.6B 可达**，精确计数，不是瓶颈 |
| **code 带执行标注** | 7% | 2.0B | 0 | 新增 | **从 73.6B raw 里挖配对 impl+test** | **7.6% 的 repo 有 impl+test，26.96% 的文档落在这种 repo 里**（`ds.code_exec_supply`）——可过滤，零生成 |
| **math 可验证** | 18% | 5.5B | 0.082B | ×67 | OpenWebMath / proof-pile | 开放，未落地计数 |
| **长 CoT** | 15% | 4.5B | 0 | 新增 | OpenThoughts / Skywork-OR1 | Apache，未落地计数 |
| **英文通用** | 18% | 5.5B | **0.027B**（真英文） | **×204** | Ultra-FineWeb en / fineweb | 开放 |
| **中文网页** | 11% | 3.3B | 1.434B | ×2.3 | fineweb2-HQ + CCI3-HQ | 已下，sha 钉 |
| **中文教材** | 3% | 1.0B | 1.610B | **×0.6（缩）** | cosmopedia，截断 | 有余 |
| **wiki + chat** | 1% | 0.2B | 0.284B | ×0.7 | 现有 | 有余 |

zh:en ≈ **35:65**（旧的 84:16 随中文 LM 目标一起作废）。

**两条今天的测量改变了这张表，不是猜的：**

1. **code 格要的不是"更多代码形状"，是"更多正确的代码"。** base 在零 demo 下已经有 **97.8%** 的生成是代码形状的，正确率 **0/497**。形状不缺，内容缺。所以 27% 的原始 code 买的是分布，而**唯一没试过的杠杆是带执行信号的那 2B**——它是表里唯一一格没有现成供给的，也是唯一值得先解决供给的。把 8B 原始码加到 10B 不会动这一格。
2. **`en` 域现在只有 0.027B 真英文**（声称 0.161B，其中 85% 是中文）。code/math/CoT 三个角色的源压倒性是英文，而模型几乎没见过英文——**×204 是这张表里最大的倍数，也是最便宜的一格**（源开放、无 gate、无许可问题）。

**cosmopedia 从 1.61B 缩到 1.0B**：理由是它 48.2% 的文档共用同样 100 个句框（6.38× 帧浓度），**不是**"它伤害表达能力"——后者未测，不写。

### 工具调用格式：把算术交出去，而不是学会算术

**我们的 `eqcheck` 实测：8% 的生成含等式，其中 81.7% 算错。** 这是一个**格式可以绕开**的问题，不是必须用规模解决的问题——只要输出格式把「写出表达式」和「写出结果」分成两件事，结果那一半就可以由执行器产生。

**不要发明模板。词表 2026-08-29 冻结，新的特殊 token 一个都加不了**，而现成的标注形式在数据里本来就有，且用现有词表就能表示（2026-08-31 实测 round-trip 无损）：

| 形式 | 来源 | 冻结词表下的开销 |
|---|---|---|
| `<<12/60=0.2>>` | **GSM8K 答案里本来就有的计算器标注** | 10 token，无损 |
| `>>> f(2)` + 下一行输出 | Python doctest，真实代码里本来就有 | 6 token，无损 |
| `assert f(2) == 4` | 测试文件，真实代码里本来就有 | 7 token，无损 |

**nanochat 走的正是这条**（`docs/lessons/nanochat_recipe.md`）：`tasks/gsm8k.py` 不发明协议，它把 GSM8K 自带的 `<<expr=result>>` 解析成 `{"type":"python"}` + `{"type":"python_output"}` 两个有类型的片段。**执行信号是从数据里挖出来的，不是造出来的。**

**训在哪一步：预训练，不是 SFT。** 依据是我们自己的测量——base 在零 demo 下已经有 **97.8%** 的生成是代码形状的，**因为预训练语料里有代码**。格式从预训练来，而且便宜；一个 200M 模型没有多少容量能从一个小 SFT 集里学会一套全新的输出约定（同一批测量里，SFT 只把准确率从 0.0% 抬到 2.2%，且 11 道全是同一个模板族）。

**因此这一格的施工方式是过滤而不是合成：**
1. 从 73.6B raw 里筛出 impl 与 test 配对的文件（3b 扫描：7.6% 的 repo、26.96% 的文档）。
2. 把每一对渲染成**一种**形态——渲染器由代码持有，形态不由源数据继承（这是 `dq.sft_termination_underdetermined` 量到 23.9% 的近重簇形态不一致之后的直接推论；`code_python` 那一档是 0%，正因为代码有确定边界）。
3. math 侧同样处理：`<<expr=result>>` 已在 GSM8K 及其中文衍生集里，直接沿用。

**更正（2026-08-31，本节写下一小时后）：单片段渲染训不出 agent，要两回合。**

`<<12/60=0.2>>` 教的是「写出表达式，然后写出结果」——一次连续生成，**两半都由模型产出**。agentic 循环要的是「写出调用，然后停住」——**结果是给它的**。同一份内容，两种相反的行为；按单片段训出来的模型会自己编造工具输出，正好是 agent 的反面。

冻结词表够用，无需新特殊 token（2026-08-31 实测，`skip_special_tokens=False` 下逐字节无损）：

```
<|im_start|>assistant
12/60 = <|im_end|>          ← 8 token，模型在这里被监督「停」
<|im_start|>tool
0.2<|im_end|>               ← 给它的，必须从 loss 里 mask 掉
<|im_start|>assistant
...继续<|im_end|>
```

**关键在 loss mask：tool 回合和 prompt 一样屏蔽。监督了它，就是在教模型生成工具输出。** 这条约束落在 `scripts/test_sft_pack.py`，不落在文档里——那里已经在检 loss mask，加一条 tool 段必须为 -100 即可。

**上下文长度因此成为设计约束，不是实现细节。** 预算的单位从「一道题」变成「一个循环」：系统提示 + 问题 + 调用 + 工具回合 + 继续，多轮叠加。两个数决定 4096 够不够，都在测（`t18`/`t19`）：模型在 4096 内外的 per-position loss（NoPE 没有 RoPE 可外推，三层完整因果注意力与九层固定状态的 KDA 在超长时可能表现相反），以及一个真实循环在**我们自己的词表**下的 token 开销——我们中文约 1.7 chars/token，nanochat 是 4.8，**别人的「4096 够用」不能直接搬**。

**chat 现在是 mix 的 1.18%，而本表 `wiki + chat` 合起来只有 1%。对 agentic 目标这一格是在缩。** 多轮结构和「停下等外部输入」与代码、数学一样要靠预训练量堆出来，1% 堆不出来——具体权重等 `t19` 的循环开销落地后重定。

**没测的一条，必须标明：「把模板训进预训练、模型就会在推理时使用它」这一步没有测量。** 现有证据只支持前一半（格式来自预训练，97.8%），不支持「一个被显式渲染的新约定会被同等地学会」。3.24B 上无法测——所有生成式指标都在地板。**这是 30B 上第一个该测的东西，也是它唯一一个不靠规模的假设。**

**λ 的定位因此改变（2026-08-31）：** 合成不服务这一格。2.0B / 500 token 一个样本 = **400 万个样本**，即使 λ=1 也在任何预算之外；pilot 实测 λ=1.38 次调用/可用样本（26/36）。**λ 服务的是 SFT 与评测规模（万级），在那里它是真约束。**

**未测且必须标明的三件：**（a）30B 上 code 生成是否越过 12.6% 仪器阈值——3.24B 上是零，那是干净零点不是预测；（b）词表在自身单位下对 code 的闸口位置（见解冻条件 2 节）；（c）math/CoT 两个源的实际落地 token，现在是权重不是计数。

- **synthetic 轴重组**：math+code 的 synthetic = 机制非瑕疵（`run score` 验证器 = 比一切 filter 好的过滤）。cap 只打**模板化百科散文**（cosmopedia 6.38× 帧浓度），不打 synthetic 推理链——两者我们此前混为一谈，分离。
- **渲染器原则（2026-08-30，fb）+ 实测依据**：SFT 终止欠定实测 23.9% 近重簇答案形态 >1（集中 school_math 81%/s1k 82%/openo1 67%，code_python 0/math_gsm8k 0），机制解释=code 有确定边界不欠定。**修法 = 每源过渲染器、形态由代码定（nanochat 式），非打包归一时**——归一是补救，渲染器是预防。λ 合成目标：**每簇一种形态，由生成器强制**（`dq.sft_termination_underdetermined`）。code 重复 55.8% 成因在解码器（f8c08c1），不在数据。
- 残余的通用中文从 16B 缩到 ~4B（支撑读题），合成占比升（math+code+CoT ~70%）。


**具体源 + 许可（2026-08-30 实测可达）：**
| 域 | 候选源 | 可达/许可 | 状态 |
|---|---|---|---|
| code | 见 `ds.code_source_reachability` + `ds.code_fertility_redpajama`（2026-08-30 实测） | **直达 host 可达、必须 `curl -4`**（IPv6 出口坏）。RedPajama-1T github 未压缩 jsonl（每文件 2.66GB 751.3M tok，98 个）可达；hf-mirror resolve 通、API 403 反爬；huggingface.co 真堵；the_pile 证书过期需 -k 不可信 | **预训练 code 格量够（~73.6B tok 精确，分支 ≥8B 定）**；真码走预训练（code-SFT 实测 200M 只 0.0%→~3%，装不下能力，见 fb 2026-08-30）。格闸口未测 |
| math | OpenWebMath / DeepSeekMath / proof-pile | 开放/开源 | 语义以中文刷卡；英文为主，可验证 |
| CoT | OpenThoughts / Skywork-OR1 / r1 类 | Apache 系 | 英文长链 |
| en 通用 | WanJuanCC(46B en) / Ultra-FineWeb en / fineweb en | Apache/CC | 修 en 域（现 85% 中文） |
| zh web | fineweb2-HQ + CCI3-HQ | 已下 sha 钉 | web 留出验过 |
| wiki | zh wiki | CC | 现有 |

## 2. 36B 跨源去重（工程主矛盾，`harness run dedup` 第 4 步）

设计：**exact content-hash O(n) + MinHash 带状 LSH**（build_corpus 内嵌 MinHashLSH）。**实测修正（2026-08-30）**：
- **exact 独立对（fineweb2-cmn vs CCI3-HQ）= 0.00%**（15K×2 采样全 0）。**对「两爬是否重叠」无信息**——已知答案已示 exact 对现实重复（whitespace/boilerplate/truncation/encoding 变体）盲，0.00% 只证无字节相同、恰非现实形式。**答案重叠的数 = MinHash 测独立对**（同 15K×2、同已验代码，决定 36B 多少 fetch 被浪费，待跑）。
- **MinHash 已知答案 = 100% 召回 / 0% 假阳**：构造 4 类现实变体（whitespace 折叠/boilerplate 剥离/截断/编码空格）× 74 原 doc = 296 近重全抓到，异 doc 0/72 错。**exact 对这些 ~0。** 即 near-dup 抓的正是 exact 抓不到的现实类。
- **结论：near-dup 进 v1（非 v2）**——已知答案证 exact alone 做不了现实跨源去重。派生对（SkyPile vs fineweb-edu，上游关系保证重叠）仍要跑定两派生源实际重叠，但已不负责证「exact 足够」。
- 优先对：CCI3-HQ vs fineweb2（已测 0% exact，测 near-dup）、SkyPile vs fineweb-edu（**SkyPile 是它上游，最可能 embarrass**，derived-pair 测试用）。
- **`harness run dedup --domains <a,b,c>` → `scripts/dedup_corpus.py`**（de 第 4 步）。**全局跨源 pass**（dedup 后 clean、score 前）。输出是一个 dedup 清单加一份统计，都带 `dedup_fp`（**计划中的产物，尚不存在**；路径在实现时定，此处不写路径 —— `doc_commands_exist` 会把文档里的路径当成应当存在的文件）。CPU-only、按域续跑。

## 3. "scoring" 在 36B 该指什么 + 磁盘守卫

现有质量头在 cosmopedia 上打低于原始 web——**跨 register 不迁移**（facts/data_quality.json），不照它规划。打分 = 格式/编码闸（现，precision 高）+ 跨源去重（dedup_fp）+ 组合权（mix 定）+ register 校准（44 研究定）。`harness run score --scorer` 可换后端，不硬编码今日质量头。44 拥研究问题，我拥管线实际跑什么。
- **磁盘（2026-08-30 更正，前一版把语料指向了一个会消失的层）**：`data/raw` 和 `data/corpus` 都留在 **`/work`** 下。容器里的 `/data00` **不是挂载点**——它和 `/` 同一个 `st_dev`，就是容器自己的 overlay，重启即清空；真正耐久的 `/data00`–`/data03` NVMe 在**宿主机**上，容器里看不到（`scripts/harness.py:45`）。两条路径报同样的可用空间，因为 overlay 的上层落在同一个底层文件系统上——**可用空间从来不是"另一块盘"的证据**，这就是上一版和 `fetch_corpus.py:29 BIG_RAW_TARGET` 一起被骗的地方。证据是 token cache：`TOKEN_CACHE = /data00/pretrain_1b_tokens.pt`，六个阶梯点跑完之后全库找不到任何 `tokens_*.pt`，而它消失时没有任何东西报错。
  磁盘守卫因此要查两件事，不是一件：`shutil.disk_usage(target).free >= target_bytes*1.5`（够不够），**且** `os.stat(target).st_dev != os.stat("/").st_dev`（是不是真盘）。只查前者的守卫会在一个会消失的层上愉快地放行 260GB。`--target_bytes` = 磁盘字节非 token（code 实测 3.04 B/tok，中文 UTF-8 未知则估 3-4）。

## 5. harness 契约（与 de 协同，一条管线）

所有步走 `harness run {fetch,clean,score}`，同 run pretokenize 契约：check 红即拒、写 exp 行、产物钉指纹、可重跑。我供实质（源/组合/去重/scoring 逻辑），de 供 step（harness 集成/缓存/增量）。fetch+正则清洗是 CPU，GPU 先按 44 的 register 研究再定。

## 未决（需测/需协调）
- CCI3-HQ/SkyPile/TeleChat 交叉去重率（入库时 MinHash）。
- natural 权重上移是否真的收窄 lambada open/双向 gap（阶梯 + 36B 早期点测）。
- textbook 压到 ~10% 的具体 -keep/截断规则（面板/44 定）。
- en 真源选 WanJuanCC vs Ultra-FineWeb vs fineweb（可达性/许可）。
