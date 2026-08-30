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

| 角色 | 量(tok) | 源 |
|---|---|---|
| code（执行信号） | ~10-12B（~35-40%） | The Stack v2 / starcoderdata 真码 + 带测试解题库（train 高峰） |
| math（可验证） | ~5-6B（~18-20%） | OpenWebMath / DeepSeekMath / proof-pile（英文，可验解） |
| 长 CoT/推理链 | ~4-5B（~15%） | OpenThoughts / Skywork-OR1 / r1 类（英文） |
| 通用文本（读题） | ~8-10B（~30%） | 真英文（修 en 更急）~5-6B + 通用中文网页 ~3-4B + wiki/百科少量 |

- **synthetic 轴重组**：math+code 的 synthetic = 机制非瑕疵（`run score` 验证器 = 比一切 filter 好的过滤）。cap 只打**模板化百科散文**（cosmopedia 6.38× 帧浓度），不打 synthetic 推理链——两者我们此前混为一谈，分离。
- 残余的通用中文从 16B 缩到 ~4B（支撑读题），合成占比升（math+code+CoT ~70%）。


**具体源 + 许可（2026-08-30 实测可达）：**
| 域 | 候选源 | 可达/许可 | 状态 |
|---|---|---|---|
| code | 见 `ds.code_source_reachability`（2026-08-30 实测） | **大体量 ungated code 源在 pod 上全不可直接拉**：.co 堵、镜像 API 403、the_pile/RedPajama 是 URL-manifest、SlimPajama 元数据 401、stack/starcoderdata gated | **预训练 code 格存疑**：唯一干净 ungated 直接源 codeparrot-clean 很小且路径未定。code 能力主战场可能落 Fable+沙箱的 SFT/RL 验证道（已通） |
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
- **`harness run dedup --domains <a,b,c>` → `scripts/dedup_corpus.py`**（de 第 4 步）。**全局跨源 pass**（dedup 后 clean、score 前）。输出 `data/dedup/dedup_manifest.json` + `dedup_stats.json` 带 `dedup_fp`。CPU-only、按域续跑。

## 3. "scoring" 在 36B 该指什么 + 磁盘守卫

现有质量头在 cosmopedia 上打低于原始 web——**跨 register 不迁移**（facts/data_quality.json），不照它规划。打分 = 格式/编码闸（现，precision 高）+ 跨源去重（dedup_fp）+ 组合权（mix 定）+ register 校准（44 研究定）。`harness run score --scorer` 可换后端，不硬编码今日质量头。44 拥研究问题，我拥管线实际跑什么。
- **磁盘**：`data/raw` symlink → `/data00/aupai_raw/`（fetch 脚本建 target+symlink；`/data00` 1.9T 余）；语料 `data/corpus` 留 /work（866G 余）。`fetch_corpus.py` 起前 `shutil.disk_usage("data/raw").free >= target_bytes*1.5`，不足即拒（便宜上检，贵在小时六凑）。`--target_bytes` = 磁盘字节非 token（中文 UTF-8 未知则估 3-4 B/tok）。

## 5. harness 契约（与 de 协同，一条管线）

所有步走 `harness run {fetch,clean,score}`，同 run pretokenize 契约：check 红即拒、写 exp 行、产物钉指纹、可重跑。我供实质（源/组合/去重/scoring 逻辑），de 供 step（harness 集成/缓存/增量）。fetch+正则清洗是 CPU，GPU 先按 44 的 register 研究再定。

## 未决（需测/需协调）
- CCI3-HQ/SkyPile/TeleChat 交叉去重率（入库时 MinHash）。
- natural 权重上移是否真的收窄 lambada open/双向 gap（阶梯 + 36B 早期点测）。
- textbook 压到 ~10% 的具体 -keep/截断规则（面板/44 定）。
- en 真源选 WanJuanCC vs Ultra-FineWeb vs fineweb（可达性/许可）。
