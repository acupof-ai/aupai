---
question: "3.6B -> 36B 预训练语料：源组合、跨源去重、register 迁移的打分——语料半的设计（不是把当前权重放 10x）"
status: recorded
source: "aupai-fb hypothesis (discrimination/generation gap is composition not volume); multilingual.json supply; measured 2026-08-30"
---

# 36B 语料计划（语料半）

目标：3.6285B → ~36B。36B/172.51M 非嵌入参数 = 209 tok/param，仍约 30x 低于现代实践。供给不是约束（9.73TB 都在）；**约束是 36B 规模下的打分与跨源去重，不是抓取**。

## 组合假设（fb，待阶梯验证，不据此排工）

当前 53.8% chinese-cosmopedia（模板化合成，loss 2.79 vs web_hq 5.05，**2.26 nat 差距**）——语料中最易预测的文本。假设：判别/生成 gap 是**组合问题而非体积问题**；10x 同 mix 提知识不提表达能力。阶梯的 six-point loss + lambada 开放/双向 gap 测它：gap 随 D 收窄则体积修复；平台则组合。

**cap 的理由不建立在上述假设上**（fb 明示假设未测、不据它排工）。cap 的理由是：**54% 合成从不是被选择的数** —— 它 = 49.6% textbook 权重 + 85% 一个大家都以为是英文的域（today 才发现的意外）。没人决定建一个过半模板化合成文本的语料。~10% 是把意外校正到标准做法：SmolLM2 用 Cosmopedia ~11%，没有任何认真的预训练语料停在 54%。**若阶梯显示 gap 随体积收窄，证伪 fb 的机制，但这不改变「54% 合成曾被用来训」这件事本身不可辩护。**

## 词表解冻条件 2（material，需自主裁，非默认）

词表冻结的三个解冻条件之一是「语料分布 material 变化」。0.75%→16% en、54%→10% 合成，material。**在 fetch 提交前**：对提议组合取样本跑 `scripts/tokenizer_eval.py`，验冻结词表仍过闸（round-trip 无损 / 256 字节 / hanzi≥0.95 / en fertility≤1.55 / never-used≤0.01）。过→保持冻结并记录「条件已查非忽略」；不过→重建决策（会作废所有旧 checkpoint）→归属用户。方向有利：`tok.en_share_sweep` 量 14/33/50% 的 fertility，14% 是我们拟合的值；语料现仅 0.75% en = 词表已与语料很不匹配——16% en 反而比 0.75% 更贴合 14% 拟合的词表，是不重建的论据。

## 1. 36B 源组合（自然文本前移）

提案：自然 ~68% / 合成 ~32%（当前估反转了——现 46% 合成当道）。zh:en ≈ 84:16。

| 组 | 量(tok) | 源 | 现状→36B | 为什么 |
|---|---|---|---|---|
| 自然 zh web | ~16B | web_hq(fineweb2) 扩到全量 + **CCI3-HQ**(22-24B, 已下 33.9GB) + SkyPile zh / TeleChat-PTD 切片 | 现 ~1.4B → 16B | 真实中文网页=自然文本核心。CCI3-HQ 已污染/去重验证过（0.32x web_hq）、sha 钉死 |
| 自然 zh wiki | ~1.5B | 现有 wiki 域扩（zh wikipedia） | 0.23B → 1.5B | 实体/结构，人写 |
| 自然 en | ~5B | **真 en**：WanJuanCC(46B) 或 Ultra-FineWeb en 或 fineweb en 切片 | 现 en 名义 0.16B 实际 85% 中文 | **修 en 域**：现在 en 是 cosmopedia 冒充的，得换成真英文自然文本 |
| 合成 math | ~4B | 现有 mathbank + NuminaMath/MetaMath(已审) | 现 ~0.08B → 4B | 数值/步骤推理，需保留 |
| 合成 code | ~4B | 一个真 code 源切片 | 现 ~0.06B | 程序文本结构 |
| 合成 CoT/en-math | ~3B | en math/CoT 小包(7B 可选) | 0 → 3B | 迁移赌注，zh:en 兼容 |
| **textbook/cosmopedia 封顶** | ~3.5B(~10%) | opencsg chinese-cosmopedia | 现 1.7B(53.8%)→ **3.5B(**~10%**)** | **关键组合改动**：模板化合成文本**压到 web 之下**（SmolLM2 用 Cosmopedia ~11% 的理据；假设它拖表达能力）。现有 2.26 nat 差距=rec可预测，应减权 |

合成占比从现 ~54% 压到 ~32%；natural:synthetic 从现 ~46:54 反转为 ~68:32。**zh:en 用真 en 修到 ~84:16**（现 en 域是脏的）。

## 2. 36B 跨源去重（工程主矛盾，`harness run dedup` 第 4 步）

现有 near-dup O(n²) 在 36B 跨源不可行。设计：**exact content-hash O(n) + MinHash 带状 LSH**（build_corpus 内嵌 MinHashLSH 128-perm/16-band），shard 并行、跨源分桶→候选 verify Jaccard，近 O(n)。优先对：CCI3-HQ vs fineweb2（不同上游）、SkyPile vs fineweb-edu（**SkyPile 是它上游，最可能 embarrass**）。
- **`harness run dedup --domains <a,b,c>` → `scripts/dedup_corpus.py`**（de 第 4 步，a535cca 包裹）。**全局跨源 pass**，dedup 后 clean、score 前（打重浪费贵步）。输出 `data/dedup/dedup_manifest.json`（doc IDs 标记重复 + 重复自哪个源，mix/训练查 manifest 跳过）+ `dedup_stats.json` 带 `dedup_fp`（算法+参数哈希，算法变则变）。CPU-only、按域续跑。

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
