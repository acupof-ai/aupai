---
question: "3.6B -> 36B 预训练语料：源组合、跨源去重、register 迁移的打分——语料半的设计（不是把当前权重放 10x）"
status: recorded
source: "aupai-fb hypothesis (discrimination/generation gap is composition not volume); multilingual.json supply; measured 2026-08-30"
---

# 36B 语料计划（语料半）

目标：3.6285B → ~36B。36B/172.51M 非嵌入参数 = 209 tok/param，仍约 30x 低于现代实践。供给不是约束（9.73TB 都在）；**约束是 36B 规模下的打分与跨源去重，不是抓取**。

## 组合假设（fb，待阶梯验证，不据此排工）

当前 53.8% chinese-cosmopedia（模板化合成，loss 2.79 vs web_hq 5.05，**2.26 nat 差距**）——语料中最易预测的文本。假设：判别/生成 gap 是**组合问题而非体积问题**；10x 同 mix 提知识不提表达能力。阶梯的 six-point loss + lambada 开放/双向 gap 测它：gap 随 D 收窄则体积修复；平台则组合。
**若成立，36B 应把组合移向自然文本，而非复刻当前权重。**

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

## 2. 36B 跨源去重（工程主矛盾）

现有 near-dup O(n²) 在 36B 跨源不可行。设计：
- **exact**：全源 content-hash（现有 exact_key），O(n) 流式。
- **near-dup**：MinHash **带状 LSH**（build_corpus 已内嵌 MinHashLSH 128-perm/16-band），跨源分桶——shard 并行，每 band 一张表。36B/文档 ~10^9 级：按 100M doc 分片 + banded 表并行进，跨源桶碰撞即候选、verify Jaccard。降到近 O(n)。
- 源间重复的优先对：CCI3-HQ vs fineweb2（不同上游，但都抓中文网页）、Skypile(是 fineweb-edu 上游) vs 别的。
- 和 de 协调 harness `clean` 步的去重子步（harness run clean 契约、指纹输出）。

## 3. "scoring" 在 36B 该指什么

现有质量头在 cosmopedia 上打低于原始 web——**跨 register 不迁移**（facts/data_quality.json），不照它规划。44 拥研究问题。管线实际跑什么（我拥）：
- 不依赖单一 curriculum 头。分段打分：**register 感知**——一个 register 一档（web/散文/百科/wiki/代码/数学），各 register 内用对应判据，不全局一刀。
- 或者把"打分"退化为三层：格式/编码闸（现有，precision 高）+ 跨源去重 + 组合权（mix 定，不用头）+ 一层 register 校准（44 研究定）。打分的目标从"评单文档质量"改为"评 register 能否被当前模型学"——即选择性更高 LOSS 的自然文本，而非 cosmetic。
- 36B 的 scoring 是否还需 GPU 头，取决于 44 的最新研究；管线留接口（harness run score 可换后端）。

## 4. harness 契约（与 de 协同，一条管线）

所有步走 `harness run {fetch,clean,score}`，同 run pretokenize 契约：check 红即拒、写 exp 行、产物钉指纹、可重跑。我供实质（源/组合/去重/scoring 逻辑），de 供 step（harness 集成/缓存/增量）。fetch+正则清洗是 CPU，GPU 先按 44 的 register 研究再定。

## 未决（需测/需协调）
- CCI3-HQ/SkyPile/TeleChat 交叉去重率（入库时 MinHash）。
- natural 权重上移是否真的收窄 lambada open/双向 gap（阶梯 + 36B 早期点测）。
- textbook 压到 ~10% 的具体 -keep/截断规则（面板/44 定）。
- en 真源选 WanJuanCC vs Ultra-FineWeb vs fineweb（可达性/许可）。
