---
question: Which HumanEval base primitives does the SEG137 textbook arm actually have token dose for, and which are predicted flat by construction?
status: measured
source: runs/ab_coverage_prediction.md (genB, 2026-09-15), from scripts/ab_coverage.py keyword title-match over textbook_claude_v41_dc (1,938 rows / 8,122,669 tok); failure counts from primitive_gap_table.md (156 clean HE / 138 fail)
---

# A/B 预期强弱项：教科书池基础原语覆盖量化（零生成，纯读）

**任务**：基于已 vet 的 `textbook_claude_v41_dc`(1,938 行/8,122,669 tok)+ `primitive_gap_table.md`(HE 失败原语),量化 SEG137 A 臂教科书注入对每个 HumanEval 基础原语的 token 覆盖,预测 A/B 强弱项,供 E0/ET 数字对照判读。
**来源**：`scripts/ab_coverage.py`(pod,关键词分类,只读)。**红线遵守**:不生成、不占 GPU、不改 corpus。

## 判读口径：title-match 才是真实剂量

每个原语两个数：**title 命中**(章节 seed_topic/topic/subtopic 就是讲这个原语 = 真实教学剂量)和 **body 提及**(关键词出现在正文任意处 = 附带噪声)。body 数普遍虚高——如 palindrome body 427 行,因"reverse/symmetry"在大量无关章节里顺带出现,而真正以回文为主题的只有 5 行。**下文全部以 title-match 为准。**

## 结构性事实(最大的"为什么可能弱")

A 臂 = 教科书 weight 0.2999 → 7,888 训练行 = 1,972 可训池 × 4 epoch。但**整段 SEG137 只有 137 步 / 全程 38,207 步 = 0.36%**。即便 A 臂 30% 权重,绝对训练量极小;能否移动某原语,取决于该原语在池里占多少 token——而池的主题质量压在软件工程题材上,不在 HE 考的原子算术原语上。

## 逐原语覆盖表（fb 点名 6 个在前）

| 原语 | HE 失败题(gap failP) | title 行 | title tok | 占池 % | A臂4ep剂量 | 预测 A vs C |
|---|--:|--:|--:|--:|--:|---|
| **loop-counting**(计数/频率) | 11 (MED) | 63 | 246,844 | **3.04%** | ~987K | **A 可能强**：剂量最大 + 真失败 |
| **boundary-threshold**(谓词/边界) | ~10 (boolean MED) | 24 | 103,566 | 1.27% | ~414K | A 略强：有量、对口 |
| **fractional-decimal**(分数/小数) | 4 (**HIGH**) | 19 | 75,130 | 0.92% | ~300K | A 或略强:剂量中等、HE 高频失败 |
| **early-exit-control**(早退控制) | ~7 (filter) | 20 | 78,010 | 0.96% | ~312K | A 微弱:量薄 |
| **palindrome-symmetry**(回文) | 5 (**HIGH**) | 5 | 20,028 | **0.25%** | ~80K | **A/B 预期平**:HE 最常失败却几乎没教 |
| **geometry-poly**(几何/多项式) | 3 (**HIGH**) | 1 | 2,629 | **0.03%** | ~10K | **A/B 预期平/无信号**:池里近乎为零 |
| sorting-variant | 14 | 111 | 447,834 | 5.51% | ~1.79M | A 强但失败复合(补料帮助有限) |
| encoding-cipher | 4 | 73 | 300,788 | 3.70% | ~1.20M | A 强但 HE 失败少 |
| string-parse-split | 6 | 44 | 181,297 | 2.23% | ~725K | A 中 |
| recursion | 3 | 54 | 218,571 | 2.69% | ~874K | A 强但 HE 失败少 |
| base-conversion | 2 | 34 | 142,384 | 1.75% | ~570K | A 中,HE 失败少 |
| primality-factors | 8 | 9 | 36,529 | 0.45% | ~146K | A 弱:失败多、覆盖薄 |
| dict-mapping | 0 | 11 | 43,474 | 0.54% | ~174K | 无 HE 失败 |
| set-membership | 3 | 5 | 22,955 | 0.28% | ~92K | A 弱 |

## 结论：A/B 预期强弱项清单

**A 臂预期能拉开(E0<ET，A 更好)**：
- **loop-counting / counting-frequency**：池里覆盖最厚(3.04% / 63 章),且 HE 失败 11 题(MED)。最可能出现 A>C。
- **boundary-threshold / boolean-logic**(1.27% / 24 章,HE 10 失败)、**fractional-decimal**(0.92% / 19 章,HE 4 失败 HIGH)。有对口剂量 + 真失败,次可能。

**A/B 预期平(E0≈ET，教科书帮不上)——恰是 HE 失败最狠处**：
- **palindrome**(HE 失败 5=HIGH,池仅 0.25% / 5 章)和 **geometry-poly**(HE 失败 3=HIGH,池仅 0.03% / 1 章)。这两个 HIGH-gap 原语教科书池几乎没教,A 臂无剂量可注入 → **A/B 在 HE 最常错的原语上预计无差异**。这是 A/B 整体信号可能偏弱的核心原因:池的主题质量(sorting 111 / encoding 73 / recursion 54 章)压在软件工程题材,不在 HE 考的原子算术原语。

**教科书池几乎没有的高频失败原语**(title < 10 章且 HE 失败 ≥3)：
- geometry-poly(1 章 / 失败 3)、palindrome(5 章 / 失败 5)、primality-factors(9 章 / 失败 8)、set-membership(5 章 / 失败 3)。

## E0/ET 对照判读指引
- 若 ET 相对 E0 的提升**集中在** loop-counting/boundary/fractional 类题 → 与覆盖预测一致,A/B 机制成立但受限于池主题。
- 若 palindrome/geometry 类题 E0≈ET → **不是 A/B 方法失败,是池里没料**,别据此否定教科书路线;要动这些得定向补料(gen_exercises,须批)。
- 若整体 ET−E0 幅度小 → 归因于 SEG137 仅占全程 0.36% + 池主题错配,而非教科书无效。

**口径**：覆盖为关键词 title-match(章节主题),偏保守;A臂剂量=title_tok×4epoch;失败数引自 primitive_gap_table.md(156 clean HE / 138 失败)。池 1,938 行是本 A/B 实际引用的 `textbook_claude_v41_dc`,与 gap 表用的 5,916 行更宽池不同——本表以 A/B 实际池为准。
