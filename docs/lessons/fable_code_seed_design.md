---
question: "Fable 生成的 code-with-tests 语料：种子分布设计（多样性是真正的风险，不是正确性）"
status: recorded
source: "aupai-fb 2026-08-30：code 用 Fable 生成、沙箱执行作独立信号、核心是选择非生成"
---

# Fable code-with-tests 种子分布设计

目标：code+math 能力。T回来生成 code（题目+解+测试），沙箱只留通过的。**风险是多样性不是正确性** —— 自由生成塌缩到少数模式（fizzbuzz/二分），执行过滤拦不住（一万段通过的雷同代码执行率 100%、教不会任何东西）。种子空间决定多样性，比生成量重要。

## 设计原则：用真实作种子，不"写个函数"

生成从**真实种子**出发：
1. **真 API/库签名** —— stdlib（repeatertools/itertools/functools/datetime/re/collections）+ 常见包（numpy/datetime 风格）的真实函数签名，题 = 在签名约束下实现/修补/h接。
2. **真问题陈述** —— LeetCode/HackerRank/Codeforces 风格**但构造改题**（换数字/条件/边界、拼装两题、改输入域）避免记忆 + 过 code_holdout_500。
3. **真 bug/错误场景** —— 崩溃（IndexError/KeyError/除零/类型错）、超时（死循环/二次复杂）、边界（空输入/单元素/负/极值）、竞态/状态污染 —— 题 = 修这个 bug。
4. **真数据处理** —— 从我们已有 JSONL 抽取真实数据做题（排序/过滤/聚合/映射，输入域真实），非合成数字。

## 多样性分布（语言/难度/主题）

- 语言：**Python 为主（~80%）**，少量 C++（~15%）/JS（~5%）（C++ 的 STL/template 是 Codeforces 主流）。
- 难度分层（~权重）：简单热身（15%）/ 中等算法（45%）/ 难构造+复杂（30%）/ 极难竞赛（10%）。
- 主题带权（从 LeetCode pattern 加权 + 我们 code_holdout 未覆盖的）：
  arrays/hashing(20%)  two-pointers/sliding(10%)  string(10%)  linked-list/tree/图(15%)
  DP(12%)  greedy(8%)  math/number-theory(10%)  sort/binary-search(8%)  stack/queue(7%)
  模拟/实现(10%)
- 种子模板数：**~500 真实种子签名/问题骨架**，生成时按模板组合（题→多解→多测试变体），防"同一种子生成同质题"。
- 每题的测试集：1 黄金解 + 3-8 测试（含边界/极端/随机），测试需要独立于实现（输入→期望输出的 example-based，非比"答案字符串"）。

## 注入的多样性防塌缩度量
每个种子模板的生成配额 cap（防单一模板过生成）；生成的 prompt 多样性用 n-gram/结构哈希测（若同模板的下游题目高相似 → 降该模板配weight + 换组合）。**多样性门 = 每批 code_holdout pass + 模板熵不塌缩**。

## 与 44/沙箱/管线配合
- 生成 → `sandbox_exec.run_sandboxed`（44，硬前提）→ 只留 rc==0 且实测与黄金解一致的。
- 每批过 `code_holdout_500`（生成污染 = 模型主动召回，比爬取更需要）。
- 全部走 harness（fetch/clean/score/dedup，含 filters_fp + 去重）。

## 种子分布先给数（待 fb Fable 指定后跑 1000 探针验证）
- 先 100 真实种子签名人工审（我提供清单）。
- #4 通过率 probe：1000 生成 → 沙箱 → 我报 λ，进 #3 成本。
