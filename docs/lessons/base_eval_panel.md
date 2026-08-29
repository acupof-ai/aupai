---
question: "第一次 pretrain 之前定稿的 200M 基座评分面板:每个指标能分辨多大差异、需要多少样本、什么条件下给假信号"
status: recorded
source: "project measurements 2026-08-30 + stats; companion to docs/lessons/base_eval_at_200m.md; facts in facts/base_eval.json"
---

# 200M 基座评分面板(预注册,定稿后冻结)

定稿于第一次 0830v1 pretrain 之前。之后不许改指标——事后挑指标是上一轮最烂的地方。
要改只能发新版本 + 书面理由。架构前提:KDA + full causal MLA + AttnRes 默认开,
滑窗已删(b3cad87),eval attention == training attention(此前 infer_local 跑的
注意力比训练宽,生成式 eval 因此被污染;似然 eval 在短上下文上不受影响,面板冻结的是修正后的状态)。

## 面板(似然优先)

| # | 指标 | 类型 | 地板 | 样本量 | 200M 分辨力(80% power) | 假信号条件 |
|---|---|---|---|---|---|---|
| 1 | 分域 NLL(loss) | 似然 | 无(值 3-5 nat) | 每域 ≥10M token,按文档分块 | Δ ≈ 2·SE_block;16× 阶梯实测动 1.66-1.99 nat,分辨力远小于此 | 训练/eval 重叠(污染);tokenizer 漂移;FoNE value channel 关闭时数字 NLL 不可比;eval 集太小且 token 自相关(σ/√n 高估分辨力);checkpoint 间域配比漂移 |
| 2 | 中文最小对(CLiMP 16 范式) | 似然二选一 | 50% | 1000 对/范式(共 16000) | 4.4 pt/范式 | BPE 跨编辑点合并(skip 率 >5% = 构造 bug,我们首稿 22%);标签错误(人工一致率 <85% 的范式整组丢弃,CLiMP 先例);swap 测试不反转;长度/词频混杂;单范式 n=1000 看不见 <4.4pt 的运动 |
| 3 | LAMBADA-zh 末词预测 | 似然(受限目标) | ≈0 | 1000 题 | p=0.3 时 3.6 pt | 目标多 token 分词;词频效应;上下文记忆;英文 LAMBADA 测中文模型(直接禁用) |
| 4 | math-hard v2 似然孪生 | 似然二选一 | 50% | 1080 对(现量)→ 5000+ 才到 4pt | 现量 8.4 pt(欠功效);5000 对时 3.9 pt | 对错解长度不等(必须长度匹配);数字 BPE(0830v1 非 FoNE,15/17 分词可能不同,逐对 token 对齐);错解质量(必须是 plausible 的错解,不是垃圾解);gold 正确性(construction-from-answer 已保证) |
| 5 | ceval MC | 似然 4 选 1 | 25% | dev 全集 | z>2 才算信号;200M 上预期 z≈0-1 | 英文 MC(已移出面板);随机涨落(z 已量化);中文暴露量 |
| 6 | 生成式(math-500、v2 pass@1) | 生成 | ≈0 | — | **不是基座指标**。基座上是已知的零,记零等于"四个零和测过了长得一样"。仅 SFT 后启用 | 指令格式失配(基座没有格式可服从);退化输出(165 冒号循环)被读成难度 |

## 每条指标的已知答案对(known-answer pair)

没有 60pt 以上双读数的指标不进面板(repo 规则,38af944):

1. 分域 NLL:同一 checkpoint 在训练域 vs 留出域外文本上的 NLL 差 >1 nat(域外必然更差);swap 不适用。
2. 最小对:正确标签 vs 交换标签,最强 checkpoint 上差值 ≥60pt(预期 80%+ vs ~20%)。
3. LAMBADA-zh:末词预测 vs 随机同长度词,差值 ≥60pt。
4. v2 似然孪生:正确解 vs 错解,最强 checkpoint 上 ≥60pt;swap 必须反转。
5. ceval:z 值,最强 checkpoint 上 z>2;z<2 即绊线未触发,不读成能力。

## 分辨率判定(阶梯跑完后)

六个 checkpoint(0.2b→3.24b,16× 数据)上,指标的 best-worst 跨度:

- **<10 pt:无分辨率,砍掉**,不留着凑数。
- 10-30 pt:弱分辨率,只报趋势不报点值。
- ≥30 pt:全分辨率,报点值。

分维报,不只报总分——一个维度饱和了另一个还在动,合并会把它抹掉。

## 样本量公式(附录)

- 二项指标(最小对、似然孪生、LAMBADA、MC):SE=√(p(1-p)/n),50% 地板时 SE=0.5/√n。
  80% power、双侧 α=0.05 可分辨 Δ:n=((1.96+0.84)·0.5/Δ)²。Δ=0.02 → 4900;Δ=0.044 → 1000。
- NLL:按文档分块 bootstrap,SE_block=σ_block/√(n_blocks),可分辨 Δ=2·SE_block。
  token 级 σ/√N 因自相关高估分辨力,禁用。
- MC 的 z:z=(acc-chance)/√(chance(1-chance)/n),|z|>2 才报。

## 冻结清单

- 指标 6 个,样本量如上,已知答案对如上,分辨率判定如上。
- 第一次 pretrain 之后:只允许加新版本(v2 面板),不允许改本版的任何数字。
- 例外条件:某个指标的 known-answer pair 在最强 checkpoint 上达不到 60pt → 当场砍,这不算改面板,是面板自己的执行规则。
