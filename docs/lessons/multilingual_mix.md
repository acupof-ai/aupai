---
question: "中英文都要、不过滤、不翻译的前提下：30B 扩展的中英配比怎么定，英文 fertility 代价和迁移收益各是多少"
status: recorded
source: "literature search + project measurements 2026-08-30; 供给侧见 token_supply.md"
---

# 中英配比（P2）与 fertility 代价 / 迁移（P3）

## P2 配比

**前提修正：中文供给不构成约束。** "英文是唯一可获得体量" 不成立——ModelScope 公开中文语料 >500B token（SkyPile 150B、CCI3-HQ 100B+、Fineweb-Edu-Chinese 1.5T，见 `token_supply.md`）。真正的约束是带宽和磁盘，不是可达性。配比问题因此不是"能拿到多少英文"，而是"英文占多少还不伤中文"。

**证据状态：200M 尺度没有实测的中英配比最优点**（`facts/multilingual.json#mlm.ratio.sub1b_optimum`）。最近的三条证据：

1. 多语言诅咒在 sub-1B 真实存在：XLM-R 270M 从 7 语扩到 100 语，XNLI -4.1（`#mlm.curse.sub1b_measured`）。
2. en→zh 数学迁移在 200M 无任何实测；最近的正向证据在 1.5B（en→ko +28.5，`#mlm.transfer.enzh_math_200m`）；测试时 CoT 迁移 8B 才涌现（`#mlm.transfer.testtime_emergence`）。
3. 家族级最优：He et al.（arXiv:2410.12883）85M-1.2B 中文家族 ~0.22——但那是"多语言家族"的最优，不是"中文模型加英文"的最优，口径不同。

**建议（照做）**：30B 扩展以中文为主（≥90%），英文只加 math/CoT 小包（`token_supply.md` P1 加购，~7B token 可用，按需截取使 en 占比 ≤10%）。理由：中文供给充足且便宜；英文迁移在 200M 是无证据的赌注，下注金额控制在小包级别。

**证伪实验（已排队）**：e1 的 0/30 sweep 扩成 0/10/30 英文三臂，同一跑同时裁决配比和迁移。这是唯一能闭合 `#mlm.ratio.sub1b_optimum` 的实验。

## P3 fertility 代价与迁移

**代价（实测）**：冻结分词器英文 fertility 1.87 tok/word（`#mlm.fertility.en_project`），对原生 32k 英文分词器 1.32 是 1.42×（`#mlm.fertility.multiplier_vs_native32k`）。含义：每个英文 token 按 1.42× 计入训练预算——30B 扩展若含 3B 英文，实际消耗 4.3B token 预算。

**伤害证据（混合，未裁决）**：

- 因果正向：修 fertility 后 BPB -4.43%（`#mlm.fertility.harm_causal`，Nepali，非 200M）
- 成本正向：英文中心分词器治德文，训练成本最高 +68%（`#mlm.fertility.harm_cost`，2.6B）
- 反例：fertility 差 2.3× 与跨语言准确率不相关（r=-0.14, p=0.24，`#mlm.fertility.null_result`）

**迁移收益（无 200M 证据）**：见 P2 第 2 条。英文 math/CoT 的迁移收益在 200M 是开放问题，值 ~8h 传输的赌注，不值 30B 档的主仓位。

**与 b0 的接口**：fertility 1.87 是分词器决策的条件输入——若 b0 的分布统计支持重做分词器，1.42× 代价随之改变；分词器冻结决策落地前，英文占比的任何结论都是条件性的。
