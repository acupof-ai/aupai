---
question: "固定算力下怎么换更多能力:MFU 31% 是不是太低、加参数还是加数据、混合架构的成本账、便宜的吞吐优化"
status: open
source: "e1 literature review 2026-08-30; all numerical claims in facts/efficiency.json (referenced as #<id>)"
---

# Architecture efficiency at 200M

现状:166M non-embedding,12 层 (3 KDA + 1 gated MLA)×3,d=1024,seq 4096,FP8,8×H20,31% MFU / 73K tok/s/gpu (facts/efficiency.json#eff.fb_mfu)。每条结论给出动作和依据;文献只在 1B+ 测过的标 **[extrapolation]**。

## P0 — MFU 31% 是不是太低,瓶颈在哪

**结论:31% 无法从文献判定高低——H20 上 200M 级 dense 训练 MFU 没有公开数字(未找到出处)。但瓶颈位置可以从代码路径直接推断,不需要文献。**

- 未找到出处:所有 H20 公开 benchmark 都是推理(TensorRT-LLM、SGLang serving),没有 200M 级训练 MFU (facts/efficiency.json#eff.h20_mfu_200m)。没有同硬件 dense 对照,31% 这个数本身不可解释。
- 代码路径给出的嫌疑链:训练时 KDA 走 fla 的 Triton chunk_kda kernel;FlashKDA CUTLASS kernel 只用于推理(train.py:225 明说 "unblocks FlashKDA at inference, zero training cost");dense 模型同硬件走 cuBLAS/CUTLASS (facts/efficiency.json#eff.kda_kernel_path)。**Triton 对 CUTLASS 的差距是 MFU 差距的首要嫌疑,但未实测。**
- 另一个已知损耗:torch.compile 在 AttnRes 的 .chunk() 路径上是关的(train.py:351)(facts/efficiency.json#eff.kda_kernel_path)。

**动作(按成本排序):**

1. **batch 32→48/64 扫一遍**(分钟级)。batch 32 no-ckpt = 90K tok/s,batch 72 OOM,48–64 区间从未测过 (facts/efficiency.json#eff.batch_ceiling)。这是零代码、零风险的第一杠杆。
2. **profile 一次**(小时级)。用 torch.profiler 或 nsys 抓 10 步,看 chunk_kda 占 step time 的比例。如果 KDA kernel 占比 >50%,MFU 差距就是 kernel 差距,不是架构开销。
3. **把 profile 结果给 de/3b**,问 FlashKDA 训练 kernel 的可行性——这是唯一可能把 31% 拉到 dense 水平的改动,但属于 kernel 工程,不是调参。

**不做的事:** 不要为了追 MFU 去改架构(架构已冻结);不要在没有 dense 对照的情况下宣称 31% "太低"。

## P1 — 固定算力:加参数还是加数据

**结论:加数据(或重复数据),不加参数。你的数据指数 β=0.555 远陡于文献的参数指数 α≈0.34,固定算力下数据的边际收益更大。**

- 你的 6 点拟合:L(D)=2.157+0.842/D^0.555,β=0.555 (facts/efficiency.json#eff.fb_data_curve)。Chinchilla 的联合拟合 α=0.34、β=0.28 (facts/efficiency.json#eff.chinchilla_alpha)。你的数据指数是文献的 2 倍——在 200M/中文/KDA+MLA 这个点上,数据比参数值钱得多。
- 数据-约束前沿也指向同一方向:Muennighoff 的 R_N*=5.3 < R_D*=15.4,重复数据比多余参数衰减慢,前沿选择"更小模型+更多 epoch" (facts/efficiency.json#eff.chinchilla_alpha 的姊妹事实在 data_scaling.json#ds.muennighoff_size_epoch_interaction)。
- 你只剩 0.43 nat 的数据红利(E=2.157 vs 当前 2.587),但重复到 4 epoch 能把有效数据从 19.5 拉到 ~71 tok/param,成本 ~0.5%——这是 data_scaling_design.md 的 P0,不重复。

**如果一定要跑参数曲线**(比如 4-epoch 实验失败、需要买参数):点取 **83M / 166M / 332M non-embedding**(2× 几何,3 点)。理由:3 点恰好识别 L(N)=E+A/N^α,零残差自由度,只能验证不能证伪;要证伪得 5 点,但 5 点的算力够把 4-epoch 实验跑两遍。方法学上,Meta 的混合架构 scaling 论文在 100M/350M/1B/3B × 5 算力预算上用了 **Trapezoid scheduler + 复用大预算 run 的中间 checkpoint**(WSD 式一读多) (facts/efficiency.json#eff.meta_hybrid_scaling_method)——和 data_scaling_design.md P1 的校准方案一致,已在 100M–3B 验证过。

**动作:** 先跑 4-epoch(已在跑),不跑参数曲线。4-epoch 成本 >2% held-out loss 才考虑买参数。

## P2 — 混合架构的成本账

### KDA:MLA = 3:1 有没有依据

**有。3:1 = attention:linear = 1:3,正是 Jamba 实测过的比例。**

- Jamba 发布比例 1:7(4 attn + 28 Mamba),但 1.3B/250B token 的消融里 **1:3 和 1:7 质量几乎相同**(HellaSwag 37.2/37.2,WinoGrande 65.1/65.1,NQ 61.7/61.7,C4 -0.533/-0.533),两者都 beat pure Attention 和 pure Mamba;1:7 只是等质量下更省算力 (facts/efficiency.json#eff.jamba_ratio)。
- Meta 的系统研究(350M/1B,60B DCLM):1:1 质量最好,~1:5 是质量+效率最优;attention 块必须在中后层,放前层会显著掉点 (facts/efficiency.json#eff.meta_hybrid_ratio)。你的 (3 KDA + 1 MLA)×3 是均匀交错,MLA 在第 4/8/12 层——符合"中后层"原则。
- 边界:Jamba 是 1.3B(你的 8 倍),Meta 最小消融 350M(你的 2.1 倍)——**[extrapolation]**;两者用的是 Mamba/SWA,不是 KDA,但比例问题在 linear-attention 家族内是 primitive-agnostic 的。

**结论:3:1 不需要改,它落在两条独立文献曲线的交点上。**

### AttnRes 的算力税

**论文声称可忽略,但只在 48B 上验证过;你内部从未测到——两次 A/B 都 OOM 了。AttnRes 目前是"按决策进配方",不是"按测量进配方"。**

- 论文(arXiv 2603.15031):Full AttnRes 在普通训练里不增显存(和反向传播激活重叠),计算开销"tiny",推理延迟 <2%;Block 变体把 O(Ld) 降到 O(block) (facts/efficiency.json#eff.attnres_paper)。
- 验证规模:Kimi Linear 48B total / 3B active,1.4T token。**没有 sub-1B 测量** (facts/efficiency.json#eff.attnres_paper 的 boundary)。
- 内部:两次 A/B 都 OOM,从未拿到 step-time delta (facts/efficiency.json#eff.attnres_internal)。OOM 是显存现象,论文的设计预测计算税很小,但"预测"不是"测量"。

**动作:** 要测就用 **Block AttnRes + 小 batch**(显存 O(L·d)→O(block),绕开 OOM 原因),拿 step-time delta。不测就承认它是决策不是测量,在报告里写明。**不要**用 Full AttnRes + grad_ckpt 再试第三次——grad_ckpt 下显存 O(Ld) 增长,正是前两次 OOM 的疑似原因。

### tied embedding 要不要解开

**不要。naive 解开在 200M 上是亏的;Leviathan 式解开能赢但 200M 上代价 26% step overhead,不划算。**

- Leviathan(arXiv 2601.22040):naive untied 在 8.4B token 内全程输给 tied,尽管多 ~50% 参数 (facts/efficiency.json#eff.leviathan_tied)。
- Leviathan(连续输入生成器 + untied output head)在 200M 上赢 tied(3.052 vs 3.073 nats),但 **26% per-step overhead**(净 18% wall-clock 惩罚),crossover 要 ~3B token,且 LAMBADA PPL 在 200M 上**差 33%** (facts/efficiency.json#eff.leviathan_tied)。
- 你的 tied 省 33.6M 参数,FLOPs 不变。解开 = 多 20% 参数换一个在 200M 上 mixed 的结果。

**结论:保持 tied。** 这是 P2 里唯一有明确负面文献答案的问题。

## P3 — 便宜的吞吐优化

| 改动 | 成本 | 预期收益 | 依据 |
|---|---|---|---|
| batch 32→48/64 | 分钟级 | 未知,但 48–64 是唯一未测区间;batch 32→72 之间必有最优点 | facts/efficiency.json#eff.batch_ceiling |
| profile 定位 KDA kernel 占比 | 小时级 | 决定 MFU 上限是 kernel 还是架构 | facts/efficiency.json#eff.kda_kernel_path |
| FlashKDA 训练 kernel | 周级(kernel 工程) | 唯一可能把 31% 拉到 dense 水平的改动 | facts/efficiency.json#eff.kda_kernel_path |
| torch.compile 重开 | 天级 | 未知;.chunk() 路径上被关,可能是 AttnRes OOM 的 workaround 残留 | train.py:351 |
| grad_ckpt | 已测 | -25% wall-clock 换 ~15GB 显存,当前 batch 32 不需要 | facts/efficiency.json#eff.batch_ceiling |

**动作顺序:** batch 扫 → profile → 拿 profile 结果决定 compile 还是 kernel。前两步今天能做完。

## 本次报告的具体收益(不是描述做了什么)

1. **P0 去伪:** "31% 太低"这个前提无法从文献证实(未找到出处),真正的嫌疑是 Triton chunk_kda vs CUTLASS——给了一个分钟级动作(batch 扫)和一个小时级动作(profile),不需要再读论文。
2. **P1 定方向:** β=0.555 vs α=0.34 → 加数据不加参数;参数曲线只在 4-epoch 失败时才跑,点取 83/166/332M。
3. **P2 清账:** KDA:MLA 3:1 有 Jamba 1:3 实测背书,不改;AttnRes 税从未测到,要测就用 Block 变体绕开 OOM;tied 保持,naive 解开在 200M 上亏(Leviathan)。
4. **P3 排序:** batch 扫 → profile → compile/kernel,前两步今天完成。

## Open items

- H20 上 200M dense 训练 MFU 的公开数字——未找到出处,只能内部跑 dense 对照或 profile。
- AttnRes 在 200M 的实测 step-time delta——两次 OOM 后未再测。
- KDA Triton kernel vs CUTLASS 的具体差距——profile 之前是嫌疑,不是结论。
