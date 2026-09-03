---
question: SMELT 与 DeepLoop 对我们（200M–500M、KDA+MLA+AttnRes、30B token 预算、H20）是否可用、什么时候用、收益和成本各是什么
status: open
source: arXiv 2609.01343 (SMELT, 2026-09-01) + arXiv 2607.13491 (DeepLoop, 2026-07-15) 原文 2026-09-03 取;repo 实测见 facts/smelt_deeploop.json 与 facts/efficiency.json
---

# SMELT 与 DeepLoop:可执行判决

**判决:现在不做。500M@20B 落地后做一次纯 looping A/B(DeepLoop 的 α/β 缩放 + 中间半层 loop 2×,不带 MoE);MoE 部分留到 token/param > 40 之后。A/B 之前必须先测的第一个数是 AttnRes 跨循环的 O(L²) 成本——那是硬约束,测不过就整条线不做。**

用户给的链接(2607.13491)是 DeepLoop,不是 SMELT;SMELT 的真实 ID 是 2609.01343。两篇都做,DeepLoop 是第二个正主不是背景文献。

## 0. 第一性:两个交易在合成什么

| 脉络 | 交易 | 买的东西 |
|---|---|---|
| MoE(Shazeer 2017 → Switch 2021 → Mixtral/DeepSeekMoE) | 参数涨、FLOP 平 | 每 FLOP 的参数量(知识容量) |
| Looped(Universal Transformer 2018、ALBERT、Mixture-of-Recursions) | 参数平、FLOP 涨 | 每参数的有效深度(计算深度) |

两个方向相反。SMELT 把它们放在同一条 compute-matched 曲线上:固定 per-token FLOPs + 参数量 + KV cache 三者,looping 买深度,收窄 hidden 补 FLOP,加专家补参数。**它在权衡的是:同样的 FLOP 预算,深度(loop)和宽度(expert 参数)怎么分。** 它的答案:中间一半层循环两次 + top-8 稀疏专家,在 compute-optimal frontier 上省 6.8–18.0% 训练 FLOPs(smelt.ce_gain)。DeepLoop 不做这个权衡——它是纯 looping 的稳定性修法,回答"loop 怎么才不炸"。

## 1. 四个问题

**什么时候好用(论文报告值)。** SMELT 的证据在 10²⁰–10²¹ FLOPs(600M–1.6B active,205B stable tokens,TPP 56–91),收益随规模涨(每 10× 算力 +8pp),代码(CE gain 20.4%)和长上下文(长桶收益 1.52×)迁移最强。DeepLoop 在 GPT-2 small/medium(12/24 层,与我们 L12 同构)R=3 即有效:-0.016 nats(small)/ -0.015(medium),R=7 到 -0.028(medium)。

**我们现在能不能用(我们实测 + 外推)。** 不能直接用,三个不匹配:
1. **规模**:200M@4B ≈ 2×10¹⁹ FLOPs,在 SMELT 拟合窗口(1.3×10¹⁹–2.2×10²¹)的下沿之外;其 10²⁰ 行 CI 下界已到 1–4pp,外推到我们规模点估计几个百分点、下界近零。
2. **数据**:我们 20 tok/param(200M)、40(500M),SMELT 的 compute-optimal TPP 是 56–91。MoE 买参数,参数在数据不足时买不到东西——在 20–40 上专家欠训练。
3. **架构**:KDA 带循环状态,论文明确没覆盖(smelt.stateful_architectures);AttnRes 的 O(L²) 在 loop 下平方涨(见 §2-Q1)。

**收益是什么(论文报告值)。** SMELT:等算力下 loss 更低,等 loss 下省 6.8–18.0% FLOPs(10²⁰–10²¹);DCLM Completion 96/96 配对全胜;机制是 attention sink 在第二次访问时近消失、质量重定向到内容 token。DeepLoop:R=3–7 省 0.016–0.028 nats,下游 7/8 任务胜。两者都单 seed(DeepLoop 作者自己声明)。

**成本是什么(我们实测 + 推算)。** 见 §2 五个 repo 专属问题。四个成本项,全部有我们自己的数:AttnRes 跨循环墙钟 L12 26–35%、L32 47–57%(两变体,见 §2-Q1;推算,未在 loop 模型上测);MoE all-to-all 在 H20 上是一步的 1.63–1.86%(L12)/4.35–4.97%(L32)下界,且串行暴露不能像 DDP allreduce 那样叠进 backward(repo.moe_a2a_cost_h20);权重重读可忽略(0.005%),真实代价是 1.22× 激活流量加在已饱和的 30.6% elementwise 上(repo.loop_weight_reread_verdict);KDA 状态跨访问语义论文没答,自测。

## 2. 五个 repo 专属问题

**Q1(最优先):DeepLoop 的残差缩放修法 vs AttnRes,是同一问题的两种解法还是可叠加?**

可叠加,不是二选一——但叠加的代价是稳定性证明失效。DeepLoop 管**写**:共享权重的多次访问把梯度聚合成一个更新,残差流被相关更新 inflate,修法是初始化增益 β=(8N)^{-1/2} + skip scale α=(2N)^{1/2},把分支相对残差流压到 1/(4N)(deeploop.kappa_and_exponent)。AttnRes 管**读**:残差求和无结构,修法是学一个 pseudo-query softmax over 所有前面子层输出。一个管写尺度、一个管读权重,作用点不同。

但有两个真实交互:
- DeepLoop 的 κ_R 分析假设标准残差流;AttnRes 把残差流换成 softmax 加权和,bound 不直接覆盖。叠加要自己测稳定性。
- AttnRes 的"前面的子层"在 loop 下有一半是同一权重的上一轮,两种读法都不免费(repo.attnres_x_loop_corrected)。**跨循环**:attend 所有执行过的子层,read 数 650→1406(L12,2.16×)/4290→9506(L32,2.22×;数的是 reads,宽度因子另算——SMELT 收窄 d 配 FLOPs 时字节比再 ×0.816,换 GQA 配 KV cache 时宽度不动,两个量不能焊死)。墙钟不能拿占比当速率乘(6e 2026-09-03 退订):share' = kA/(kA+R'),分母含 AttnRes 自己。固定宽度(我们 A/B 的设计)L12 30–35%、L32 52–57%;SMELT 式匹配宽度 L12 26–30%、L32 47–52%。**按物理层**:循环对 AttnRes 不可见,比 dense 还便宜——但这是更坏的结果:looping 买来的深度落在 AttnRes 不读的残差流上,SMELT 自己报告的机制(attention sink 第二次访问消失)就发生在那条流上。哪种读法成立是架构决策,不是测量性质。**跨循环读法的推算区间已压在 30% 门线两侧(L12)和远超门线(L32):A/B 前的测量从"会不会超"变成"定哪种语义 + 实测落在区间哪端";语义定为跨循环且实测 >30% 墙钟,整条线停。**

**Q2:我们是 token 受限还是参数受限?** 200M@4B = 20 tok/param,在 Chinchilla 最优点上;500M@20B = 40,偏 token 受限。MoE 的最优 TPP(56–91)远高于 dense(~20),因为每参数只被部分 token 训练。在 20–40 上 MoE 专家欠训练,收益曲线的甜区在我们当前预算之外。**门槛的来源说清:判决里的"TPP>40"是我们这侧的现状上界(500M@20B = 40),不是论文的下界折算——论文的 56–91 是在 10²¹ FLOPs 处测的甜区,两个数不是一个量。40 的意思是"越过我们现在到过的最远位置再评估",不是"论文说的最低条件";论文没有给出 MoE 收益的最低 TPP。**

**Q3:MoE 与 KDA + gated MLA 的兼容性?** 结构上部分成立:专家放 FFN 是常规做法,不碰 KDA 状态(KDA 状态在 DeltaRecurrence 内部,从不穿过 block 接口,model.py:6)。但 looping 中间层时同一 KDA 层被访问两次,**状态是接着算还是重置,论文没答**(smelt.stateful_architectures)——这是我们要自己付的成本,且答案影响数值正确性,不只是性能。

**Q4:H20 上的实际成本?** 我们的步:50.8% GEMM、30.6% 逐元素撞带宽屋顶(实测 1.30× 屋顶理想值)、10.6% KDA、2.3% nccl(repo.tpp_and_step_profile)。四项成本,de 2026-09-03 全部从我们自己的 trace/拓扑推完(repo.nv18_topology_measured / repo.moe_a2a_cost_h20 / repo.loop_weight_reread_verdict / repo.attnres_x_loop_corrected):
1. **MoE all-to-all**:NV18 全互联 450 GB/s/方向(实测拓扑、额定速率,非实测 achieved)。字节/带宽定价:L12 1.63–1.86%、L32 4.35–4.97% 一步(N=4/N=7),按 50% 链路效率 L32 到 8.7–9.9%;permute 项 0.98%(L12)/2.61%(L32)落在 elementwise 类。结构性的点不是数字:DDP allreduce 的 2.3% 已叠在 backward 里,dispatch→expert→combine 串行,同样的百分比是完全暴露的。这是下界——机器上从没跑过 all-to-all,无效率因子。per-collective 成本模型(4.6–12.8%)已被我们自己的 bucket A/B(50 vs 25MB = 75K vs 75K tok/s/gpu)证伪(repo.collective_count_cost_refuted)。
2. **权重重读**:循环层权重不进 L2(L12 1.40×、L32 5.6× 于 60MB L2),第二次访问从 HBM 重读;但只占一步的 0.005%——seq 4096 下权重比激活流量小三个数量级。缓存理由不成立,重读本身确实便宜。
3. **激活流量(真实代价)**:1.5L 执行 × 0.816 宽度 = 1.22× dense 激活流量,落在 elementwise_cast——实测 497.24ms、30.6% 步、已 1.30× 屋顶,该项推到 ~37.5%。
4. **AttnRes**:见 Q1,两读法 26–35%/47–57%(区间,方法见 Q1)。
SMELT 用 GQA ratio 调 KV cache 到 ±4%,我们的 MLA KV 本来就小,这一项成本低。

**Q5(6e 提):"中间一半层"这个位置选择对 L12 成立吗?** SMELT 自己的消融:loop 跨度 0/2/4/6/8/10/12 层,最优点在 6(50%),全栈循环回退(1.9322 vs 1.9257)。理由是浅层做特征、深层做输出、中层做迭代精化。L12 有中间 6 层可循环(SMELT 的 200M 尺度就是 L12→18),位置选择在我们深度上成立——但我们的 12 层是 9 KDA + 3 MLA(attn_every=4,MLA 在 4/8/12,train.py:162),中间 6 层(4–9)的 KDA/MLA 构成和 SMELT 的全 dense 不同,循环哪 6 层要按我们的层类型重选,不能照搬"中间 6 层"。

## 3. 落地路径(如果判决执行)

1. **先测 Q1 的数**:L12 上搭一个 loop 2× 模型,实测 AttnRes 跨循环的墙钟成本(fused kernel 3ed56306),同时定掉跨循环/按物理层的语义。推算区间 26–35%(L12)已压在 30% 门线上——测量的任务是定语义 + 定位落在区间哪端,不是探边界;语义定为跨循环且实测 >30% 墙钟,整条线停。
2. **500M@20B 落地后**:纯 looping A/B——DeepLoop α=(2N)^{1/2}/β=(8N)^{-1/2} 初始化 + 中间半层 loop 2×,不带 MoE。对照 = 500M dense 臂。预登记出口:val NLL 差 ≥ 2.28σ(我们的噪声带)且退化率不降。
3. **MoE 留到 TPP>40**:即 500M 训过 20B 之后的下一个规模决策点;届时 all-to-all 成本先在 H20 上微基准。
4. DeepLoop 的 α/β 是零成本的默认初始化——只要上 looping 就用它,不需要额外实验。

## 4. 来源类型声明

- 论文报告值:SMELT 全部数字(arXiv 2609.01343v1,2026-09-03 取),DeepLoop 全部数字(arXiv 2607.13491v2,2026-09-03 取)。两篇都晚于知识截止,先验零,全部从原文取。
- 我们实测:AttnRes read 律 n(n+1)、n=2L+1(gate_failure_shapes.md:833)、H20 步剖面(facts/efficiency.json:2361)、tok/param(data_scaling_design.md:9)、AttnRes 13.5% 吞吐成本(eff.attnres_internal)、NV18 全互联拓扑(repo.nv18_topology_measured)。
- 推算(de 2026-09-03,facts/smelt_deeploop.json,全部从我们自己的 trace/拓扑推,无论文数字):all-to-all 步占比下界(repo.moe_a2a_cost_h20)、权重重读 0.005% 与激活 1.22×(repo.loop_weight_reread_verdict)、SMELT 宽度匹配形状(repo.smelt_shape_correction)、per-collective 成本模型已被我们自己的 bucket A/B 证伪(repo.collective_count_cost_refuted)。
- 推算(44):loop 后 read 数 2.16×/2.22×、固定宽度墙钟区间 30–35%/52–57%(repo.attnres_loop_cost_derived,从实测 read 律 + share 公式推,未在 loop 模型上测)。
- 裁决(6e 2026-09-03):占比不是速率,share' = kA/(kA+R');naive 乘法(42.6%、de 的 34.8%/68.4%)已退订,能写的只有区间。
- 同行报告未入库:AttnRes 19.7%/37.8% 墙钟(tilerl 2026-09-03)、AttnRes softmax 12% bf16 误差。
- 两篇都是单 seed(DeepLoop 作者声明;SMELT 消融同),收益数字的 run 间方差未测。
