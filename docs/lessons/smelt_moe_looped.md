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

**成本是什么(我们实测 + 推算)。** 见 §2 五个 repo 专属问题。最大单项:AttnRes 跨循环的 read 数 2.16×(L12 loop2×,推算);KDA 状态跨访问语义论文没答,自测;MoE all-to-all 在 H20 上零测量;我们 30.6% 的步已撞带宽屋顶,looping 的激活流量加在饱和侧。

## 2. 五个 repo 专属问题

**Q1(最优先):DeepLoop 的残差缩放修法 vs AttnRes,是同一问题的两种解法还是可叠加?**

可叠加,不是二选一——但叠加的代价是稳定性证明失效。DeepLoop 管**写**:共享权重的多次访问把梯度聚合成一个更新,残差流被相关更新 inflate,修法是初始化增益 β=(8N)^{-1/2} + skip scale α=(2N)^{1/2},把分支相对残差流压到 1/(4N)(deeploop.kappa_and_exponent)。AttnRes 管**读**:残差求和无结构,修法是学一个 pseudo-query softmax over 所有前面子层输出。一个管写尺度、一个管读权重,作用点不同。

但有两个真实交互:
- DeepLoop 的 κ_R 分析假设标准残差流;AttnRes 把残差流换成 softmax 加权和,bound 不直接覆盖。叠加要自己测稳定性。
- AttnRes 的"前面的子层"在 loop 下有一半是同一权重的上一轮。若 AttnRes attend 所有**执行过的**子层,read 数从 n(n+1)/2 = 325(L12)涨到 703(L12 执行 18 层),2.16×;按 tilerl 实测的 19.7% 墙钟占比线性外推到 ~42%(repo.attnres_loop_cost_derived;19.7%/37.8% 是 tilerl 2026-09-03 实测,未入库)。**这一条测不过,SMELT 对我们不可用,不管曲线多好看。**

**Q2:我们是 token 受限还是参数受限?** 200M@4B = 20 tok/param,在 Chinchilla 最优点上;500M@20B = 40,偏 token 受限。MoE 的最优 TPP(56–91)远高于 dense(~20),因为每参数只被部分 token 训练。在 20–40 上 MoE 专家欠训练,收益曲线的甜区在我们当前预算之外。500M@20B 的 40 是我们第一次接近可试的位置——但仍只有甜区下界的一半。

**Q3:MoE 与 KDA + gated MLA 的兼容性?** 结构上部分成立:专家放 FFN 是常规做法,不碰 KDA 状态(KDA 状态在 DeltaRecurrence 内部,从不穿过 block 接口,model.py:6)。但 looping 中间层时同一 KDA 层被访问两次,**状态是接着算还是重置,论文没答**(smelt.stateful_architectures)——这是我们要自己付的成本,且答案影响数值正确性,不只是性能。

**Q4:H20 上的实际成本?** 我们的步:50.8% GEMM、30.6% 逐元素撞带宽屋顶、10.6% KDA、2.3% nccl(repo.tpp_and_step_profile)。MoE 的 all-to-all 每 token 路由、小消息多,在 H20 互联上零测量——nccl 2.3% 是 dense all-reduce,不能外推。Looping 不增权重读取(权重 tied 驻留),但增激活带宽流量,落在已饱和的 30.6% 那一侧。SMELT 用 GQA ratio 调 KV cache 到 ±4%,我们的 MLA KV 本来就小,这一项成本低。

**Q5(6e 提):"中间一半层"这个位置选择对 L12 成立吗?** SMELT 自己的消融:loop 跨度 0/2/4/6/8/10/12 层,最优点在 6(50%),全栈循环回退(1.9322 vs 1.9257)。理由是浅层做特征、深层做输出、中层做迭代精化。L12 有中间 6 层可循环(SMELT 的 200M 尺度就是 L12→18),位置选择在我们深度上成立——但我们的 12 层是 9 KDA + 3 MLA(attn_every=4,MLA 在 4/8/12,train.py:162),中间 6 层(4–9)的 KDA/MLA 构成和 SMELT 的全 dense 不同,循环哪 6 层要按我们的层类型重选,不能照搬"中间 6 层"。

## 3. 落地路径(如果判决执行)

1. **先测 Q1 的数**:L12 上 AttnRes attend 执行深度 18 的 read 成本(fused kernel 3ed56306 上实测,不用推算)。读数 >30% 墙钟,整条线停。
2. **500M@20B 落地后**:纯 looping A/B——DeepLoop α=(2N)^{1/2}/β=(8N)^{-1/2} 初始化 + 中间半层 loop 2×,不带 MoE。对照 = 500M dense 臂。预登记出口:val NLL 差 ≥ 2.28σ(我们的噪声带)且退化率不降。
3. **MoE 留到 TPP>40**:即 500M 训过 20B 之后的下一个规模决策点;届时 all-to-all 成本先在 H20 上微基准。
4. DeepLoop 的 α/β 是零成本的默认初始化——只要上 looping 就用它,不需要额外实验。

## 4. 来源类型声明

- 论文报告值:SMELT 全部数字(arXiv 2609.01343v1,2026-09-03 取),DeepLoop 全部数字(arXiv 2607.13491v2,2026-09-03 取)。两篇都晚于知识截止,先验零,全部从原文取。
- 我们实测:AttnRes read 律 n(n+1)/2(gate_failure_shapes.md:833)、H20 步剖面(facts/efficiency.json:2361)、tok/param(data_scaling_design.md:9)、AttnRes 13.5% 吞吐成本(eff.attnres_internal)。
- 推算:loop 后 read 数 2.16×、墙钟 ~42%(从实测 read 律线性外推,未在 loop 模型上测)。
- 同行报告未入库:AttnRes 19.7%/37.8% 墙钟(tilerl 2026-09-03)、AttnRes softmax 12% bf16 误差。
- 两篇都是单 seed(DeepLoop 作者声明;SMELT 消融同),收益数字的 run 间方差未测。
