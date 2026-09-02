---
question: 业界提 tok/s 的方法逐项对本栈(KDA+MLA、FP8 torchao、DDP 8×H20、seq 4096)给预期收益和改动代价
status: open
source: facts/efficiency.json 实测 + train.py 实读(两者都必须查,见文首)。e1,2026-09-02,60 分钟冲刺
---

# 训练提速方法逐项对照

**这份文档不是提案,是索引:八项全部已测、已 ship,或结构性不适用,内核层没有剩余项。**
把已测的结论重新包装成"建议尝试"会让人第二次花钱买同一个答案,所以每一项先标状态,再谈还剩什么。

**这份草稿自己犯过一次那个错,值得写在最前面。** 第一版把 `chunk_size=32` 和 flash 缝的
`_dynamo.disable` 列为"本轮可试、性价比最高",而两者早已分别在 train.py:174(`43c4110`)和
train.py:164(`36328ab`)。原因是可复现的:**我只读了 `facts/efficiency.json`,没读 `train.py`。**
一条 fact 记录的是"这个杠杆值多少",不**是**"它有没有被拉下去",而我把前者读成了后者。
判据是一行 grep,不是一份 fact 表——这一条对任何以 facts 文件为唯一输入的调研都成立。

标注口径:**已测**(本栈有实测数)/ **已否决**(实测后 NO-SHIP)/ **未测**(本栈没数,可试)/
**不适用**(结构性,附原因)。收益一律按本轮实测基线折算,不引用厂商峰值。

## 基线,先钉死

| 量 | 值 | 出处 |
|---|---|---|
| 当前 500M 运行 | 12K tok/s/gpu, MFU 12%, batch 32 accum 1, grad_ckpt ON, FP8, 8 卡 | 运行日志(2026-09-02) |
| 200M 参照 | 73K tok/s/gpu, MFU 31%, batch 32, **no grad_ckpt** | `eff.fb_mfu` |
| 稳态 step 分解 | busy 1600.25 ms / span 1676.63 ms,idle 76.38 ms(95.44% busy) | `eff.steady_state_composition` |
| fp8 GEMM | 493.1 ms/step,**已在本 pod 实测 fp8 峰值的 98.2%** | `eff.fp8_gemm_at_realizable_peak` |
| H20 实测 fp8 峰值 | 279.6 TFLOPS(厂商标 296) | 同上 |
| 架构上限 | **2× 不可达**;软件四杠杆理想值合计 1.104×,机制明确的两项 1.084× | `eff.two_x_not_reachable_by_architecture` |

最后一行是这份文档的天花板,也是它存在的理由:**没有任何一项能单独给 2×,而"把所有理想值相加"
是双重计数**——重塑形状和 KDA 重调优动的是同一个 917.2 ms 区间。

## 逐项

### 1. torch.compile + CUDA graphs — 部分已测,部分不适用

**compile 已在用**;真正的发现是编译**不会摊销**。稳态 trace(步 56-76)里 8 个 >1 ms 的空隙,
7 个在同一个 `rms_norm -> flash-attention` 缝上,每个 150-166 ms,合计 54.9 ms/step = 全部稳态
idle 的 72%,其中 CPU 编译/guard 时间 69.3 ms/step。步 260 的重编译和步 31 一样多
(`eff.steady_state_composition`)。

已有 SHIP CANDIDATE:把 `flash_attn_varlen_func` 包进 `torch._dynamo.disable`,flash 重编译
70 → **0**,总重编译降 85%,数值中性(`eff.seam_dynamo_disable`)。**已 ship**:train.py:164
(`36328ab`)。本轮没有剩余动作。

**这一项最重要的一句话是它的验收方式**:两臂 tok/s 都是 81K,**完全一样**。一个恒定的税对稳态
吞吐不可见,只看 tok/s 会把这个修复判成"没用"——当初把平的 77K 读成"重编译税不存在"就是这么错的。
这条判据教训对后面任何一次内核修复仍然有效,即使这一项本身已经合了。

CUDA graphs:**不适用**。doc-mask 的动态 cu_seqlens 形状让 `mode=max-autotune` 已被实测为不可用
(`eff.max_autotune_dynamic_shape_noship`),graph capture 对动态形状是同一堵墙。要用得先放弃
doc-mask,那是改数据语义不是提速。

代价:`_dynamo.disable` 一行,已有 A/B。**本轮可试**(判据必须 trace-based)。

### 2. fused / chunked CE(liger 类) — 已在用,且是最大单块

Liger `FusedLinearCrossEntropy` 已经是训练损失路径(train.py:2557,验证路径 train.py:1021)。
LM head 仍是最大单项:190.0 ms/step(`eff.lm_head_is_compute_bound`,注意这个数从 224 更正过——
224 是整个 nvjet 核而不是 head)。

**fp8 LM head 已否决**:roofline 预测 +5.6%,A/B 实测 **-3.9%**(`eff.fp8_head_ab_noship`)。原因
在下一项。**不要重试**,除非量化税那 135.1 ms 先被解决。

### 3. FP8 覆盖面 — 已测,且方向是"别再扩"

GEMM 侧没有空间:493.1 ms/step 已在实测峰值的 98.2%,GEMM 库替换从 2.06% 降到 0.53%(旧数除的是
硅片达不到的 296)。

扩覆盖面的真实代价被量化过:elementwise 组 250.61 ms/step 里 **99.98% 被 correlation id 归因到
量化税**——div 84.01 + add_ 82.00 + copy_ 67.17 + abs 13.84 + clamp 0.65,正是 `_fp8_mm` 的
`abs().amax().clamp()` → `(x/s).to(fp8)` → `.contiguous()` 逐op签名(`eff.quant_tax_is_the_elementwise_group`)。
每步 343 个 clamp = 343 次 scale,而其中 ~93% 是 head 一个模块贡献的。

所以 fp8 head 和"copies"不是两级台阶,是**同一级**;合并后天花板仍是 75.5 ms = 4.44%。
另有 `eff.fp8_transpose_cast_no_config_lever`:torchao 0.17.0 每步把 input/weight/grad_output 各
cast **两次**,没有配置开关关掉。**不适用**(需要改 torchao 或自己写 epilogue)。

`eff.evt_epilogue_has_nothing_to_attach_to`:EVT 是 GEMM epilogue,而承载这笔开销的三个张量
没有一个挂在 GEMM 上——机制上不成立,不是大小问题。

### 4. 选择性重算 / grad_ckpt — 已测,且随深度反号

这是本清单里唯一**已经在按深度做正确选择**的一项,而且反号幅度超过决策需要
(`eff.grad_ckpt_inverts_with_depth`,单卡 d1024 heads8 ffn3072 AttnRes Full seq4096,200 计时步):

| 深度 | grad_ckpt | 结果 |
|---|---|---|
| L=12 | ON | **2.4× 慢**,当时被否决 |
| L=32 | ON | 1.116× 慢,峰值显存 **降 4.1×** |

L=32 batch4:OFF 2069.5 ms / 54.50 GiB,ON 2309.4 ms / 13.36 GiB;按倍增找最大 batch,OFF 只装下 4
(56.38 GiB),ON 装下 16(49.18 GiB)——**4 倍 batch 而峰值还低 7.2 GiB**。存储的激活占 54.5 GiB 里
约 41 GiB。

对本轮两条新启动行的直接结论:200M(L12)用 `--no-grad_ckpt`,这也是 `eff.fb_mfu` 那个 73K 基线
的设置,所以基线可比;300M(L18)在两者之间且**从未实测**,显存探针该进冲刺清单。

**选择性**(只 ckpt 一部分层)本栈未测。L=32 的 1.116× 已经很便宜,而 L=18 可能连 ckpt 都不需要,
所以选择性重算的收益空间被两头挤掉了。**未测,低优先**。

### 5. fused 优化器 — 已在用

`fused=True` 已在三处(train.py:1124/1131/1141)。**没有剩余空间**。foreach 是 fused 的次优
退化路径,不是升级。

### 6. ZeRO-1 vs DDP + 通信重叠 — 通信侧已测到饱和,ZeRO-1 未测

通信重叠已经调过并**饱和**:`bucket_cap_mb` 100→50 给 +14.1%(64K→73K),50→25 **没有进一步收益**,
7 卡上 50 与 25 同样打平(`eff.bucket_cap_mb_ab`, `eff.bucket_cap_mb_7gpu_ab`)。NCCL
`proto=Simple` 另给 +14.1%(`eff.nccl_proto_simple_ab`)。两者合起来 75K,而**单卡无通信是 80K**——
也就是说 8 卡通信的全部残余代价只有 6.25%,且这 5K 至今未被归因(`eff.ddp_5k_not_identified`:
static_graph=False 无变化,bucket_view=False 无变化)。

**ZeRO-1 的收益上界因此是那 6.25% 的一部分,不是"省显存换速度"**。它省的是优化器状态显存,而本轮
的显存约束已经被 grad_ckpt 解决(峰值 69.63 GiB / 96 GiB,还有 26 GiB)。**未测,但上界已知很小**;
在 6.25% 未归因之前,先归因比先换实现便宜。

### 7. fla 内核参数 — 一项已测可用,一项被 upstream 封顶

`chunk_size=32`:实测 fla 核 T=4096 时 **-19.1%**(12.12 → 9.81 ms),T=1024 -17.8%,T=2048 -18.2%。
折算到 in-model fla 组约 19.5 ms/step = 整步 978.5 ms 的 2.0%(`eff.kda_chunk_size_32`)。数值已验:
每步 loss 差 ≤0.0017,无增长,符号振荡(`eff.chunk_size_parity`)。

**已 ship**:`Cfg.chunk_size = 32`(train.py:174,`43c4110`),经 :301 → :353 传进 `chunk_kda`。
本轮没有剩余动作。train.py:176 的注释还留了一句我这份草稿漏掉的限定:**那 +19.1% 是单层隔离
测的,不是七卡真模型**,合并后的运行必须自己把它显示出来——所以这一项的正确状态是"已 ship,
待真模型确认",不是"可试"。

`num_warps`:fla 把 KDA backward 的 autotune 在 Hopper 上封在 [2,4],我们的 H20 吃不到更大的配置
(`eff.kda_num_warps_capped_on_h20`)。**不适用**(需改 upstream)。

三个"KDA 是瓶颈"的假设都已被实测**否决**:并行度不足(`eff.kda_parallelism_not_the_bottleneck`)、
occupancy 受限(`eff.kda_occupancy_bound`,三个独立测量,时间随 B*h 线性)。所以 KDA 侧除
chunk_size 外没有已知空间,而 chunk_size 已经在里面了。

### 8. 通信/计算重叠 — 见第 6 项;另一类重叠已被否决

同卡再塞一个进程**不适用且是结构性的**:第二个 PROCESS 用不上训练卡的空闲 SM
(`eff.colocation_second_process_cannot_backfill`),空 SM 槽位不是可填的容量
(`eff.low_occupancy_is_not_free_capacity`)。并且我们的 "95.44% busy" 是**驻留**度量,不能读成 SM
利用率(`eff.busy_is_residency_not_sm_work`)——把它读成"SM 有 4.6% 空闲"是这一族错误的入口。

启动开销**已被否决为杠杆**:6684 launches/step 在 2 µs 是 13 ms、10 µs 是 67 ms,而平均核 138 µs,
差两个数量级;最短的 elementwise-copy 组均值 58.9 µs 也是单次 launch 的 30 倍
(`eff.launch_overhead_is_not_a_cost`, `eff.launch_reduction_not_a_lever`)。95.3% 的 138460 个
gap 在 5 µs 以下、合计只有 10.2 ms,而最大的 20 个 gap 合计 164.9 ms 全在同一条 flash 缝上——
**所以进程侧的杠杆是"修一条缝",不是"减少启动次数"**。

## 本轮该做什么

**内核层没有剩余项。** 八项全部已测、已 ship 或结构性不适用——包括我这份草稿原本列为"可做"
的两项(`chunk_size=32` 和 flash 缝的 `_dynamo.disable`),它们已经分别在 train.py:174 和 :164。
草稿的错误值得写下来,因为它是可复现的:**我只读了 `facts/efficiency.json`,没读 `train.py`。**
一条 fact 记录的是"这个杠杆值多少",不是"它有没有被拉下去",而我把前者读成了后者。判据是一行
grep,不是一份 fact 表。

所以本轮唯一的提速来源是**形状**:

| 运行 | 形状 | grad_ckpt | 预期 | 依据 |
|---|---|---|---|---|
| 500M(当前) | d1024/L32,493.64M | ON | 12K tok/s/gpu, MFU 12% | 运行日志 |
| 200M | d1024/L12,206.13M | **OFF** | 73K 基线可比 | `eff.fb_mfu` 同形状同设置 |
| 300M | d1024/L18,293.05M | OFF | 未测,介于两者 | 无 |

3× 来自 **L12 无重算 vs L32 重算**这个组合,不是任何一个内核技巧。500M 的 12K 和 200M 的 73K
差约 6 倍,而两者的差就是深度加 grad_ckpt(`eff.grad_ckpt_inverts_with_depth`:L=12 时 ckpt 慢
2.4×,L=32 时只慢 1.116×)。这也是为什么 200M 那条启动行必须带 `--no-grad_ckpt`——它是 73K
那次测量的设置,换掉就失去可比性。

已 ship 的两项仍有一件事没做完:`chunk_size=32` 的 +19.1% 是**单层隔离**测的,七卡真模型上
还没确认(train.py:176 自己的注释)。合并后的运行应当把它读出来。

## 未测清单(留给下一轮,附为什么现在不做)

| 项 | 为什么现在不做 |
|---|---|
| ZeRO-1 | 上界是 8 卡通信残余 6.25% 的一部分;那 5K 尚未归因,先归因更便宜 |
| 选择性重算 | L=32 全 ckpt 只 1.116×,L=18 可能不需要 ckpt,两头挤掉空间 |
| L18 batch32 显存 | 需要探针;`eff.microbatch_32_oom` 是 500M 形状的结果,不可外推 |
| batch 48-64 | `eff.batch_ceiling` 明确标 untested;32 no-ckpt=90K,72 无 ckpt OOM |

## 引用说明

本文档的每一个数字来自 `facts/efficiency.json` 的实测条目或本轮运行日志,条目 id 在正文行内。
外部来源只在本栈无实测时引用,而这次**没有一项需要**:fb 点的八项本栈都已有测量或结构性判定。
`eff.h20_mfu_200m` 是唯一的"未找到出处"——公开文献里没有 H20 上 ~200M 稠密模型的训练 MFU,
所以 31% 既不能被文献判成高也不能判成低。这条不影响上面任何一项。
