---
question: 30B 预训练的读数怎么判——每个里程碑、每个指标,多少算动、动了之后干什么?
status: recorded
source: fb tasking 2026-08-31; docs/lessons/scale_36b_plan.md §1b (recipe), §1 (roles); facts/base_eval.json (be.panel_expressive_seed_variance, be.degeneration_rate, be.ctx_length_p324); facts/data_scaling.json (sigma-hat=0.0516)
---

# 30B 预训练读数:预注册(2026-08-31,t22 启动前冻结)

用户的 ask:"看 30B 预训练的效果"。本文在 run 启动前写死:里程碑、每个指标的
n 和判变阈值、以及判完之后的动作(continue / stop / change mix)。**没有决策规则
的指标不进读数。**

实现:`eval/readout_30b.py`(t34)——本预注册即代码,里程碑落地时直接跑判定,不争论。

Run 配方(scale_36b_plan.md §1b 定稿):30B tokens,seq 4096,warmup 300 步,
7 卡 × batch 16 × accum 2(有效 batch 224,与阶梯一致),~16h @ 524K tok/s,
mix = `data/mix_scale_30b.json`(8 能力角色,zh:en ≈ 35:65)。

## 1. 里程碑

| 里程碑 | tokens | 步数 | 读法 |
|---|---|---|---|
| **3.24B** | 3.24B | ~3,531 | **配对读数**:对阶梯点 ckpt_p324(同 arch、同 tokens、旧 mix)。arch+tokens 匹配,**隔离的是语料变更** |
| 8B | 8B | ~8,720 | 趋势读数 |
| 16B | 16B | ~17,440 | 趋势读数 |
| 30B | 30B | 32,697 | 终判 |

每个里程碑落一个 checkpoint,跑全量 score_matrix(base 行 + sft 行视情况)。

### 3.24B 配对的一个混杂,必须写在前面

配对两边 **warmup 不同**:阶梯点 warmup=20(六冻结点),30B run warmup=300。
3.24B 处 300 步 = 全程的 8.5%,阶梯点 20 步 = 0.57%——30B 里程碑有更长一段
跑在低 LR 上。方向已知(长 warmup 拖低早期 loss,0.2b smoke:warmup 2 vs 20
差 0.52 val),量级在 3.24B 处未测。

**预注册的读法**:
- 30B 里程碑 **好于** 阶梯点且过阈值 → 语料效应为真(warmup 拖拽只会衰减它,
  不能制造它)。
- 30B 里程碑 **差于** 阶梯点 → **此配对对语料问题不可读**,不得判"mix 更差";
  干净读数在 8B/16B/30B(warmup 占比降到 3.7%/1.9%/0.9%)。
- 逃生舱:**仅当** 3.24B 里程碑读"差于"阶梯点时才跑同配 warmup=300 的 3.24B
  对照 run(~1.7h,7 卡);在此之前不跑——block time 是本周最稀缺的东西,
  配对读"好"或"平"时这个对照买不到任何东西。

## 2. 每个指标的 n、判变阈值、决策规则

| 指标 | n | 判变阈值 | 决策规则 |
|---|---|---|---|
| **code-500 accuracy** | 500 | 12.6pt(二项 2δ,δ=1.4/√500=6.3pt) | 30B 处 >12.6% → 代码生成存在,continue 进 RL 闸;≤12.6%(地板)→ 30B mix 没买到可测代码生成,**stop 推理轴主张**,按阴性报。8B/16B 只看趋势,早过阈值不改动作 |
| **math-hard pass@1** | ≥1000 | 8.8pt(二项 2δ @ n=1036;δ=1.4/√n) | 同上:30B 处非零(>2δ)→ continue;地板 → stop。n 以实际跑的题数为准,阈值随 n 写死在记录里 |
| **pass@8−pass@1 gap** | 同上 | **15pt(RL 闸口)** | ≥15pt → 有可放大的东西,RL 可开;<15pt → RL 不开(读数:"不是 RL 没用,是这个 checkpoint 上没有 RL 可放大的东西")。这是 continue-to-RL 决策,不是 stop |
| **MC 套件**(ceval + mmlu + arc-easy) | ceval 1,050 题(ceval.py val split,52 科;2026-09-01 更正:原写 1.3k 是错的) | ceval 5.9pt(seed SD 1.27pt × 4.65,be.panel_expressive_seed_variance);**2026-09-01 起前向阈值 provisional 13.5pt**(见下,be.adjacent_checkpoint_jitter);英文三件 **chance+2δ 地板**(各 25% 机会率) | **tripwire,不是 stop**:ceval 掉 ≥阈值 → mix 伤了通用能力,change mix;升 ≥阈值 → 佐证,不单独构成结论。英文三件在 chance+2δ 以内 = 地板,不判读(三选一是机会率,不是模型性质) |
| **per-role domain loss**(8 角色) | 4000 行/角色(打包后截断 64 seq × 4096 = 262,144 tokens;记录以 tokens 字段携带,domain_loss.py:34-35) | 0.1176 nat(2.28×σ̂,σ̂=0.0516,ds.seed_variance_0p2b) | 对阶梯点同头比较:角色改善 ≥0.1176 → mix 买到了该角色,continue;30B 处仍平(<0.1176)→ 该角色权重没买到可测价值,**change mix 候选**(进重配 A/B)。这是唯一能逐角色说"change mix"的指标 |
| **degenerate_rate** | 同各 accuracy 的 n | 二项 2δ @ n | **永远并排报,永不替代 accuracy**(v2:55.8% 退化@贪心 vs 2.2% 正确)。accuracy 升且退化率升 ≥2δ → 格式不稳定,报;退化率降 → 佐证。本身不构成 stop |

格式类指标(degenerate_rate、围栏率)是贪心解码器的性质,报的时候必须带温度;
能力类指标(accuracy、pass@k、domain loss)不受影响。

**2026-09-01 ceval 阈值修正案(be.adjacent_checkpoint_jitter,b0 8d5f5ee,fb 转交 44 裁决)。** 同一 run、同 seed、同配置的两个 checkpoint(step16000 与 run-end save,相隔 281 步,log-attributed)在 stage-1 头上打分:per-role domain loss 移动 0.005–0.027 nat(自身阈值 0.1176 的 1/4–1/20,**不涉**),但 C-Eval 移动 22.5→25.4 = **2.9pt**,退化率移动 0.522→0.628 = 10.6pt。281 步内不可能有真实能力变化,所以生成/MC 指标在相邻 checkpoint 间的抖动主要来自"取了哪个 checkpoint",而非能力。2.9pt 是 seed SD 1.27 的 2.3×——5.9pt 阈值建立在两个独立噪声项中较小的一个上。**裁决:接受修正案。** 5.9pt 保留为预注册记录(已记录的判决全部是地板,更大阈值只让地板更确定,无判决改变——包括 25.4,它读作"22–23 带外的首次移动"但本身就在抖动带内,22.5→25.4 就是抖动本身)。前向阈值 provisional 提至 **13.5pt**(4.65×2.9,jitter-only,适用 run 内比较;跨 run 比较在独立性未证前取 max(5.9, 13.5)=13.5,combined 14.7 不值得那个假设)。**n=1 临时态**:2.9pt 是单个相邻对的一次观测,不是带 df 的估计,抖动可能随 gap 缩放;3+ 对后重审,end-of-run checkpoint 使多对很便宜。**结构警告**:该对是 step-save vs run-end-save(run-end save 带 step: None,是不同代码路径),更干净的估计需 step-save vs step-save 对。**退化率连带修正**:8B(0.412)→15B run-end(0.628)的 +21.6pt 趋势被 +10.6pt 相邻抖动污染——+21.6 是抖动的 2.04×,真实上升可能存在但幅度未解析;"训练越久退化越重"的趋势主张软化,SFT 阶段含义(base 是退化的,0.52–0.63)不变。**预注册判决规则(2026-09-01,数字出来前写死;打分 step15000/15500,两个 step-save 同代码路径对,500 步 gap,fb 批准、de 执行)**:ceval step-save 抖动 ≥1.5pt → run-end 的 2.9 是抖动,13.5 以 n=3 站住;<1.0pt → run-end save 是系统性偏移,2.9 是 offset 不是 jitter,阈值降回 4.65×max(seed_sd 1.27, step-save SD)(run 内,大概率回 5.9–8);1.0–1.5pt → 不确定,**ceval 保持 13.5 provisional 且退化率判决同时悬置**(b0 补充:两个指标来自同一对,ceval 中间结果不独立解决退化率的 5-vs-8pt 问题)。退化率:8500↔16000 都是 step-save,+11.0pt 本身干净;干净对的 step-save 退化抖动 <5pt → +11.0 是 2×+ 抖动,上升真实(幅度只从 step-save 读);≥8pt → 趋势撤回到"base 退化,趋势不可读"。

### 每个指标在它的 n 上能到达哪些状态(44 重点核)

终判三态:**动了**(过阈值,带方向)/ **地板**(不可判读)/ **平了**(可读但
没过阈值)。不是每个指标都能到达全部三态——判变阈值和可读性下限的关系决定:

| 指标 | 可达状态 | 为什么 |
|---|---|---|
| code-500 accuracy | 动了 / 地板 | 判变阈值 12.6pt = 可读性下限(2δ@500),阈值之下皆地板,**无 flat 带** |
| math-hard pass@1 | 动了 / 地板 | 同上,阈值 = 2δ@n |
| pass@8−pass@1 gap | **动了 / 平了 / 地板** | 决策阈值 15pt > 可读性(gap 是配对差,pass@8⊇pass@1 同题相关,SE ~1-2pt@1036);3.5pt 的 gap 可读但 <15pt = **平了**,200M 上的读数就是平了不是地板 |
| ceval | 动了 / 地板 | 5.9pt = 4.65×seed SD = 可读性下限,无 flat 带 |
| per-role domain loss | 动了 / 地板 | 0.1176 nat = 2.28σ̂ = 可读性下限,无 flat 带 |
| degenerate_rate | 动了 / 地板 | 2δ@n = 可读性下限;并排指标,不单独构成 verdict |

**只有 RL gap 能判"平了"**;其余指标的阈值就是它们的可读性下限,阈值之下
只能是"地板"。读数时不许把"地板"报成"平了"——前者是仪器没分辨率,后者是
有分辨率但效应不够大,两种阴性的后续动作不同(地板 → 换仪器或加 n;平了 →
效应真小,换杠杆)。

## 3. 3.24B 配对不能说什么

1. **30B mix 在 3.24B tokens 之后的任何事。** 执行标注(7%)和长 CoT(15%)
   是这张 mix 的区分性赌注;3.24B tokens 时模型只见过 0.23B / 0.49B。它们对生成
   的效应(如果有)在 3.24B 不可测——生成本来就在地板。
2. **任何 base-eval 分辨率小于预期效应的判断。** 每个指标旁边已写分辨率:
   code-500 12.6pt、math-hard 8.8pt、英文 MC chance+2δ、domain loss 0.1176 nat。
   语料效应小于阈值时配对读"无可读效应"——**这不是"无效应"**。
3. **角色之间的因果归因。** 新 mix 一次换了 8 个角色,per-role domain loss 定位
   "哪里动了",不定位"哪个角色导致的";因果归因要重配 A/B(scaling_decision_tree
   §2a 机制),不在本读数内。
4. **(warmup 混杂,见 §1)配对读"差"时不能说 mix 更差。**

## 4. 读数规则(写死)

- 地板 ≠ 无效应:永远读零的指标等于没有指标;地板读数标"地板,不可判读",
  预注册文本不删(删 = 事后改判据)。
- 退化率并排报,永不替代 accuracy。
- MC 是 tripwire:掉 = change mix,升 = 佐证。
- 30B 终判时,每个指标给三态之一:**动了(过阈值,方向)/ 地板(不可判读)/ 平了
  (可读但没过阈值)**——"平了"和"地板"是两种不同的阴性,不许合并。

## 5. Stage-1(15B WSD)里程碑集(t46,2026-08-31)

Stage-1 run 配方:15B tokens,seq 4096,warmup 300(同 §1b),7 卡 × batch 16 × accum 2
(有效 batch 224,与阶梯一致),mix = `data/mix_15b_stage1.json`(math_owm/cot 绑定 pod
现有种子数据 math_seed(81.68M 实测),cot = numina 直接落地(424M 实测);cot_seed
(fable5)已 drop——4,665 段全 <100 字符、轻过滤 0 留存,不可训练(fb 2026-08-31);
缺口按 8:5.5 转 code_rp1t+en_c4),WSD:
`--warmdown 0 --anneal_frac 0`(stage 1 保持 lr 稳定,终在 lr_mult 1.0;stage 2 在此
resume,`--warmdown 0.10`,resume 机制以 t47 排练为准)。

| 里程碑 | tokens | 步数 | 读法 |
|---|---|---|---|
| **3.24B** | 3.24B | ~3,531 | **配对读数**:对阶梯点 ckpt_p324(同 arch、同 tokens、stage-1 mix vs 阶梯 mix)。warmup 混杂同 §1(stage-1 warmup=300,与 30B run 同),读法规则不变 |
| 8B | 8B | ~8,720 | 趋势读数 |
| 15B | 15B | ~16,349 | **stage-1 终判** |

阈值与决策规则同 §2,逐字不改(code-500 12.6pt、math-hard 8.8pt、RL gap 15pt 闸、
ceval 5.9pt、per-role domain loss 0.1176 nat、degenerate_rate 2δ)。**终判动作映射不同**:
30B 表的"continue 进 RL 闸"在 stage-1 终判处读作"continue 进 stage 2(resume)";
"stop"读作"stage 2 不 resume,按阴性报";"change mix"读作"stage-2 mix 调整候选"。
3.24B 配对的逃生舱(同配 warmup=300 对照 run,~1.7h)同样**仅当**配对读"差于"阶梯点
时才跑,在此之前不跑。

Stage-1 3.24B 配对隔离的对比与 30B run 的 3.24B 配对**不是同一个对比**:后者隔离完整
30B mix,前者隔离 stage-1 mix(种子 math/cot + 重配 code/en)。两者都是"固定 arch+tokens
下的 mix 变更",§1 的读法规则(好于且过阈值 = 语料效应为真,warmup 拖拽只会衰减它;
差于 = 此配对对语料问题不可读)通用。Stage-1 的 per-role domain loss 对 7 角色读数
(code_tests 已 drop——mining 未落地,1.0B 折进 code_rp1t,stage 2 回归;fb 2026-08-31),
其中 math_seed/cot 是 stage-1 角色名(math_seed 替代 math_owm;cot = numina 直接落地;
cot_seed 已 drop——fable5 不可训练,fb 2026-08-31),其余 6 角色同名。

## 6. 启动后发现的混杂:stage-1 实际抽取(2026-08-31,tilerl 挑战 / 44 核验)

stage-1 实际调度 3,647,072 行 = **14.9384B tokens,不是合同的 15.000B**。cot 的 epoch cap 在 pool 行上截断(want 310,546 > pool×3 = 295,512),切掉 15,037 行 = 61.6M tokens;cot 是唯一被 cap 的域。cot 实际跑 **2.855 raw-supply-epochs**(3.00 pool-epochs,295,512 行),不是配方的 3.00。步数交叉验证:16,281 步 × 224 × 4096 = 14.9379B,与计划一致。

读法后果:

- 若 math-hard 或任何 cot 敏感指标在任一里程碑读 **flat**,"cot 抽得比配方少"是预注册此前未命名的**活替代解释**——flat 不可直接读作 mix 无效。3.24B 里程碑处绝对差小(cot 占实际调度 8.10% vs 名义 8.48%),随里程碑增大。
- 所有里程碑的 **n 以实际步数计**(k × 224 × 4096),不以名义 tokens 计;3.24B 配对的 p324 同 tokens 配对不受影响(p324 是独立 run)。
- 根因已立档:check_mix_supply 用 raw supply × epochs 验证,而 build_mix 在 pool(supply 减 val)× epochs 上 cap——任何被 cap 的域都会少抽。修复已报 de(pool 模型 + broken world)。

## 7. per-role domain loss 的适用条件(2026-08-31,step3500 读数争议时立)

该指标的判读**要求头集合是被判 mix 拥有权重的域的子集**。不是"两侧同头",也不是"各自本 mix 的头"——同头可以同时是两侧都不对的头。

理由在 §2 的决策规则本身:角色改善 → mix 买到了该角色;角色持平 → 该权重没买到可测价值,进 change-mix 候选。**没有权重的角色无法进入这条规则**,对它的判决喂不给任何决策。

由此得出零重叠情形的构造性结论:两个 mix 的域名无交集时,**任一方向的配对都不可判读**。在被判 mix 的头上打分对另一侧是分布外,反之亦然,delta 混合了能力与分布偏移,事后无法分离。此时记**地板(不可判读)**,不记方向、不记幅度。

阈值同受此约束。0.1176 nat = 2.28 × σ̂,σ̂ = 0.0516 测自四个 0.2b 种子(`mix data/mix_scale_0.2b.json`、`corpus fp web_hq 30838d423348b2e5`),是**同分布下的种子噪声**。用在模型未训练过的文本上,它计的不是同一个量。头集合更换后,阈值标记为 provisional-not-transferred,判决列只写方向与幅度,直到在该头集合上测出 σ̂。

2026-08-31 记录:stage-1 3.24B 里程碑按 `mix_scale_3.24b.json` 的头打分(web_hq/textbook/wiki/en/math/code/chat),与 stage-1 训练域零重叠。该读数无效,标 superseded 保留,不删除;"6 of 7 degraded" 不作为结论出现在任何地方。stage-1 头上的重打分作为**无配对的本 mix 基线**:8B/15B 里程碑对它做趋势差(同 mix 同头,按 §1 只读方向);30B 处与 stage-2 配对(stage-1↔stage-2 共享域,配对有效,阈值适用)。

前向规则:任何在**已退役 mix 的头**上对新 mix 打分的里程碑读数,适用同一条件——阶梯头的存在不是错,用错了 mix 才是错。代码侧已强制执行:readout_30b 对 head 集合不一致的配对直接拒绝输出判决(de 19bea53,selftest 第 4 例即今夜这对)。

**基线丢失(2026-08-31 深夜)**:step3500 checkpoint 在重打分前被 newest-3 轮换窗口删除,3.24B 本 mix 基线不可恢复(fb 裁决)。own-mix 趋势序列从 8B 开始——8B 无更早本 mix 参照,读绝对值;15B 对 8B 做差;30B 处与 stage-2 配对不变。里程碑 checkpoint 自此钉出轮换(de,step 8500 前落地);ad-hoc step6000 基线被否(非里程碑 token 数,无预注册地位,事后造基线是事后 instrumentation)。

### 7.1 跨阶段配对:逐角色,按权重变化设闸(2026-08-31,b0 提案 / 44 落地)

stage 2 与 stage 1 域集相同、权重不同,因此跨阶段的 per-role delta 中混有**重配效应**:剂量变了,loss 变化里机械地含有剂量项,语料效应分离不出来。de 的 19bea53 异头拒绝管不住这一类(头集合相等,筛子通过)。规则改为逐角色:

**仅当该角色的权重在容差内未变时,跨阶段 delta 可判读。** 容差取 5% 相对变化——剂量-损失弹性(α≈0.05–0.1 的 scaling 律)下,5% 剂量变化的机械损失效应约 0.005–0.01 nat,低于 σ̂=0.0516,不构成阈值外的混杂。

| 角色 | w1 | w2 | w2/w1 | 判读 |
|---|---|---|---|---|
| math_owm | 0.18333 | 0.32058 | 1.75× | 地板(重配不可判读) |
| code_rp1t | 0.37200 | 0.29330 | 0.79× | 地板(重配不可判读) |
| en_c4 | 0.20987 | 0.15604 | 0.74× | 地板(重配不可判读) |
| cot | 0.08480 | 0.08069 | 0.95× | 可判读 |
| zh_web | 0.11000 | 0.10955 | 1.00× | 可判读 |
| textbook_30b | 0.03333 | 0.03320 | 1.00× | 可判读 |
| wiki_chat | 0.00667 | 0.00664 | 1.00× | 可判读 |

不可判读的三个角色记**地板(重配不可判读)**,旁打印 `w1 → w2`,只做阶段内比较(stage-2 里程碑对 stage-2 基线)。

cap 可以压缩权重变化:cot 的 w1 想要 310,546 行、被 pool×3 = 295,512 截断,w2 想要 295,495 行、未触 cap——权重差 5.1%,实际抽取差 17 行(0.006%)。故容差判的是权重,精确条件是 **cap 后的实际抽取之比在容差内**(抽取 = min(want, cap));cap 绑定处以抽取为准,不以权重为准。

两条推论:

一、不可判读的三个角色正是承载 stage-2 配比论点的三个(math 加、code/en 减)。**跨阶段 per-role 读数回答不了 stage 2 要回答的问题**——那个答案来自 stage-2 内部里程碑和 30B 的生成指标(code-500、math-hard、RL gap 不受本节约束),不要在 8B 处期待跨阶段 per-role 给出它。

二、容差是**这一对 mix 的性质**,不是常数。未来的 stage 3 若权重不同,可读角色的划分随之重分,规则须**逐边界重新应用**,不可把上面这份名单当作结论记住。

实现:两个 mix 自带权重,`readout_30b.py` 可逐角色计算 w2/w1 并自动拒绝超容差角色——与 de 19bea53 同形,使不可判读的读数不可能打印出来,而不是靠读者记得本节。归 de 的 shipment。

### 7.2 同头同权重但语料变了:.srcfp 变更的判读(2026-08-31,de 提问 / 44 裁决)

同头、同权重、同域名下语料字节变了(stage-2 的 `*_stage2` 重建即此类),per-role delta 测的是**语料效应**——这正是 mix 归因要的,本身不是混杂。但有三个后果:

1. **阈值不可迁移**:0.1176 = 2.28×σ̂ 测自固定语料的种子噪声(ds.seed_variance_0p2b);语料变了,σ̂ 不含跨语料方差项。按 §7 同理,阈值标 provisional-not-transferred,判决列只写方向与幅度,直到在新语料上测出 σ̂。
2. **被测文本必须同一**:domain loss 的 4000 行/角色来自 val 切分,语料重建后 val 切分不保证是同一批行。**默认拒绝**该角色的判决,除非记录证明两次打分用同一批 held-out 文本(文本指纹一致),且该文本在新语料中缺席(holdout 哈希对新构建重探;命中 = 污染,拒绝,code-500 v1 类)。
3. **打印 banner**:语料变了的角色,读数旁打印 `srcfp A→B`,让读者知道 delta 是语料效应。

即:不是一律拒绝,也不是沉默脚注——**默认拒绝,验证后降级为注记**(同文本 + 无污染 + 阈值 provisional)。30B 跨阶段读数对每个 `*_stage2` 重建域都会撞上本条。

### 7.3 守卫必须对真实产物自检(2026-08-31,b0 提案 / 44 落地)

本节的每条规则都要有一个**跑在真实文件上**的自检,不能只跑合成夹具。

2026-08-31 的实例:重配闸的首个版本(e178f17)合成用例全绿,对真实的 stage-1/stage-2 文件却**每个角色都返回 None**——三处原因合成夹具都看不见(stage-2 键带 `_stage2` 后缀、六个角色尚在 `_blocked`、池字段叫 `stage1_pool_rows`)。夹具是按作者**对文件的设想**写的,所以它无法证伪那个设想。

失败不是"夹具太简单",是**共享出处**:同一个人写代码和夹具,同一个假设编码了两次,夹具抓不到假设本身错了。写更丰富的合成夹具照样抓不到。修法是 `readout_30b.py` 的 selftest 5b:直接读 `data/mix_15b_stage1.json` 与 `data/mix_30b_stage2.json`,断言没有角色不可读;pod 实测 3 refuse、cot 按 cap 后抽取相等判可判读(ratio 0.952,draws_equal True)。harness 的 broken world 是同一模式的另一实现(44)——**变异真实产物,而不是构造一个想象中的产物**。

fail-open 方向让失败是沉默的而不是响亮的:只读 `domains` 使未落地的角色"因缺席而不可判",而不可判不产生拒绝——守卫在最需要它的角色上最宽松。守卫对缺席数据的行为必须是写明的选择,不能继承自某个恰好返回 None 的分支。

判据:**一个读不到它所守护的数据的守卫,与不存在无异,且更糟——它显示为绿。**

### 7.4 复用域的治疗标注与 pool 算术约定(2026-08-31,tilerl 实测)

**治疗标注**:复用域的 epoch 数必须写明新鲜/重复行的拆分。cot 的 stage-2 抽取 295,512 行 = pool×3,全部是 stage 1 已读过 3 次的行——**零新鲜行**;"6 pool-epochs" 算术正确但读作比实际更多的数据,表格须写 "6 epochs, no fresh rows" 或等价。de-7 的 cursor 修复也不改变这一点(cot 的 want 恰为 3×pool,无行可读)。对照:zh_web/textbook_30b/wiki_chat 的 stage-2 抽取在 cursor 下全部是未读行(supply 余量 13×/3.2×/2.8×)。同样写 "N epochs",治疗不同——重复是 A′ 依据(2605.12715,constrained-domain r 15–20)许可的,但必须可见。

**pool 算术**:pool = cache_rows − n_val,n_val = min(int(cache_rows × 0.05), 5000),**从 cache 行数起算,不从 pool 起算**。wiki_chat 是第一个 5% 侧绑定的域:69,295 cache 行 → n_val 3,464,pool 65,831;cot 是 5,000 cap 侧绑定(103,504 → 5,175 > 5,000 → 5,000,pool 98,504)。用 "5% of pool" 反推是循环论证,只在 5,000 cap 绑定时碰巧抵消;sub-100K 行的域 cap 不绑定,循环算法会出错。d633dee 的 pool-model 检查从 cache_rows 起算,是对的。

**2026-08-31 验证(tilerl,caf4b4b)**:五个复用域与 stage-1 训练字节一致,§7.2 的范围声明成立。决定性证据是 `ckpt_pretrain_15b_s1.pt.step5500` 的 corpus_fp(save_checkpoint 从活 run 写入),五域全部等于 live;live `_corpus_fp` 对 cache 旁 `.srcfp` 5/5 匹配——但后者只证 cache 现行,不证 stage-1 身份(两者可一起移动),故两者都跑。**复验走 checkpoint 的 corpus_fp,不走 mix 文件**(后者只在散文里记了 code_rp1t 一个域)。两个已知边界:checkpoint 截至 step 5500,其后由 pod_drift + 语料写入方已完工覆盖(且现在跑的 live-vs-.srcfp 能抓到 5500 之后的改写);`_corpus_fp` 只哈希 shard 名/大小/首尾 64KB,shard 中部改动两侧都看不见——文档化设计,接受。

### 7.5 语料更名后跨阶段 per-role 的可读域划分;进度/保留/组成三轨分立(2026-09-01,b0 提问 / 44 裁决;同日更正:更名域为 **2/7** 非 5/7——fb 据 stage2_composition.md 与 §7.4 修正,b0 初稿的"five of seven"与其自身指纹表矛盾)

**触发**:16B(step17500)对 8B 自有 mix 基线(step8500)的 per-role 被 head guard 整指标拒绝。pod 指纹实证四套语料(en_c4/en_c4_stage2、math_owm/math_owm_stage2 各异)。**拒绝正确**:更名的 2 个角色会跨不同 held-out 文本比较,正是 3.24B 失败类。guard 先于 §7.1 触发也是正确顺序——head 不配比权重变更更根本。

**结构事实(更正后)**:stage 2 更名了七个域中的**两个**(en_c4、math_owm → `*_stage2`);其余五个(cot、zh_web、textbook_30b、wiki_chat、code_rp1t)按 §7.4 复用 stage-1 目录与 cache、训练字节一致。故:**5 个复用域跨阶段仍可读**(同头、同字节,逐角色过 §7.1 权重闸);**2 个更名域跨阶段熄灯**(16B、22B、30B 皆然,永久,非临时)。16B 读数可直接打印 5 个可读域对 8B 基线,无需重打分(fb 2026-09-01 裁决,取代此前"16B 无 per-role"的说法)。

**裁决——三轨分立,现在写明,不在 30B 时才发现**:

1. **进度轨(per-role)**:两条可读边界——(a) 阶段内:16B→24B→30B,同头、同语料字节、同 mix,7 个角色全可读,不触发 §7.1/§7.2,就是同一分布上的规模进度,显著性用 σ̂ 阈值;(b) 跨阶段:5 个复用域(cot、zh_web、textbook_30b、wiki_chat、code_rp1t)在字节同一的 stage-1 头上对 8B 基线可读,**逐角色过 §7.1 权重闸**(stage-2 配比权重与 stage-1 差超容忍 = 该角色拒判,闸的首个真实用例);2 个更名域(en_c4、math_owm)跨阶段**熄灯**——16B、22B、30B 皆然,永久,非临时。30B 产出里没有更名域的跨阶段 per-role 进度读数——如此而已,不是缺口。
2. **保留轨(标注 OOD)**:仅为 2 个更名域而设——stage-2 里程碑(30B final 一次重打分)在 **stage-1 头**上打分,列名 `retention (OOD)`,回答的问题是"stage-2 模型是否灾难性遗忘旧分布",不是"stage-2 配比是否更好"。它**不进阈值机械,不叫进度,不进 per-role 判决表**。前置条件:按 §7.2 第 2 条,stage-1 holdout 哈希对 stage-2 语料重探,命中 = 污染 = 拒绝该角色的保留读数(stage-2 重建可能与 stage-1 语料有重叠文本,不验就打是裸奔)。5 个复用域的跨阶段读数直接在进度轨(b)出,不走保留轨。
3. **组成轨**:"stage-2 语料重建值不值"不由 per-role 回答——不可归因的比较不如缺席(3.24B floor 的同一逻辑,b0 的 lean 正确)。它由:聚合指标对 8B/15B 趋势(§7 line 152,direction-only)+ 保留轨 + 语料级事实(near-dedup 移除率、污染扫描)共同回答。

**§7.1 的状态:阶段内休眠,跨阶段首次真实开火**。阶段内同 mix 无可闸,读数须写 `dormant: no mix change between these checkpoints`,**永不写 "passed"**。跨阶段 5 个复用域是 §7.1 的首个真实用例(此前从未在真实数据上跑过)——故首个真实用例前仍须先有合成开火测试(一个没跑过的闸等于没测过的闸,P6);若 stage 3 改权重,阶段内也会开火。

**守卫改逐角色,不整指标拒绝**:现 guard 在 head 集不一致时拒绝整个 metric,把 5 个干净角色(未更名域)的信号一起扔了。改为逐角色判定:头匹配的角色判,不匹配的拒绝,拒绝消息逐角色列出。干净角色的阶段内/跨阶段/保留读数照出。

**预注册规则(持久,逐边界重新应用)**:*per-role 比较仅当两模型在字节同一的 held-out 头上打分时有效。训练语料跨阶段变更时,更名域的跨阶段 per-role 进度熄灯,复用域在字节同一的头上逐角色过权重闸后可读;阶段内 per-role 是进度指标;更名域的跨阶段保留是单独标注的 OOD 指标,且 holdout 哈希须对新语料重探。*

**另注**:16B 读数表 math_hard/rl_gap 列 ABSENT——读数须写明缺席原因(未跑/未打分),空列与"跑了是零"不可区分。
