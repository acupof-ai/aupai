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
| **MC 套件**(ceval + mmlu + arc-easy) | ceval 1.3k/科 | ceval 5.9pt(seed SD 1.27pt × 4.65,be.panel_expressive_seed_variance);英文三件 **chance+2δ 地板**(各 25% 机会率) | **tripwire,不是 stop**:ceval 掉 ≥5.9pt → mix 伤了通用能力,change mix;升 ≥5.9pt → 佐证,不单独构成结论。英文三件在 chance+2δ 以内 = 地板,不判读(三选一是机会率,不是模型性质) |
| **per-role domain loss**(8 角色) | 4000 行/角色 | 0.1176 nat(2.28×σ̂,σ̂=0.0516,ds.seed_variance_0p2b) | 对阶梯点同头比较:角色改善 ≥0.1176 → mix 买到了该角色,continue;30B 处仍平(<0.1176)→ 该角色权重没买到可测价值,**change mix 候选**(进重配 A/B)。这是唯一能逐角色说"change mix"的指标 |
| **degenerate_rate** | 同各 accuracy 的 n | 二项 2δ @ n | **永远并排报,永不替代 accuracy**(v2:55.8% 退化@贪心 vs 2.2% 正确)。accuracy 升且退化率升 ≥2δ → 格式不稳定,报;退化率降 → 佐证。本身不构成 stop |

格式类指标(degenerate_rate、围栏率)是贪心解码器的性质,报的时候必须带温度;
能力类指标(accuracy、pass@k、domain loss)不受影响。

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
