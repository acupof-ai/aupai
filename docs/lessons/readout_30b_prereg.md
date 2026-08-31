---
question: 30B 预训练的读数怎么判——每个里程碑、每个指标,多少算动、动了之后干什么?
status: recorded
source: fb tasking 2026-08-31; docs/lessons/scale_36b_plan.md §1b (recipe), §1 (roles); facts/base_eval.json (be.panel_expressive_seed_variance, be.degeneration_rate, be.ctx_length_p324); facts/data_scaling.json (sigma-hat=0.0516)
---

# 30B 预训练读数:预注册(2026-08-31,t22 启动前冻结)

用户的 ask:"看 30B 预训练的效果"。本文在 run 启动前写死:里程碑、每个指标的
n 和判变阈值、以及判完之后的动作(continue / stop / change mix)。**没有决策规则
的指标不进读数。**

实现:`scripts/readout_30b.py`(t34)——本预注册即代码,里程碑落地时直接跑判定,不争论。

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
