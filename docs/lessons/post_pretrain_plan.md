---
question: base checkpoint 落地后做什么——SFT → RL 门 → RL 配方的可执行计划,轨迹校准方案按落点分段
status: open
source: readout_30b_prereg.md (RL 门) + algorithms/README.md (RLVR 配方) + trajectory_calibration_survey.md (方案分级) + scale_36b_plan.md (SFT 约束) + facts/base_eval.json (200M 读数)
---

# 预训练后计划(post-pretrain plan)

目标:base checkpoint 落地那一小时,有人能照本文档开始执行,不需要再等任何决定。
读法:每段写四件事——入口闸、出口数、失败决策、token/卡预算。预算按 8 卡(7 训练块 + 1 lane)、20B base 估。

## 0. 入口条件(base 落地时必须同时成立)

- base checkpoint 落地且 `eval_panel`(be.eval_panel_200m 六指标:per-domain NLL / CLiMP 最小对 / LAMBADA-zh / math-hard likelihood / ceval MC / 退化率)跑完——panel 是 SFT 前的基线,没有基线的 SFT 读数不可解释。
- math-hard pass@1 与 pass@8 已测(n≥1000,同一份题)——RL 门的两个数在 base 上先读一次,SFT 后再读一次,门看的是 SFT 后的 gap。
- SFT pack 已过 `scripts/test_sft_pack.py`(loss mask:tool 回合全 -100;holdout 零污染)。

## 1. SFT 段

- **数据**:3b 的 v1 pack(agentic 格式,`datagen/` 渲染器产出;格式约束见 scale_36b_plan.md:每源过渲染器、形态由代码定)。pack 路径与行数由 3b 在落地前填进本节——这是唯一的开放项。
- **入口闸**:base checkpoint + panel 基线(§0)。
- **出口数**:SFT 后重跑 panel 六指标 + math-hard pass@1/pass@8。出口判据不是「SFT 比 base 好」,是「SFT 后 pass@8−pass@1 gap 是否过 RL 门」——SFT 的产品是 RL 的起点,不是终点。
- **失败决策**:
  - panel 指标 SFT 后显著退化(任一指标动过 2.28σ)→ pack 或学习率有问题,停,查 pack 污染与 mask,不进 RL。
  - panel 持平、gap < 15pt → RL 门不开(§2),SFT 段产出的是「这个 checkpoint 上没有可放大的东西」这个读数。
- **预算**:pack 安装格式 ~192M tokens(scale_36b_plan.md 实测量级);20B SFT 1-2 epoch,8 卡,估 ~1-2 天(按 200M 的 2.2% 读数外推有风险,预算按实测吞吐落地前重算)。

## 2. RL 门(pass@8−pass@1 ≥ 15pt)

- **判据**(readout_30b_prereg.md:52,预登记):math-hard 上 pass@8−pass@1 **≥ 15pt** → RL 可开;< 15pt → RL 不开,读数「不是 RL 没用,是这个 checkpoint 上没有 RL 可放大的东西」。这是 continue-to-RL 决策,不是 stop。
- **读法**:n≥1000;gap 是配对差(pass@8⊇pass@1 同题相关,SE ~1-2pt@1036),15pt 阈值远在可读性之上。三态:动了(≥15pt)/ 平了(可读但 <15pt)/ 地板(pass@8 本身在地板,不可判)。200M 上的读数是 3.5pt = 平了,不是地板。
- **预算**:pass@8 = 8 次生成/题,n=1000 → 8000 次生成;lane 卡上跑,生成式评测与训练块互斥(06:50 规则:run 活着时 block 卡只跑似然类),所以门的测量窗口 = run 结束后或空卡期。

## 3. RL 段(algorithms/rlvr)

- **配方**(algorithms/README.md,已实现):RLVR GSPO 循环,fp32 master 权重(1e-6 AdamW 更新低于 bf16 ULP),FP8 训练副本 + bf16 生成分离(FP8 量化噪声降采样质量),DDP 同 prompt 不同响应、adv=0 时 loss 精确为 0。
- **数据**:`data/rl/rlvr_math.jsonl`(school_math_r1_zh + gsm8k_zh,`algorithms/prepare_rlvr.py` 产出)。
- **入口闸**:RL 门 ≥15pt(§2)+ SFT checkpoint。
- **出口数**:math-hard pass@1 相对 SFT 起点的提升(预登记阈值:8.8pt @ n≥1000,2δ);退化率 beside 报告。
- **失败决策**:pass@1 不动或退化 → 回 SFT 段查 pack,或判 RL 配方在这个规模不成立(读数,不是猜)。
- **预算**:RL 是生成式训练,8 卡全占;按 GSPO 每步 8 卡生成+更新,具体步数以落地前的实测吞吐填。

## 4. 轨迹校准方案按落点分段(trajectory_calibration_survey.md)

我们的仪器:teacher-forced 0.727/0.6875 vs free-running LCS ~0.23,误差即时非渐衰。0/14 篇干预论文测过轨迹级统计——想要轨迹级证据只能自己造,每段的读法必须预登记。

| 方案 | 落点 | 预登记读法 | 预算 |
|---|---|---|---|
| Scheduled sampling(BSS 变体) | **SFT 段**(离线数据交错,不改循环;唯一有正面证据的) | SFT 后 free-running LCS vs 纯 SFT 对照,同题 n≥200;仪器 = 我们自己的 0.727→0.23 对 | 交错进 SFT pack,零额外卡;BSS 离线交错 ~1x SFT 数据量 |
| Unlikelihood(退化/重复) | **SFT 段**(辅助损失,token 级直接加) | 退化率 beside pass@1;它治退化不治一致性,不读 LCS | 辅助损失,~5-10% 训练步开销 |
| DAgger | **RL 段**(需要已训练策略 rollout + oracle 重标注) | 同 SS 的 LCS 读法;DAgger 迭代外环,每轮 = 一次生成+重标注 | 每轮 = lane 卡生成 + 标注;轮数开放 |
| 自蒸馏 / rationale-SFT(Orca 式) | **SFT 段**(大学教师 trace 上的 SFT) | 同 SFT 段出口数;trace 来源是预算大头 | trace 采购/生成成本,未测 |
| 序列级 MRT/margin | **RL 段**(每步多候选打分,大成本乘数) | 同 RL 段出口数 | 每步 N 候选,N=4-8 倍生成成本 |

**深度赌注**:深度对轨迹校准的已发表测量不存在(两个方向都没有)——unsupported,不是 contradicted。不在本计划的任何一段里,除非先有自己的轨迹级测量。

## 5. 开放项(落地前必须填)

1. SFT pack 的确切路径与行数(3b 填)。
2. SFT 与 RL 的实测吞吐 → 预算的卡时数字(落地后用实测填,不用外推)。
3. pass@8 的生成温度与 max_new(预登记,进 panel 配置)。
4. RL 门的题目版本(math-hard v2,与 200M 读数同一份)。
