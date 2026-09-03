---
question: base checkpoint 落地后做什么——SFT → RL 门 → 两条并行 RL 轨道(数学 RLVR / 代码 agent RL)的可执行计划,轨迹校准方案按落点分段
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

- **数据**:e1-24 的 Claude Code transcripts pack(agentic 格式,`scripts/loader.format_agentic`:assistant 文本监督、tool 输出 mask)。最新实测 v10:4,002 行 / 16,090 对 / 5,766,589 token / 76.7% 工具循环;v11 重跑中(13:40Z 因磁盘满死过一次),预计多约 40 行。as of 13:40Z,v11 终报与 pack 路径 pending——落地前替换成 fact id 与确切路径。
- **入口闸**:base checkpoint + panel 基线(§0)。
- **出口数**:SFT 后重跑 panel 六指标 + math-hard pass@1/pass@8。出口判据不是「SFT 比 base 好」,是「SFT 后 pass@8−pass@1 gap 是否过 RL 门」——SFT 的产品是 RL 的起点,不是终点。
- **失败决策**:
  - panel 指标 SFT 后显著退化(任一指标动过 2.28σ)→ pack 或学习率有问题,停,查 pack 污染与 mask,不进 RL。
  - panel 持平、gap < 15pt → RL 门不开(§2),SFT 段产出的是「这个 checkpoint 上没有可放大的东西」这个读数。
- **预算**:pack 安装格式 ~192M tokens(scale_36b_plan.md 实测量级);20B SFT 1-2 epoch,8 卡,估 ~1-2 天(按 200M 的 2.2% 读数外推有风险,预算按实测吞吐落地前重算)。

## 2. RL 门(pass@8−pass@1 ≥ 15pt)

- **判据**(readout_30b_prereg.md:52,预登记):math-hard 上 pass@8−pass@1 **≥ 15pt** → RL 可开;< 15pt → RL 不开,读数「不是 RL 没用,是这个 checkpoint 上没有 RL 可放大的东西」。这是 continue-to-RL 决策,不是 stop。
- **读法**:n≥1000;gap 是配对差(pass@8⊇pass@1 同题相关,SE ~1-2pt@1036),15pt 阈值远在可读性之上。三态:动了(≥15pt)/ 平了(可读但 <15pt)/ 地板(pass@8 本身在地板,不可判)。200M 上的读数是 3.5pt = 平了,不是地板。
- **预算**:pass@8 = 8 次生成/题,n=1036 → 8288 次生成;lane 卡上跑,生成式评测与训练块互斥(06:50 规则:run 活着时 block 卡只跑似然类),所以门的测量窗口 = run 结束后或空卡期。
- **生成参数(2026-09-03 钉死,与 200M 读数同源)**:k=8、temperature 0.8、max_new=512——`eval/code_l0prime.py:199` freeze_hard 的默认值,200M 读数用的就是这组(`eval/score_math_formatfree.py:7` 记录 "t=0.8, k=8")。pass@1 与 pass@8 同一份题、同一次运行的两个统计量,不另跑。

## 3. RL 段:两条并行轨道

用户 2026-09-02 16:05 裁定(docs/lessons/claude_code_rl.md Ruling,main 52ff763):RL 主线 = Claude Code 外壳 + tileRL 引擎/训练器 + 容器 + 测试奖励。数学 RLVR 是两条并行轨道之一,不是唯一。

### 3a. 数学 RLVR(GSPO,boxed 奖励)

- **配方**(algorithms/README.md,已实现):RLVR GSPO 循环,fp32 master 权重(1e-6 AdamW 更新低于 bf16 ULP),FP8 训练副本 + bf16 生成分离(FP8 量化噪声降采样质量),DDP 同 prompt 不同响应、adv=0 时 loss 精确为 0。
- **数据**:`data/rl/rlvr_math.jsonl`(school_math_r1_zh + gsm8k_zh,`algorithms/prepare_rlvr.py` 产出)。
- **入口闸**:RL 门 ≥15pt(§2)+ SFT checkpoint。
- **出口数**:math-hard pass@1 相对 SFT 起点 +8.8pt(n≥1000,2δ);退化率 beside 报告。
- **失败决策**:pass@1 不动或退化 → 回 SFT 段查 pack,或判 RL 配方在这个规模不成立(读数,不是猜)。
- **预算**:8 卡全占的生成式训练;吞吐按 tilerl-17 实测中位数 11.87K tok/s/gpu 换算(见 §5②)。

### 3b. 代码 agent RL(tileRL GRPO,测试奖励)

- **形态**:Claude Code 是 agent 外壳,tileRL 是推理引擎和训练器,容器是环境,测试是奖励。策略先 27B(Qwen3.8-27B NVFP4,LoRA on frozen fp4 base),500M 在能驱动 tool loop 后作为被服务策略进同一 API,不进 tileRL 引擎做模型类。
- **任务库**:3b-9 已 admitted 134 个 Exercism 任务(≥5 测试、MIT;as of 13:40Z,在 3b worktree,提交 pending);Aider polyglot 与 EvalPlus 还在 fetch。阶段 5 的 20 任务试点从 admitted 集里抽,落地前替换成 fact id。
- **五阶段**(裁定固定顺序,前一阶段的 done 成为产物才开下一阶段;1-4 不需要 GPU,现在就能开):
  1. API 适配(tilerl-19):tileRL OpenAI server 前的 Anthropic Messages 兼容层;done = `claude -p` 对 tiny 模型完成一次 tool call + 最终回答。
  2. 镜像(tilerl-19):tileRL server + shim + Claude Code CLI + python3 + 任务库,一命令启动;done = 冷启动端到端跑通阶段 1。
  3. 沙箱(de-28):每 rollout 一容器,任务库读写挂载,测试在容器内跑,K=8 并行;done = 8 个 rollout 的 diff 与测试结果收齐。
  4. 奖励+轨迹(e1-24):奖励 = 测试通过;轨迹 = 会话 transcript JSONL → token ids(同 shim 渲染的 ChatML)+ server 返回的 per-token logprobs;done = 一条 rollout 的 transcript 复现 server 采样的精确 token 序列。
  5. 训练(tilerl-19):tileRL GRPO over K rollouts,LoRA on served weights,sampler=trainer 无权重同步;done = 20 任务试点奖励上升且 MMLU 不降(噪声内)。
- **入口闸**:阶段 4 的 done(一条 rollout transcript 复现采样序列)。
- **出口数**:20 任务试点奖励上升 + MMLU 不降。
- **失败决策**:500M 不进这条轨道,只做 SFT。
- **预算**:阶段 1-4 零 GPU;阶段 5 一卡(run 结束后),27B GRPO pilot 约一卡日。

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

1. SFT pack 的确切路径与行数:v10 已测(16,090 对 / 5,766,589 token / 76.7% 工具循环),v11 终报 + 路径 pending(e1-24,2026-09-03 仍 open,owner e1)。任务库 134 个 Exercism admitted(3b-9),提交 pending。
2. SFT 与 RL 的实测吞吐 → 预算的卡时数字:数学 RLVR 轨道用 tilerl-17 的 11.87K tok/s/gpu 中位数换算,公式 `卡时 = 训练 token 数 / (11.87K × 8 × 3600) × 生成开销倍数`;`data/rl/rlvr_math.jsonl` 尚未入库(prepare_rlvr.py 已实现,数据未构建),token 数与生成开销倍数落地前填。代码 agent 轨道阶段 5 按 27B GRPO pilot 实测填。
3. ~~pass@8 的生成温度与 max_new~~ **已钉(2026-09-03)**:k=8 / t=0.8 / max_new=512,与 200M 读数同源(见 §2)。
4. ~~RL 门的题目版本~~ **已钉(2026-09-03)**:`data/synthetic/math_hard_eval_1k.jsonl`,1032 行,sha256 `3ce9b0ff7fc6253c0d23c41cd360f09242f8a9a67ed187f4f56f812957ac703b`。预注册登记的 n=1036 与磁盘 1032 行差 4 行——落地前以跑通的过滤后计数为准,门的判据不变(n≥1000)。
