---
question: "200M 上数据质量到底值多少：过滤强度三臂 + web_hq 权重消融，共用对照的可执行实验设计"
status: recorded
source: "fb tasking 2026-08-30; noise anchors from facts/data_scaling.json#ds.kaplan_noise (seed 0.05 nat) and docs/lessons/base_eval_panel.md (eval 0.01 nat); supply numbers from fb (fineweb2 1.8B at keep=0.40)"
---

# 数据质量 A/B：可执行实验设计

两个并列候选实验，共用同一对照臂。fb 在六个点跑完后一起裁。

## 问题

固定 token 预算、同架构、同配比，**只改过滤强度**，200M 上能不能测出差异？预注册双侧检验——"更严更好"和"更严更差"（DCLM 人读一致率反相关 R²<0.3，`#dq.classifier.human_agreement_counter`）都是活假设。**效应小于 MDE 本身就是一等结论**：它告诉我们 200M 阶段别在过滤上花钱。

## 实验 F（过滤强度）：三臂（固定 T，同一打分池）

| 臂 | web_hq 选择规则 |  treatment |
|---|---|---|
| A 对照 | 质量头 top-40% + 全 garbage 链（含 Step 0+1） | 当前生产 |
| B 严切 | 质量头 top-20% + 全 garbage 链 | 阈值强度 |
| C 无正则 | 质量头 top-40%，garbage 链关（`AUPAI_NO_GARBAGE=1`） | 正则层定价 |

三臂从**同一份 fineweb2 打分快照**取数（3b 重建正在产出 `data/web_scores.npy`，直接复用），T 相同、域权重相同、每域 token 数相同、同 seed 的跨域交织顺序相同——**唯一差别是 web_hq 的内容选择**。A vs B 定价阈值，A vs C 定价正则层。

## 实验 W（权重消融）：固定过滤，变 web_hq 权重

- **问题**：继承来的 web_hq 32.4% 对不对？（从来没有实验支持）
- **臂**：A（共用对照，当前配比）vs W（web_hq 32.4%→20%，+12.4% 给 textbook——E2 曲线上斜率最陡的大域）
- **为什么可能比 F 更值钱**：E2 的斜率对权重是**直接决策量**——无论 A_d/B_d 谁因，web_hq 边际 token 换的 loss 最少，挪给陡的域就赚。行动只改一个 mix json，不需要新语料、不需要等打分。
- **供给**：web_hq 用量减少，无打穿问题；textbook 是合成语料，供给弹性。
- **成本**：0.2b × 8 seed = 8 run（A 共用），边际 0.8 GPU-h。

## 两实验对比

| | 实验 F（过滤强度） | 实验 W（权重消融） |
|---|---|---|
| 回答 | 过滤在 200M 值不值 | 32.4% 权重对不对 |
| 动机 | DCLM 反相关 + 数据线的存在理由 | E2 斜率直接读数 |
| 臂 | A/B/C 三臂 | A/W 两臂 |
| 0.2b 跑次 | 24 run（含 A） | +8 run（A 共用）→ 同批合计 32 run ≈ 3.2 GPU-h |
| 行动 | 过滤链改不改 | 一个 mix json |
| 依赖 | 3b 的 web_scores 快照 | 无 |

**顺序**：可同批跑（共用 A，边际成本只 8 run）。若被迫排序：**W 先**——更便宜、直接读 E2、A_d/B_d 不分也能决策；F 后——问题更大，0.8b 全面板才有完整读数。fb 裁定。

## 混杂控制（fb 的问题 1，两实验通用）

"更严的臂要用更多 raw → raw 来源不同"这个混杂**只在池子被打穿时存在**。供给核算：

- fineweb2 在 keep=0.40 下供给 1.8B token；keep=0.20 下 0.9B
- 预算点 T 的 web 需求 = 0.324×T
- B 臂不打穿的条件：0.324×T ≤ 0.9B → **T ≤ 2.78B**

3.24b 点（web 需求 1.05B）会把 B 臂逼到 CCI3 补量 = 混杂，**排除**。≤1.6b（需求 0.52B，余量 1.7×）三臂都在同一池内，无来源混杂。

## 预算点（fb 的问题 2）

噪声账：

- 种子噪声 σ_seed ≈ 0.05 nat（`facts/data_scaling.json#ds.kaplan_noise`）
- eval 噪声 0.01 nat（domain_loss.py，262K token/域，±0.01 nat）——比种子噪声小 5 倍，不瓶颈
- 两臂各 n seed 的可分辨 Δ ≈ 2·σ_seed·√(2/n) = **0.14/√n nat**

分阶段，先便宜后贵：

| 阶段 | 预算点 | 每臂 seed | 可分辨 Δ | 成本 | 读法 |
|---|---|---|---|---|---|
| 1 | 0.2b（6 min/run） | 8 | ≈0.05 nat | 24 run ≈ 2.4 GPU-h | 只读 NLL（面板下游指标在此规模无分辨率，面板规则 <10pt 砍） |
| 2（门控后） | 0.8b（~24 min/run） | 4 | ≈0.07 nat | 12 run ≈ 4.8 GPU-h + 全面板 eval | NLL + CLiMP/LAMBADA/math-twins/ceval |

阶段 1 的门控：NLL 出现 ≥0.05 nat 的信号 → 上阶段 2；三臂都在 ±0.05 内 → 效应小于 0.05 nat，阶段 2 的价值存疑，回来再议。

## 显著性（fb 的问题 3），预注册

- **主指标**：冻结面板 eval  mix 上的留出 NLL（`scripts/domain_loss.py`，每域首 shard 头部 4000 行打包，262K token/域，所有臂同一 eval）
- **MDE = 0.05 nat**（≈种子噪声；16× 阶梯实测动 1.66-1.99 nat，比 MDE 大 30 倍——值得有的质量效应应该能越过它）
- **判定规则**（每对臂）：Δ = mean(NLL_treatment − NLL_control)，跨 seed
  - Δ ≤ −0.05 且 seed 分布不重叠（或 Mann-Whitney p<0.05）→ 处理臂胜
  - Δ ≥ +0.05 同样规则 → 处理臂负
  - |Δ| < 0.05 → **无可检测效应**（价值结论，见下）
- **次指标**：只在 0.8b 读，按面板分辨率规则；次指标动而主指标不动 = 报告，不采纳
- **方向预注册：双侧**

## 零结果的价值

0.8b 上三臂都落在 ±0.05 nat 内 → **"200M 上，top-20% 到无正则之间的过滤强度效应小于 0.05 nat"** → 200M 这轮停止在过滤上投资，力气花在数据量和域配比上。这是本实验的一等产出，不是失败。

## E2（零成本，先跑）

六个 0830v1 checkpoint 的分域 loss 对各自 token 数画同一坐标系（`scripts/domain_loss.py --ckpt A --ckpt B ... --json runs/domain_loss.json`）：

- 坍缩到一条 L(C)=A+B·C^−β → 各域按 token 数走同一曲线，当前配比无明显错配
- 某域在曲线上方 → 该域在"喂了多少"之外还有质量问题，是 B 臂的天然靶子

成本：6 ckpt × 7 域 × 262K token ≈ 11M token 前向，~30 min。六个点跑完即跑，无需等 A/B。

## 执行清单

- [ ] E2：~~六个点跑完后跑 domain_loss.py~~ **已测**（`#dq.e2.matched_token_protocol`）；新架构六个点落地后重跑（旧架构数不可混用）
- [ ] 复用 3b 的 `data/web_scores.npy` 建三臂池：A=top-40% 去正则命中；B=top-20% 去正则命中；C=top-40% 不去
- [ ] 管线小改：score 阈值参数化（现在写死 top-40%）；arm C 用现有 `AUPAI_NO_GARBAGE=1`
- [ ] 实验 W：建 `mix_scale_0.2b_web20.json`（web_hq 20%、textbook 62%，其余不变）
- [ ] 阶段 1：0.2b × 8 seed ×（A/B/C/W 四臂，A 共用）= 32 run
- [ ] 门控判定 → 阶段 2（0.8b × 4 seed × 胜出臂 + 全面板）
- [ ] 全部数字进 facts，预注册规则不许事后改

## 边界

- 本实验定价的是 **web_hq 域内的过滤强度**。textbook（49.6% 合成）的质量问题是另一个曝光对齐两臂（mix 注释里已挂账），不在这里。
- CCI3 解禁路径挂在 fastText 上，但 CCI3 不是瓶颈（fineweb2 余量 0.75B），不赶。
- 3.24b 点不做本实验（B 臂打穿池子）；要做只能先补 CCI3 打分，那是另一个设计。

---

## 重算 v3（2026-08-30，fb 终版）：4+4 对称，F 臂白拿 e1 的 seed

**裁决**：质量头出局，W 臂的"那一刀" = Step 0/1 模式匹配（纯 CPU，用户定的抽样+模式匹配路线）。控制臂不是阶梯六点——单点 1 seed 时 MDE 地板 0.10 nat（n→∞ 仍 0.10，往 W 臂加 seed 是往被单点卡死的比较里倒钱）。控制臂是 **e1 的 A/B 在 0.2b 上本来就要跑的 4 个 seed**（attn_every 4 臂），边际成本 0。

**设计**：
- F 臂（控制）= e1 的 4 seed，语料无 Step 0/1
- W 臂 = `web_hq_step01/`（web_hq 过当前 filters 含 Step 0/1）× 4 seed = 24 GPU-min
- MDE = 2.8σ·√(1/4+1/4) ≈ 1.98σ，**σ 用实测**（seed 0-3 方差，3 df），不用 0.05 的文献假设
- 排在 e1 A/B 之后（F 臂 = e1 的一个臂）

**σ 实测先行**：seed 0-3 方差出来先报 fb——它决定 b0 拟合协议 RMS≤0.05 门槛能不能达到（6 点 3 参数拟合的期望 RMS ≈ 0.71σ，σ=0.035 → 期望 0.025，门槛可达；σ 更大则门槛要重议）。

**路径**：
1. web_hq 冻结（指纹落地、PROVENANCE 不再是 in progress）后测 Step 0/1 命中率（纯 CPU，5 万篇抽样估计）
2. 命中率够高 → 建 `web_hq_step01/`（**新目录，不重指现有域**——0.2b 事故买来的规矩）
3. W 臂 4 seed（24 min）

**被否方案留档**：
- 六点阶梯当控制：单点 1 seed，MDE 地板 0.10 nat，8+1 用 9 run 只换 0.104；对称 4+4 用 8 run 换 0.069——对称永远赢
- 拟合曲线当控制：0.2b 是阶梯最小端点，三参数曲线在端点杠杆最高；"等效 2 seed"和 σ=0.035 都是假设（拟合残差含模型误设，不是 seed 间方差）

**W（权重消融 32.4%→20%）是另一个问题**：W/F 问"这一刀值多少"，权重消融问"web 域的权重值多少"。若 Step 0/1 命中率 <2%，正确记法是"这一刀已在语料里，W/F 无对比度"，权重消融**另起**为独立实验——不顶替 W/F 的位置（否则半年后会有人以为测过质量切而其实没有）。

**预注册（2026-08-30，看到命中率之前定死）**：Step 0/1 在 web_hq 上的命中率 **≥2%** → W/F 成立，建 `web_hq_step01/`，4 seed。下限的唯一理由是物理上限：两臂 98% 文档相同时，效应上界 = p × Δ_marginal，其中 Δ_marginal 是"训练没见过这些文档"对 val NLL 的改变——二阶量，远小于文档间 loss 差（2-3 nat）；以 1 nat/文档作宽松上界，p=2% → 上界 <0.02 nat < MDE 0.07-0.10。**事前就能算出低于分辨率的实验不该跑。** （fb 的 <2%/>5% 框架与此一致是巧合，不作独立证据。）p<2% → W/F 不跑，报"这一刀已在语料里"。精度人读（100 篇命中，3b）条件触发（p≥2%）、与 W 臂并行——是 W 臂结果的解读前提，不是开跑前提。

---

## 普查结果（2026-08-30）：W/F 死亡，预注册触发

全 62 shard 普查（1,366,324 篇，指纹 30838d423348b2e5；50,000 篇 text[:600] reservoir，过滤器精确作用窗口；`facts/multilingual.json#mlm.corpus.step01_hit_rate`）：

- **Step 0/1 命中率 0.326% ± 0.050%**（163/50000）≪ 2% 预注册线 → **W/F 死于自己的预注册阈值，不是零结果**——零结果是"看了没看到"，这是"没有对比度可测，且判定规则事前定死"。两臂 99.67% 相同，效应上界 p × Δ_marginal ≪ MDE。记法按预案："这一刀已在语料里"——旧 garbage 链在构建时已把 web_hq 清干净，Step 0/1 的边际捕获只剩 0.33%。
- **pre-df2b774 链命中 0/50000**：3b 的 PROVENANCE 构建记录（全链跑过）在确定性层面成立；预测的零回来了。union = Step 0/1 精确相等（一致性 163 = 163+0−0）。
- **权重消融（32.4%→20%）另起为独立实验**，不顶替 W/F 的位置——半年后没人会以为测过质量切。
- e1 的 4 个 0.2b seed 照跑（e1 自己的 A/B），实测 σ 仍给 b0 的 RMS≤0.05 把门。
- 精度人读（100 篇命中，3b）是 p≥2% 的条件触发项，随 W/F 一并取消。

**副产物（同一次扫描）**：繁体篇级 17.715% ± 0.064%（vs fb 前 3 shard 侦察 15.20%，差 = 源排序，shard 1 = 20.9%；**语料属性：web_hq 按源排序未洗乱，任何按 shard 的抽样必须跨 shard 范围，不能取前缀**——见 `mlm.corpus.web_hq_ordered_by_source`）；fertility 1.161×（最小对 n=30）/ 1.191×（语料级），merge 分割效应非 byte-fallback；whole-char gate 口径复核——`contained` 复现 gate 自测值（simp 0.9929 vs 0.9913），繁体 0.9682 > 0.95 阈值，不触发 vocab-unfreeze。详见 `facts/multilingual.json` 的 `mlm.corpus.traditional_rate` / `mlm.fertility.traditional` / `mlm.tokenizer.wholechar_gate_check`。
