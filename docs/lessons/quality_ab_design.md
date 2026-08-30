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
