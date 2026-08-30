---
question: "200M 上数据质量到底值多少：过滤强度的两臂（三臂）预训练 A/B 怎么设计才可执行"
status: recorded
source: "fb tasking 2026-08-30; noise anchors from facts/data_scaling.json#ds.kaplan_noise (seed 0.05 nat) and docs/lessons/base_eval_panel.md (eval 0.01 nat); supply numbers from fb (fineweb2 1.8B at keep=0.40)"
---

# 数据质量 A/B：可执行实验设计

## 问题

固定 token 预算、同架构、同配比，**只改过滤强度**，200M 上能不能测出差异？预注册双侧检验——"更严更好"和"更严更差"（DCLM 人读一致率反相关 R²<0.3，`#dq.classifier.human_agreement_counter`）都是活假设。**效应小于 MDE 本身就是一等结论**：它告诉我们 200M 阶段别在过滤上花钱。

## 三臂（固定 T，同一打分池）

| 臂 | web_hq 选择规则 |  treatment |
|---|---|---|
| A 对照 | 质量头 top-40% + 全 garbage 链（含 Step 0+1） | 当前生产 |
| B 严切 | 质量头 top-20% + 全 garbage 链 | 阈值强度 |
| C 无正则 | 质量头 top-40%，garbage 链关（`AUPAI_NO_GARBAGE=1`） | 正则层定价 |

三臂从**同一份 fineweb2 打分快照**取数（3b 重建正在产出 `data/web_scores.npy`，直接复用），T 相同、域权重相同、每域 token 数相同、同 seed 的跨域交织顺序相同——**唯一差别是 web_hq 的内容选择**。A vs B 定价阈值，A vs C 定价正则层。

## 混杂控制（fb 的问题 1）

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

- [ ] E2：六个点跑完后跑 domain_loss.py（GPU 4-7 闲时插）
- [ ] 复用 3b 的 `data/web_scores.npy` 建三臂池：A=top-40% 去正则命中；B=top-20% 去正则命中；C=top-40% 不去
- [ ] 管线小改：score 阈值参数化（现在写死 top-40%）；arm C 用现有 `AUPAI_NO_GARBAGE=1`
- [ ] 阶段 1：0.2b × 8 seed × 3 臂 = 24 run
- [ ] 门控判定 → 阶段 2（0.8b × 4 seed × 3 臂 + 全面板）
- [ ] 全部数字进 facts，预注册规则不许事后改

## 边界

- 本实验定价的是 **web_hq 域内的过滤强度**。textbook（49.6% 合成）的质量问题是另一个曝光对齐两臂（mix 注释里已挂账），不在这里。
- CCI3 解禁路径挂在 fastText 上，但 CCI3 不是瓶颈（fineweb2 余量 0.75B），不赶。
- 3.24b 点不做本实验（B 臂打穿池子）；要做只能先补 CCI3 打分，那是另一个设计。
