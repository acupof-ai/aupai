---
question: "e1 的 KDA vs 全注意力 A/B(--attn_every 4 vs 1,4 seed × 2 臂,0.2b)结果出来按什么读——预注册,看到数之前写死"
status: recorded
source: "aupai-fb 任务 2026-08-30;e1 的实验设计;docs/lessons/scaling_fit_protocol.md v1.4(噪声锚);facts/data_scaling.json"
---

# attn_every A/B 读法(预注册)

e1 的设计,fb 的三条读法,本文件坐实判据并补上 fb 要求自算的三项。
**设计是 e1 的,判据不是它自己的**——读数规则在看到数之前冻结。

## 0. 设计(照录,责任在 e1)

- 臂 A:`--attn_every 1`(12 层 GatedMLA 全注意力),194M 参数
- 臂 B:`--attn_every 4`(9 KDA + 3 GatedMLA),206M 参数(+5.8%)
- 4 seed × 2 臂 = 8 run,0.2b 点(218 步量级),主指标分域 NLL,e1 声称 MDE≈0.035

## 1. MDE 自校准(替代 e1 的 0.035——它是估的)

4v4 两样本比较,MDE = (t₀.₉₇₅,₆ + t₀.₈,₆)·s·√(2/4) = **2.371·s**,s = seed 间 SD。

- e1 的 0.035 要求 s = 0.015。**项目内部 seed 方差从未测过**(拟合协议 v1.4 自己声明
  "无内部重复 run");唯一锚点是 Kaplan run-to-run 0.05 → MDE 应为 **0.118 ≈ 0.12**。
  0.035 是把未测的方差当成了已测,按 3.9pt 先例记为**不可复现**,不采用。
- **冻结公式**:决策阈值 = 2.371·s_pooled,s_pooled 从 8 个 run 自身的 seed 散布算
  (合并臂内 SD)。MDE 不是假设,是从这批 run 里长出来的。
- **df=6 的不确定带必须报**:s 的 95% CI ≈ [0.64s, 2.20s](χ²,₆),MDE 同比例。
  gap 落进 [2.371·0.64s, 2.371·2.20s] 带内 = **无分辨率**,不读。
- Kaplan 0.05 是外部锚(3M 参数、他们的数据),只作 sanity check:s_pooled 若 >0.10,
  说明 0.2b 点 seed 噪声比 Kaplan 还大,整批 8 run 的解读降级为"趋势"。

## 2. 主指标:聚合 val NLL,事前定死

- **主指标 = 聚合 val NLL**(固定 val 集,与拟合曲线同量、同集)。一个数,无多重比较。
- **分域 NLL 全部降为二级诊断**:七个域(web_hq/textbook/wiki/en/math/code/chat)全报,
  禁止挑最好看的域报。任何"某域赢了"的决策性主张须过 Bonferroni(0.05/7),
  否则只是诊断,不是证据。
- 理由:七域挑一 = 七次比较报一次,family-wise α 涨到 30%。

## 3. 次指标:0.2b 上只报趋势,不进决策

CLiMP(4.4pt/范式)、LAMBADA-zh(4.1pt)、math v2 似然孪生(2.5pt)的 MDE 是
**采样分辨率**,在任何 checkpoint 上都成立;但:

- known-answer 校准在 3.24b **旧架构**上做的,0.2b 新架构的读数贴地板,
  臂间 gap 未知且大概率小;
- **冻结规则**:次指标同时满足 (a) 臂间 spread > 该指标 MDE 且 (b) 读数在地板+MDE 之上,
  才可写进决策;否则只报趋势,**事后不得当证据引用**(空结果不认证格子,
  也不允许事后把趋势读数升级成证据)。

## 4. 决策树(fb 三条,坐实)

```
gap = 聚合 val NLL(B) − 聚合 val NLL(A),阈值 = 2.371·s_pooled(§1)
├── A 赢 ≥ 阈值(全注意力更好)→ 阶梯改跑 attn_every 1
│   ⚠ 阶梯协调:六个点若已在 attn_every 4 上跑,换臂 = 曲线混架构,
│     必须在点之间切换并记录,或等下一轮阶梯——fb 裁决时序
├── B 赢 ≥ 阈值(KDA 更好)→ 不可直接读,先跑配平臂
│   混淆是二元的:206M vs 194M(+5.8%)的容量差对任何 margin 都存在,
│   margin 大小不改变混淆是否存在(fb 的推导,复核成立;补一句:没有 200M 的
│   容量弹性实测,margin 也无法用来校准混淆——配平臂是唯一解法)。
│   配平臂:--ffn_hidden 3392(全注意力臂加宽到 206M 量级),4 seed。
│   验收:两臂参数数从 run 日志实测,差 ≤1%(信日志不信配置名)。
│   ├── 配平后 B 仍赢 ≥ 阈值 → KDA 留下,容量混淆排除
│   ├── 配平后平或输 → 之前的赢是容量,KDA 不留
│   └── 配平臂自身 seed 方差同样按 §1 读
└── |gap| < 阈值 → KDA 留下,记"0.2b 无分辨率"
    到 3.24b 用阶梯自己的 checkpoint 重问(阶梯两臂都在跑,零额外成本)。
    e1 原设计"无显著差异 → 删 KDA"违反空结果纪律,作废。
```

## 5. 这份读法不决定什么

- 不决定阶梯的时序(换臂点)——fb 裁决。
- 不决定配平臂之外的架构消融——另一份预注册。
- 次指标无论结果如何都不进本决策(§3)。

## Sources

- e1 的 A/B 设计(2026-08-30,经 fb 转述)
- docs/lessons/scaling_fit_protocol.md v1.4(Kaplan 0.05 噪声锚、空结果纪律)
- docs/lessons/base_eval_panel.md v1.3(次指标 MDE、known-answer 校准的架构归属)
- facts/data_scaling.json(ds.kaplan_noise)
