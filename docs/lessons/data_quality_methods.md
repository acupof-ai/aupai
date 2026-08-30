---
question: "给定新语料，训练前怎么判断现有过滤器是否仍有效；质量怎么提上去并能证明提上去了"
status: recorded
source: "literature search 2026-08-30 (3 agents, every arXiv ID fetch-verified) + project measurements; numbers in facts/data_quality.json"
---

# 数据质量方法：过滤器迁移判据 + 检测/测量协议

## 0. 交付物：过滤器迁移判据（recall-ratio）

**规则拆分**（`facts/data_quality.json#dq.criterion.rule_split`）：语料无关规则（字节/编码/重复度）永远有效；语料相关规则（pass1-3 正则表、SPAM 关键词、质量头阈值）换语料必然失效——CCI3-HQ 上保留 97.1% 而手读 40% 垃圾（`#dq.cci3.filter_retention`、`#dq.cci3.handread_150`）。

**判据**（`#dq.criterion.recall_ratio`）：

1. 新语料分层抽 N=150 篇手读标注（可训/垃圾+类型）；
2. 同一批跑现有语料相关过滤器，算 recall_new = P(过滤器=drop | 手读=垃圾)；
3. 对照过滤器在老家语料的召回 recall_home（web_labels.jsonl，已实测 3.4%，`#dq.criterion.positive_control_protocol`）；
4. 比值 ≥0.8 **且 recall_home ≥0.5 绝对下限** → VALID 可进训练；<0.5 → RECALIBRATE 不可进；中间且垃圾率 >10% → 也 RECALIBRATE；
5. 重标定是最小流程不是重新调参：手读垃圾按失效模式聚类，每类加 1-2 条正则或给质量头加几百个新标签，在**新样本**上重测到比值 ≥0.8。

**绝对下限的来由**（`#dq.criterion.ratio_needs_floor`）：光有比值会被"什么都不切"的过滤器骗过——任何语料上召回都是 0，比值恒为 1，处处 VALID。阳性对照恰好抓到了这个退化情形的镜像：现有正则层在老家语料上 recall_home 只有 3.4%（5/147，precision 5/5），0.8×3.4%=2.7% 的通过带等于给"结构性失明"发合格证。加上 recall_home ≥0.5 后，正则层自己在老家语料上就读作 RECALIBRATE 级——质量决策本来就住在质量头里（AUC 0.823，`#dq.quality_head.auc_vs_hand`），正则层只是 precision 1.0 的格式闸。

**Known-answer 验收**（`#dq.criterion.known_answer_cci3`、`#dq.criterion.cci3_measured`）：CCI3-HQ 是已知失效的语料，判据必须输出 RECALIBRATE。3b 实测 2×2 混淆矩阵：手读垃圾 59/150，过滤器只 drop 其中 1 篇 → **recall_new = 1.7%**，比值 0.017/0.034 = 0.50 且远低于绝对下限 → **RECALIBRATE，与已知答案一致**。（口径修正：1.7% 是在 120 字摘录上跑的；全文本生产路径上同一过滤器的基线为 11.9%，见 `#dq.regex.recal_step01`——结论不变，仍远低于 0.5 下限。）过滤器对 58/59 篇农场/洗稿/SEO/模板垃圾放行——是对这个类别的结构性盲，不是阈值差一点。手读标签由 3b 为质量审计产生，先于判据存在，不与判据共享假设。

**阳性对照（已测，`#dq.criterion.positive_control_protocol`）**：web_labels.jsonl 180 篇跑同一 `reject_reason()`，recall_home = 3.4%（5/147，全部 garbage_topic），precision = 5/5。注意两点：(1) 正则表从未在这批标签上调过（cosmopedia 取向、早于这批标签），所以这是独立测试不是 in-sample——低召回是更强的证据；(2) web_labels 的 y=0 口径是"无教育价值"（比 3b 的 usable/junk 严，垃圾率 81.7% vs 39.3%），召回差距有一部分是口径差，但 3.4% 仍说明正则层不做质量决策。

**第一个客户**：3b 的 CCI3 质量头打分。接口：3b 手读的 150 篇跑同一批过滤器出 2×2 混淆矩阵，判据出 go/no-go。

## 1. P0 — 跨文体质量判据不存在，但有可迁移的做法

**结论：没有被验证过的通用质量判据。** 三条独立直接测量（`#dq.universal.criterion_absent`）：

| 失效维度 | 实测 | 出处 |
|---|---|---|
| 跨语言 | FineWeb-Edu 英文头上中文网页 P 0.91 / R 0.02 / F1 0.03 | `#dq.fineweb_edu.chinese_failure` |
| 跨风格 | 改写成 Wikipedia 风格翻转 7% 过滤决定，26 个域名全部得分上升 | `#dq.register.style_flip` |
| 跨判据 | 三个主流分类器对同一语料的 HQ 交集仅 10.1% | `#dq.criterion.hq_intersection` |

机制：主流系统把"质量"操作化为"与目标分布的相似度"（Llama 3/SkyPile/RedPajama 都是 Wikipedia 适合性）。目标文体变了判据就该变。本项目质量头给 cosmopedia 打分低于原始网页，与风格敏感性机制一致（无直接发表记录，`#dq.register.style_flip`）。

**可迁移的做法**（文献反复出现的模式）：

1. **DCLM 式 fastText**：正例=你想要的文体样本，负例=新语料低分样本+农场样本。零 LLM 标注，下游效果超过 AskLLM 和困惑度过滤（Core 30.2 vs 28.6 vs 29.0，`#dq.classifier.sample_complexity`）。
2. **LLM 标注+蒸馏**：prompt 必须按文体重写，每个文体单独验证。已发表标注预算 140k-600k（`#dq.classifier.sample_complexity`），下限无文献答案，要自己跑学习曲线。
3. **多分类器集成**：Nemotron-CC 把 HQ 产出从 8-14% 提到 25%。
4. **按域分别设阈值**：CCI4 的 26 子域名各取 99.5 分位。

**成本**（100GB 中文规模，`#dq.cost.zero_shot_vs_distill`）：零样本 70B 直打 ~1100 H100-hours；蒸馏路线 ~6 H100-hours 标注 + CPU 训练。差两个数量级，且零样本每换一次文体/判据重付全价。

**一个指标澄清**：FineWeb-Edu 官方只报 F1 82%，从未报 AUC（`#dq.fineweb_edu.metrics_gap`）。本项目头的 AUC 0.823 与之不可比。

## 2. P1 — 损失降幅不能作为加权依据

**降幅模式是幂律学习曲线的机械预期**（`#dq.scaling.drop_mechanical`）：L_d(C)=A_d+B_d·C^−β 下，16× 跨度的降幅 Δ_d=(L_d−A_d)(1−16^−β)，与"当前剩余损失"成正比。小占比域离地板远，即使最优混合是均匀的也降幅更大。决策相关量是边际 w_d·dL_d/dC_d，不是 Δ_d。没有任何论文直接陈述过"小占比域降幅大"——这是幂律推论。

**零成本判别实验 E2**（`#dq.e2.matched_token_protocol`）：7 个域的损失对各自域 token 数 C_d=w_d·T 画同一坐标系。共享 (B,β) 下应坍缩到一条曲线；math/code 在 matched token 上位于曲线上方 → B_d 真更大 → 该加权；坍缩 → 纯假象。数据用 `scripts/domain_loss.py` 跑六个 0830v1 checkpoint（pod，3b/de），无需重训。

**math/code 加权的证据是真实的但有边界**：最强因果证据是 DeepSeekMath 有/无消融（1.3B：GSM8K 2.9→23.8，MMLU-STEM 19.5→33.1，通用 MMLU 49.1→54.9，`#dq.math.deepseekmath_ablation`）；Llama 3 配方 25% 数学+推理/17% 代码但无消融（`#dq.math.llama3_mix`）。反方向：DCLM 向已过滤数据混域平均 -1.2pp，phi-1.5 窄域加权牺牲知识，Kimi k1.5 证明 math/code 能力可后期 RL 注入（`#dq.math.dclm_counter`）。文献处方一致：DoReMi/RegMix/Data Mixing Laws 都用小 proxy 搜索权重，不读损失降幅（+6.5pp few-shot / 匹敌多训 48% 步数，`#dq.reweight.*`）。

**建议**：先跑 E2。坍缩 → 维持原比例，math/code 交给后期 SFT/RL；在曲线上方 → 做 math+code 2× 的小 proxy 对照（matched tokens），收益显著再加权。任何情况下不按降幅比例加权。

## 3. P2 — 三类垃圾的检测栈（比正则可靠的部分）

| 垃圾类型 | 方法 | 实测基准 | 验证方式 |
|---|---|---|---|
| 洗稿/搬运（10%，自认 2.07%） | MinHash+LSH（5-gram, sim>0.8） | C4 3.04%、RealNews 13.63% 近重复（`#dq.dedup.minhash_rates`） | 抽 200 篇被切文档人读查精确率 |
| 同上，深度洗稿 | DSpin 式功能字不变量锚点（50-100 个中文功能字，阈值 0.75） | 英文：洗稿页 77-93% vs 非洗稿 59.4%（`#dq.dedup.dspin_anchor`） | **在 2.07% 指纹文档上测召回（应近 100%——自认洗稿是现成验证集）**，再在人读样本上测精确率 |
| 同上，换词保义 | 语义嵌入去重（BGE-M3 聚类，余弦 0.85-0.9） | LAION 可移除 50% 性能无损；中文网页无公开实测（unmeasured） | precision@200 + 与 MinHash 重叠分析 |
| 内容农场/SEO 软文（30%） | 质量分类器（DCLM 式 fastText 或 CCI3 管线扩维度） | CCI3 管线 macro F1 0.73（`#dq.classifier.cci3_pipeline`） | 按文体设留出集 + 风格改写探针（`#dq.register.style_flip`） |
| 模板/题库/作文（17%） | 行级重复 + ExactSubstr + 结构指纹 | FineWeb 重复行规则切 12.47% token（`#dq.filter.fineweb_rules`） | 人读抽样 |

正则对模板的结构有效、对内容无效；n-gram 对深度洗稿失效是设计使然（DSpin：洗稿后 4-gram 相似度 20.8-38.8%，与无关文章不可区分，`#dq.dedup.spin_evades_ngram`）。中文洗稿检测无专门文献（arXiv/ACL/OpenAlex 均未找到）。

**未承认洗稿的患病率**（`#dq.hidden.prevalence`，已测）：400 篇无指纹人读审计（`#dq.audit.protocol_400`）得 p_hidden = 8.0%（±2.7pp），总洗稿 = 2.07% + 0.9793×8.0% = **9.9%**——未承认的洗稿是自认率的 ~4 倍。同批审计 junk = 30.3%（±4.6pp），与 3b 的 39.3% 同量级。两样本零重叠，免费 IAA 不可得（`#dq.audit.protocol_400`）。

**定权重前先定目标函数**：DCLM 人读研究里与人读最一致的过滤器（ROC-AUC 82%）下游效果反而最差，人读一致率与下游 R²<0.3（`#dq.classifier.human_agreement_counter`）。权重服务的是人读"可用率"还是 R6 式下游 eval，两者可能给出相反的过滤器排序。

## 4. P3 — 收窄 60%±10 的协议

**样本量**（`#dq.sample.size_table`）：n=150 实际精度 ±7.8pp（不是 ±10）；±5pp 需 369，±3pp 需 1025，±2pp 需 2305。前 30 篇 vs 后 120 篇的翻转（19pp ≈ 1.9σ，p≈0.06）无法区分标注漂移与语料顺序异质——阅读顺序未随机化。

**协议**（`#dq.sample.kim_two_stage`，Kim 2026 两阶段）：

1. **随机化**（零成本，最高优先）：先打乱语料顺序再抽，否则一切测量被顺序污染；
2. LLM-judge 全量打 3000-5000 篇（成本可忽略）；
3. 分层抽 300-500 篇双人标注（层=LLM 分桶×域名×正则命中），估 ρ²（人-LLM R²）和 Cohen's κ（文献中位 0.60）；
4. 双稳健估计：人读需求 n = n*×(1−ρ²)。ρ²=0.7 时 ±3pp 只需 ~308 篇人读（而非 1025）。用 pilot R² 的置信下界，高估会欠配；
5. 漂移控制（`#dq.sample.drift`）：每批 ≤50-100 篇，每批插 10-15 篇校准金标，节末复测同一校准集，监控标注速度（"自动驾驶"连标是漂移信号）。

文献主流模式：模型标注为主，人读 300-1000 篇做校准（FineWeb 0 人读/460k LLM 标注；ChineseWebText 2.0 1000×5 人；DCLM 人读研究 500×3）。本项目 150 篇处于区间下沿，且无 LLM 辅助层，精度最差。

## 5. 可执行结论清单

| # | 做什么 | 预计切掉/收益 | 验证方式 |
|---|---|---|---|
| 1 | 每个新语料进训练前跑 recall-ratio 判据（含 recall_home ≥0.5 绝对下限） | CCI3-HQ 已知答案 RECALIBRATE（recall_new 1.7%，比值 0.50） | 判据在 CCI3-HQ 上输出 RECALIBRATE 即通过；3b 混淆矩阵为输入 |
| 2 | web 质量头不用于 CCI3（直接证据 F1 0.03）；用 DCLM 式 fastText 或 CCI3 管线重标 | 切掉比例 unmeasured——阈值是决策变量不是检测属性（DCLM top-10%、FineWeb-Edu 切 91% 都是阈值选择） | 按文体留出集 + 风格改写探针；下游用 Ultra-FineWeb 式 1B WSD 退火验证（~110 H100-h/次）。**fastText 已测失败**（`#dq.fasttext.cci3_failed`）：150 标签训出锁定 400 AUC 0.574、web_labels 0.577 vs 旧头 0.823——质量分类器路线在 n=150 上三个特征家族全灭 |
| 3 | DSpin 功能字锚点检测洗稿 | 英文基准洗稿页 77-93% 命中；中文切掉比例 unmeasured | 2.07% 指纹文档上召回应近 100% |
| 4 | ~~400 篇无指纹审计估隐藏洗稿患病率~~ **已测：p_hidden 8.0%，总洗稿 9.9%；junk 30.3%**（`#dq.audit.protocol_400`） | — | 混淆矩阵已出并已被重标定超过：Step 0+1 正则上线（`#dq.regex.recal_step01`），锁定样本上 recall 17.4%→21.5%、0 新增 FP；Step 2 质量头重训失败（`#dq.head.recal_step2_failed`）——特征空间是瓶颈，正则层是当前唯一有实测的垃圾过滤器 |
| 5 | Kim 两阶段收窄 60%±10 | ±3pp 需 ~308 人读（ρ²=0.7） | pilot R² 置信下界 + κ ≥0.5 |
| 6 | math/code 加权前先跑 E2 | 零成本 | matched-token 坍缩→维持；在曲线上方→2× proxy 对照 |
| 7 | 定权重前先定目标函数（人读 vs 下游 eval） | — | R²<0.3 警告（`#dq.classifier.human_agreement_counter`） |

## 6. 未闭合数字

- ~~recall_home（过滤器在 web_labels 上的召回）~~——已测 3.4%（`#dq.criterion.positive_control_protocol`），连带发现比值判据需要绝对下限
- 3b 的 150 篇混淆矩阵——判据在 CCI3 上的实测输入
- ~~隐藏洗稿患病率 p_hidden~~——已测 8.0%，总洗稿 9.9%（`#dq.hidden.prevalence`）
- ~~400 篇审计混淆矩阵~~——已测：recall 17.4%、precision 53.8%（FP 中 10/18 是 not_zh 英文文档，语言门正常工作）；该样本锁定为重标定的 fresh re-test（`#dq.audit.confusion_400`）
- IAA（与 3b 的 150 篇）——两样本零重叠，未测；如需 κ，3b 需从 400 篇中抽读 ~50 篇
- ~~质量头标注量下限~~——已测：150 手读标签训出的头 AUC 0.555，与 20K 教师标签的旧头 0.541 无差异；头/尾池化都是 0.54。下限不在标签量，在 mean-hidden 特征空间本身（`#dq.head.recal_step2_failed`）——recall≥0.5 且 precision≥0.8 的门槛在此架构下不可达，下一步杠杆是 DCLM 式 fastText 或 27B 直打
- E2 六 checkpoint 分域损失——pod 上跑 domain_loss.py
- 语义去重在中文网页的切掉率——无公开实测
- ~~质量分类器路线（CCI3 junk/usable）~~——已测，三个特征家族全灭：mean-hidden 头 0.541（20K 教师标签）/0.555（150 手读）、尾池化 0.542、hashed-n-gram fastText 0.574（均为锁定 400 AUC）；fastText 在 web_labels 上 0.577 vs 旧头 0.823，CCI3 训出的分类器对 fineweb2 是净损失（`#dq.fasttext.cci3_failed`）。正则层是当前唯一有实测的 CCI3 垃圾过滤器
