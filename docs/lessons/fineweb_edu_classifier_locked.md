---
question: 官方 FineWeb-Edu 340M 强分类器（非自训头）在我们锁定手读集上是否可用于中文 web / 代码过滤，还是同样漂移失效
status: measured
source: laptop CPU 离线推理（无 GPU、无重训、无阈值选择），scripts/score_fineweb_edu_locked.py + models/fineweb-edu-classifier（HF HuggingFaceFW/fineweb-edu-classifier，model.safetensors sha 见 fact），2026-09-16，0e
---

# 官方 FineWeb-Edu 分类器 · 锁定集一手实测

## 结论（回答方向问题）

**现成强模型同样不可用——跨语言漂移在官方 340M 头上一手复现，在锁400 上甚至方向反了，且对代码是 100% 误杀。** 这补上了 `#dq.fineweb_edu.chinese_failure`（二手，arXiv CCI3 Table 4：P.91/R.02/F1.03）缺失的本仓一手测量：我们在自己锁定的手读集上跑官方权重，而不是引用论文。

1. **中文 web（web_labels）：判别力≈随机，官方阈值切光一切。** 180 篇（33 有教育价值 y=1 / 147 无 y=0）上 raw-logit 对人读标签的 **ROC-AUC = 0.5725，bootstrap 95% CI [0.474, 0.677]，跨 0.5**。分数全压 0.03–1.64，**180 篇无一 ≥3**；官方切教育 recall 0%、非教育 kill 100%。
2. **中文 CCI3 锁400：方向反了（最强证据）。** 逐行 junk/rewrite 标签在 `cci3_audit_400_labels.jsonl`（按 id join，400/400，121 junk / 279 keep / 32 rewrite，与 fact 的 30.3%/8.0% 一致）。分数预测"应保留(not-junk)"的 **AUC = 0.4220，95% CI [0.359, 0.483]，整段低于 0.5**——**垃圾的分数反而更高**（junk 中位 1.041 vs keep 0.925；junk 中 4.1% ≥3 vs keep 仅 1.1%）。not-rewrite AUC 0.556（≈随机）。比"随机"更糟：这个头在中文 CCI3 上是反向指标。
3. **代码：官方阈值误杀率 100%。** 100 篇真实 OSS 源码分数 −0.07–1.42（中位 0.52），**100/100 <3，kept=0%**，2% 甚至 <0。

## 分数口径（易错点）

模型头是**单回归输出，分数 = 原始 logit，量纲 0–5，不是 sigmoid，无 prompt 前缀**——以官方模型卡示例为准（`score = model(**inputs).logits`，`int_score=round(clamp(score,0,5))`，curation 切 int_score≥3）。任务书提到的 sigmoid/prompt 与模型卡不符，已按模型卡实现。输入硬上限 512 token（Snowflake-arctic-embed-m BERT，position embedding 512，1024 直接 shape error），39/180 篇中文文档超 2000 字符会被尾截——但 AUC 已≈随机，尾截不是结论的驱动因素。

## 数字表

| 集合 | n | 人读标签 | AUC（raw logit，正=应保留） | 分数中位 [min,max] | ≥3 占比 | 官方切下行为 |
|---|---:|---|---|---|---:|---|
| web_labels（中文 web） | 180 | 33 教育 / 147 无 | **0.5725 [0.474,0.677]** 随机 | 0.86 [0.034,1.638] | 0% | 教育 recall 0%，非教育 kill 100% |
| └ y=1 教育 | 33 | — | — | 0.92 [0.597,1.634] | 0% | 全被切 |
| └ y=0 无教育 | 147 | — | — | 0.86 [0.034,1.638] | 0% | 全被切 |
| cci3 锁400（中文 CCI3） | 400 | 121 junk / 279 keep（id join） | **0.4220 [0.359,0.483] 反向** | 0.97 [0.29,4.234]；junk 1.041 / keep 0.925 | junk 4.1% / keep 1.1% | junk 分更高，几乎全切 |
| └ not-rewrite | 368/32 | 32 rewrite | 0.5563 [0.449,0.670] 随机 | — | — | — |
| code OSS-100（应保留） | 100 | 正样代理 | — | 0.52 [−0.07,1.415] | 0% | **误杀 100%** |

## 判据归属与边界

- 这是"现成强模型对照"的负结果，与自训质量头/fastText 全灭（`#dq.fasttext.cci3_failed`，锁400 AUC 0.54–0.58）独立：前者证明不是我们训练规模/特征工程的问题，而是**英文教育文体回归头跨 register/跨语言不迁移**这一更根本的失败（机制同 `#dq.register.style_flip`、`#dq.universal.criterion_absent`）。锁400 AUC 0.422 且 CI 整段 <0.5 还说明：在中文 CCI3 的 register 上该头不只是不迁移，而是反向——营销/SEO/模板文本更像它训练目标里的教科书英文。
- web_labels（180，CI 宽跨 0.5）与锁400（CI 整段 <0.5）都是中文 web/CCI3 文体；仓库无英文人读标签，故"英文 web"类无法在本仓出 AUC。
- 代码"误杀"用人读正样代理（dedup band 抽样的真实 OSS 文件头，非随机），是分布/下界证据而非标注混淆矩阵；要精确代码 FP 率需对代码样本单独人读打"训练可用"标签。
- 锁400 逐行标签此前一度被当作"只在会话笔记"，实际在 `data/corpus/sample/cci3_audit_400_labels.jsonl`（fasttext_junk.py 的 --audit400-labels 输入），按 id join 400/400；本次据此补上真 AUC。注意该 id 实测不等于 sha256(text)[:16]，跨评分 ledger 回连人读标签时需另存 join 键，不能假设 doc_id 直接命中现有 id。
- 不重训、不定阈值：≥3 是官方自己的 curation 切，这里只报告在该切和 raw 分布上的表现；若有人想按中文/代码重定阈值，那是新的 RECALIBRATE 任务，不在本对照范围。
