---
question: "L1 多语言困惑度质量层——零标签、不重复已死分类器路线的统计异常检测是否成立"
status: measured
source: "3b 2026-09-16, datagen/l1_ppl_kenlm.py (stdlib-only interpolated Kneser-Ney); KenLM 不可装（离线 pod pip 无 index 且重启丢包），故自带 ARPA 同语义 scorer"
---

# L1 困惑度质量层（n-gram backoff LM）

## 为什么是这一层

L2 质量分类路线在本仓语料已实测**全灭**，不重做：

- 自训 mean-hidden 头 AUC 0.541（20K 教师标签）/ 0.555（150 手读标签），尾池化 0.542；
- hashed-n-gram fastText 0.574（锁定 400）/ 0.577（web_labels），旧 cosmopedia 头 0.823 不可迁移；
- 三个特征家族的瓶颈是特征空间本身，不是标签量（`docs/lessons/data_quality_methods.md` §0/§6，
  `facts/data_quality.json#dq.fasttext.cci3_failed`、`#dq.head.recal_step2_failed`）。

L1 困惑度是**正交**信号：不需要标签，度量 token 流在"域内语言/代码"下的统计意外度，
直接打分类器漏掉的三类——乱码（mojibake）、行级重复、模板拼接，它们都扭曲 n-gram 统计。

## 实现

`datagen/l1_ppl_kenlm.py`，纯标准库（无 numpy/native deps，离线可复跑）：

- **插值 Kneser-Ney**（order 默认 3），Chen-Goodman 折扣 d = n1/(n1+2·n2)；
  P(w|ctx) = [max(0,C(ctx,w)−d) + d·N1+(ctx,·)·P(w|更短历史)] / Σ_w C(ctx,w)，
  一元续接概率 P_cont(w)=不同前文数/不同 bigram 类型数；未见上下文整层 backoff，未见词给 floor。
- 分词：unicode `\w+` 词 + 每个非空白标点单独成 token；同一分词器打代码和散文，
  两类各自训出不同统计。
- CLI：`--train file --score file --order N --hist`，jsonl 读 `content` 字段；
  `--selftest` 钉已知答案。输出 ARPA 同语义，未来 KenLM 二进制模型可无缝替换 scorer。

## 已知答案自测（--selftest，小英文语料）

| 文本 | 现象 | PPL（order 3） |
|---|---|---:|
| `the cat sat on the mat`（域内正常句） | 基线 | ~1.3 |
| `{{ }}} %s %s … #### ---- ####`（模板/标点拼接） | 高 | ~38 |
| `zxqw vlmp krntt …`（未见 token 乱码） | 高 | ~97 |
| `the the the the …`（行级重复） | 最高 | ~228 |

契约：三类垃圾 PPL 都 ≥ 正常句的 5×。**不预设两类垃圾谁更高**——重复常见词违反其
bigram 搭配（限定词后不接限定词），可能比未见 token 的平坦 floor 更意外；这正是要抓的信号。

## 真语料已知答案（pod，5000 篇训练，order 3，每篇自损坏配对，3b 2026-09-16）

把每篇真实文档人为损坏成三种，用**该域自己的模型**打分（控制主题，只改损坏类型）：

**域错配验证（不同模型）**：正确域内 median PPL ≪ 错配域——
NL 模型打散文 12.9、打代码 2708；代码模型打代码 6.8、打散文 5390（~300–800×）。
域模型确实学到各自分布，且**域错配本身就是最大假阳来源**。

| 损坏类型 vs 同篇原文 | NL(en_c4) median | CODE(starcoder) median | 判读 |
|---|---:|---:|---|
| 原文 real | 21.1 | 12.2 | 基线 |
| 整篇重复一行 repeat | 23.2 | 25.9 | **弱**（文档级均值被长度稀释；代码本就常重复） |
| 模板拼接 template | 2,887 | 231 | 强信号 |
| 编码乱码 mojibake（25% 字符→Ã¤â€�） | 59,750 | 21,237 | 极强信号 |

结论：**mojibake/template 两类，文档级词 n-gram PPL 在两个域都是数量级级别的可分信号；
整篇行重复不是**——它要靠已有的行级重复规则（FineWeb 规则切 12.47% token）而非文档 PPL。
短测试语料里"the the the"那种**句内相邻重复** PPL 高，但生产里重复通常跨整行且被均值稀释。

## 训练语料来源与覆盖率（两个模型）

| 模型 | 训练源（pod） | 性质 | 已知偏差 |
|---|---|---|---|
| 英文自然语言 | `data/corpus/en_c4_stage2_dc/`（packed 1.99B tokens / 484,552 packed 行；已 13-gram 去污染） | C4 英文网页 | 本身是网页语料，**农场/SEO 文本会显得 PPL 不高**——PPL 抓"统计异常"抓不了"通顺但无价值" |
| 英文代码 | `data/corpus/code_py_starcoder_dc/`（7.95B / 1,940,182 行；去污染，188 hit 0.025%） | Stack 风格 Python | 注释/字符串/自然语言夹杂，代码模型对纯散文 PPL 会高 |

代码模型交叉验证也可用 `code_ultra_l2_dc`（UltraData 自然代码 15.3B）。
覆盖率随训练文档数变化；当前先各取 shard 子集验证可分性，扩全量是 CPU 时间问题不是方法问题。

## 盲区（读数字前必读，翻车点 1：代码为什么会被误杀）

1. **域错配是最大假阳**：好代码在自然语言模型上 PPL 巨大（反之亦然）。
   **每个域必须用为它建的模型打分**；绝不能拿一个全局模型切所有语料。
2. **通顺但低质抓不到**：SEO 软文、农场文在 C4 训练的模型上 PPL 可能很低——
   它统计上"正常"。PPL 只覆盖结构损坏，不覆盖价值判断（价值是分类器/LLM 层的事，而那条路线已死，
   所以这类目前只有手读/正则层覆盖）。
3. **高技术密度合法文本 PPL 高**：数学公式、密集标识符、罕见 API 名、生成式长标识符会被抬高，
   不能按高 PPL 直接删。
4. **分词与语言**：word+标点分词对中文按字/词边界处理粗；中文若要做需要单独的字/词模型，
   本层先只交付英文两个模型。
5. **小训练集 floor 主导**：训练文档少时常驻词少，未见 token floor 偏大、绝对 PPL 不稳；
   分布形状比绝对值可靠，**不报阈值**。
6. PPL 是**文档级均值**，长文档内局部污染会被稀释；必要时配合行级重复规则（§3 已有 FineWeb 12.47%）。

## 不交付什么

- **不交付阈值**。切/留是下游数据负责人的决策变量（DCLM top10%、FineWeb-Edu 切 91% 都是阈值选择，
  不是检测器属性）。本层只给每文档 PPL 和在手读集上的**分布**。
- 不宣称在锁定手读集上的 AUC/召回——那是下游决定是否采用、在何种权重下采用时再测的。

## 在手读集上给分布的口径（待定，见交付说明）

锁定的手读集目前是**中文**：`data/web_labels.jsonl`（180 篇中文，y=0/1）与
`data/corpus/sample/cci3_audit_400.jsonl`（400 篇中文，文本在、手读标签存于 session notes）。
英文两个 PPL 模型不能直接给中文集分布。要给英文分布需要英文手读标签集（候选：
code_rp1t 手读只有语言标签无质量标签，不可用）——这是开工时已识别、需 fb 确认的口径缺口。
