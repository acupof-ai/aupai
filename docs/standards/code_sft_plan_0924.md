---
question: CED 预训练结束后的代码 SFT 用哪些数据、什么配比、什么格式——候选源清单与三档配比方案
status: planned
source: 用户任务 2026-09-24（via aupai-1e 交接）；失败拆分实测自 pod step14500 八个 CPU shard 产物；配方计划 #684 之后的下游段
---

# 预训练后代码 SFT 数据方案

This is a plan, not a build. 本文不下载、不构建、不占 GPU。配比由用户拍板。
每个外部源的规模是 card 数字，token 量在我们 32,768 词表下**未测量**；仓内源的数字带 fact 引用。

> **用户选定（2026-09-24，via aupai-1e）：方案 A（停止优先）。**
> 配比 code_if 55% + sc2-exec 30% + APPS 5% + 非 code 散文 10%，按 §4 共同口径构建。
> 方案 B 保留为 SFT 后第一轮读数（81 类下降、73 类不动时）的第二轮加料。

范围：CED run（`v41_ced_0923`）结束后，面向 HumanEval pass@1 ≥ 30% 门的**一段**代码 SFT。
只覆盖数据。SFT 入口走 `sft.py` 还是 `sft_math.py`、CED 架构的 decoder 段如何接 pack，
是 de 的实现决定，本文不替代。

## 1. step14500 的失败结构（今天实测，不是交接转述）

读数对象：`ckpt_v41_ced_0923.pt.step14500`，HumanEval 164 题，rstrip continuation 臂，
greedy，`max_new=280`，8 CPU shard，评测器 `eval/humaneval_gen.py --rstrip_nl`，
preds 为 `data/eval/preds_humaneval_ckpt_v41_ced_0923.pt.step14500.rstripnl.shard*of8.*.jsonl`（pod）。
合并行 `runs/experiments.jsonl#heval_ced_s14500_rstrip_loop`：FULL 10/164 = 6.10%，CLEAN 10/156 = 6.41%。

逐题计 `(ok, stop_reason, empty_kind)`，164 行全部非空：

| 类别 | n | 占比 | 含义 |
|---|--:|--:|---|
| 通过 | 10 | 6.1% | 9 个 stop-string 停，1 个 eos 停 |
| **到上限停（max_new=280）** | **81** | **49.4%** | 写满 280 token 仍未终止，被判错；SFT 的首要目标 |
| 正常停止但逻辑错 | 73 | 44.5% | 66 个撞 stop-string（下一个顶层结构），7 个发 eos 后判错；全部多行非空 |
| 空完成 | 0 | 0% | 与 experiments 行 "empty completions: 0" 一致 |

交接口径是「81 到上限 + 63 逻辑错」，artifact 实测逻辑错为 **73**，差 10 题。本文用 artifact 数。

两条读法：

1. **停止是最大的单一错误类（49.4%），且形态与历次 base checkpoint 不同。**
   旧 30B checkpoint 的失败是空完成（step34000：160/164 空，其中 127 个被 `\ndef ` 截断、
   33 个 eos-first）；CED step14500 已经肯写函数体，缺的是**在 280 token 预算内收尾**。
2. **逻辑错（44.5%）全部有实质输出，是解题正确性问题，不是格式问题。**
   这一类要靠带测试信号的解题数据，不靠接口对。

停止类的历史干预证据（不同 checkpoint，按因果方向读，不跨 checkpoint 比绝对值）：

| 干预 | 数据/格式 | 停止相关结果 | 出处 |
|---|---|---|---|
| format_sft_0909，30B base | 5k 裸 code_if 对，continuation，EOS 监督 | 空 160→72；stop_at_0 127→2，但 eos-first 33→70；pass 0→3/164 | `experiments.jsonl#format_sft_0909` |
| v41 ChatML SFT 0913，step16000 | code/EN/zh ChatML pack，2 epoch | ChatML 臂 pass 0→11/164；空 103→28，残留 28 个**全是 max_new**，0 eos-first | `facts/v41.json#v41.humaneval_chatml_sft_step16000_0913` |
| phi 式 continuation SFT n6，r3-final | L3 留出 2% 重建函数对，~42M 监督 token | 净变化不显著（HE +1.73pt 下界 −1.41，MBPP −1.27），判 INEFFECTIVE | `facts/v41.json#v41.phisft_n6_psft_vs_e0_0915` |

第三条是负面先验：**L3 解池的窄 continuation SFT 在真留出上也没拉动**。叠加 §3.1 的
新颖性问题，三个方案都不放 L3。

## 2. 格式：用 continuation，不用 ChatML

建议：**门 SFT 用裸 continuation 对（prompt = 函数签名+docstring，target = 函数体+EOS），
不走 ChatML。** 四条理由：

1. **门的评测臂是 continuation。** 自动评测循环 `runs/heval_auto_loop.sh` 与 step14500 读数
   都是 `--rstrip_nl` 裸续接；`--chatml` 是另一条打分列。SFT 格式必须与打分格式一致，
   否则 280-token 预算内的终止分布学在 `<|im_end|>` 上，评测序列里没有这个串。
2. CLAUDE.md Chat format 节的规则：预训练语料中 ChatML 近乎不存在（0/4000 行实测上界），
   ChatML 必须从无到有教；base 评测不得引入预训练未见的 token 序列。
3. **裸 continuation 的打包边界已修对。** `datagen/prepare_format_sft.py` 用
   split_encode 分开编码 prompt/body，推理序列是训练序列的逐 token 前缀；
   默认打包路径在 ~98% 的对上把 prompt 尾换行与 body 缩进合并成一个 BPE token，
   第一个监督位置就对不上（该文件 docstring 实测 4,892/5,000）。
4. ChatML 臂 0→11 的移动是在 `--chatml` 打分列下测到的，中途换门的打分列等于换指标。
   如果产品目标含聊天，那是门之后的第二段 SFT，ChatML pack 构建器
   （`build_post30b_chatml_pack{,_v2}.py`）保留可用。

ChatML 唯一实测优势是终止校准更好（残留空完成全是 max_new，没有 eos-first 提前终止）。
若 continuation 方案复现 format_sft 的 eos-first 上涨（33→70），回退选项是在 continuation 对里
加短输出占比（方案 A 已含），不是换 ChatML。

## 3. 候选源

许可一列只写 card/实测层面的结论；外部源均未下载，行数为 card 数字。

### 3.1 仓内/ pod 上已存在

Pod 数据面实测于 2026-09-24（`~/bin/pod` 容器视图，`ls` 计数）：0916 销毁后
`data/sft/` 为空，`/mnt/data02` 备份未挂载，vetted 教科书池和 code_if_pairs jsonl
**当前不在 pod 上**。可重建性逐源写明。

| 源 | 内容与规模 | 许可 | 测试 | 13-gram 状态 | 24h 就绪（实测后） |
|---|---|---|---|---|---|
| **code_if_pairs**（重抽） | 原版 99,996 对 + 10,000 控制对已随 0916 销毁丢失；**283 个 `code_py_starcoder_dc` jsonl shard 在 pod 上现存**，`extract_code_if_pairs.py` 直接吃 jsonl（docstring 自述全量 2–5 分钟 / 48 workers），seed 42 确定性重抽，数量会略少于 99,996（源是 _dc 版） | starcoderdata card 为 "other"，从属于各 repo 许可；**该来源已在 CED 的 6 域 mix 中被接受**，不引入新许可决策 | 无 | 原版实测 4/100,000 = 0.004% 命中，全在 solution 侧、prompt 0 命中（`experiments.jsonl#code_if_pairs_0909`）；扫描器是抽取器自带的双向 whitespace-normalized containment（带长度下限），**不是** `decontam_ngram` 的 13-gram DROP；重抽后在最终对文本上用 `filters/decontam_ngram.py` 按 13-gram 复扫，以新掉率为准 | **就绪**：纯 CPU 分钟级抽取 + 打包 |
| **非 code 散文** en_c4_stage2_dc / math_owm_stage2_dc | 242 / 333 个 jsonl shard 现存，预训练同形短文 | 同 gate mix 已接受 | 无 | 域级已扫（gate _dc） | **就绪**：截短文档直接续接打包 |
| **L3 task/solution 对** | 147 个 L3 raw parquet 在 `data/raw/ultradata` 现存，`datagen/build_l3_stub.py` 可重建 stub 域（RAW 默认路径 `/data00/aupai_raw` 不存在，需改路径或软链） | UltraData apache-2.0 + 源 repo 许可 | 带 test 字段；noexec 未经执行 | aggregate 已扫 | **不建议用**：r3 的 2% 留出只对 stub 域成立；**CED 的 mix 是 6 域全量 noexec_dc，`full_content` 含 solution，这批函数 CED 已在预训练里见过**，无新颖性；且同一池在真·留出的 r3-final 上已测为 null（n6） |
| **vet 教科书** textbook_claude_v41_dc | pod 上不存在，备份未挂载；旧 `/data00/tokens_textbook.pt` 是 0830 旧域不是该池 | 内部 Fable 生成 | 代码类执行通过率曾测 96.38% | vet 漏斗内曾扫 | **24h 不可得**：重建需 Fable 生成（用户审批）；且 gen-A 认证不过，题材与 HE 失败原语错配（§5），不作为主料 |
| **code_tests_v1** | src+真实测试 repo 内配对；59 个 starcoder parquet 在 `data/raw/ms_starcoder_py` 现存；产率试验 490,666 对 / 2.82B 源行 token，`cs.code_tests_supply` 当前供给=0 | 同 starcoder | **有真实测试**，构建器不执行 | 未构建未扫，两道都要扫 | **偏紧**：`build_code_tests_v1.py` 两遍读 22 GB parquet，纯 CPU 可在 24h 内启动但不在最稳路径 |

### 3.2 外部候选（均未下载、未扫；第一步固定是 `curl -4` 镜像链抓取 + `filters/decontam_ngram.py` 扫描）

| 源 | 规模（card） | 许可 | 测试/执行信号 | 形状契合度 |
|---|---|---|---|---|
| **bigcode/self-oss-instruct-sc2-exec-filter-50k** | 50,661 行 instruction/response | **ODC-By** | response 经执行验证（数据集名即 exec-filter）；不随数据单列测试 | 高：指令→完整代码响应，长度可控，直接补逻辑错 |
| **codeparrot/apps** train | 5,000 题，多解 | **MIT** | `input_output` 带 IO 测试；195 个 train 题缺测试需筛掉 | 高：竞赛小题，多解可截短；与 HumanEval 风格不同（stdin/stdout 为主，需渲染成签名/docstring 题面）；13-gram 掉率未测 |
| **BAAI/TACO** | 26,443 题 / 1.55M 已验证解 | card **apache-2.0**；聚合 APPS、CodeContests、GeeksforGeeks 等来源，**下游许可混杂，入项需法务口径确认** | IO 测试齐全，解经执行验证 | 中高：量大可挑短题；GfG 来源段建议直接不取 |
| **ise-uiuc/Magicoder-OSS-Instruct-75K** | 75,197 行 | **MIT** | seed 代码片段创建时跑过；response **不保证执行过滤**，测试只是偶发出现在片段里 | 中：教学式 instruction/response，偏一般编程任务 |
| **Kelexine/Fable-5-traces** | 4,665 行 agent trace | **AGPL-3.0**（训练使用本身不分发代码，但产出物的 copyleft 争议需要用户接受） | 人工/脚本试玩验证，无系统单测 | 低：是工具调用 agent 轨迹，不教函数签名续接；与 HumanEval 错误形态不对口。列为已知不选项 |

所有外部源在扫描前状态一律写 **未扫**，不假定干净。下载体量只核实了 sc2（card 90.1 MB）；
APPS、TACO 的实际体积未测，24h 抓取走 `curl -4` 镜像链没有已知阻断，风险在许可确认和人工抽检，不在带宽。

## 4. 配比方案（三档，用户拍板）

共同口径：

- **目标序列**：prompt = HumanEval 同形的签名+docstring 前缀；target = 完整解 + EOS，裸 continuation，
  split_encode 打包（`prepare_format_sft.py` 路径）。
- **长度闸**：监督 body ≤ **256 token**（评测预算 280）。该阈值已核：164 个 HE canonical
  solution 在当前 32,768 词表下 body 长度 min 6 / p25 27 / median 46 / p75 77 / p90 108 /
  max 251，**256 截掉 0/164 个金解**（pod，`Tokenizer.encode(canonical_solution)`，2026-09-24）。
  到上限的 81 个失败生成长度全是 280，是最长金解 251 的 1.12 倍以上、中位金解 46 的 6 倍，
  形态是收不住尾，不是题本身需要更长输出。外部源的解超长则丢弃或截到完整语句。
- **规模**：总包 10k–30k 对、1–2 epoch 为建议区间。依据：格式类学习在小模型上百条/行为即启动
  （`docs/lessons/sft_data_size_small_models.md`：2B 上 100–1,000/API，本栈同模板 ≥120 行门槛）；
  ≤360M 容量受限有反向证据（422M 上 IT 伤害 held-out；SmolLM 在 ≤360M 删掉工具调用段）。
  30k 对 × ~400 token ≈ 12M 监督 token/epoch 是**估算**。
- 每个方案都混入 10% 非 code：gate 域短文档做**预训练同形的纯散文续接**（en_c4_stage2_dc、
  math_owm_stage2_dc 缓存 pod 现成、已 13-gram 去污），不用 v1/v2 ChatML pack 的 alpaca 切片
  （那是 `<|im_start|>` 渲染，continuation 包不能用）。作用是正则、防纯 code SFT 掉语言面板；
  面板六指标是 SFT 出口闸（`docs/lessons/post_pretrain_plan.md` §0–1）。
- 统一门控：13-gram 在**最终渲染文本**上复扫（不是源字段）；holdout 哈希闸；
  `scripts/test_sft_pack.py` loss mask；pack 带当前 vocab_id（32,768 词表，
  fp f1f860970d15d623），`sft_math.py` 不匹配会拒。

### 方案 A —— 停止优先（我的推荐）

针对 81/164 到上限（49.4%）。格式/终止是本栈唯一被反复证实「便宜且可动」的一档，
最大错误类先治。

| 成分 | 监督 token 占比 | 来源 |
|---|---:|---|
| 短身 code_if 对（body ≤256 tok，从重抽池按 seed 抽，50–150 token 长度分层） | ~55% | code_if_pairs 重抽，13-gram 复扫 |
| exec 验证短指令解（响应 ≤256 tok 优先） | ~30% | sc2-exec-50k，外部，抓取+扫描 |
| APPS 短题解（≤256 tok 的解优先） | ~5% | 外部 |
| 非 code 散文续接段（en_c4_stage2_dc / math_owm_stage2_dc 短文档截段，前段 mask，预训练同形） | ~10% | pod 现存 jsonl |

逻辑覆盖靠 sc2 的 30%；若 81 类显著下降而 73 类不动，第二轮再加 B 的料，分两次读数。
风险：format_sft 已证明纯接口数据会抬 eos-first；长度分层（混入 50–150 token 的中短身）
是针对该失败形态的，要求每批长度分布落地时打印，不靠配比声明。

### 方案 B —— 逻辑优先

针对 73/164 正常停止但答错（44.5%）。

| 成分 | 占比 | 来源 |
|---|---:|---|
| sc2-exec-50k 验证解 | ~35% | 外部 |
| APPS 题解对（prompt 渲染成签名/docstring 形态，不是裸 stdin） | ~25% | 外部 |
| TACO 短题解（仅 apache/MIT 可确认段，GfG 段不取） | ~15% | 外部，许可前置 |
| Magicoder-OSS 75K 短 response | ~10% | 外部，MIT；无 exec 保证，靠人工抽检+长度闸 |
| code_if 短身对 | ~5% | pod 重抽，仅作终止锚 |
| 非 code 散文续接（en_c4 / math_owm 短文档） | ~10% | pod 现存 |

不放 L3：对 CED 它不是未见过的数据（§3.1），n6 的 null 又已覆盖真留出情形。
若四个外部源的许可/抓取任一卡住，后备是 code_tests_v1 的 repo 内真实测试对（需先构建）。
风险：350M active 的容量先验是「逻辑不便宜」（206M 三臂：换来源 40.0→0.0，量不是变量）。
B 的成败判据是逻辑错类是否下降，不是总分。

### 方案 C —— 均衡

A 与 B 各半：code_if 30% + sc2 30% + APPS 20% + Magicoder 5% + 非 code 散文 15%。
用于「24h 内只跑一轮 SFT、没有第二轮窗口」的情形：两类错误各治一半，
但按本栈剂量证据（同模板 ≥120 行才学会），每类分到的剂量都更接近门槛而非饱和点。

### 建议与判读顺序

**推荐 A，且预留 B 为第二轮。** 理由：81 是最大类；停止干预在三个 checkpoint 上都动了
（格式类有因果证据），逻辑干预有一次 null 先验；SFT 段的产品是 RL 的起点
（post_pretrain_plan §1），先把 pass@8 的采样从 280-token 截断里放出来，RL 才有可放大的轨迹。

SFT 后读数固定拆三类报（pass / max_new / 正常停止但错），沿用 step14500 这张表的口径，
同时报 MBPP 同臂与面板六指标。总分涨了但 81 没降，不算 A 起效。

## 5. 未测量项与风险

- 外部四源在我们词表下的 token 量、长度分布、13-gram 掉率：未下载未测。
- TACO 下游许可、Fable traces 的 AGPL：需用户/法务口径，方案默认都不取。
- 教科书池与旧 code_if_pairs jsonl 已在 0916 销毁，备份未挂载；对的重抽耗时是分钟级（jsonl 现存），
  教科书池重建需 Fable 生成审批，三方案均不依赖它。
- gen-B 教科书即使重建，题材也偏软件工程（`ab_coverage_prediction.md`
  的 title-match 结果：palindrome/geometry 等 HE 高频失败原语在池中近乎零剂量），不进主料。
- 方案只覆盖数据；CED 架构的 SFT 入口、decoder 段掩蔽、EOS id 行为由 de 确认后才构建。
- code_tests_v1（22 GB parquet 两遍）是「外部源不可用、需要 repo 级真实测试对」时的后备，
  不在 24h 关键路径上。
