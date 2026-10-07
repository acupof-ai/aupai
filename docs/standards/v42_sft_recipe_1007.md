---
question: v42_s24（3.26B 总 / 614M 激活 MoE）step54000 之后的 SFT 配方——格式遵循 + 数学 + 代码的数据、超参、冻结决策
status: planned
source: 用户任务 2026-10-07（awb 转任务，d1）；短板实测自 pod eval_all #898749 与 preds artifacts；业界调研 2026-10-07（arXiv 原始论文，见 §6）
---

# v42_s24 step54000 SFT 配方方案

This is a plan, not a build. 本文不下载、不构建、不占 GPU；开工数据与超参由主控拍板后另走数据任务与训练任务。
所有业界数字带 arXiv 出处；所有本仓数字带 artifact/fact 出处。估算显式标「估算」，实测标口径。

## 0. 结论（先给可执行配方）

- **格式用 ChatML（`<|im_start|>…<|im_end|>`），completion-only loss，pack 4096，回答末尾必须监督到 `<|im_end|>`**。这是修 step54000「复读题目 / 问答回环、不停止」的直接手段，不是风格选择。
- **数据一段式混合（stacked），不做分阶段**：代码推理 50% + 中文数学分步 30% + 英文/通用 MC-格式与短指令 20%，去污染后约 **150M–250M 监督 token**，**2 epoch**。
- **峰值 LR 2e-5（约为预训练 1e-4 的 0.2×，落在 sft_math.py 的 `--lr_scale 0.2`），cosine 衰减到 1/10，warmup 3%，全局有效 batch ≥256 序列**。
- **不冻结任何参数**：v42_s24 配置 `engram_layer_ids=()`、`n_mtp_layers=0`（`v41f/config.py:351`），**模型里没有 engram/MTP 模块可冻**；embedding/router/专家全部全参微调，用低 LR 而非冻结控遗忘。
- **喂给这个 614M-active 小模型的推理链要短**：CoT 目标 600–1500 token、硬上限 ~2500 token；不要直接灌 OpenCodeReasoning 的 8K–24K 原始 R1 长链（§6 的小模型反证）。
- **先跑一个 20M-token 的格式先导 smoke**（同一混合的 1/8 子采样，1 epoch）确认复读消除、`<|im_end|>` 正常停止，再放全量——本仓 v41 的历史是格式干预有效但天花板低（§4），别一次押满。

## 1. step54000 的短板实测（本仓 artifact，不是转述）

评测对象 `ckpt_v42s2.pt.step54000`（stage-2 终点，train loss 1.225、val@54000=1.487，lr 归零，全程无发散；
训练事实见 awb 看板 2026-10-06 21:18 完成行）。

| 维度 | 分数 | 口径 / artifact |
|---|---|---|
| HumanEval pass@1 | **44/156 = 28.21%**（164 口径 44/164），164 题全部非空 | `data/eval/preds_humaneval_ckpt_v42s2.pt.step54000.rstripnl.shard*of8.jsonl`（pod，rstrip_nl greedy）；8 个去污排除题口径同 `eval_watch.jsonl` |
| math-hard | **0/1032 = 0.0%**（L3 0/598、L4 0/434） | eval_all #898749，`runs/hard_ckpt_v42s2.pt.step54000.*.jsonl` |
| math-500 | **0/500 = 0.0%** | 同上 |
| gsm8k | **3.6%**（4735s） | eval_all，format_continuation |
| C-Eval 中文填空 | **27.34%**（1346 题 acc_norm） | `runs/ceval_cloze.log` |
| 英文 MC（平均 40.0%） | ARC-E 42.9 / ARC-C 21.4 / WinoGrande 50.7 / BoolQ 61.4 / OpenBookQA 23.6 | `runs/mc5b.log`（OpenBookQA 23.6 低于 25% 随机） |
| eqcheck | 428 式中 47.4% 错，仅 2% 含等式 | eval_all #898749 |

**失败形态定性（#898749 实读）**：数学 0% 不是 harness 故障，预测实读为**复读题目 / 问答回环退化**；
OpenBookQA 低于随机、eqcheck 几乎不产等式，都是同一症状——**不会进入「按格式作答并在终止符收尾」的模式**。
代码侧 164 题全非空、28.21%，说明「肯写、会写一点」已具备，缺的是稳定收尾与推理正确性。

根因在本仓已被反复证实，不是新假设：

- ChatML 前缀在预训练语料中出现 **0 次**（4000 行抽样口径，AGENTS.md「ChatML 格式」节）。
  base 模型拿到 `<|im_start|>user…<|im_start|>assistant` 是在续写一条没见过的 token 序列，
  所以复读输入或漂向 web 样板——`scripts/loader.py:219` `format_continuation` 的 docstring 直接记载
  「ChatML 臂代码块出现率 1.6% vs 1-shot continuation 94.4%，同一个 checkpoint」。
- 数学 0% 同时叠加**数学分步能力未被蒸馏**（预训练含 cot/math，但从没在「提问→分步作答→终止」的监督对上训过）。
- MC 弱叠加**不会按要求输出选项/结论格式**（eqcheck 2% 含等式是硬证据）。

所以 SFT 要同时教三件事，且**格式是其余两件能在生成式评测上显形的前置条件**。

## 2. ChatML 与 loss mask（机制，不可省）

统一走 `scripts/loader.format_prompt / format_example`（单一事实源，`loader.py:206/252`）：

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

- **completion-only loss**：labels 在 prompt（含 `<|im_start|>assistant\n`）为 -100，
  仅 `{answer}` **和末尾 `<|im_end|>`** 参与损失。`<|im_end|>` 必须在监督区间内，
  否则模型永远学不会「停」——这正是 step54000 复读/写满上限的直接修复。
  本仓 `datagen/prepare_sft.py` 已实现 prompt-mask + split-encode（prompt 与 answer 分别编码再拼接，
  避免 `\n`+缩进 BPE 合并导致 mask 错位），并有 `scripts/test_sft_pack.py` 断言每个 mask 段止于 `assistant\n`。
- **一个问题只允许一种答案终止格式**。math 一律以「答案是：X」或 `\boxed{X}` 收尾再 `<|im_end|>`，
  不要混用（FoNE 一课：多格式并存让终止判定欠定，AGENTS.md「Numbers」节）。
- **system 用固定一句**（如「请逐步推理，最后给出最终答案。」），全数据集一致；不要每题换 system。
- 评测必须配对换 ChatML：SFT 后的 ckpt 用 `--chatml`（`humaneval_gen.py` 已支持）、
  math/MC 走 `format_prompt`。继续用 continuation prompt 评测 SFT 模型会得到假低分（AGENTS.md 已记载此坑）。
- v41f 词表 eos id=1，`<|im_end|>` 是独立特殊 token；packer 用 tokenizer 解析，不硬编码（`prepare_sft.py` 已如此）。

## 3. 数据配比与供给（本仓现有原料 + 需新增）

目标去污染后约 **150–250M 监督 token、2 epoch**。监督 token 只计 answer 段（-100 之外）。

### 3.1 建议混合（stacked，一段式）

| 桶 | 占比 | 监督 token（200M 目标） | 内容 | 现有供给（pod 实测） |
|---|--:|--:|---|---|
| **代码推理 CoT（英文）** | 50% | ~100M | 赛题→短-中推理链→**可执行 Python 块**；ChatML | `data/corpus/code_opencot_dc/` 23,856 篇 / 168.8M pool token（d1 2026-10-05 建，含 40K 字符长链，**用前必须按长度截断/重选到 §3.3 短链**）；`data/distill/ocr_sft.jsonl` 1,893 行已是 {prompt,output} |
| **中文数学分步** | 30% | ~60M | gsm8k 中文题→分步算式→「答案是：X」；含少量 MetaMath 式英文算术 | `data/gsm8k_zh.jsonl` 7,471 题（有 gold）；`data/distill/gsm8k_claude_sh*.jsonl` 共 1,231 行 Claude 生成已校验 gold；`data/sft/sft_reason_v1/cot_reason.jsonl` 48k 英文数学 CoT |
| **通用格式 / MC / 短指令** | 20% | ~40M | 中英短问答、选项作答（「答案：X」）、代码解释/改 bug/函数体短对 | `data/sft/code_if_pairs_dc_train.jsonl` 100k 代码指令对（已去污）；`data/sft/sft_func_v2/verified.jsonl`；可补 Code-Feedback（§6） |

代码:数学:其他 ≈ 5:3:2，数学占比比 Qwen2.5-Coder（近纯代码）高，因为本模型数学是 0 分、代码已 28%，短板优先；
但不超过 30%，避免压住代码基本盘（OpenCoder 实测代码专域数据不可用通用推理替代，§6）。

### 3.2 各桶构建要点

- **代码推理**：OCR 原始 R1 链中位 24K 字符（d1 实测 med 23,151 chars），远超小模型该学的长度。
  二选一：(a) 用 d1 已建的 24k 篇里**推理段 ≤1,500 token、code ≤300 行**的子集（需在 packer 侧加长度切分）；
  (b) 更稳妥——保留每题 **1 个**结构最干净、最短且 `ast.parse` 通过的解（现在是 per_id=2）。
  代码块必须与 `solution` 字段一致（d1 的 `ocr_shards.py` 已校验链尾 code==solution）。
  **不要**为「正确性」只留执行通过的解：OpenCodeReasoning 消融证明全量（含错误解）比只留正确解高 7.1 LCB
  （§6），多样性更重要；我们的语法闸已够，别加运行时硬过滤。
- **中文数学**：7,471 题的 gold 都能抽末位数字。每题配 1 条 600–1200 字的分步解（模板：列条件→分步算式→「答案是：X」）。
  1,231 行 Claude 蒸馏已是这个形态，扩到 ~8–10k 行即可（可再派 4 个生成 worker，或用教师 ckpt batch 生成）。
  末位数字必须 == gold（已有 `distill_gen.py` 的 last-number 校验逻辑）。
  英文数学 CoT 用 cot_reason 48k 池，按 §3.3 长度筛后采样，别全灌。
- **通用/MC**：目标是教「按要求格式给结论」。code_if 对是 continuation 风格，pack 时也要包 ChatML、
  answer 段保持纯代码/短结论；另构造 ~2 万条**选项作答**对（从 ceval/arc 风格题干生成「答案：C」式短答），
  专治 OpenBookQA 23.6 / eqcheck 2%。这部分短、便宜，是格式先导的主力。

### 3.3 长度纪律（对 614M-active 学生最关键的一条）

业界最直接的反证（§6，arXiv:2502.12143）：Qwen2.5 学生里 **≤3B 从超长 R1 链学到的反而更差**
（0.5B −4.74、1.5B −7.13、3B −3.08 平均分），交叉点在 3B→7B；
1:4 难长:易短的「Mix Distillation」把 3B 从 40.3 拉到 45.9、MATH 50.7→64.7。
对应到本模型（激活 614M）：

- answer 段目标分布：**70% 落在 300–1000 token，25% 1000–1800，5% 最长 2500**；pack SEQ 保持 4096（`sft_math.py` 现网路径）。
- 硬上限 2500 token/回答；超长的截断到完整步骤边界并补终止符，绝不留没写完的链（没结尾的链教复读）。
- OpenCodeReasoning 报告本身：蒸馏模型比 QwQ 少用 20–30% 推理 token，16K→32K 预算对难题无增益——长度不创造能力。

### 3.4 去污染与防泄漏（硬门，沿用本仓管线）

- 所有 SFT jsonl 先过 `scripts/filter_gate_domains.py`（13-gram，HumanEval+MBPP；
  数学加 `--extra_math` 对 gsm8k_test/math_500）。OCR 域实测仅 0.28% 命中、6-8 题，量级正常。
- gsm8k_zh 用 **train 题**，math-500/gsm8k 测试题绝不进 SFT（AGENTS.md：math-500 的 SFT 后分会被语料重叠污染）。
- 本仓 `prepare_sft.pack_and_save` 已带 holdout_fp，`sft_math.py:384` 会拒绝未声明 holdout 的包；新包必须盖戳。
- 沿用现有 packer，不新写（packer 已过 `test_sft_pack` / `test_arch_compat`）。

## 4. 本仓历史先验（同一套模型家族，先读再跑）

| 干预（旧 30B CED，非 v42） | 数据 / 超参 | 结果 | 出处 |
|---|---|---|---|
| ChatML SFT 0913 | code/EN/zh ChatML pack，2 epoch，B4，**lr_scale 0.1**，4804 step | ChatML 臂 HumanEval **0→11/164**；空 103→28（残留全是 max_new，0 eos-first）；判「RL-amplifiable signal」，非过门类 SFT | `facts/v41.json#v41.humaneval_chatml_sft_step16000_0913` |
| phi 式 continuation SFT n6 | L3 留出对，~42M 监督 token | 与 E0 无统计可检差异，**INEFFECTIVE** | `facts/v41.json#v41.phisft_n6_psft_vs_e0_0915` |
| zh_think_v1 SFT | 4487 packed rows / 15.28M 监督 token，2 epoch，lr_scale 0.1 | 训练完成（中文分步） | `runs/experiments.jsonl#v41_sft_zh_think_v1` |
| sft_reason_v1 | 14980 rows / 33.12M 监督 token | 训练完成 | experiments.jsonl |
| sft_func_v2 | 1867 rows / **3.39M** 监督 token | 训练完成；量太小 | experiments.jsonl |
| 当前 distill_code/distill_math 包 | 905 rows / **2.94M**、93 rows / **0.27M** 监督 token | 已 pack 但量级远不够 SFT | pod `data/sft/distill_*.pt` labels 实测 2026-10-07 |

读法：

1. **ChatML 格式干预确实把 0 打到非 0 并修了停止**（0913：空 103→28），所以 §2 的格式路径有效；
   但单次 ChatML SFT 只到 11/164，**格式不是过门类能力本身**——必须配上 §3 足量的数学/代码推理对。
2. **窄 continuation SFT 无效**（phi n6），且与 AGENTS.md「base 用 continuation、SFT 用 ChatML」一致——
   v42 走 ChatML，不重做裸 continuation。
3. 现有 distill 包（2.9M / 0.27M token）比历史有效包还小一个数量级，不能直接拿来当正式 SFT，
   只能当 smoke。

## 5. 超参建议（v42_s24，MoE，sft_math.py）

`sft_math.py` 已支持 `--arch v42`（带 `refuse_v42_unsupported`：禁 fp32_master / stochastic_round / fone / loop / prefix，
走纯 bf16 + V4.1 优化器）。下列值可直接映射：

| 项 | 建议值 | 依据 |
|---|---|---|
| 峰值 LR | **2e-5（`--lr_scale 0.2`，预训练 1e-4 的 0.2×）** | 1–4B SFT 主流 5e-6–2e-5（Qwen2.5 7e-6、Tulu3 5e-6、OpenMath/MetaMath/Numina 2e-5）；OCR/代码可到 5e-5。小 MoE 激活低、专家易漂，取 2e-5 而非 5e-5；smoke 从 1e-5 起 |
| schedule | cosine 衰减到峰值 1/10，warmup **3%** | R1 distill 衰减到 1/10；Qwen2.5-Math 2e-5→7e-7；Tulu3 warmup 0.03 |
| epoch | **2**（smoke 1） | Qwen2.5/Numina/OpenMath/OCR(3) 普遍 2–3；§4 本仓历史用 2；>3 仅 LIMA 式极小数据 |
| 有效 batch | **≥256 条 4096 序列**（micro B4 × accum × world 凑齐） | OCR batch 256、OpenMath 512、Qwen2.5-Math 128；MoE 用大 batch + 低 LR 稳专家 |
| seq len | **4096**（不碰 32K） | 现有 pack/训练路径与 §3.3 短链纪律；OCR 用 32K 是因为其长链+TP/CP，本模型刻意学短链 |
| 精度 | bf16，`sft_math.py` v42 路径（无 fp32 master） | 代码已强制，勿传 `--fp32_master`（否则 master 回写静默还原权重，见其 docstring） |
| 训练量 | 200M 监督 token ÷ (256×4096×~0.5 监督占比) ≈ **380–450 optimizer step / epoch ×2 ≈ 800–900 step** | 估算，监督占比按 completion-only 约一半；用 pack 实际 supervised token 复核 |
| 保存/选择 | 每 1/4 epoch 存档 + **末 4 点权重平均** | OpenMath/OCR-Nemotron 用 4-checkpoint avg；选点看 math-500/gsm8k/HumanEval 不看 train loss（LIMA：生成质量不跟踪困惑度） |
| optimizer | AdamW，wd 0.1（若 sft_math 暴露），grad clip 1.0 | Qwen2.5 / R1 distill 通行 |

### 5.1 冻结决策：不冻结（且 engram 无对象可冻）

- `v41f/config.py:342-353` 的 `v42_s24()` 显式 `engram_layer_ids=()`、`n_mtp_layers=0`、
  `dspark_target_layer_ids=()`：**v42 没有 engram 表、没有 MTP/dspark 头**。「是否冻结 engram」在 v42 是伪命题。
- 业界被调研的所有一手 SFT 配方（Phi-3/4、Qwen2.5/3、Tulu3、Granite、OpenMath、OCR、OpenCodeInstruct）
  **无一冻结 embedding / 早层 / MoE router**，全部全参微调；唯一相关技巧是 Granite 把 EOS embedding 重置为均值
  以解「预训练 EOS 粘连、逐 turn 不停」——这与 step54000 不停机同源，可作为备选（先不做，靠 `<|im_end|>` 监督解决；
  若 smoke 后仍不停止，再考虑重置 token 1 / im_end 的 embedding）。
- 用**统一低 LR**控灾难性遗忘，不靠冻结。若 smoke 显示 val（预训练 MC/代码 bpb）大幅退化，
  再做「router 不冻、只把 LR 降到 1e-5 + 混入 10% 预训练文本回放」的备选（Granite stage replay 思路），
  而**不是**先验冻结。

### 5.2 遗忘控制

- 低 LR + 2 epoch 本身是主闸。
- 混合里的 code_if / 通用短答（20%）天然起 replay 作用。
- smoke 与全量都要在训练前后测同一组**非目标**指标（ceval、英文 MC、HumanEval bpb），
  退化超过 ±1 个 Math-eval 分辨率（~1pt）才触发 §5.1 备选，不要凭单次噪声调参（AGENTS.md：±1pt 是 math-500 分辨率）。

## 6. 业界依据（2026-10-07 调研，均为一手 arXiv/模型卡，含与小模型相关的反证）

小模型直接相关：

- **arXiv:2502.12143 Small Models Struggle to Learn from Strong Reasoners**：≤3B 学生从超长 R1 链学到的更差
  （0.5B −4.74 / 1.5B −7.13 / 3B −3.08），交叉点 3B→7B；1:4 难长:易短 Mix Distillation 显著修复
  （3B MATH 50.7→64.7）。→ §3.3 短链 + 长短混合的直接依据。
- **arXiv:2501.12948 DeepSeek-R1 §B.4.3 Table 6**：纯 SFT（无 RL）800K R1 样本即可蒸馏；
  初始 LR 随尺寸反比：1.5B 用 **1e-4**、7B 8e-5、8B 5e-5、32B 6e-5；cosine 到 1/10，batch 64，ctx 32K，2–3 epoch。
  Distill-Qwen-1.5B 达 AIME24 28.9 / MATH500 83.9；HF 卡建议无 system、强制推理前缀。→ 小模型可吃长 CoT 但需短上下文适配与高 LR（我们用激活口径折算后取保守 2e-5）。
- **arXiv:2409.12122 Qwen2.5-Math**：1.5B 用 2M 英文+0.5M 中文 CoT，3 epoch、batch 128、seq 4096、
  **LR 2e-5→7e-7**，13-gram+LCS>0.6 去污。→ 中文数学 + seq4096 + 2e-5 最贴近的尺寸锚点。
- **arXiv:2504.16891 OpenMath-Nemotron**：1.5B 峰值 LR **3e-4**、6 epoch、batch 1024，540K 题/3.2M 长链；
  末 4 checkpoint 平均；1.5B 用纯 CoT（TIR 长工具链小模型吃不消，未完成率 40%）。→ 小模型避免工具型长链。
- **arXiv:2505.09388 Qwen3**：0.6B/1.7B/4B 小模型走强→弱蒸馏 + 全参更新（无冻结），冷启动 SFT「样本数与步数都要尽量少」；
  显式过滤复读/回环/猜测。→ 支撑不冻结 + 短而精 + 反复读。
- **arXiv:2505.22425 PromptCoT-Mamba-7B**：非标准架构 7B，两阶段 1.88M 对，AdamW LR 5e-6、betas(0.9,0.95)、
  batch 64；消融：去掉 OCR 代码推理数据 LCB 29.9→13.6——代码推理数据不可替代。→ 代码桶保留 50%。
- **arXiv:2409.12186 Qwen2.5-Coder**：SFT 把 0.5B HumanEval 28.0→61.6、1.5B 43.9→70.7、3B 52.4→84.1；
  两阶段（广覆盖→高质量）+ 少量 FIM。→ 28.21% 的 base 经代码 SFT 有 30–40pt 上行空间。
- **arXiv:2504.04030 OpenCodeInstruct**：1.5B 用 5M 样本、**保守 LR 5e-6**、3 epoch、batch 2048，
  HumanEval +8.0 / MBPP +11.0。→ 大数据多 epoch 时 LR 要更低（我们 200M 属中小量，用 2e-5）。
- **arXiv:2411.04905 OpenCoder**：两阶段顺序 > 混合；去污染删 10-gram 测试重叠。→ 去污门 + 若做第二阶段再分序。
- **arXiv:2412.15115 Qwen2.5 / 2411.15124 Tulu3 / 2412.08905 Phi-4 / 2305.11206 LIMA / 2412.13337 Secret-Recipe**：
  chat SFT LR 1e-6–7e-6、2 epoch、大 batch、completion-only；LIMA 证小而精；Secret-Recipe 证低 LR 大 batch 堆叠训练、
  未见冻结/LoRA 收益。

代码蒸馏主案例（OpenCodeReasoning）：

- **arXiv:2504.01943 (COLM 2025)**：735K Python / 28K 题，DeepSeek-R1 蒸馏；学生 7B/14B/32B，
  3 epoch、AdamW batch 256、seq 32768、**LR 5e-5（网格 [1e-5,1e-4] 选出）**、cosine warmup 0.1、packing、bf16；
  OCR-32B LCB 61.8 / CodeContests 24.6（SFT-only，超 R1-Distill-7B 38.0/11.1）。
  **关键反直觉消融**：全量 445K（含错解）54.1 LCB > 只留执行通过 151K 的 47.0，错解 SFT 甚至比正解高 5.3——
  指令多样性优先，结构过滤（标签/单代码块/Tree-sitter 语法/推理段无代码）即可。
  蒸馏模型省 20–30% 推理 token，16K→32K 无增益。数据集**不带 chat 模板**（要自己包 ChatML）。
- **OpenCodeReasoning-2（HF 卡）**：1.4M Python，新增 judgement/pass_rate，可供后续正确率感知采样。

数学/代码 SFT 数据源案例：MetaMath（2309.12284，395K，3ep/2e-5）、Orca-Math（2402.14830，answer-only loss/不 pack/1e-6）、
OpenMathInstruct-2（2410.01560，13.97M，2ep/2e-5/batch512，容忍 ~20% 错解，去污 50K）、
NuminaMath（860K，2e-5/3ep/warmup0.1/ChatML apply_chat_template）、Code-Feedback（2402.14658，
156K 短代码指令 Qwen-72B 复杂度 4–5 过滤）、Magicoder（2312.02120）。

未在一手来源核实的数字一律不写；上表均给了可复查的 arXiv id 与章节。

## 7. 执行顺序与验收（拍板后）

1. **数据任务（不占 GPU，可并行）**：①OCR 短链重选（`ocr_shards.py` 加 answer≤2500 token/题取1最短解，重建一个 SFT 用 jsonl，
   与 pretrain 域 `code_opencot_dc` 分开）；②中文数学扩到 ~10k gold 校验行；③code_if 抽 ~30k + MC 短答 ~20k。
   全部过 13-gram（+extra_math）去污染、holdout 盖戳。
2. **pack（CPU）**：用 `datagen/prepare_sft.pack_and_save`（ChatML + split_encode + prompt-mask），
   出 `data/sft/v42_sft_v1.pt`，stats 记录各桶 supervised token 与长度分布。
3. **格式 smoke（GPU，8 卡空闲窗口）**：20M token / 1 epoch / lr_scale 0.1；
   验收硬门：gsm8k-style 20 题 ChatML 采样复读率 0、`<|im_end|>` 正常停止（对照 step54000 的回环）。
4. **全量 SFT**：200M token / 2 epoch / lr_scale 0.2 / cosine→0.1 / warmup 0.03，末 4 点平均。
5. **评测（全用 ChatML prompt）**：math-hard、math-500、gsm8k（看 0→? 与复读率）、
   HumanEval `--chatml`（28.21% 基线）、ceval/英文 MC（格式分）、eqcheck（等式产出率）；
   同时回归代码 bpb / ceval 防遗忘。
   **门建议（主控拍板）**：math-500 从 0 到 ≥30%、gsm8k ≥30%、HumanEval ChatML 不低于 28%（不退化）、
   MC 格式类（OpenBookQA/eqcheck）显著离开随机/近零。
6. 数据/包/评测行全部进 `facts/` 与 `runs/experiments.jsonl`，负结果也结案（沿用本仓规矩）。

## 8. 风险与不做的事

- **不灌超长 R1 链**（§3.3/§6 小模型反证）；不因「答案可执行验证」硬过滤 OCR（OCR 自身反证）。
- **不冻结**（无 engram 对象；业界全参）；不上 fp32 master / stochastic_round（sft_math v42 已拒）。
- **不用裸 continuation 教对话**（AGENTS.md + phi n6 无效）；SFT 后不用 continuation prompt 评测。
- 不一次全量押注：先 smoke 修格式，再全量。
- 本文不决定是否在 SFT 后接 RL/DPO（那是下一阶段；Qwen2.5/OCR 都是 SFT 后再 DPO/GRPO）。
