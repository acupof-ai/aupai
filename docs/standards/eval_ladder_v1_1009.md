# 契约：E0-E6 评测阶梯（阶段 A + 阶段 B）

来源：`v42_Engram_训练与评测交接文档.md`（2026-10-09 16:24），§5 P0/P1、§6 评测阶梯、§7 多 GPU 规则、§8 阶段 A/B。
主理人：主控会话（awb 看板 `AWB_DIR=/Users/bytedance/.aupai-team/awb`）。
契约是唯一接口，口头约定无效。

## 1. 一句话目标

先修评测器并逐级自测（阶段 A），再对正确基座 `/work/aupai/ckpt_v42s2.pt` 跑 E0→E1→E3→E4→E5→E6（阶段 B）。前一级不过，后一级不放行。

## 2. 硬边界

- 只在 `/work/aupai` 与本地仓库内工作，只操作用户自己的 `sglang-test` 容器。
- **不碰** `ckpt_v42_mathfix_v6_1009.pt`：它是失败对照，不进任何评测结果表、不进训练。
- 不启动训练（SFT / 预训练 / RL），不修改 `data/mix_*.json`、`train.py`、`model.py`、`sft_math.py`。
- 不改他人文件；每个 worker 只写本契约 §4 分配给自己的路径。
- 不删除用户原始文件；不 force push。
- Pod 命令纯 ASCII；多行脚本先写到文件再执行。

## 3. 统一入口与产物

入口：`python3 eval/eval_ladder.py <stage> [--ckpt PATH] [...]`，`stage ∈ {e0,e1,e2,e3,e4,e5,e6,selftest}`。
`e2` 只对 SFT checkpoint 开放；base checkpoint 调 e2 必须 `REFUSE` 退出非零。

每个 stage 的产物固定写在 `runs/ladder/<tag>/<stage>/`：

| 文件 | 内容 |
|---|---|
| `items.jsonl` | 逐题/逐样本：`id`、`prompt_sha256`、`output`、`stop_reason`（token id 判定）、`tokens`、`passed`、`fail_class` |
| `summary.json` | `stage`、`ckpt`、`ckpt_sha256`、`step`、`protocol`、`n`、`passed`、`gates{}`、`inputs{}`、`gpu`、`pid`、`claim`、`shards`、`exit_code` |
| `log.txt` | 原样日志 |
| `seal.json` | 输入数据 hash、tokenizer `vocab_id`、命令、GPU、PID、claim、分片 i/n、题目 ID 列表、开始/结束时间 |

退出码：`0` 全门通过；`2` 前置条件不满足（`REFUSE:` 行说明原因）；`3` 门未通过（结果仍写入）；`1` 崩溃。

`eval/ladder_common.py`（主理人维护，**worker 不得修改**）提供：
`stage_paths`、`write_items`、`write_summary`、`seal`、`require`、`stop_reason`、`load_for_eval`、`verify_coverage`。

## 4. 文件所有权（一个文件一个作者）

| owner | 文件 |
|---|---|
| 主理人 | `eval/eval_ladder.py`、`eval/ladder_common.py`、`eval/ladder/e0_identity.py`、`runs/v42_*` 启动器与 verify 脚本、本契约 |
| w1 | `eval/ladder/e1_base_gen.py`、`data/eval/ladder/e1_prompts.jsonl` |
| w2 | `eval/ladder/e3_arith.py`、`eval/ladder/e4_wordprob.py`、`data/eval/ladder/arith_v1.jsonl`、`data/eval/ladder/wordprob_v1.jsonl` |
| w3 | `eval/ladder/e5_code.py`、`eval/ladder/e6_bench.py` |
| v1 | 只读复算，写 `runs/ladder/_verify/verify_v1.json` 与 `runs/ladder/_verify/` 下的对抗样本 |

## 5. 各级检查项与门（逐条来自交接文档 §6）

| stage | 检查项 | 门 |
|---|---|---|
| E0 | checkpoint 可加载；step/cfg/vocab/tokenizer 指纹一致；Engram 层 `1,5,9,13,17,21`；Engram 权重键 24；state dict 键 627；固定输入 logit 有限；无 NaN/Inf | 任一项不一致即停止 |
| E1 | 中文短句续写、英文短句续写、基础问答 continuation；不回显整段输入；不无限复读；能在固定 token 上限内结束 | base 协议下生成稳定 |
| E2 | ChatML 前缀正确；不输出用户段；不重复 system/user；正常 `im_end`；空输出可识别；复读可识别；停止原因以 token id / `stop_reason` 为准 | 格式与停止同时通过 |
| E3 | 整数加减乘除、小数一步、百分比一步；精确答案匹配；每题算术校验；负例覆盖错误运算与错误精度 | 正确率与算术校验同时通过 |
| E4 | 两到三步中英文应用题；GSM8K 小样；每步可核验；最终答案唯一；不跳题不切题 | 短链正确率、步骤校验、停止行为同时通过 |
| E5 | 语法可编译；函数签名正确；单测可运行；HumanEval 小样；不生成无关解释或密钥样式文本；base 用 continuation，SFT 用 ChatML | 语法、单测、小样正确率同时通过 |
| E6 | GSM8K 全量 1319；官方英文 MATH-500 500；HumanEval clean 156 / all 164；排除题 8 题；MC 保留；逐题 JSONL 与分片日志齐全 | 缺分片、缺 ID、分母错误或 seal 失败，均不得出总分 |

HumanEval 排除题：`HumanEval/105`、`/123`、`/129`、`/156`、`/19`、`/66`、`/71`、`/78`。

## 6. 正例与负例（阶段 A 硬性要求）

每个 stage 必须有 `--control` 自测，正例与负例都要跑，且负例必须 FAIL：

- 正例：已知正确的输入（例如 E5 的 canonical solution、E3 的已知正确答案）→ 必须 PASS。
- 负例：构造一个必须失败的输入（例如 E5 的 constant-False 实现、E3 的错答案、E6 缺一个分片）→ 必须 FAIL，且失败原因落在**该级自己的**判定上。
- 自测结果写进 `summary.json` 的 `controls{}`，`--control` 不过则 stage 退出 3。

## 7. 分片与合并（E6 与任何多 GPU 级）

- 每个分片记录：GPU、PID、claim 名、启动命令、日志路径、输入 hash、`shard_i/shard_n`、题目 ID 列表、逐题 JSONL、退出码。
- 合并只认**行数 + 精确 ID 集 + 分片覆盖**三者同时成立；任一分片失败或缺 ID，不出总分（退出 3）。
- 失败分片单独重跑，不得用缺失分片汇总。

## 8. 验收（主理人跑，worker 自称不算）

`awb check <id> -- <命令>`；命令必须能在 CPU 或单卡上跑，且对**被验收的那个文件**有效。示例：

- w1: `python3 eval/eval_ladder.py e1 --control --ckpt ckpt_v42s2.pt --device cuda:0`
- w2: `python3 eval/eval_ladder.py e3 --control ... && python3 eval/eval_ladder.py e4 --control ...`
- w3: `python3 eval/eval_ladder.py e5 --control ... && python3 eval/eval_ladder.py e6 --merge-selftest`
- v1: 独立复算：对 w1/w2/w3 的负例各跑一次，确认 FAIL 且原因正确。

## 9. 交付格式

每个 worker 回复一行结果：`[awb <id>#<key>] <做了什么> | <产物路径> | <验收命令> | <退出码>`。
产出必须能独立 commit（一个里程碑一个 commit 单元）。

## 10. P0 状态（交接文档 §5）

| P0 | 状态 | 证据 |
|---|---|---|
| v6 使用非 Engram 基座 | 已停止 | 08:30 UTC 启动的 v42_mathfix_engbase_1009 按 pid 1115258 kill，卡全空，claim 已 release，exp 行 08:32 以 `aborted` 关闭；v6 不进任何评测表 |
| verify 脚本缺 `sys` 导入 | 当前副本无法复现 | 本地与 pod 的 `runs/`、`eval/`、`scripts/` 全量 AST 扫描：除 vendored CPython 外无 `sys.*` 缺 import 的文件；`runs/verify_mathfix_v6_gate.py` 在 pod 上已不存在。新路径的控制在 `eval/eval_ladder.py selftest`：缺 checkpoint 必须 rc=2 且 summary 带 `refused`（不许 rc=1 + NameError） |
| 显卡 claim 失败仍可能继续 | 已修 | `runs/claim_pattern_selftest.sh` 三世界全过：`--pid $$` REFUSE、先起作业再 claim 其 pid held、同卡二次 claim REFUSE；启动器模板改为「先起 torchrun 拿 pid，再 claim，claim 失败 kill 作业并 exit 4」（`runs/v42_mathfix_engbase_1009.sh`） |
| 评测协议混用 | 由阶梯固定 | E1/E3/E4/E5 对 base 一律 continuation；E2 只对 SFT checkpoint 开放，base 调 e2 必须 REFUSE |
| 输出停止行为失控 | 已实测，E1 判 FAIL | `ladder_common.stop_reason` 以 token id 判 `eos`/`im_end`/`max_new`/`other`，不看解码文本。基座实测：`runs/ladder/ckpt_v42s2/e1/` n=16 passed=5 exit=3，**11/16 复读**（题干整句回环、句子循环、公式复读），16/16 `stop_reason=max_new`（按 R8 只记数）。问题清单：`runs/ladder/ckpt_v42s2/e1/PROBLEMS.md` 与 `docs/audits/eval_ladder_e1_base_1009.md` |

## 11. 主控裁定（2026-10-09，主控巡检）

四处冲突/漂移已裁定，涉及方按此执行，不再各自定义。

**R1 协议随 checkpoint 类型，不随级号。** 阶段 B 的对象是 base 基座 `ckpt_v42s2.pt`，E1/E3/E4/E5 一律 **continuation**；E2 只对 SFT checkpoint 开放。`data/pretrain_100b/audit/EVAL_LADDER.md` 把 E3/E4/E5 的对象默认写成 SFT/ChatML，与交接文档 §6/§8 冲突：auditor 校验的是**协议与 checkpoint 类型匹配**，不是"必须 ChatML"。base 跑 ChatML 或 SFT 跑 continuation 才是 FAIL。

**R2 基线的门是证据完整性，不是能力数值。** 阶段 B 产出基线，门 = 协议正确 + 逐题证据 + ID 覆盖 + seal + 分母正确。`EVAL_LADDER.md` 里的 GSM8K ≥80%、MATH-500 ≥40%、HE clean156 ≥40% 单列为**目标值**（对训练后 checkpoint），不作基线门；当前基座实测 GSM8K 3.34%、官方英文 MATH-500 4.00%，用目标值当门会让 E6 永远 FAIL。保留的能力门只有两个：项目既有验收 HumanEval clean ≥30%，以及 E6 的回归基线 **48/156 = 30.77%**（v42_gate_1001r step34000）。

**R3 逐题证据接口（auditor 从这里独立复算）。** 主控阶梯写 `runs/ladder/<tag>/<stage>/{items.jsonl,summary.json,seal.json,log.txt}`。`items.jsonl` 每行至少：`id`、`prompt`、`response_raw`、`parsed`、`passed`(bool)、`stop_reason`、`n_tokens`、`fail_class`、`meta`(seed/temperature/耗时)；聚合只在 `summary.json`，且必须能由 `items.jsonl` 现场重算一致。id 以固定文件为准，禁止改 ID 或补造。

**R4 原始语料不进 git。** `data/pretrain_100b/*/raw/` 与 `*/src/` 的克隆（oi-wiki、mdn-zh、leetcode、trpl-zh、react-zh、Python-3.12.7）保持未跟踪；仓库有 `no_oversized_blob`（跟踪文件 >5MB 即 FAIL）。交付物是 `manifest.json` + `report.json` + 有界样本，token 数一律按 `f1f860970d15d623` 实测。

**R5 巡检发现的缺陷，已退回 s4：** `data/pretrain_100b/audit/gates/E1_prompts.json` 不是合法 JSON——`ctrl:garbage` 的 prompt 使用了 `\x00`（JSON 没有 `\x` 转义），且该字段字节是乱码；改为 `\u0000` 或存 hex + 解码规则，改完必须 `json.loads` 通过并同步 `pins.json` 的 sha256。同一批 `E2_prompts.json`、`E3_arith.json`、`E5_code_tiny.json` 已验证可解析。

**R6 资源（用户 2026-10-09 口头授权）：** 必要时可以使用 H20B；交接文档 §12 原写"已归还、禁止重新使用"被此授权覆盖。仍不变的是：不碰他人容器 `linyu-dev`、临时数据用完即删、不在 H20B 上重建历史 cron。当前阶梯只需要 ≤4 张卡，本机 8 张 H20 空闲，暂无必要启用。

**R7 两套题集，两个角色，不合并。** 阶梯的操作集是 `data/eval/ladder/*`（w1/w2 自建，日常跑门用）；审计的交叉核对集是 `data/pretrain_100b/audit/gates/*.json`（s4 钉，独立复核用）。同一套协议与判定，两套数据各自钉 hash，报告 id 一律取输入文件的 id。stage 模块必须提供可选 `--prompts PATH` 且能读审计集的 JSON 结构（`{"prompts":[...],"negative_controls":[...]}`），默认仍用自己的操作集；**不得依赖审计集文件存在**。

**R8 E1 的门不含 EOS 终止。** 证据：`ckpt_v42s2.pt` 的 E1 首跑（`runs/ladder/ckpt_v42s2/e1/`）16 题的 `stop_reason` **全部是 `max_new`**。base continuation 在散文前缀上没有 EOS 线索，要求它吐终止符测的是"提示不完整"，不是"生成是否稳定"；交接文档 §6 E1 的门写的是"base 协议下生成稳定"，独立审计 s4 的 E1 门同样不要求 EOS。因此 `stop_reason` 逐题记、`stopped_rate` 进 summary，**不当门**；E1 的门 = `nonempty` + `protocol` + `no_echo` + `no_repeat` + `lang`（输出语种与 prompt 一致）。后续 base 级（E3/E4/E5）沿用同一条：`stopped` 只记数。

**R9 `no_repeat` 的判据要能抓住已实测到的退化。** 同一份 E1 证据里，zh-qa-01 的 8-gram 出现 3 次、最长重复 span 占输出 **96%**（"地球到月球的距离大约是多少？ 答：…"循环），zh-qa-04 占 **75%**，en-cont-01 占 **50%**，按旧的"8-gram 重复次数 > 4"判据全部记成 pass——判据太松，等于把复读放行。改为两条**任一成立即 FAIL**：

1. 最大 8-gram 重复次数 > 2；或
2. **重复覆盖比例** > 0.5。覆盖比例 = 落在"出现超过一次的 8-gram"内的位置数 ÷ 生成 token 数（不是"最长重复 span 占比"；两者只在 [0.5, 1.0] 区间不同，覆盖 ≥ 最长 span，取更严的一侧）。

三个数都要进 `meta` 以便审计：`max_ngram_count`、`repeat_span_frac`（覆盖比例）、`longest_repeat_span_frac`（最长重复 span 占比）。**"评测器不评任何东西也必须失败"**：`--skip-forward` 这类只校验题集、不加载模型的运行不得返回 0（`n=0` 的运行不是通过的梯级，`check_prereq` 也据此拒绝上一级）。

**R10 items.jsonl 字段以 R3 为准，各 stage 对齐。** 首跑的 E1 文件写的是 `output`/`tokens`/`prompt_sha256`，缺 `prompt`（逐字）、`parsed`、`meta`、`n_tokens`——审计员无法据此独立复算。`prompt_sha256` 可保留为额外字段，但 R3 列的九个字段每个 stage 都要有。

**R11 门链是顺序的，诊断跑走 `--skip-prereq` 并必须留痕。** w2 问：`check_prereq` 让 e3/e4 因 E1 `exit_code=3` 直接 REFUSE，这是否与 R2（基线的门判证据完整性）冲突。**不冲突**：
- R2 说的是**每一级自己的门**不设能力阈值；R11 说的是**级与级之间**的顺序。交接文档 §6 的 E1 门是"base 协议下生成稳定，才进入后续评测器验证"，§8 阶段 B 也是逐级——后面的分数在不稳定生成上是噪声，这是技术理由，不是记账。
- 因此：**E1 失败时 E3-E6 不出门结果**；要拿诊断数字，用 `--skip-prereq`，并把 `--skip-prereq` 写进 `seal.cmd`、在报告里标"诊断，非梯级通过"。w2 已经这么做了，合规。
- 这条随 E1 转绿自动解除：E1 一行 `exit_code=0` 之后，E3-E6 的正常调用不再需要逃逸口。

**R12 两套 E3 题集不可互换，只有操作集进梯级的门。** w2 在 pod 上实测同一基座：操作集（w2 建，题面 `问：计算 94 + 41 的值，只写数值。\n答：`）**4/72 = 5.56%**，审计集（s4 建，题面 `计算：7 + 6 = ?`）**11/20 = 55%**。差异 10 倍，原因不是评测器 bug 而是题面协议：审计集是"补全算式"，模型只需吐数字；操作集要求遵循指令与 `答：` 格式。裁定：
- 两套各自记分，**只有操作集进梯级的门**；审计集是审计员对**评测器**的交叉核对（三态、ID 覆盖、复算），不与操作集比分数。
- 任何把两个数字并列比较的表述都是错的；报告里出现审计集分数时必须写它自己的题面口径。
