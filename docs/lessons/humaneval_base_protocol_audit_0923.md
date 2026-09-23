---
question: "eval/humaneval_sample.py / humaneval_gen.py 是否等价于发布 base-model HumanEval 数字所用的协议（bigcode-evaluation-harness、DeepSeek-Coder、Qwen2.5-Coder、OpenAI Codex）；不等价的地方会不会把 base 分数读低"
status: recorded
source: "ae 协议审计 2026-09-23：4 个外部源 fetch 原始码并二次独立复核（8 agent，源 sha 见各节）；本仓 eval/humaneval_gen.py、eval/humaneval_sample.py @ main；机制与升降分证据 facts/base_eval.json#be.humaneval_rstrip_tokenizer_artifact"
---

# HumanEval base-model 评测协议审计

2026-09-23，ae。问题（aupai-1e）：step2000 CPU greedy 大部分题目在 docstring 后立刻 EOS，
少数重声明同名 `def` 被 `\ndef` stop 切成空；历史 base model 在本 harness 也是 0/164、97.6% 空。
这是模型差，还是评测协议和发布 base HumanEval 数字所用的协议不一致？

## 结论（先读）

**有一处差异实质压低 base 分数：prompt 末尾换行。** 标准 HumanEval prompt 164/164 以
docstring 闭合引号后的一个裸 `\n` 结尾（147 `"""\n`、17 `'''\n`，0 条带缩进）。发布 base 数字的
harness 在**生成前**处理掉它——bigcode 默认 task 用 `doc["prompt"].strip()`，DeepSeek-Coder
两次 strip 后在结束于 `"""` 的文本上生成，Qwen2.5-Coder 是 `strip()` 后补回恰好一个 `\n`。
本仓 base 默认臂（`standard`）**原样喂裸 `\n`**（`eval/humaneval_gen.py:496` 的 else 分支）。

这个裸换行在本仓已被实测为 tokenizer 伪迹，把 base HumanEval 从约 12/164 读成 0–1/164
（`facts/base_eval.json#be.humaneval_rstrip_tokenizer_artifact`，同一 r3 ckpt、CPU greedy、
同 truncate/STOPS/judge）：

| 臂 | 6k | 12k | 13k | empty |
|---|---|---|---|---|
| standard（裸 `\n`，默认） | 0/164 | 1/164 | 1/164 | 82/164 → 63/164（探针确认 40/40 是 stop_at_0：模型写下一个 column-0 顶层 def） |
| `--rstrip_nl`（去末尾换行） | 15/164 | 11/164 | 12/164 | 0/164 |
| indent+4（末尾补 4 空格） | — | — | 1/164 | 129/164，全 stop_at_0 |

机制：tokenizer 在 docstring 后把「换行+缩进」并成一个 token；裸换行 token 在训练语料里只出现在
column-0 行之前，等于命令模型 dedent 开一个新顶层 def，于是命中 `\ndef` stop、completion 为空。
补 4 空格同样 OOD（换行+4 空格不是训练过的合并），也救不回来。合规的续写是 `prompt.rstrip("\n")`，
让模型自己吐「换行+缩进」——这正是外部协议等价物，也是本仓已有的 `--rstrip_nl` 臂。

**推荐协议（结论，依据见末节）：base-model 的 HumanEval pass@1 在生成前对 prompt 做
`rstrip("\n")`（等价于外部 strip 后补一个 `\n` 喂入、judge 仍用原始 prompt）；其余四项本仓
与发布协议一致或更严，不改。**

**裁决（用户，2026-09-23，经 aupai-1e 转达）：gate 采用行业协议——生成前 rstrip 掉末尾换行；
旧协议（standard 裸换行）作为并列对照列同时报告，不删。** 落于 runs/prereg.jsonl
v41_ced_0923 amendment_1（fc35cfd9）：验收数 = rstrip（--rstrip_nl）HumanEval pass@1，standard
臂并列报告、不作 gate；阈值 ≥30% 与 n=164 不变；并要求最终 read 前先修 sampled 臂 entry_point bug。
genA 的控制臂已从「末尾 4 空格」改为 `--rstrip_nl`，其读数归 genA，本 lesson 不代报。

另有一个**代码 bug**（不属于协议差异）：sampled pass@1 臂的 truncate 漏传 entry_point，见第 3 节；
已派给 genA 修（reviewer de），本 lesson 只记录、不改码。

## 逐项对照

外部四源均 fetch 了原始文件并由第二个 agent 重新拉取复核（confidence 全 high）。源版本：
openai/human-eval `6d43fb9`（master 2025-01-17）+ 首发 `463c980`（2021-07-08）；
bigcode-project/bigcode-evaluation-harness `8fc5bae`（main 2025-07-22）；
deepseek-ai/DeepSeek-Coder `2f9fd85`（main）；
QwenLM/Qwen2.5-Coder `33bc6aa`（main 2026-03-24）+ 历史树 `29743d2`。

### 1. prompt 末尾换行 —— 不一致，实质降分（见结论）

| 源 | 生成时喂什么 |
|---|---|
| OpenAI human-eval 包 | 原样（`read_problems` 只解析 jsonl；README stub 直接传 `problem["prompt"]`）。但包不含生成器，真正的停止靠 Codex API |
| bigcode（默认 humaneval task） | `doc["prompt"].strip()`（`bigcode_eval/tasks/humaneval.py` get_prompt，`create_task(True)`）。另有一个非默认 `humaneval-unstripped` task 才原样喂 |
| DeepSeek-Coder | strip 两次：数据集构建 + eval 循环；生成结束于 `"""`。judge 时用**未 strip 的原 prompt + 多一个 `\n`** 拼回（`Evaluation/HumanEval/`） |
| Qwen2.5-Coder | `prompt.strip() + "\n"`（multiple-eval 与 evalplus 路径皆然；`--no_new_line_at_last` 才只 strip 不补）。即规范化成恰好一个末尾换行 |
| 本仓 standard | 原样裸 `\n`（`humaneval_gen.py:495-496` else）；`--rstrip_nl` 才 `pr.rstrip("\n")` |

外部三家（bigcode/DeepSeek/Qwen）在生成条件上都不让模型看到「结束于裸换行的 docstring」；
OpenAI 包原样喂，但它是补全 API 且配合 `\ndef` 解码停止，形态不同。本仓 base 默认臂是唯一把
裸换行当正常续写条件的，而这正是被实测读低的那一格。

### 2. stop 序列 —— 同族、列法不同，本仓更宽，不构成读低

发布协议的停止词都是「column-0 顶层结构」语义，列法各家不同：

- Codex 论文（arXiv:2107.03374 §3.2）解码停止：`\nclass`, `\ndef`, `\n#`, `\nif`, `\nprint`（不带尾空格）。
- bigcode（`humaneval.py` GeneralHumanEval.stop_words，8 个，经 `od -c` 确认无尾空格）：
  `\nclass`, `\ndef`, `\n#`, `\n@`, `\nprint`, `\nif`, "  \n```", `<file_sep>`，再加 tokenizer eos。
- DeepSeek-Coder：解码**不传**停止词（只靠 eos / max_new=500），5 个标记
  `\ndef`, `\nclass`, `\nif`, `\n#`, `\nprint` 在解码后做字符串截断。
- Qwen evalplus：vllm 解码停止 = eos 6 个 + `extra_eos_for_direct_completion`
  `\ndef `, `\nclass `, `\nimport `, `\nfrom `, `\nassert `（带尾空格）+ 代码围栏 2 个。

本仓 `STOPS`（`eval/humaneval_gen.py:60-61`）：
`"\ndef "`, `"\nclass "`, `"\nif __name__"`, `"\nprint("`, `"\n#"`, `"\n@"`, `"\nassert "`,
`'\n"""'`, `"\nimport "`, `"\nfrom "` —— 10 个，语义是同族的 column-0 截断，且比各家都更宽
（多了 `\nif __name__` / `\nassert ` / `\n"""` / `\nimport ` / `\nfrom `）。更宽只会更早切掉
跑题写额外顶层语句的 completion，不会把空 completion 算对，方向上不是 0/164 的来源。
列法差异（带不带尾空格、`\nprint(` vs `\nprint`）只影响极少数边角，不是主要矛盾。

### 3. 同名 def 重声明 —— greedy 主路径与发布协议相反（更宽松，不读低）；sampled 臂有 bug

模型从 docstring 闭合处续写时，常见输出是换行重开一个 `\ndef <同名函数>(`。发布协议对此
**一律截断为空**：bigcode `_stop_at_stop_token` 对 completion 取最早停止词，`\ndef` 在 index 0
即返回空串；DeepSeek `_truncate_code_at_stopwords` 同样 `code[:0]==""`；Qwen 把 `\ndef ` 当解码
停止，body 根本不生成；OpenAI 包不截断（重声明会因 Python 晚绑定而被执行、第二个 def 胜出），
但 Codex API 的 `\ndef` 解码停止让它产生不了。**没有一个发布变体保留并给同名重声明的 body 打分。**

本仓分两条路径：

- **greedy 主路径（base 数字走这条）保留 body**：66-14 修复后 `truncate(s, entry_point)`
  （`humaneval_gen.py:99-120`，跳过 `f"\ndef {entry_point}("` 的自命中，`:114-116`），并在
  `run_control` 里用「包一层自己 def header 的 canonical」专门守住这条路径
  （`humaneval_gen.py:315-364`）。这比发布协议**宽松**——发布协议会切空的，本仓给 body 打分。
  方向上只会多给分、不会把分数读低；但意味着本仓 greedy 数与发布协议在这一格**不可直接对比**。
  164 条里 154 条 canonical 是 body-form（非整函数形式），重声明场景普遍。
- **bug（sampled 臂）**：`eval/humaneval_sample.py:110` 调 `truncate(tok.decode(new))` **没传
  entry_point**，其循环内停止判断 `:108` `any(st in ... for st in STOPS)` 也没有自命中豁免。
  于是 sampled pass@1（n=20, t=0.2）会把同名 `\ndef self(` 重声明切成**空串**——与文件 docstring
  自称的「argmax 换成 nucleus，其余逐字复用 greedy 路径」（`:93-94`）矛盾。实测：对一个
  自重声明串，`truncate(s)` 得 `''`，`truncate(s, entry_point)` 保留 84 字符 body。greedy 与 sampled
  因此在同一个 completion 上判分不一致。已派 genA 修（reviewer de，2026-09-23）。

### 4. pass@1 采样参数 —— greedy 与「论文 pass@1」是两种数，本仓两者都给、标注清楚

- 发布 base 数字的实际做法分两派：
  - **DeepSeek-Coder / Qwen2.5-Coder：greedy 单样本 pass@1**（`do_sample=False`/temperature=0，
    n=1，estimator 在 n=1 退化成正确数/总数）。Qwen in-repo 结果 1.5b-base HumanEval 43.9、
    7b-base 61.6 即此协议。
  - **bigcode 文档 / Codex 论文：温度采样 pass@1**。bigcode 默认 `do_sample=True, temperature=0.2,
    top_p=0.95, n_samples=1`，文档协议 n=200（batch 10）报 pass@1/10/100；Codex 论文 base 用
    T*=0.2、n=200、top_p 0.95 的无偏估计 `1-C(n-c,k)/C(n,k)`（T*=0 greedy 只用于微调后的 Codex-S）。
- 本仓：base 主数字走 greedy argmax（`humaneval_gen.py:545`，eos tid=1，max_new=280）；
  sampled 臂 `humaneval_sample.py` 默认 n=20、temperature=0.2、top_p=0.95（`:64-66`），p1 规格
  见 `docs/standards/p1_data_recipe.md`，estimator 用 c/n（k=1 的无偏形式，`:12-13`），且强制先
  复现 greedy（`:153-159`）再跑采样。

结论：本仓 greedy 数与 DeepSeek/Qwen 发布 base 数**同口径**（都是 greedy 单样本），与 Codex 论文/
bigcode 文档的 t=0.2 多样本 pass@1 **不是一个量**，但本仓两条都报且共享同一 judge，标注是诚实的。
n=20 不是论文的 n=200；这是下游预算选择，不影响 greedy 对比。

### 5. 生成后截断规则 —— 同族最早停止词切尾，本仓多两项小差异

- bigcode：对 completion 单独取最早停止词，无 dedent/rstrip/AST（plain humaneval；这些只在
  humanevalpack）；拼 `prompt + cut_completion`（其 prompt 是 strip 过的）。
- DeepSeek：按 **strip 后 prompt 的字符长度**切出 completion，cleanup_code 最早停止词截断，
  无 dedent/strip，再用**未 strip 原 prompt + `\n` + 后缀**拼回（引入一个空行）。
- Qwen：multiple-eval 同 DeepSeek 5 词截断、拼未 strip prompt；evalplus 路径**完全不做**停止词
  截断（vllm 已剔除停止串），但对每个 completion 做 tab→4 空格归一化。
- OpenAI 包：无任何截断，prompt+completion 原样 exec。

本仓：decode 后 `truncate()` 最早停止词切尾（`humaneval_gen.py:99-120`），greedy 路径带同名 def
豁免（第 3 节），judge 用 `prompt + completion + "\n" + test + check(entry)`（`:86`），6 秒 SIGALRM
in-process exec（`:88-96`）。同族，差异在同名豁免（更宽）与 6s/ in-process vs 外部 10s/ 子进程
（更严的超时沙箱，不影响正确性判定方向）。

两个本仓特有的生成侧设置外部都没有，记在这里以免被当成协议等价：**max_new=280**
（外部 500/512/768），以及**每 16 token 才批量检查一次停止词**（`humaneval_gen.py:555-559`、
`humaneval_sample.py:107-109`）——停止最多晚发现 16 个 token，但最终仍做整串 truncate，不改变
切点；280 的长度上限对 HumanEval 函数体（外部 500）偏紧，理论上可能切掉个别超长正解，本次未发现
0/164 由长度导致（empty 已归因 stop_at_0/eos），但作为与发布协议的残留差异保留。

## 推荐协议与依据

base-model 的 HumanEval gate 数字，建议固定为：

1. **生成 prompt = `prompt.rstrip("\n")`**（`--rstrip_nl` 已是这个）；judge 仍用喂入的同一文本
   （`judge(p, c, prompt_text=...)` 已支持，`:82-86,607,629`）。依据：(a) 生成条件与
   bigcode/DeepSeek 一致、与 Qwen 的 `strip()+"\n"` 在「模型自己吐换行+缩进」上等价；(b) 仓内
   受控测量显示这一个字节把 0–1/164 变成 11–15/164 且 empty 归零，机制是 tokenizer 合并伪迹而非
   模型能力；(c) 不加 4 空格——外部无此做法，实测也救不回。
2. **greedy 单样本 pass@1 作为主数**（与 DeepSeek/Qwen base 发布同口径）；t=0.2/n=20 的 sampled
   pass@1 作为第二列继续报，但要在 genA 修掉 `humaneval_sample.py:108,110` 的 entry_point 漏传后
   才能与 greedy 共享 scorer。
3. stop/截断/judge 其余保持现状；同名 def 豁免建议在发布口径数字里**同时给一列发布协议口径
   （切空）**，或在脚注声明本仓对自重声明更宽，避免和 DeepSeek/Qwen 的数不可比。
4. 残留差异登记但不改：max_new 280（外部 500+）、16-token 停止检查粒度、6s in-process exec。

## 边界

- 本审计只读 + CPU，无 pod I/O；外部源是 fetch 的原始码，引用到 repo 路径/论文节，行号随上游漂移，
  以源 sha 为准。
- rstrip 在 r3 12k/13k 的 11/12 与 6k 的 15 在单 greedy 164 题上只差 3–4 题，约 ±0.6pp/题的
  分辨极限内（见 `be.humaneval_rstrip_tokenizer_artifact` 的 uncertainty）；但 **standard 0–1 vs
  rstrip 11–15 的差远超这个噪声**，降分结论稳。
- genA 的 `--rstrip_nl` 控制臂（step2000）在跑，读数归 genA；gate 已裁决用 rstrip（见上）。
  本 lesson 不代报、不重复其臂。
