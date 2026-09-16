---
question: v2 续训方案 PR #371 的数据余量，从盘上独立重算是多少；尤其 code_ultra_l3_noexec_dc 的 26.7B 里"文档全新"与"已见 stub"各占多少
status: measured
source: 独立第三方核对（0e，2026-09-15），不看 3b/66 结论；pod 直读，只读零生成零 GPU 零删除。cursor 用 ckpt_v41_r3_0914.pt（final，sha f76ddeb9，step38070）的 row_cursor（=fb 转述六值，逐字一致）
---

# v2 pool audit — r3 未见数据余量（独立硬表）

## 口径（先定，避免把行/文档/token 混着加）

- **cursor 单位是 packed row**，且已对齐：final `sum(row_cursor)=7,309,440 = 38070 steps × batch4 × accum6 × world8 = 7,309,440`，逐字相等。
- **packed row**：cache `/data00/tokens_<d>.pt`（torch zip 容器）`data/0` member 字节数 /4（int32）/4097。**1 packed row ≠ 1 文档**；文档在 pretokenize 时多行拼成 4097-token 序列。token 总量以 cache 内 int32 计数为准（= 字节/4）。
- **val 切分**：每域 `min(int(packed_rows×0.05), 5000)`（train.py:298/437 `val_frac=.05 val_rows_max=5000`，r3 mix 无域设 val_frac）。val 前缀 r3 不训。
- **多 epoch 域的"cap 剩余行"是重复，不是新文本**：build_mix 的 epochs cap 是 `trainable_rows×epochs`，cap 内每文档最多见 epochs 次；cap−cursor 的正余量只是"还没重复到的次数"，文本已见过。下表把它与单 epoch 域的"真新行"分列。
- **r3 实际吃的 stub cache 是排除版**：mix_v41_r3 对 code_ultra_l3_stub_dc 带 `cache_exclude 56d12083dc30bdf2`，故池是 `/data00/tokens_code_ultra_l3_stub_dc.excl56d12083dc30bdf2.pt`（273,879 文档排除），不是同名 plain。见末节证据。

## 硬表（每值带文件/字段）

| 域（cache 文件 `/data00/tokens_<域>.pt`） | 池 packed token | packed 行 | 源文档数（字段） | r3 epochs | r3 cursor 行（final ckpt `row_cursor`） | r3 未读 token | 未读构成 |
|---|---:|---:|---|---:|---:|---:|---|
| code_ultra_l3_stub_dc（**.excl56d**） | 4,894,050,865 | 1,194,545 | 13,682,629−273,879=13,408,750（fact config kept_docs/manifest 行数） | 3 | 3,427,726 | **0.577 B** | cap 3×(1,189,545)=3,568,635，余 140,909 行=重复，非新文本 |
| code_keep_p1_dc | 2,626,977,570 | 641,195 | 3,059,135（stats rows_scanned−rows_dropped） | 3 | 1,908,579 | **≈0 B**（余 6 行） | cap 3×636,195=1,908,585，已到顶 |
| code_ultra_l2_dc | 15,340,877,471 | 3,744,417 | 16,359,418（stats `kept`） | 1 | 922,846 | **11.560 B** | 3,739,417 trainable−922,846=2,816,571 真新行 + 5,000 val |
| math_owm_stage2_dc | 5,857,969,755 | 1,429,819 | 4,132,693（scanned 4,135,793−dropped 3,100） | 1 | 603,515 | **3.385 B** | 1,424,819−603,515=821,304 真新行 + 5,000 val |
| en_c4_stage2_dc | 1,985,211,248 | 484,552 | 3,887,752（scanned 3,887,759−7） | 1 | 212,401 | **1.115 B** | 479,552−212,401=267,151 真新行 + 5,000 val |
| cot_dc | 399,994,717 | 97,631 | 851,661（scanned 851,965−304） | 3 | 234,373 | **0.180 B** | cap 3×92,631=277,893，余 43,520 行=重复，非新文本 |
| **code_ultra_l3_noexec_dc** | **26,697,332,099** | 6,516,312 | 15,308,313（stats `kept`） | **0（mix 无此域）** | 0 | **26.697 B 全未读** | 见下节拆分 |
| code_py_starcoder_dc | 7,948,927,615 | 1,940,182 | 6,178,621（6,180,175−1,554） | 0（r3 无，gate mix 才有 .07） | 0 | **7.949 B 全未读** | gate 域，r3 schedule 未列 |
| code_py_rp1t_dc | 379,600,071 | 92,653 | 209,659（209,669−10） | 0 | 0 | **0.380 B 全未读** | 同上 |
| zh_c4_dc | 500,599,208 | 122,186 | — | 0 | 0 | **0.501 B 全未读** | r3/gate 均无 |
| zh_wiki_dc | 270,763,843 | 66,088 | — | 0 | 0 | **0.271 B 全未读** | r3/gate 均无 |

**合计 r3-schedule 未读 ≈ 52.6 B**（26.70 noexec + 11.56 l2 + 7.95 starcoder + 3.39 math + 1.12 en + 0.58 stub + 0.50 zh_c4 + 0.38 rp1t + 0.27 zh_wiki + 0.18 cot）。

口径警示：
- 这 52.6 B 是**字节池余量**，不是"有效新 token 供给"。其中 0.76 B（stub 0.58+cot 0.18）是 epoch cap 内的重复曝光，不是新文本；keep 已耗尽。
- 真"全新文档文本"主要三块：noexec 26.7B（质量分层见下）、l2/math/en 未训尾部 16.06B、starcoder/rp1t/zh 9.10B。

## 专项核对：noexec 26.7B 里"全新 vs 已见 stub"

3b 的断言（~2.84B 全新、~23.9B 是见过的短 stub）我**复现了它的 uuid 算术，但在文本/token 层证伪了"已见"**。

**血缘（两域同源、不同渲染）**——`facts/corpus_supply.json#cs.code_ultra_l3_stub_dc_landed_0913` + 两域 build_corpus_stats：
- code_ultra_l3_noexec_dc：openbmb UltraData L3 的 15,308,313 个 survivor uuid 的**完整 solution** 渲染，26,697,332,099 tok（均 1,744 tok/文档）。
- code_ultra_l3_stub_dc：join 同 15,308,313 survivor，过"最后顶层 def 必须有真实 body"检查后留 13,682,629 uuid，渲染成 **last-def + docstring 短形态**，4,993,975,636 tok（plain 池，均 365 tok/文档）。
- uuid 差集 = 15,308,313−13,682,629 = **1,625,684（10.62%）**。3b 的 2.84B = 0.1062×26.697 = 2.835B，是**用整体均值 1,744 tok 外推差集 uuid**。

**uuid 归属拆分（算术层，成立）**：

| 层 | uuid 数 | 占比 | 按各自实测均值 token |
|---|---:|---:|---:|
| noexec 独有 uuid（stub 检查丢弃：no_last_top_func 1,543,984 等） | 1,625,684 | 10.62% | 1,859 tok/个（抽样）≈ **3.02 B** |
| 两域共享 uuid（不同渲染） | 13,682,629 | 89.38% | noexec 侧 1,723.4 tok/个 ≈ **23.58 B**；stub 侧 365.5 tok/个 ≈ 5.00 B |

**文本重叠实测（runs/v2_overlap_probe.py，确定性 hash 抽样，n=379,686 配对，占共享池 2.8%，无偏：抽样 stub 均值 365.5≈全量 365.0）**：

| 指标 | 实测 | 含义 |
|---|---:|---|
| 共享 uuid 数占抽样比 | 89.38% | 与全量 89.38% 一致，抽样无偏 |
| stub 均 token / noex 均 token（同 uuid） | 365.5 / 1,723.4 | 完整 solution 是短函数渲染的 **4.7×** |
| stub 文本**逐字**是 noex 子串的比例 | **5.95%** | 仅 6% 的短渲染能在完整 solution 里原样找到 |
| 去全部空白后是子串 | 11.47% | 放松空白后仍只 11% |
| stub AST 节点集 ⊆ noex AST 节点集 | 48.7% | 即便只比语法符号，也只有一半被覆盖 |
| noex 行 `ast.parse` 失败 | **194,682/379,686 = 51.3%** | 完整 solution 一半不是可解析模块（多 def/脚本/非纯函数），这正是被 stub 检查过滤的类 |

**结论（与 3b 的实质分歧）**：
- 2.84B 的**文档-uuid 归属**拆分可复现，但用均值估差集 token 偏高一点（差集实测 1,859 tok/个 → 3.02B，不是 2.84B；均值差异因被 stub 丢弃的多为非函数长 solution）。
- "~23.9B 是见过的短 stub"**在文本层不成立**：共享 uuid 的 noexec 内容是 4.7× 长的完整 solution，逐字重合仅 6%、去空白 11%、AST 覆盖 49%。模型在 r3 见到的是 365-tok 的 last-def 短形态（且经 273,879 排除+3 epoch 上限只实际跑到 2.88 epoch），**没见过**这 23.6B 里的完整多函数/脚本上下文。
- 但"没逐字见过"≠"高价值新增"：这层一半 AST 不解析、与已见短函数同题目（uuid），新颖性主要在**函数外的编排/多函数上下文**，需要 v2 自己定是否收。建议的诚实口径：**26.7B 全部是 r3 未读 token；按 uuid 分 3.0B 新题目 + 23.6B 同题目的完整解；其中逐字重复极低但同分布、且约半数 AST-脏。** 不要用"23.9B 已见"给 1.7 天续训的有效增量打折到 2.84B，也不要把 26.7B 全当干净新语料。

## r3 吃排除版 stub 的证据（余量口径前提）

- final ckpt `cfg.mix=data/mix_v41_r3.json`，该文件 stub 带 `cache_exclude:"56d12083dc30bdf2"`；train.py:2708 填 `_CACHE_EXCLUDE`、2143 拼 `.excl<id>` 路径。
- 启动日志 `runs/v41_r3_0914.log`：cache-read 块 stub=**18.23 GiB** == 磁盘 `.excl56d…pt`（19,576,220,099 B/1024³=18.232）；plain 18.60 GiB。计划 `code_ultra_l3_stub_dc 3427733 rows = 2.88 epochs` == excl 可训池（1,194,545 packed−5,000 val=1,189,545）×2.8815。
- 故上表 stub 池用 excl 的 4.894B/1,194,545 行，不用 plain 的 4.994B/1,218,934 行。

## 附：本次读取的脚本与工件

- 每 cache token：torch zip `data/0` member `file_size/4`（只读中央目录，不 torch.load，避开 E0 磁盘 IO）。
- 文档数：各 `data/corpus/<d>/build_corpus_stats.json` 的 `kept`/`rows_scanned`/`rows_dropped`（字段名随域不同，已逐域取）。
- 文本重叠：pod `runs/v2_overlap_probe.py` + `runs/v2_overlap_probe.log`（抽样脚本，可复跑改 K=40）。
- 未写/未改任何数据文件；探针只在 runs/ 落日志。
