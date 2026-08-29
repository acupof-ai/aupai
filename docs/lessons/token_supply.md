---
question: "到哪去拿 30B-100B token：可达性、传输、磁盘、许可、重复率的供给清单与最省采购单"
status: recorded
source: "ModelScope dataset API (StorageSize/ApprovalMode), arXiv, project measurements 2026-08-30"
---

# Token 供给清单（P0）与 30B 最省采购单（P1）

方法：token 量 = ModelScope StorageSize ÷ 4.5 bytes/token（`facts/multilingual.json#mlm.calib.zh_bytes_per_token`，UTF-8 3B/字 × 冻结分词器 1.5 字/token；SkyPile 自述 620GB/150B=4.13 校验通过）。传输时间按唯一实测带宽 1.16 MB/s（`facts/multilingual.json#mlm.bandwidth.measured_relay`，HF 经 tn 中继）算悲观上界，h = GB × 0.24；ModelScope 直连速度未实测。可达性以 API `ApprovalMode` 为准（null=公开，1=需申请）。每行数值见对应 fact id。

## P0 供给清单（公开可达）

| 源 + 路径 | 语言 | token(估) | 可达性 | 传输(h) | 磁盘 | 许可 | 与现有重复 |
|---|---|---|---|---|---|---|---|
| BAAI/CCI3-HQ `#mlm.src.cci3_hq` | zh | 115B（自述100B） | 公开 | 124 | 518GB | Apache 2.0 | 低，待实测（与 fineweb2 不同上游） |
| modelscope/SkyPile-150B `#mlm.src.skypile` | zh | 148B（自述150B） | 公开 | 160 | 665GB | Apache 2.0 + Skywork CLA | 中（chinese-fineweb-edu 的上游；与 fineweb2 不同） |
| TeleAI/TeleChat-PTD `#mlm.src.telechat_ptd` | zh | 110B | 公开 | 119 | 497GB | Apache 2.0 | 低，待实测 |
| BAAI/CCI3-Data `#mlm.src.cci3_data` | zh | 242B | 公开 | 261 | 1.09TB | Apache 2.0 | 低（全量未 HQ 过滤，质量低于 HQ） |
| opencsg/Fineweb-Edu-Chinese-V2.1 `#mlm.src.fineweb_edu_zh_v21` | zh | 538B（自述1.5T） | 公开 | 580 | 2.42TB | Apache 2.0，商用需邮件 | 中（上游含 SkyPile/MAP-CC/TeleChat-PTD） |
| BAAI/CCI4.0-M2-CoT-v1 `#mlm.src.cci4_m2_cot` | zh+en | 571B | 公开 | 616 | 2.57TB | Apache 2.0 | 与现有 math/chat 域可能重叠 |
| BAAI/CCI4.0-M2-Base-v1 `#mlm.src.cci4_m2_base` | zh+en（zh 15%） | 1.27T（zh ~190B） | 公开 | 1371 | 5.72TB | Apache 2.0 | 待实测 |
| OpenBMB/Ultra-FineWeb `#mlm.src.ultra_fineweb` | zh+en | 2.16T（自述1.12T） | 公开 | 2332 | 9.73TB | apache-2.0 | 待实测 |
| m-a-p/MAP-CC `#mlm.src.map_cc` | zh | 358B（自述800B） | 公开 | 386 | 1.61TB | CC-BY-NC-ND（禁商用） | 待实测 |
| Shanghai_AI_Laboratory/WanJuan1_dot_0 `#mlm.src.wanjuan1` | zh | 74B | 公开 | 80 | 334GB | CC-BY-NC | 待实测 |
| whynlp/WuDaoCorpus-200G-shuffled `#mlm.src.wudao` | zh | 21B | 公开 | 23 | 96GB | CC-BY-NC-SA | 待实测 |
| Shanghai_AI_Laboratory/MiChao `#mlm.src.michao` | zh | 20B | 公开 | 21 | 88GB | CC-BY-NC-SA | 待实测 |
| Shanghai_AI_Laboratory/WanJuanCC `#mlm.src.wanjuancc` | en | 46B | 公开 | 50 | 193GB | CC-BY-4.0 | 待实测 |
| en math/CoT 小包 ×7 `#mlm.src.numina_math_cot` 等 | en | ~7B | 公开 | ~8 | ~31GB | Apache/MIT/cc-by | 与现有 math 域待去重 |

en 小包 = NuminaMath-CoT 1.23GB + MetaMathQA 0.23GB + DAPO-Math 0.30GB + Skywork-OR1 0.82GB + OpenThoughts-114k 3.55GB + OpenR1-Math-220k 12.6GB + OpenMathInstruct-2 12.6GB。

两处前提纠正（代理 C 核实）：Ultra-FineWeb 是 fastText 过滤语料，中文 120B token，不存在 "anchored rewrite" 说法；CCI4 在 ModelScope 以 M2 三件套（Base/CoT/Extra）发布，中英双语（中文仅 15%），无纯中文 `BAAI/CCI4` 路径。

## P1 — 30B 最省采购单

30B 中文 token ≈ 135GB 下载（`#mlm.calib.zh_bytes_per_token`）。**实测修正 2026-08-30：CCI3-HQ 真实 = 96 LFS 分片 / 108.03GB（非 135GB 高估）→ 约 22-24B token，非 30B。** 若必须凑 30B，在 `BAAI/CCI3-Data` 同源补 2-3 片（每片 ~1.1GB，几分钟）或改计 TeleChat-PTD。

**主单：BAAI/CCI3-HQ 全量（96 片 / 108.03GB）→ ~22-24B token**

- 许可 Apache 2.0，公开，无需账号；同档许可里唯一 HQ 过滤子集
- 与现有 web_hq（fineweb2 zh）不同上游，预期重复低
- 传输：ModelScope 直连实测 **22.9 MiB/s**（训练节点 115.190.184.36）→ 108GB ≈ **80 分钟**，非 32h 悲观上界
- 磁盘 108GB + 解压/去重工作空间（`/work` 2.0T，余 1.1T，非约束）
- 入库前 MinHash vs mix_v3 去重 + b0 污染扫描（重复率未实测）

**可选加购（en 迁移赌注，~8h / 31GB / ~7B token）**：en math/CoT 小包，按需截取使 en 占比 ≤10%。理由见 `multilingual_mix.md` P3。

**合计：~108GB 磁盘，~80 分钟（ModelScope 直连实测），换 ~22-24B zh + 可选 ≤7B en。**

备选（若 CCI3-HQ 重复率实测过高）：TeleChat-PTD 切片 135GB（Apache 2.0）或 SkyPile 切片 135GB（注意 SkyPile 是 Fineweb-Edu-Chinese 的上游，若未来要用 V2.1 则二者取其一）。

## REJECT（含毙因判据）

| 源 | 毙因 |
|---|---|
| Chinese-FineWeb 原版 `#mlm.src.rejected_chinese_fineweb` | 可达性：ModelScope 无，HF 不可达 |
| ChineseWebText `#mlm.src.rejected_chinesewebtext` | 可达性：同上 |
| Nemotron-CC 中文 `#mlm.src.rejected_nemotron_cc_zh` | 可达性：同上 |
| OpenWebMath / DeepSeekMath / OE-MATH `#mlm.src.rejected_openwebmath` 等 | 可达性：同上 |
| Fineweb-Edu-Chinese-V2.2 `#mlm.src.rejected_fineweb_edu_zh_v22` | 可达性：ApprovalMode=1 |
| liwu/MNBVC `#mlm.src.mnbvc` | 可达性：ApprovalMode=1（MIT 但需申请；840B，申请通过可复活） |
| COIG-CQIA `#mlm.src.rejected_coig_cqia` | 类型：SFT 指令数据，非预训练爬取 |
| wikipedia 系列 `#mlm.src.rejected_wikipedia_dupe` | 重复率：现有 wiki 域已是 wikipedia zh 20231101 |
| MAP-CC `#mlm.src.map_cc` | 许可：CC-BY-NC-ND 禁商用（研究用可复活，358B） |
| WanJuan1.0 / WuDao / MiChao | 许可：CC-BY-NC 系，商用毙；研究用保留在 P0 表内 |
| CCI3-Data 全量 / Ultra-FineWeb / CCI4-M2 全量 | 非 REJECT：30B 档用不上，100B+ 档备选 |

## 未闭合数字

- ~~ModelScope 直连带宽~~ **已测 2026-08-30：22.9 MiB/s**（训练节点 115.190.184.36，CCI3-HQ part_000059 45s 拉 1.08GiB；与服务节点 Qwen weights 23.1 MiB/s 一致）
- CCI3-HQ vs mix_v3 重复率（MinHash，入库时跑）
- ~~pod 可用磁盘~~ **已测**：/work 2.0T，余 1.1T

## Known issues（实测记录 2026-08-30）

| question | status | source | note |
|---|---|---|---|
| pod 的 curl 太老不认 `--retry-all-errors` | recorded | curl 7.64.0（两节点实测） | `--retry-all-errors` 是 7.71+ 才加。7.64 上它是**未知选项**：`curl` 立刻退出、0 字节，循环里全部行记为 FAIL 空转，`-s` 吞掉错误 → 静默失败教科书形状。所有 pod 下载脚本不要带新式 curl flag |
| 被截断的输出会无声藏掉内容 | recorded | `du -sh corpus/* \| head`（本会话二次中招） | `head` 按字母序截断，`batch_*` 之后的 `textbook/web_hq/wiki` 全部看不见且**无任何截断信号**——"这台没有语料"是错的。与 b0 的 `+8 more` 藏红同形。任何 `| head`/`+N more` 的输出，先问"被截掉的是什么"再下结论 |
| 两个 H20 静态 pod 共享 hostname | recorded | `iv-yeozpb5g5cbw80bls64e`（服务节点 + 训练节点同报） | **hostname 不能区分节点**。服务节点=tn隧道/arle（810G 推理权重、无语料），训练节点=115.190.184.36（15G v3 语料 + 训练在跑）。访问用 `podx`（直 SSH 115.190.184.36）；操作前先验 `/work/aupai/data/corpus/*` 是否有 v3 域 |
| CCI3-HQ 真实规模与带宽 | recorded | ModelScope API + 45s/60s 实测 | 96 LFS 分片 / 108.03GB（非 135GB）；直连 22.9 MiB/s（非 1.16 中继上界）。108GB → 约 22-24B token 非 30B |
