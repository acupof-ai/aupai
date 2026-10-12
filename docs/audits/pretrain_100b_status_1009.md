---
question: 累计 100B Engram 预训练这条线现在到哪一步：已训练多少、供给够不够、还缺什么？
status: measured
source: data/pretrain_100b/supply/{manifest,report}.json（s1 交付）、data/pretrain_100b/{codemath,zh,audit}/ 目录现状、aws 看板 events
---

# 累计 100B 线状态（2026-10-09）

目标口径（`/Users/bytedance/.aupai-team/awb/CONTRACT_100B.md`）：累计训练到 **100B token**，每 10B 评测一次，连续两次无提升或明显退化就暂停；**预算不是消耗**。

## 1. 已训练量（s1 台账，已按日志核验）

| 段 | 步数 | token | 依据 |
|---|---:|---:|---|
| v42_gate_1001r（从零，`mix_v41_gate.json`） | 38,146 | **29.999B** | 38,146 × 786,432；日志自报 29.99B @ step 38,140 |
| v42s2 stage-2（从 step34000 续，`mix_v42_stage2.json`） | 20,050 | **15.768B** | 日志 + `record.jsonl` |

- **累计 join 口径：42.507B**；**优化器处理口径：45.767B**；唯一抽样下界 **42.796B**。
- 已确认重复抽样 **2.834B**：gate 跑到 38,146，stage2 从 step34000 续跑同一绝对 cursor，区间 34000..38146 被排了两次。
- 未确认同源风险 **3.604B**：`math_owm_stage2_g4_dc` vs `math_owm_stage2_dc`（3.22B）、`cot_g4_dc` vs `cot_dc`（0.38B）——硬链接孪生域，文档级重叠取决于 `train.py` 的抽样排列，日志无法判定。
- 预算 ≠ 消耗：`mix_v41_gate.json` 30B、`mix_v42_stage2.json` 40B，而 stage2 的 WSD join 实际只排了 15.74B。

## 2. 距 100B 的缺口

| 口径 | 缺口 |
|---|---:|
| 按累计 join | **57.49B** |
| 按优化器处理 | **54.23B** |

## 3. 供给够不够

| 项 | 值 |
|---|---|
| 六个 gate 域唯一 token 总量 | 86.46B |
| 其中**剩余**唯一 token | **50.19B** |
| 六个域第 1 epoch 能否填满缺口 | **不能**，差 **7.30B** |
| pod 上全部独立供给（含额外 `_dc` 缓存） | 约 **141.6B** |
| 判定 | 用全部独立供给**够**填缺口，**但前提是许可核验与近重复审计** |
| 头寸告急 | `cot_dc` 已耗尽（0 唯一头寸，gate 期已 1.42 epoch） |

## 4. 四家交付物的落地状态

| 线 | 交付 | 状态 |
|---|---|---|
| s1 账本与供给 | `supply/{manifest.json,report.json,pod_supply_measured.raw.json,evidence/,scripts/}` | **完整**（本轮唯一完整交付） |
| s2 代码与数学 | `codemath/scripts/{common,code_gen,math_gen}.py`、`qc/code_gen_selftest.json`、holdouts | **脚手架**：`shards/` 空、**无 manifest.json / report.json** → 数据没落地 |
| s3 中文高质量 | `zh/raw/`（oi-wiki、mdn-zh、leetcode、trpl-zh、react-zh 克隆）、`zh/scripts/01_clone.sh`、`zh/work/` | **脚手架**：`out/` 空、**无 manifest.json / report.json** → 数据没落地 |
| s4 独立审计 | `audit/auditor.py`、`gates/{E1,E2,E3,E5}*`、`pins.json`、`allowed_licenses.json`、`DATA_GATES.md`、`EVAL_LADDER.md` | 审计器与门在，**无 manifest.json / report.json**；`gates/E1_prompts.json` 曾是非法 JSON（已退回，待复验） |

三条数据/审计线的 agent 已被用户停掉（看板记录 `done s1..s4 用户要求结束全部子agent`），所以现在是**停在中途**的状态。

## 5. 要恢复这条线，先要三件事

1. **许可核验**：每个域的许可现在没有记录（s1 把这条列为第一风险）。契约写明"不使用许可不清的数据"。
2. **近重复审计**：现在只有精确去重 + 13-gram 去污染，没有 MinHash；而缺口要靠额外 `_dc` 缓存补，那份供给正好没做近重复。
3. **孪生域重叠判定**：`*_g4` 与本体是硬链接孪生，跨段是否重复抽样取决于 `train.py` 的抽样排列，日志判不了——要么读代码给出结论，要么把它们当同一份供给计。

另有两条小的：stage-2 最后 36 步（0.028B）未按域分配；`step 54051` 的最终行未在日志观测到，台账停在 54050。

## 6. 与训练主线的关系

`s1` 的台账回答的是**阶段 F 的入账问题**（缺口、供给、口径），而当前主线卡在**阶段 D**：基座在 E1/E2 上行为不达标，D1b 已经证明格式数据能改行为（E2 1/10 → 7/10）。两条线不冲突，但**阶段 F 启动前必须先过阶段 E 的全量评测**（交接文档 §8）。
