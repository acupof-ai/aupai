---
question: 交接文档（v42 Engram，2026-10-09 16:24）的阶段 A 与阶段 B 是否完成，证据是什么？
status: measured
source: runs/ladder/**（pod，2026-10-09 09:00–09:30 UTC）、eval/eval_ladder.py、eval/ladder_common.py、eval/ladder/*.py、awb 看板 events.jsonl
---

# E0-E6 评测阶梯：阶段 A / B 验收记录（2026-10-09）

主控会话执行，契约 `docs/standards/eval_ladder_v1_1009.md`（含裁定 R1–R10）。看板 `AWB_DIR=/Users/bytedance/.aupai-team/awb`。

## 1. 阶段 A：修复并自测评测器

| 交接文档 §8 阶段 A 项 | 状态 | 证据 |
|---|---|---|
| 修复 verify 脚本缺 `sys` 导入 | 已闭合（当前副本无法复现） | 本地与 pod 的 `runs/`、`eval/`、`scripts/` 全量 AST 扫描无 `sys.*` 缺 import；新路径由 `selftest` 的错误路径控制兜住：缺 checkpoint 必须 rc=2 且 summary 带 `refused`，且输出必须有 `REFUSE:` 行 |
| 修复 claim 失败仍继续 | 已修 + 三世界控制 | `runs/claim_pattern_selftest.sh` 全过：`--pid $$` REFUSE / 先起作业再 claim 其 pid held / 同卡二次 claim REFUSE；启动器模板「先起 torchrun 拿 pid → 再 claim → 失败即 kill 并 exit 4」写在 `runs/v42_mathfix_engbase_1009.sh` |
| 建立 E0-E6 统一入口 | 完成 | `eval/eval_ladder.py`（子命令 e0…e6 + selftest），产物固定 `runs/ladder/<tag>/<stage>/{items.jsonl,summary.json,log.txt,seal.json}`，退出码 0/2/3/1 |
| 每个评测器加入正例和负例 | 完成 | 7 个 stage 模块各有 `control_selftest`：E0 1 基座 + 1 SFT 正例 + 9 负例；E1 1 正例 + 6 负例 + 1 证据世界 + skip-forward 控制；E2 1 正例 + 6 负例（每条判定都有世界）；E3 22 + 8 世界；E4 29 世界；E5 1 + 9 世界；E6 四类假分片世界 + 已知答案对照 |
| 校验分片合并、ID 覆盖和 seal | 完成并活体验证 | E6 合并只认行数 + 精确 ID 集 + 分片覆盖；smoke（20 题）显示 `rows/ids/shards/merge/denominator/seal/stage_seal` 全 true，`duplicates/missing/extra` 均为空 |
| CPU 与单卡自测 | 完成 | CPU：`python3 eval/eval_ladder.py selftest` → rc 0，7 个模块全过。单卡：E0（基座 + v3 SFT）、E1（基座）、E2（v3 SFT）、E3（基座）、E5（基座）、E6（smoke）、E4（基座）各有一次真实运行 |

### 交接文档 §13 交接验收清单

- [x] 8 张 GPU 状态与文档一致——评测期间按 pid 分配单卡，跑完释放；当前无遗留进程
- [x] `/work/aupai/ckpt_v42s2.pt` 存在且身份一致（step 54051、627 键、24 Engram 键、层 1/5/9/13/17/21、vocab f1f860970d15d623、zip CRC 通过、logits 有限）
- [x] v6 未被放入主线训练或评测（`ckpt_v42_mathfix_v6_1009.pt` 原样，不在任何结果表）
- [x] E0 评测器可运行（基座 PASS；v3 SFT 按 producer_identity 规则 PASS）
- [x] E1-E6 评测器协议明确（base 一律 continuation；E2 只对 SFT，base 调用即 REFUSE）
- [x] 每个评测器有正例和负例（见上表）
- [x] 分片评测记录 GPU、PID、日志、hash 与 ID 覆盖（`shards/<i>.json` + `seal.json`）
- [x] 缺分片时不出总分（E6 的 gates 设计 + 四类假世界控制）
- [x] holdout 缺失问题已明确限制（E6 控制三态：本机 humaneval 缺失记 SKIP 并写明路径，不读成 PASS）
- [x] s1-s4 产物经主控验收——数据线另册；本轮验收对象是 w1-w4 的阶梯模块
- [x] 完整评测通过后才进入 RL（未进入 RL）
- [x] 100B 训练按累计口径和 10B 门控执行（未启动长跑）

## 2. 阶段 B：对正确基座的逐级记录

| 级 | 对象 | n | 退出码 | 门 | 说明 |
|---|---|---:|---:|---|---|
| E0 | ckpt_v42s2.pt | 11 | 0 | identity=true | 身份与有限 logits 全过 |
| E0 | ckpt_v42_sft_v3_v1fmt_abslr_1008.pt | — | 0 | producer_identity=true | SFT 无 step，按 corpus_fp/env_fp/vocab_id 认身份 |
| E1 | ckpt_v42s2.pt | 16 | 3 | nonempty/protocol/no_echo/lang=true；**no_repeat=false** | 11/16 复读；16/16 `stop_reason=max_new`（R8 只记数） |
| E2 | v3 SFT | 10 | 3 | controls=true；**stopped/format/quality=false** | SFT 线停止行为同样失控 |
| E3 | ckpt_v42s2.pt | 72 | 0 | arith_check/controls/coverage/evidence/protocol=true | 实测正确率 0/72，抽取率 0.92；按 R2 基线门不含能力阈值 |
| E5 | ckpt_v42s2.pt | 4 | 3 | controls=true；syntax/unit/sample=false | 实测 0/4，3 题空输出 |
| E6 | ckpt_v42s2.pt（诊断态 smoke） | 20 | 3 | rows/ids/shards/merge/denominator/seal=true；contract_complete=false | 小样本按设计不出总分 |

**停级结论**：交接文档 §6「前一级失败，后一级不放行」。E1 判 FAIL，阶段 B 停在此级。E3/E5 的调起与 E6 的 smoke 都标为 **harness 检查**（`--skip-prereq` 或诊断态），不作为门结果；`check_prereq` 现在会拒绝任何在 E1 失败后启动 E3-E6 的尝试（pod 实测 `e5` → `REFUSE: e5 needs e1 to pass …`）。

## 3. P0（交接文档 §5）状态

| P0 | 状态 |
|---|---|
| v6 用错基座 | 已停止；v6 不进任何结果 |
| verify 脚本缺 `sys` | 无法复现 + 新路径控制覆盖 |
| claim 失败仍继续 | 已修 + 三世界控制 |
| 评测协议混用 | 阶梯固定（R1） |
| 输出停止行为失控 | **已实测**：基座 11/16 复读（E1）、v3 SFT 10/10 不停止（E2） |

## 4. awb 验收记录

`awb check` 全部 accepted：`w1 → selftest --only e1`、`w2 → --only e3,e4`、`w3 → --only e5`、`w4 → --only e6`。worker 交付与主控复验的差异都当场退回并重验（E1 的 R8/R9/skip-forward 三处、E6 的 `--limit` 与三态、e5/e6 的 R3 字段）。

## 5. 遗留与下一步

1. E4 的活体单卡记录在收尾时仍在跑（4 题 × 128 token，无 KV cache，O(T²)）。
2. 阶段 C（重建格式与停止数据）与阶段 D（逐级 SFT）是 E1/E2 FAIL 的直接后续，不在本记录范围。
3. 阶梯代码尚未 commit：主 worktree 处于 detached HEAD，且暂存区有他人文件，提交由主控统一处理。
