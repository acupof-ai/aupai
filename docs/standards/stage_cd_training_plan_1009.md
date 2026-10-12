# 阶段 C/D 训练方案（2026-10-09）

依据：`docs/audits/eval_ladder_ab_record_1009.md`（阶段 B 记录）与 `docs/audits/eval_ladder_e1_base_1009.md`（E1 问题清单）。

## 1. 为什么必须换数据

| 证据 | 数字 |
|---|---|
| 基座 `ckpt_v42s2.pt` E1（base continuation） | n=16 passed=5，**11/16 复读**，16/16 撞上限不停止 |
| v3 SFT `ckpt_v42_sft_v3_v1fmt_abslr_1008.pt` E2（ChatML） | n=10 passed=0，**stopped/format/quality 三门全 false**；失败类 `no_echo`、`no_extra_turn`、`no_repeat`、`stopped` |

现有 SFT 数据（`data/sft/v42_sft_v1.pt` 660 MB、`data/sft/v42_sft_v2.pt` 770 MB、`data/sft/math_v6/math_sft_v4.pt` 169 MB）教出了回显、加轮次、不复读失败的行为。**不从这些包续训**，按交接文档 §8 阶段 C 重建。

## 2. 阶段 C：数据（关键路径）

每一项交付：`manifest.json`（来源、许可、revision、行数、token、哈希）+ `report.json`（去重、去污染、抽检、失败明细）+ 可复现命令；token 按 `f1f860970d15d623` 实测。

| 集 | 规模 | 做法 | 门 |
|---|---|---|---|
| C1 格式与停止 | 1–2 万条 | 短答（一两个短句）+ 正确 `<|im_end|>` 终止 + 无复读；prompt 用 ChatML 但 system 固定为空（与评测一致） | 每条有来源、长度上限、终止符位置；抽检 200 条人工可读；负例（复读、无终止）必须不存在 |
| C2 一步算术 | 2–5 万条 | 复用 E3 的生成器扩量，答案独立复算 | 100% 可复算；答案唯一 |
| C3 短链应用题 | 1–2 万条 | GSM8K train 改写 + 合成，2–3 步 | 每步可复算；最终答案唯一 |
| C4 基础代码 | 1–2 万条 | 函数级 + 自带可运行单测 | 单测在沙箱里跑过；无人为长解释 |
| C5 中文技术 + 跨域保留 | 各 2–5 千 | 短句、主动语态、术语统一；跨域样本覆盖预训练已有能力 | 抽检 + 去污染 |

**污染**：C2–C4 必须先过 HumanEval/MBPP/GSM8K/MATH-500 的 13-gram 去污染（保留集 hash 见 `data/pretrain_100b/audit/pins.json`）。

## 3. 阶段 D：逐级 SFT

起点固定 `ckpt_v42s2.pt`（step 54051，627 键 / 24 Engram 键）。每级一个独立 checkpoint 与一套独立评测记录。

| 级 | 数据 | 训练 | 出门条件 |
|---|---|---|---|
| D1 | C1 | 8 卡，micro 4 × accum 8，2–4 epoch，peak 2e-5 余弦退火 | E2 三门（stopped/format/quality）通过；E1 不退化 |
| D2 | +C2 | 同上，混采 | E3 功能门（≥18/20 一步算术） |
| D3 | +C3 | 同上 | E4 功能门（≥14/20 短链） |
| D4 | +C4 | 同上 | E5 门（语法 8/8、小样 ≥4/8） |
| D5 | 混合巩固 | 低 lr 短训 | E1–E5 全部不退化 |

**每级失败就停在那一级**，回到数据而不是加步数。

## 4. 时间估计（基于本机实测，不是外推）

| 环节 | 实测依据 | 估计 |
|---|---|---|
| SFT 80 步（8 卡，micro4×accum8，63 GiB 峰值） | v4/v6 各 80 步约 60 min | ~1 h/级 |
| E1（16 题 × 64 token） | pod 实测 16.5 s 解码 | ~3 min |
| E2（10 题 × 256 token） | pod 实测约 10 min | ~10 min |
| E6 smoke（GSM8K 20 题，单卡） | pod 实测 169 s + 合并 | ~3 min |
| C1 数据构建 + 自检 | 新任务 | ~2–3 h |
| C2/C3 数据 | E3/E4 生成器已在，扩量 + 复算 | ~2–3 h（与 C1 并行） |

**结论**：C1 数据过门后 **1.5 小时内**可以跑出 D1 的 checkpoint，随后 ~15 分钟拿到 E2/E1 的判定。C2/C3 与 C1 并行，所以 D2/D3 不必等。

## 5. 现在开始的动作

1. 派 `c1`：建 C1 格式与停止集（关键路径）。
2. 派 `c2`：C2 一步算术 + C3 短链应用题扩量（与 C1 并行）。
3. 主控在 C1 过门后按 §3 的 D1 行启动训练；启动前跑 `runs/claim_pattern_selftest.sh` 与 vocab 断言。

## 6. 边界（不变）

不继续 v6；不修改 `train.py`/`model.py`/`sft_math.py`/`data/mix_*.json`（D 级的训练只用 CLI 参数）；不启动 100B 长跑；不进 RL；原始语料不进 git。
