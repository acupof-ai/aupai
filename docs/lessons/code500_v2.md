---
question: "code-500-v2 干净评测集：构造规则、污染审查、四个 checkpoint 的地板线"
status: recorded
source: "t51 构造 + census + eval runs 2026-08-31; generator datagen/gen_code_v2.py; census scripts/census_code_v2.py"
---

# code-500-v2: 干净代码能力评测集

## 为什么需要它

code-500（v1）的 40% pass@1 是**模板召回**而非能力：v3 SFT 打包了 carve source（code_python_zh.jsonl 3000 行）的 2413 个兄弟行，模型学会了 code-fence 格式，在同族题上默写通过。换一个 carve source 里没有的题族，v3/v4/v5 全部归零。code-500-v2 的唯一目的：**在 SFT 和预训练语料都没见过的题族上，测真实代码能力**。

## 构造规则

- **30 个子族**，全部与 gen_code.py 的 30 个 carve-source 模板族不同（isqrt/comb/gcd/lcm/factorial/ceil_div/floor_div/counter_*/set_*/enum_*/sort_*/str_*/nested_*/flatten_*/running_*/pair_*）。
- **整数安全输出**：math_mod 族只用 isqrt/comb/gcd/lcm/factorial/ceil/floor，输出全整数；scorer 是精确行匹配，不碰浮点。
- **确定性契约**：集合/字典输出排序后 print；flatten 深度写进题面；nested_dict 按 key 排序遍历。无裸 set/dict repr。
- **确定性生成**：`rng = random.Random(2026)`，每族 17 题，shuffle 后截 500。重跑秒级复现。
- **生成器即出处**：`python3 datagen/gen_code_v2.py` → `data/eval/code_holdout_v2_500.jsonl`。gold round-trip 500/500。

## 污染审查

- **Census（SFT 源）**：`python3 scripts/census_code_v2.py` — 归一化包含检查（题面前 256 字符），扫 11 个 SFT 源（含 carve source code_python_zh.jsonl）。**0 hits，PASS**。
- **Verbatim（预训练语料）**：code_rp1t 精确+包含扫描 clean；cci3 847 路径扫描进行中。
- **Holdout 守卫**：`scripts/holdout.py` EVAL_FILES 已含 code_holdout_v2_500.jsonl，holdout_hashes.txt 2532 条（+500）。

## 地板线（greedy, n=500）

| Checkpoint | code-500-v1（污染） | code-500-v2（干净） | 解读 |
|---|---|---|---|
| base p324 | — | 0.0% | 基座零代码能力，v2 是有效能力度量 |
| v3（含 carve source） | 40.0% | **0.0%** | 40% 全是模板召回，closed |
| v4（family-clean） | 0.0% | **0.0%** | 无格式、无能力 |
| v5（English addon） | 0.0% | *pending* | — |

v3 在 v2 上 = 0% 是**预注册判决**：v3 ≈ 0 = 模板召回 closed；v3 ≥ 12.6pt = 迁移存在、SFT 判决重开。实测 0.0%，模板召回 closed。

## 复现命令

```bash
# 生成评测集（秒级，确定性）
python3 datagen/gen_code_v2.py

# SFT 源 census
python3 scripts/census_code_v2.py

# 跑评测（GPU 7 lane）
CUDA_VISIBLE_DEVICES=7 HOLDOUT=data/eval/code_holdout_v2_500.jsonl TAG=v2 \
  bash eval/eval_code.sh <ckpt.pt> 1

# 重打分验证（需 root，pod 上跑）
python3 scripts/rescore_code.py data/eval/preds_code_v2_<ckpt>.jsonl
```
