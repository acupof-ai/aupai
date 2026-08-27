# aupai

200M 中文推理模型：**KDA（Kimi Delta Attention，线性衰减循环）+ Gated MLA（潜在 KV 压缩 + 滑动窗口）** 混合架构，可选 Attention Residuals。12 层 = `(3 KDA + 1 MLA) × 3`，d=1024，词表 32,772，FP8 训练。

架构可视化：<https://acupof-ai.github.io/aupai/>（可拖拽算参数/训练开销/KDA-vs-MLA 显存）。

## 特性

- **推理省显存**：KDA 是 O(1) 状态的线性循环（每层 262 KiB，与上下文长度无关），MLA 只缓存 256 维潜在（窗口封顶 512 KiB/层）。同规模全注意力 @ctx4096 是 62×。
- **训练快**：FP8（torchao e4m3_tensorwise）+ Muon + WSD + Liger FLCE，batch 32 无 grad_ckpt，8×H20 上 ~90K tok/s/gpu、MFU 38%。
- **数据干净**：语料构建自带 holdout 过滤（整文档 + QA body + 逐行，防 eval 污染）、跨域去重、math 近重去重。

## aupai CLI — 训练全流程中控

一个 Rust 二进制（`cli/`，薄调度层，shell out 到已有 python/bash 工具），mac + linux pod 通用。

```bash
cd cli && cargo build --release           # 构建；二进制在 cli/target/release/aupai
aupai list                                # 所有命令一览
aupai status                              # 系统总览：数据就绪度 / ckpt / 流水线 / 在跑的训练
```

常用命令（全部支持 `--dry-run` 预演）：

```bash
aupai data                                # 各域 token 分布 vs mix 目标
aupai mix                                 # 校验 data/mix.json 调度
aupai pretokenize                         # 分词进 token 缓存（多核）
aupai train --name k5 --track             # 预训练（默认 = 已验证最优配方 + trackio 记录）
aupai eval <ckpt>                         # math-hard 评测（metric of record）
aupai sft <name> <ckpt> <sft.pt>          # SFT
aupai rl --resume <ckpt>                  # RLVR / GSPO
aupai ckpt list|best|clean                # checkpoint 管理
aupai dashboard [name]                    # trackio 本地曲线面板
aupai arch                                # 打开架构页
```

**流水线一行到底**（数据就绪后）：

```bash
aupai pipeline --name k5 --track          # tokenizer→pretokenize→data→pretrain→eval，带阶段状态
aupai pipeline --resume k5                # 从上次失败/中断的阶段续跑
aupai pipeline --status k5                # 看每阶段状态 / 产物
```

> 注：`pipeline` 假设原始语料 jsonl 已在 `data/`。数据的**下载/生成**（`datagen/fetch_data.py`、`scripts/fetch_*.py`、`datagen/gen_*.py`、`mathbank/`）尚未全部纳入 CLI —— 接入 `fetch`/`corpus` 阶段后即可从零一行训练。见下。

## 数据流水线

```bash
bash scripts/build_domains.sh             # 从 data/*.jsonl 原始源构建 data/corpus/<domain>/
                                          # （清洗/去重/holdout 过滤/跨域排除；小域并行，math 两段近重去重）
python scripts/data_overview.py           # 各域 token 数 + 占比
python scripts/check_mix.py               # 干预算：主/anneal 阶段行数、epoch cap、步数
```

`data/mix.json` = 各域权重 / epoch cap / anneal 权重。存在时 train.py 按调度消费（主阶段→最后 `anneal_frac` 用 anneal 权重），`epochs` 强制为 1。

## 直接调用（不经 CLI）

```bash
torchrun --nproc_per_node=8 train.py --fp8 --attn_res --attn_res_blocks 4 --name k5   # 预训练
scripts/run_sft.sh <name> <resume_ckpt> <sft.pt>                                       # SFT
torchrun --nproc_per_node=8 algorithms/rlvr.py --resume <ckpt>                          # RLVR
scripts/eval_hard.sh <ckpt> [ngpu]                                                      # math-hard 评测
```

任何匹配 `Cfg.<flag>` 的 `--flag` 覆盖对应默认（`python train.py --help`）。

## 提交前

CI（`.github/workflows/ci.yml`）跑 ruff E9/F、py_compile、`scripts/test_arch_compat.py`、`scripts/eqcheck.py`、`scripts/holdout.py`。触碰模型/优化器/ckpt 路径时扩展 test_arch_compat。`ruff format && ruff check`（行宽 110）。

## 加速细节

- fla chunk_kda 内核（KDA） + FlashAttention（MLA 滑动窗口，SDPA 回退）
- FP8 前向/反向（e4m3_tensorwise，compile 安全）+ Liger FLCE
- Muon（2D 矩阵，Newton-Schulz 正交化）+ AdamW（embedding/1D）+ torch.compile
- int32 token IDs + NVMe 缓存 + 异步 H2D
