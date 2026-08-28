# aupai

200M 中文推理模型：**KDA（Kimi Delta Attention，线性衰减循环）+ Gated MLA（潜在 KV 压缩 + 滑动窗口）** 混合架构，可选 Attention Residuals。12 层 = `(3 KDA + 1 MLA) × 3`，d=1024，词表 32,772，FP8 训练。

架构可视化：<https://acupof-ai.github.io/aupai/>（可拖拽算参数/训练开销/KDA-vs-MLA 显存）。

## 特性

- **推理省显存**：KDA 是 O(1) 状态的线性循环（每层 262 KiB，与上下文长度无关），MLA 只缓存 256 维潜在（窗口封顶 512 KiB/层）。同规模全注意力 @ctx4096 是 62×。
- **训练快**：FP8（torchao e4m3_tensorwise）+ Muon + WSD + Liger FLCE，batch 32 无 grad_ckpt，8×H20 上 ~90K tok/s/gpu、MFU 38%。
- **数据干净**：语料构建自带 holdout 过滤（整文档 + QA body + 逐行，防 eval 污染）、跨域去重、math 近重去重。

## 常用命令

```bash
python scripts/build_tokenizer.py --force            # 训 data/tokenizer.json（分层采样，~5 分钟）
torchrun --nproc_per_node=7 train.py --fp8 --fone \
  --attn_res --attn_res_blocks 4 --warmup 150 --lr_scale 0.5 --name X --track   # 预训练
scripts/eval_hard.sh <ckpt> [ngpu]                   # math-hard 评测（metric of record）
scripts/run_sft.sh <name> <resume_ckpt> <sft.pt>     # SFT
torchrun --nproc_per_node=8 algorithms/rlvr.py --resume <ckpt>   # RLVR / GSPO
python scripts/exp.py start|done                     # 实验记录 → EXPERIMENTS.md
python scripts/data_overview.py                      # 各域 token 数 + 占比
python scripts/fone_probe.py                         # FoNE vs BPE 算术 A/B
```

远程执行见 AGENTS.md 的 Pod 一节——长任务必须 `setsid`，`nohup` 挡不住 crictl 的进程组回收。

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
