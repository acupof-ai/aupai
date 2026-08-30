# aupai

200M 中文推理模型。12 层混合架构 `(3 KDA + 1 MLA) × 3`，d=1024，词表 32,773，FP8 训练。

KDA 是线性衰减循环，每层状态 256 KiB，与上下文长度无关。MLA 只缓存 256 维潜在，窗口 1024，每层 512 KiB。ctx 4096 下整模型 KV 缓存 3.75 MiB，同规模全注意力是 192 MiB。

架构可视化 <https://acupof-ai.github.io/aupai/>，可拖拽算参数、训练开销、显存。

## 从零开始

```bash
uv sync
python scripts/test_arch_compat.py     # CPU 冒烟测试，不需要 GPU
python scripts/build_tokenizer.py --force
```

仓库带一份 4,992 篇、约 0.9M token 的样本语料在 `data/corpus/sample/`，配 `data/mix_sample.json`，够跑通流程但不够训出模型。真实语料由 `scripts/build_domains.sh` 构建到 `data/corpus/<domain>/`，配 `data/mix_scale_3.24b.json`（默认 mix；六个预算点见 `data/mix_scale_*.json`）。**mix 是唯一的数据路径**：扁平回退已于 2026-08-29 删除，样本和真实语料走同一套代码。

`data/tokenizer.json` 不入库。词表必须先建，train.py 缺它会直接报错——它曾经会静默训一个新的，那个词表缺 4 个 chat special 和 `[NUM]`，但 vocab_size 对得上，于是任何已有 checkpoint 配它都是乱码且不报错。

## 命令

```bash
# 词表，分层采样，约 5 分钟
python scripts/build_tokenizer.py --force

# 预训练
torchrun --nproc_per_node=7 train.py --fp8 --fone \
  --attn_res --attn_res_blocks 4 --warmup 150 --lr_scale 0.5 --name X --track

# SFT
scripts/run_sft.sh <name> <resume_ckpt> <sft.pt>

# RLVR / GSPO
torchrun --nproc_per_node=8 algorithms/rlvr.py --resume <ckpt>

# math-hard 评测，metric of record
scripts/eval_hard.sh <ckpt> [ngpu]

# 实验记录，写入 EXPERIMENTS.md
python scripts/exp.py start|done
```

任何匹配 `Cfg.<flag>` 的 `--flag` 覆盖对应默认值，见 `python train.py --help`。

远程执行见 AGENTS.md 的 Pod 一节。长任务用 `setsid`，`nohup` 挡不住 crictl 的进程组回收。

## 数据

```bash
bash scripts/build_domains.sh      # data/*.jsonl 原始源 -> data/corpus/<domain>/
python scripts/data_overview.py    # 各域 token 数与占比
python scripts/check_mix.py        # 干预算主阶段与 anneal 的行数、epoch cap、步数
```

`build_domains.sh` 做清洗、跨域去重、math 近重去重，以及 holdout 过滤防 eval 污染。

mix 给每个域三个值：权重、epoch cap、anneal 权重。train.py 按调度消费，先主阶段，最后 `anneal_frac` 换用 anneal 权重，`epochs` 强制为 1。mix 缺失是硬错误，不是回退。

## 数字表示

BPE 按词频切数字，1640 切成 `16|40`，进位规则无法跨数泛化。`--fone` 打开 Fourier Number Embedding：每个数收成一个 `[NUM]` token，值经傅里叶特征进 embedding，每位十路 argmax 解码。

```bash
python scripts/fone_probe.py                       # 同参数同步数的 FoNE / BPE A/B
python scripts/fone_digit_acc.py --ckpt X          # 读 checkpoint 测数位准确率
```

## 提交前

CI 跑 ruff E9/F、py_compile、`test_arch_compat.py`、`eqcheck.py`、`holdout.py`。触碰模型、优化器、checkpoint 路径时扩展 test_arch_compat。本地跑 `ruff format && ruff check`，行宽 110。

## 实现

- fla `chunk_kda` 内核走 KDA，FlashAttention 走 MLA 滑动窗口，SDPA 回退
- FP8 前向反向，torchao e4m3_tensorwise，Liger FLCE
- Muon 管 2D 矩阵，AdamW 管 embedding 与 1D，torch.compile
- int32 token id，NVMe 缓存，异步 H2D

7×H20 上 85K tok/s/gpu，MFU 36%。开 `--fone` 降到 73K，MFU 31%，代价来自 float64 数位提取和每步的数位交叉熵。
