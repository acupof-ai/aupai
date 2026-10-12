---
question: 本栈（H20 ×8/v42 线）在阶段 D SFT、评测生成和阶段 F 预训练上，哪些加速手段有实测依据、代价多少？
status: measured
source: runs/v42_mathfix_v6_1009.log、runs/ladder/ckpt_v42s2/{e1,e5}/summary.json、facts/efficiency.json、docs/lessons/throughput_survey.md、runs/experiments.jsonl#v42_gate_1001r
---

# 训练加速方案（2026-10-09）

## 0. 先量三个数（本机实测，不是外推）

| 场景 | 实测 | 来源 |
|---|---|---|
| SFT（world 8，micro 4 × accum 8，`grad_ckpt` 默认 **ON**） | 1.05M token/step，**43.4–45.0 s/step** → 23K tok/s 总计、**2.9K tok/s/gpu** | `runs/v42_mathfix_v6_1009.log`（80 步 ≈ 58 min） |
| 预训练 v42 线（world 16，MB4×ACC3） | 786K token/step，约 **5.9 s/step** → **8.3K tok/s/gpu** | `runs/experiments.jsonl#v42_gate_1001r` 的假设行（"beats the measured MB2 5.91 s/step"） |
| 同线 12 层 3.2B 生成（无 KV cache） | 4 题 × 512 token ≈ **141 s**；16 题 × 64 token ≈ 16.5 s | `runs/ladder/ckpt_v42s2/{e5,e1}/summary.json` |

**第一条结论**：SFT 每 token 比预训练慢 **2.9 倍**。这不是模型问题——同一份权重、同样的 fp8/compile/liger 都在——差别在 `sft_math.py` 的 `--grad_ckpt` **默认 True**（激活重算），而预训练线显式传 `--no-grad_ckpt`。

## 1. 阶段 D SFT（先做，直接决定今晚能跑几级）

**实测矩阵（2026-10-09，`runs/sft_probe_*.log`，v42 engram 基座 + `data/sft/math_v6/math_sft_v4.pt`，seq 4096）**

| 臂 | micro × accum | grad_ckpt | 结果 |
|---|---|---|---|
| 基线（v6 实跑） | 4 × 8 | ON | 43.4–45.0 s/step，峰值 68.42 GiB（engbase） |
| A1 | 4 × 8 | **OFF** | **OOM**：rank1 `Tried to allocate 256 MiB, 121.88 MiB free`，进程已占 65.25 GiB |
| A2 | 8 × 4 | **OFF** | **OOM**：同一形状，79.88 MiB free |
| A3 | 2 × 16 | OFF | 在跑（7 卡，card 1 被 w2 的评测占着） |

**结论（实测，不是外推）**：在 seq 4096 上，**显存是硬约束，不是 grad_ckpt**。去掉激活重算后 micro 4 与 micro 8 都放不下这张 96 GiB 的卡，所以"A1 关掉 grad_ckpt 换 1.33×"这条路在本形状上**不可用**（`eff.batch_ceiling` 的 25% 墙钟是 200M 的测量，不能直接搬过来）。

那么可动的只剩三样：

| # | 改动 | 状态 |
|---|---|---|
| A3 | micro 2 + accum 16 + `--no-grad_ckpt` | 在测；micro 变小通常降利用率，要看 tok/s 才能定 |
| A4 | **缩短 seq**（C1 的答案 median 3 token，4096 的包里大部分是被掩码的上下文） | 需要 `sft_math.py` 加一个 `--seq` 解析项（Cfg 有 seq，但 parser 没有 → CLI 设不了），属共享文件改动 |
| A5 | 4 epoch → 1–2 epoch | 与显存无关，纯省墙钟；交接文档 P1 已支持 |

## 2. 评测/生成（阶段 B 已暴露的成本）

| # | 改动 | 预期收益 | 依据 | 代价 |
|---|---|---|---|---|
| B1 | 实现 KV cache | O(T²) → O(T)，E6 全量从"数小时"到"数十分钟" | `scripts/export_hf.py` 的卡面明确写 "no KV cache (generate is O(n²))"；E5 的 4×512 用了 141 s | 工程量最大的一项：要动 `train.generate_batch` 的解码路径，属共享文件 |
| B2 | 8 卡分片 | 线性 | E6 已按 `--shards/--gpus` 分片，smoke 实测 169 s/20 题单卡 | 已做 |
| B3 | `max_new` 按级收口 | 线性 | E5 512 → 128 后同一批题从十几分钟降到两分钟量级 | 少收长答案；E6 的生成类必须留足 |

## 3. 阶段 F 预训练（100B 累计，还没启动）

- 现状口径：7K tok/s/gpu × 8 = **56K tok/s** → 一个 10B 分段约 **49 小时**；累计 100B 约 **21 天**（不含每 10B 一次的评测门控）。
- 用 H20B 组 world 16：**有先例**——`v42_gate_1001r` 就是 world 16、MB4×ACC3 跑完 30B 预算的。用户 2026-10-09 已授权"必要时可用 H20B"（裁定 R6）。翻倍卡数在 MB4 下预期接近减半墙钟，但要按 `v42_gate_1001r` 的失败判据验收：**s/step ≥ 9 视为 interconnect-bound，退回 world 8**。
- 每 10B 的评测成本要计入：E6 全量（GSM8K 1319 + MATH-500 500 + HumanEval 164）按 B1 之前的口径是数小时，按 B1 之后是数十分钟。

## 4. 验收方式（每个改动都要能驳回）

1. 固定 **30–50 步窗口**，量 `tok/s/gpu`、`s/step`、峰值显存；写进 `facts/efficiency.json`（带 config）。
2. **有效 batch 不变**才可与现状比；变了就另开一行。
3. NaN 与 loss 曲线是硬门：探针 0 NaN 且固定窗口 loss 与基线一致，才算"只是更快"。
4. 每个改动一个 commit 单元，跑同一段固定数据。

## 5. 不做

- **micro-batch 32 × accum 1**：已实测 OOM（93.8/95.2 GB，rank 3/6 先死）。要加有效 batch 用 accum。
- **world 16 但 s/step ≥ 9**：那是 interconnect-bound，加卡不加吞吐。
- **用 4 epoch 凑训练量**：交接文档 P1 已记为问题，不是加速手段。

## 7. 算子层（2026-10-09 追加）

**先说结论：SFT 路径上便宜的算子开关已经用满了。** base cfg 的 `v42_impl` 是 `attn_impl=fused, rope_impl=real, moe_stacked=1, hc_impl=liger, norm_impl=liger, attn_logit_softcap=50`——而 `train.py` 的默认是 `chunked/complex/0/torch/torch`。也就是说：融合注意力、Liger 的 HC/norm、堆叠 MoE、fp8、compile 全都在跑，没有"换个开关就快"的余量。

**想再快，只有两条路：**

| # | 方向 | 依据 | 代价 |
|---|---|---|---|
| O1 | **先剖析**：给 `sft_math.py` 加 `--profile_step_every N`（照抄 `train.py:1733` 的 cuda-event 分解），或写一个独立脚本 import `build_sft_model()` 跑一步进 `torch.profiler` | 没有分解就只能猜。200M 上的分解是 `eff.step_time_by_operator_complete`：fp8 GEMM 25.3% / LM head 22.1% / AttnRes 13.5% | 中（`sft_math.py` 是共享文件，要广播 + PR） |
| O2 | **CUDA graphs / `reduce-overhead`**：200M 实测稳态 idle 的 72% 来自 `rms_norm → flash-attention` 接缝的 CPU 启动开销（7 个实例、54.9 ms/step，其中编译/guard 69.3 ms/step） | `docs/lessons/throughput_survey.md` | 中；与 `torch.compile` 的 mode 相关，要验数值一致性 |
| O3 | **LM head**：`eff.lm_head_is_compute_bound`（190 ms/step）与 `eff.vocab_alignment_2x_lm_head`（对齐后腰斩） | facts | 低：vocab 32768 已经 8/16 对齐（`gemm_dims_aligned` PASS），这条可能已经吃完 |
| O4 | 形状：缩短 seq（C1 答案 median 3 token，4096 的包里大部分是被掩码上下文） | 本文件 §1 | 需要 `sft_math.py` 加 `--seq` 解析项 |

**O1 是下一步**：不剖析就动 kernel 是在猜。SFT 的 45 s/step 里，算子的 CUDA 时间、CPU 启动时间、编译/guard 时间各占多少，现在**一个都没有测**。

## 8. 实测修正（2026-10-09 下午，固定形状 A/B）

`runs/sft_shape_{gc_on,gc_off}.log`：**同一形状**（micro 2 × accum 16 × 7 卡，seq 4096，`math_v6` 包，30 步），唯一变量是激活重算。

| 臂 | step 10 窗口 | step 20 窗口 | step 30 窗口 | 稳态 |
|---|---:|---:|---:|---:|
| grad_ckpt **ON** | 506 s/10 步 | 495 s/10 步 | 347 s/10 步 | **34.7 s/step** |
| grad_ckpt **OFF** | 255 s/10 步 | 145 s/10 步 | **104 s/10 步** | **10.4 s/step** |

**结论：在这个 3.9B MoE 上，激活重算的代价是 3.3×**（不是我先前引用的 200M 上的 25%）。换算：
- 新形状：224 packs/step × 4097 = 918K token / 10.4 s = **12.6K tok/s/gpu**（7 卡）
- 旧形状（v6 实跑）：256 packs × 4097 = 1.05M token / 44 s = 2.9K tok/s/gpu（8 卡）
- **每 GPU 约 4.3×**。

代价与边界：micro 4 与 micro 8 去掉重算都 OOM（实测 121.88 MiB / 79.88 MiB 剩余），所以形状必须同时把 micro 降到 2。峰值 48.90 GiB（OFF）对 95 GiB 的卡有一倍余量。
