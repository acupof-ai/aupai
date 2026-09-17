# V4.1-Flash 忠实复刻设计（v41f，小规模可训练）

**状态：** P0（attention/compressor/indexer/engram/Hyper-Connections/MoE/rope/window/…）与 P1
（HC Block、整网 model、ckpt、loss、train smoke）已合入 main（`tests/v41f/p0_*`、`p1_*` 全绿）；
P3 DSpark draft-block 在 #454（`0e-v41f-mtp`，未合）。本文件是 v41f 的契约与真值来源。
**真值来源：** `deepseek-ai/DeepSeek-V4.1-Flash` 仓库 `inference/`（`config.json`、`model.py` 1309 行、`kernel.py`、`engram.py`，2026-09-16 拉取核对）
**范围裁定（fb，2026-09-16）：** 小规模、8×H20 可训练可评测；机制与官方参考 1:1，尺寸等比缩小。纳入核心四件套 + Engram + DSpark；**不做视觉 ViT**（与代码目标无关）。
**非目标：** 不加载官方权重、不做 TP 推理服务、不复制 tilelang fp4 推理 kernel。

---

## 1. 为什么是新模型而不是改 flag

现 v41_r3 复现的是注意力骨架（8:1 压缩条目 + 128 滑窗 + 部分 RoPE + 48 专家 MoE），与真实
V4.1-Flash 有七处结构差异，每一处都改权重语义、无法靠开关在旧 checkpoint 上叠加：

| # | 机制 | 官方参考 | v41_r3 现状 |
|---|---|---|---|
| 1 | 残差 | Hyper-Connections，`hc_mult=4` 路并行残差流，Sinkhorn 双随机 comb（20 轮） | 单路 `x = x + f(norm(x))` |
| 2 | Q / 输出 | Q 低秩压缩 `q_lora_rank`；输出**分组低秩** `o_groups=8`，块对角 einsum | 全秩 `qg` / `o` |
| 3 | KV | **MQA**：单层 `wkv: dim→head_dim`，所有头共享一条 K,V | 每头独立 K/V |
| 4 | 压缩 | softmax 门控池化（独立 `wgate`，fp32）；仅 `kv_source_layers` 产出压缩 KV、`index_source_layers` 跑索引器，**跨层共享**；两级索引（候选块粗筛→topk） | 每头线性压缩；每层各产各选；单级硬 topk + STE |
| 5 | 压缩位置 | 逐层 `compress_ratios`，压缩 KV 用独立 `compress_rope_theta` + YaRN | 统一 m=8，无 YaRN |
| 6 | 注意力标量 | 每头可学 `attn_sink`（始终可见的空 KV）；KV fp8 | 无 sink |
| 7 | MoE | `sqrt(softplus)` 路由、selection-only bias、`route_scale`、top6、SwiGLU `clamp=10` | softmax、K3 SiTU-GLU、top3、平衡 bias + aux loss |

另有 Engram（n-gram 检索）与 DSpark（MTP 草稿头）两套子系统 r3 完全没有。

结论：新建并行模型族 **`v41f`**，保留 `model.py` 现有 v41 路径不动以维持 r3 可复现；新模块放
`model_v41f.py`，只共享 RMSNorm、分词、mix、fp8 Linear、训练循环这些与架构无关的部件。词表沿用
现有 32,768 gate vocab（`vocab_id` 不变），但权重全新、开新 ckpt 命名 `ckpt_v41f_*`。

## 2. 忠实映射的机制规格

下列每一条都给出官方字段与参考实现位置；实现时以参考代码为准，不以本文转述为准。

### 2.1 Hyper-Connections（`Block`，参考 model.py:907-994）

- 残差流形状 `[B,T,hc_mult,dim]`，embed 后扩维，`pre_mix` 初始 one-hot 第 0 路。
- 每个子层（attn / ffn）各一套 `hc_fn [mix_hc, hc_dim]` + `base [mix_hc]` + `scale[3]`，
  `mix_hc=(2+hc_mult)*hc_mult`，fp32 参数。
- 一次归一化投影产出三组系数：`pre`（sigmoid+eps，把多路压成一路子层输入）、`post`
  （2*sigmoid，把子层输出扩回路）、`comb`（行 softmax+eps 后 Sinkhorn 20 轮列归一，双随机）。
- 子层用**上一个**子层产出的 `pre`（attn 用 ffn 传来的，ffn 用本层 attn 的），`Block.forward`
  的参数顺序就是契约。
- 训练实现：Sinkhorn 用纯 PyTorch（`hc_split_sinkhorn` 的 20 轮行列归一），不接 tilelang
  kernel；序列维 `[B*T]` 向量化，每 block 仅两次小 Linear，热路径可接受。

### 2.2 MQA + 低秩 Q / 分组低秩输出（`Attention`，:613-789）

- `wq_a: dim→q_lora_rank`，`q_norm`，`wq_b: q_lora_rank→n_heads*head_dim`。
- 单条 `wkv: dim→head_dim`，`kv_norm`；K,V 全头共享（MQA），sparse_attn 里 `kv` 无 head 维。
- 输出：头按 `o_groups` 分组，`wo_a` 为块对角（每组 `heads/o_groups` 个 head × head_dim →
  o_lora_rank，用 einsum，**不是** Linear），`wo_b: o_groups*o_lora_rank→dim`。
- 每头一个 fp32 `attn_sink`，softmax 分母恒加 `exp(sink - max)`，见上游 `inference/kernel.py:383`。
- Q 与输出都对末尾 `rope_head_dim` 加 RoPE；**输出要逆向 RoPE**（`apply_rotary_emb(...,inverse=True)`），
  因为压缩 KV 是 V、K 共用一条且 RoPE 过，这与 r3「K/V 分离、V 不旋转」相反，必须照抄。
- 滑窗 KV 是 128 环形缓冲，prefill/decode 两种索引（`get_window_topk_idxs`）；训练只走 prefill
  路径，decode 环形逻辑服务后续生成，先实现 prefill，环形缓冲随生成器补齐。
- 一次 `sparse_attn` 调用拼接两类 KV：`[窗口原始 KV ; 选中的压缩条目]`，单一 softmax、共享 sink。

### 2.3 Compressor + 两级 Indexer + 跨层共享（:429-610）

- **Compressor（仅 kv_source 层拥有）**：`wkv: dim→head_dim` 与 `wgate: dim→head_dim`；ratio>1
  时在 fp32 内对连续 ratio 个 token 做 `softmax(wgate)` 加权池化，再过 RMSNorm。ratio=1 是无门控
  线性投影（bf16）。跨 prefill/decode 保留不完整组的 `kv_state/score_state`；训练 prefill 处理
  尾部 partial group。
- **Indexer（仅 index_source 层拥有）**：从 Q 低秩潜变量 `qr` 出 indexer-Q（`wq_bi`），每个压缩
  位置一条共享 index-key（拥有者由压缩 latent 经 `wk+k_norm` 产生）；打分
  `relu(q·k) * weights_proj(x)` 后跨 index-head 求和，topk 后按位置重排。
- **跨层共享**：kv_source 层把压缩 KV 与 index-key 写入共享槽，中间层只读不重算（参考用进程内
  全局 `shared_attn`；训练侧改为在 `HybridLM.forward` 显式传一个 runtime 容器，避免全局态在
  checkpointing/多 backward 下串味）。
- **两级索引**：`candidate_source_layer` 先按 8 条目一块取 `candidate_topk_blocks` 个块（钉住最新
  块），后续层只在候选块内 topk。**v41f 第一版关闭**（`candidate_source_layer=-1`）：训练序列
  4096、ratio 后压缩位置仅 ~512，粗筛无收益；机制与开关保留，升长上下文（≥32k）时开启。
- 压缩 latent 用独立 `compress_rope_theta` 旋转，位置取每组首 token `j*ratio`；短上下文 YaRN
  `original_seq_len=0` 即退化为普通 RoPE，第一版不引入 YaRN 外推参数。

### 2.4 MoE（`Gate/Expert/MoE`，:792-904）

- `score_func="sqrtsoftplus"`：`scores = sqrt(softplus(logits/T))`；`indices=(scores+bias).topk`，
  `weights=scores.gather(indices)` 用**无 bias** 分，归一化后乘 `route_scale=1.5`。
- bias 仅作用于选择；可沿用 r3 的 selection-only 持久 bias 与更新，但分数函数与 route_scale 换新，
  aux-loss 平衡项取消（官方无）。
- 专家为标准 SwiGLU：up 分支双侧 clamp、gate 分支仅上界 clamp（`swiglu_limit=10`），fp32 clamp
  后回精度。**不复用** `MoEFFN` 的 K3 SiTU。
- 1 个 shared expert 无条件经过，结果与 routed 求和。
- 分发训练侧用现有 `torch._grouped_mm`（参考推理是逐专家循环，训练不可接受），专家权重先 bf16，
  不引入 fp4。

### 2.5 Engram n-gram 检索（`engram.py` + `Engram`，:328-365）

- 在指定层把「以当前位置结尾的 2..max_ngram-gram」哈希到每 (n-gram, head) 独立的素数桶，查
  fp8 嵌入行，`wkv` 映射成每路一个 key + 一个共享 value，用归一化点积 +
  `sigmoid(signed_sqrt(dot))` 门控注入残差流。
- 哈希乘数由压缩词表大小派生、按层 RNG、取奇且防 int64 溢出；回溯遇序列起点或 DEAD（图像跨段）
  填 pad。
- 依赖 `build_compressed_token_map`：对 tokenizer 做 NFKC/NFD/去音符/小写/空白折叠后把 token 映到
  压缩 id。**我们用自有 32k BPE，不是官方 129k**，需在我们的 tokenizer 上重跑该映射并据实得到
  `engram_compressed_vocab_size`（不能照抄官方 99092），该值进入配置并被断言。
- v41f 缩小：见 §3 表；不做图像 DEAD 路径（纯文本），保留序列起点截断。

### 2.6 DSpark MTP（:1032-1156，分两级交付）

- 训练侧真正需要的是**多 token 预测损失**：在主干后接 `n_mtp_layers` 个草稿块，首层拼接若干
  目标层 attn 输入（`main_proj`），草稿块用自身滑窗注意力，共享 embed/head。
- 第一版（P3）：`n_mtp_layers=1`、`block_size=5`，实现标准接续 MTP 的多 token 训练 loss 与
  `target_layer_ids` 取 hidden；**不做** Markov 秩头、confidence 头和投机解码采样回路
  （`forward_spec`），这些纯为推理加速、对训练无贡献，列为生成器阶段可选项。
- 草稿层 MoE 配置复用主 MoE（`dspark_n_routed_experts=0` 走 fallback），不单独建 128 专家。

## 3. v41f-S 配置（8×H20 可训练）

沿用 d=1024 便于和 r3 直接对照；字段为配置取值。**参数量已用
`scripts/v41f_param_count.py` 按配置公式实测（见下，非估算）**；显存占用在真机 smoke 前仍是
估算，开工后以 `ckpt_info` 同口径实测覆盖。

| 字段 | v41f-S | 官方 Flash | 取值理由 |
|---|---|---|---|
| dim | 1024 | 5120 | 对齐 r3，单卡放得下 |
| n_layers | 12 | 40 | 同 r3 |
| n_heads / head_dim / rope_head_dim | 8 / 128 / 32 | 64 / 512 / 64 | head_dim 加大到 128 容纳 nope+rope |
| q_lora_rank | 256 | 1280 | dim/4 |
| o_groups / o_lora_rank | 8 / 128 | 8 / 1024 | 每组 1 头 |
| window_size | 128 | 128 | 不变 |
| compress_ratios | `[0,0,2,2,2,2,1,1,1,1,1,1]` | 见 config 43 项 | 前 2 层纯 SWA，4 层 2:1 强压缩，后 6 层 1:1 |
| kv_source_layers | `[2]` | `[2,8,14,20]` | 强压缩段单源，跨层共享给 3,4,5 |
| index_source_layers | `[2,4,8]` | 8 层 | 源 + 后段各一 |
| candidate_source_layer | -1（关两级） | 20 | 短上下文无收益 |
| index_n_heads / head_dim / topk | 4 / 64 / 64 | 32 / 128 / 512 | 参考 small 默认档 |
| hc_mult / sinkhorn_iters | 2 / 20 | 4 / 20 | **hc_mult=2 是主要缩放旋钮**：4 路在 d1024 每层 +33M 稠密投影，2 路约 16.7M |
| MoE routed/top/shared/inter | 48 / 6 / 1 / 448 | 384 / 6 / 1 / 2304 | 7*448≈3136≈dense 3072，active 与 r3 同阶 |
| score_func / route_scale / clamp | sqrtsoftplus / 1.5 / 10 | 同 | 照抄 |
| engram_layer_ids / ngram / n_heads / head_dim | `[1]` / 4 / 4 / 128 | `[1,14]` / 4 / 8 / 256 | 单层小表 |
| engram_vocab_size / 每表行数 | 65536 / 实测算素数桶 ≈0.8M | 16M / ≈384M | 压缩词表据我们 tokenizer 实测 |
| n_mtp_layers / block / markov | 1 / 5 / 暂不做 | 3 / 5 / 256 | P3 仅训练 loss |
| vocab | 32768（现有） | 129280 | 沿用 gate vocab |

**实测规模（`python scripts/v41f_param_count.py`，2026-09-17 核对）：** total = 904,583,784
= **0.9046 B**，active/token = 210,950,760 = **210.95 M**（active fraction 23.32%）。分项：
backbone total 832.75M / active 139.12M；MoE 跨层 total 809.83M / active 116.20M（MoE 是
total 大头，但每 token 仅激活 6 路由专家 + 1 共享，active 占比低）；engram dense 投影 4.72M，
engram 表当前 rows=0（待 tokenizer 素数桶，计 0）；embedding 与 untied lm_head 各 33.55M，
final norm 1,024。这与早期 1.2–1.4B / 0.5–0.55B-active 的估算不同——估算高估了 MoE/HC 的
active，以脚本实测为准（改动任一配置字段后重跑该脚本覆盖本数）。显存仍为估算：H20 单卡
fp8 权重 + bf16 grad + fp32 master/Adam 态 ≈16–20GB，B4/seq4096 + grad checkpoint 激活 30–50GB，
8 卡 DDP 可行；若峰值越界，先降 B 到 2/accum 翻倍，不
动结构。实测后回填本节。

## 4. Kernel 与精度策略

- 官方 `kernel.py` 是 **tilelang 推理 kernel**（fp8 GEMM、fp4 专家、MXFP e8m0 标度、手写
  sparse_attn、Sinkhorn），只前向、依赖 tilelang0.1.8/SM100 调优，**不移植用于训练**。
- 训练路径：bf16 注意力（窗口复用现有 `csa2_window_flash`/flash-attn varlen，压缩条目走 gather +
  拼接 softmax，规模小）；MoE `torch._grouped_mm`；HC/索引器/门控 eager PyTorch；dense 投影可选
  挂现有 Float8Linear，但第一版全 bf16 先保正确，fp8 作为通过数值对拍后的优化项。
- 数值对拍优先于性能：任何 kernel 替换先在固定输入上与 bf16 参考比到容差内（r3 已有
  csa2 split-LSE 对拍先例）。

## 5. 分阶段交付（每阶段一个 CPU 单测，能独立合并）

1. **P0 骨架与对拍夹具**：`model_v41f.py` + dataclass 配置；从官方 `inference/model.py` 的
   `ModelArgs` small 默认（dim1024/5层/8专家）移植一个**同形状**模型；固定种子、喂同形状随机
   输入，逐模块前向 `allclose`（RMSNorm、RoPE、Compressor、Indexer、sparse_attn+ sink、HC、
   sqrtsoftplus Gate、Engram 哈希）。这是「忠实」的硬证据，不是跑通就算。
2. **P1 核心四件套**：HC Block → MQA+低秩 Q/分组输出 → Compressor/Indexer+跨层共享 → sqrtsoftplus
   MoE；组成 v41f-S，过 `test_arch_compat` 风格的旧 ckpt 不受影响检查，CPU 上一次前向+反向无 NaN。
3. **P2 Engram**：在自有 tokenizer 上构建压缩 token map（断言压缩词表大小）、素数桶、哈希态、
   注入门控；纯文本无 DEAD；对拍 Engram 输出与门控。
4. **P3 DSpark-MTP**：1 草稿层 + block5 + 多 token loss；`target_layer_ids` 取 hidden；先不做
   Markov/confidence/spec 采样。
5. **P4 训练 smoke 与 gate**：H20 小步数 smoke（显存/吞吐/无 NaN，沿用 v41 smoke 口径），再决定
   是否按 gate mix 起正式 run；生成器补 decode 环形缓冲后评 HumanEval/MBPP（沿用 E0 n=10
   CLEAN 协议）。

每阶段产物只走新文件，不改 v41 路径；CI 加 v41f 独立测试，旧 `test_arch_compat` 必须保持全绿。

## 6. 风险与待裁决

- **HC 是 active 计算与总参数的双重大头。** hc_mult 4→2 是对官方的唯一实质性缩放；若实测 active
  超预算，下一步降 o_lora/head_dim，不先砍机制。需要在 P1 后用实测参数决定 S 是否再缩。
- **MQA + 输出逆向 RoPE** 与 r3 的 head 语义不兼容，确认开新模型族、不继承 r3 权重（仅复用词表）。
- **Engram 压缩词表**必须据我们 tokenizer 实测，哈希乘数依赖该值，错配会静默整体重哈希（参考有
  断言，照搬）。
- **DSpark 训练 loss 的权重与采样**官方参考未给（仓库只实现前向），多 token loss 的加权需自定并
  prereg，不能假装来自官方。
- **两级索引/YaRN/fp4/视觉**显式排除在第一版，开关与字段保留，不在短上下文小模型上为空机制付
  实现成本。
- 是否起正式 30B run、用哪套数据 mix，待 P4 smoke 数据后另行裁决；本设计只交付可训练的忠实模型。
