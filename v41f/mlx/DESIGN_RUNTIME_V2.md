# MLX Runtime V2 设计：CED 跨层复用、Engram SSD、连续批处理

## 1. 结论

V2 一次实现完整推理链。核心原则如下：

1. 一次前向只维护一个 CED（Cross-layer Expert Decoder）共享状态。
2. KV source 层发布压缩 KV 和索引键。
3. Index source 层更新 top-k。
4. 非 index 层复用当前最新的压缩 KV、索引键和 top-k。
5. 每层只保留自己的 128 token 原始滑动窗口 KV。
6. Engram 大表放 SSD，按行读取，不整表进内存。
7. 单请求立即执行，不等待凑批。
8. 多请求进入真连续批处理，不做顺序循环。
9. 先过 PyTorch native gold，再做性能优化。

旧 V1 的逐层独立压缩缓存、三阶段状态和全前缀重跑均废弃。

## 2. 已核验事实

| 项目 | 数值 |
|---|---:|
| 总参数量 | 3,910,901,776（3.91B） |
| 路由专家参数量 | 3,019,898,880（3.02B） |
| 每 token 激活专家数 | 8 / 64 |
| 激活路由参数量 | 377,487,360（0.377B） |
| 矩阵乘核心参数量，不含 embedding 和 head | 594,273,552（0.594B） |
| 单行 embedding 口径权重读取量 | 约 627,829,008（0.628B） |
| Engram 每 token 读取行数 | 72 |
| Engram 每 token 读取元素数 | 9,216 |
| Transformer 层数 | 24 |
| 滑动窗口大小 | 128 token |
| 注意力头数 × head_dim | 16 × 256 |
| CED KV source 层 | 2、8、12 |
| CED index source 层 | 2、8、12、16、20 |
| Engram 层 | 1、5、9、13、17、21 |

压缩位置使用组首 token 的绝对位置。ratio=2 时，第 j 个 latent 的位置为 j×2。

## 3. 目录与模块

V2 目录：`v41f/mlx/runtime_v2/`

| 文件 | 职责 |
|---|---|
| `config.py` | 从 checkpoint manifest 读取配置，做字段校验。 |
| `weights.py` | 加载常驻权重，构造专家存储和 embedding/head。 |
| `expert_store.py` | 路由专家存储、按需读取、LRU 和预取。 |
| `ced_state.py` | 每请求 CED 状态、窗口环形缓存、压缩 KV 和 top-k。 |
| `attention.py` | 窗口注意力、压缩 KV 注意力、sink 和单 softmax。 |
| `compressor.py` | MLX 压缩器，支持 prefill、chunk 和 decode。 |
| `indexer.py` | MLX 索引器，支持多头评分、可见性和 top-k。 |
| `moe.py` | 路由、共享专家、分组专家计算和 SwiGLU。 |
| `engram.py` | Engram SSD 按行查询、LRU 和 decode 滚动 n-gram。 |
| `prefill.py` | 分块 prefill，写入 CED 状态。 |
| `decode.py` | 单步 decode，输出 logits 和 next token。 |
| `batcher.py` | 连续批处理调度、准入、释放和状态拼接。 |
| `server.py` | HTTP/SSE 服务和 OpenAI 兼容接口。 |
| `webapp.py` | 网页静态内容。 |
| `benchmark.py` | TTFT、单流吞吐、聚合吞吐和内存基准。 |

测试目录：`v41f/mlx/tests/runtime_v2/`

## 4. 权重策略

### 4.1 常驻权重

以下权重常驻统一内存：

- attention 非专家权重；
- shared expert；
- normalization；
- hyper-connection；
- compressor 和 indexer 权重；
- LM head；
- 路由 gate。

这些权重合计小于 1GB。

### 4.2 路由专家

M4 Pro 内存为 48GB。全部路由专家 BF16 约 6.0GB。V2 使用两级策略：

1. 默认建立 mmap 专家存储。
2. 热专家进入 LRU。
3. LRU 容量默认 12GB。
4. 若启动参数指定 `--experts-resident`，全部专家常驻内存。
5. 计算时只收集 gate 选中的 8 个专家。

系统永远不做 64 专家全量 einsum。

专家矩阵按层切分为三个堆叠张量：

- `w1`：`[64, inter, dim]`；
- `w3`：`[64, inter, dim]`；
- `w2`：`[dim, inter]`。

若源文件为堆叠格式，expert store 直接按 expert 行范围 mmap。系统不强制复制小文件。

### 4.3 Engram SSD

Engram 表不整表加载。SSD 文件按行 mmap。

每次查询执行以下步骤：

1. 维护 decode 滚动 n-gram；
2. 得到 1-gram 至 4-gram 的 hash；
3. 按 6 个 Engram 层和 4 个 n-gram head 取行；
4. 每行 128 维；
5. 进入行 LRU；
6. 按训练侧 Engram 公式注入隐藏状态。

空槽使用 pad id。LRU 默认 8,192 行。

## 5. CED 状态模型

每请求持有一个 `CEDState`。系统不使用进程全局状态。

### 5.1 每请求字段

| 字段 | 形状或说明 |
|---|---|
| `position` | 当前下一个 token 的位置。 |
| `window_kv[layer]` | `[128, 256]`，每层独立。 |
| `window_slot_age[layer]` | 环形槽位年龄，用于 valid mask。 |
| `comp_kv[source]` | 每个 KV source 的压缩 KV 列表。 |
| `index_k[source]` | 每个 KV source 的索引键列表。 |
| `partial_kv[source]` | ratio=2 的未完成组 KV。 |
| `partial_score[source]` | ratio=2 的未完成组 gate score。 |
| `topk_idxs` | 当前最新 top-k，形状 `[topk]`。 |
| `topk_source` | 当前 top-k 对应的 source。 |
| `engram_state` | 滚动 n-gram 状态。 |

### 5.2 状态推进

Layer 2：

1. 压缩当前输入；
2. 发布 source=2 的 comp_kv 和 index_k；
3. 运行 indexer；
4. 发布 topk。

Layer 8：

1. 压缩当前输入；
2. 发布 source=8 的 comp_kv 和 index_k；
3. 运行 indexer；
4. 替换 topk。

Layer 12：

1. 以 ratio=1 压缩当前输入；
2. 发布 source=12 的 comp_kv 和 index_k；
3. 运行 indexer；
4. 替换 topk。

Layer 16 和 20：

1. 复用 source=12 的 comp_kv 和 index_k；
2. 运行自己的 indexer；
3. 替换 topk。

其他层：

1. 复用当前最新 comp_kv、index_k 和 topk；
2. 只更新自己的 128 token 原始窗口 KV。

### 5.3 Decode 规则

每个 decode step、每层都执行以下动作：

1. 将当前 token 的 K 和 V 写入 `position % 128` 槽位；
2. 按环形顺序生成窗口索引；
3. 屏蔽无效槽位；
4. 若该层是 KV source，向 compressor 输入一个 token；
5. ratio=2 只在组完成时追加 latent；
6. ratio=1 每个 token 都追加 latent；
7. 若该层是 index source，重算当前 query 的 top-k；
8. 拼接窗口 KV 和压缩 KV；
9. 执行一个包含 sink 的 softmax。

旧 token 的历史注意力不重算。系统只需要当前 query 的 top-k。

## 6. Attention

### 6.1 输入

- q：`[T, n_heads, head_dim]`；
- window kv：`[T, 128, head_dim]`；
- compressed kv：ragged，按请求 padding；
- window indices：环形槽位；
- compressed indices：当前 top-k；
- attention sink：`[n_heads]`。

### 6.2 计算

单请求公式：

```text
scores = einsum(q, gathered_kv) * scale
scores = mask(scores)
logits = softmax(scores + sink)
out = einsum(logits, gathered_values)
```

窗口位置和压缩位置进入同一个 softmax。sink 始终参与分母。

### 6.3 批处理

连续批处理按最大 KV 长度 padding。每个请求持有独立 mask。

输出形状为 `[T, n_heads, head_dim]`。T 是本 step 中全部活跃 token 数。

## 7. MoE

### 7.1 路由

每个 token、每层执行以下动作：

1. gate 生成 64 个路由分；
2. 按配置使用 sqrtsoftplus；
3. 乘以 route_scale；
4. 取 top-8；
5. 归一化路由权重；
6. 收集专家 id。

### 7.2 分组计算

系统按 `(layer, expert_id)` 分组 token。

每组执行：

1. gather 输入；
2. `w1(x)`；
3. `w3(x)`；
4. SwiGLU，clamp=10；
5. `w2(hidden)`；
6. 按路由权重缩放；
7. scatter 回 token。

共享专家同时计算，并加入输出。

### 7.3 Kernel 选择

V2 按以下顺序选择：

1. MLX 量化 `gather_qmm`，用于 Q8 专家；
2. MLX grouped matmul，用于 BF16 常驻专家；
3. 按专家 einsum，用于正确性基线。

正确性基线先通过。性能 kernel 必须与基线对齐。

## 8. Prefill

### 8.1 分块

默认 chunk 大小为 256 token。原因：

- 控制 TTFT；
- 允许其他请求插入；
- 控制峰值内存；
- 保持 CED 状态连续。

### 8.2 Chunk 状态

ratio=2 时，尾部 partial group 不丢弃。系统保存：

- partial KV；
- partial gate score；
- group 内位置；
- group 首位置。

下一个 chunk 或 decode token 完成该组。

ratio=1 时，每个 token 都生成 latent。

### 8.3 多层语义

每个 chunk 在每一层按顺序执行。跨层共享状态只在同一个 chunk 内传播。chunk 结束后，持久状态写入 `CEDState`。

## 9. 连续批处理

### 9.1 请求状态

状态机：

```text
ADMITTED -> PREFILL -> DECODE -> DONE
                    -> PREEMPTED -> PREFILL/DECODE
                    -> FAILED
```

### 9.2 调度循环

调度循环固定执行：

1. 接收新请求；
2. 检查 KV 内存预算；
3. 立即启动首个 prefill chunk，不等待；
4. 拼接全部可运行 token；
5. 执行一次 prefill 或 decode；
6. 分发增量文本；
7. 移除完成请求；
8. 释放 KV 和 Engram 状态。

### 9.3 预算

默认预算：

- 最大并发请求：32；
- 单 step 最大活跃 token：64；
- 每请求最大上下文：4,096；
- 专家 LRU：12GB；
- Engram LRU：8,192 行。

### 9.4 优先级

- 新请求 prefill 优先，保证 TTFT；
- 已在 decode 的请求以轮询方式推进；
- 单请求 decode 不等待其他请求；
- 完成请求立即释放槽位。

顺序执行多个请求不属于连续批处理。该路径不允许作为性能结果。

## 10. Server 和网页

### 10.1 HTTP 接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 返回服务、权重和调度器状态。 |
| `/v1/chat/completions` | POST | OpenAI 兼容接口，支持 SSE。 |
| `/v1/engines` | GET | 返回当前模型信息。 |
| `/metrics` | GET | 返回 TTFT、吞吐和并发指标。 |
| `/` | GET | 返回聊天网页。 |

### 10.2 SSE 事件

事件顺序：

1. `role`；
2. `delta`；
3. 多次增量；
4. `done`；
5. `close`。

每个事件包含请求 id、位置和耗时。

### 10.3 网页

网页提供：

- 输入框；
- 停止按钮；
- 流式输出；
- token 速率；
- TTFT；
- 上下文长度；
- 并发请求状态；
- 错误提示。

网页不依赖外部 CDN。

## 11. PyTorch Native Gold

V2 不以旧 MLX full-recompute 路径作为黄金标准。

黄金测试文件：`v41f/mlx/tests/runtime_v2/test_native_gold.py`。

### 11.1 Gold 构造

1. 使用 `scripts/loader.load_checkpoint` 加载 checkpoint；
2. 使用 checkpoint 内配置构造 `V41FConfig`；
3. 建立 PyTorch 原生环形窗口；
4. 建立 source 级压缩 KV 和索引键；
5. 手动执行 native decode；
6. 输出每步 logits。

### 11.2 覆盖点

- prefill logits；
- layer 2、8、12 的 compressor；
- layer 2、8、12、16、20 的 indexer；
- 非 index 层 top-k 复用；
- 环形窗口；
- ratio=2 group 边界；
- ratio=1 latent；
- Engram n-gram；
- 多请求状态隔离。

## 12. 验收门

### 12.1 正确性

| 门 | 标准 |
|---|---|
| Prefill logits | MLX 与 PyTorch top-1 一致；logits max diff ≤ 0.02。 |
| Decode logits | 前 16 步 top-1 一致；logits max diff ≤ 0.03。 |
| CED top-k | source 和非 source 层 top-k 与 PyTorch 一致。 |
| 停止符 | 256 token 内按格式输出 `<|im_end|>`。 |
| 长生成 | 不出现连续复读；重复 n-gram 阈值不超过训练口径。 |
| 状态隔离 | 并发请求不共享 CED 指针和缓存。 |

### 12.2 性能

| 门 | 标准 |
|---|---:|
| 单流 decode | ≥ 200 token/s。 |
| 8 路聚合 | ≥ 6× 单流结果的 60%。 |
| 16 路聚合 | ≥ 10× 单流结果的 60%。 |
| 256 prompt TTFT | ≤ 0.5 秒。 |
| 1024 prompt TTFT | ≤ 1.5 秒。 |
| 峰值内存 | ≤ 24GB，experts-resident 模式 ≤ 18GB 权重内存。 |

性能测试使用 32 token warmup、256 token 计时、5 轮中位数。

### 12.3 失败处理

任一正确性门失败，不切换生产端口。

任一性能门失败，报告分层 profile：

- gate；
- attention；
- MoE；
- compressor；
- indexer；
- Engram SSD；
- scheduler；
- Python sync。

## 13. 实施顺序

| 里程碑 | 交付物 | 退出条件 |
|---|---|---|
| M0 | Native gold | PyTorch prefill/decode 可复现。 |
| M1 | Weight pack | 权重、专家存储和 Engram SSD 可读。 |
| M2 | MLX prefill | Prefill logits 通过。 |
| M3 | MLX decode | 前 16 步和长生成通过。 |
| M4 | Continuous batching | 8 路和 16 路通过。 |
| M5 | Server 和网页 | SSE、网页和指标可用。 |
| M6 | Benchmark | 单流和聚合性能达标。 |

M0 至 M3 使用本地 SFT v1 checkpoint 验证。最终 v2 SFT checkpoint 生成后，只替换权重包并回归全部门。

## 14. 生产切换

- 旧服务端口：8731。
- 实验服务端口：8733。
- 8733 通过全部门后，再替换默认服务。
- 不删除旧服务代码。
- 不 force push。
- 不推进 HF 导出。

## 15. 不允许重复的旧错误

1. 不逐层维护独立压缩 KV。
2. 不让非 index 层缺少 top-k。
3. 不每步重算完整前缀。
4. 不一次量化并计算全部 64 个专家。
5. 不把顺序请求伪装成连续批处理。
6. 不把旧 MLX full-recompute 作为最终黄金标准。
7. 不把 Engram 整表加载进算子。
8. 不在正确性门失败时报告性能成功。
