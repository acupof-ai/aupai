# v42 mlx-lm 推理后端冻结设计

## 1. 结论

生产后端只保留一条链路：`mlx-lm 0.32.0 + v42 自定义模型`。

旧 MLX 文件只作黄金对照。新后端不调用旧生成器、旧 HTTP 服务或旧增量实现。实施完成后，生产入口只有 `python -m v41f.mlx.runtime.server`。

## 2. 目标与口径

| 指标 | 硬门 | 目标 |
|---|---:|---:|
| PyTorch 首 token argmax | 必须一致 | 一致 |
| 64-token argmax 一致率 | ≥95% | 100% |
| 256-token 生成 | 连贯、无异常复读 | 正常 `im_end` |
| TTFT，短 prompt | ≤2 s | ≤1 s |
| 单流 decode | ≥20 token/s | 40–80 token/s |
| 批量总吞吐 | ≥100 token/s | ≥200 token/s |
| RSS | ≤24 GB | ≤16 GB |
| HTTP | SSE 正常结束 | OpenAI 兼容 |

单流速度和批量总吞吐分开报告。不得把两者混写为一个 token/s。

## 3. 不可变模型语义

以下计算必须保持 PyTorch 语义：

- 24 层，维度 1024。
- HyperConnection 倍数 4。
- 64 个路由专家，每 token 激活 8 个。
- 路由函数为 `sqrtsoftplus`。
- 路由使用偏置、归一化和 `route_scale=1.5`。
- 窗口注意力大小为 128。
- 压缩源层为 2、8、12。
- 索引源层为 2、8、12、16、20。
- Engram 层为 1、5、9、13、17、21。
- Engram 表可以驻留 SSD。

## 4. 生产架构

```text
Web / OpenAI API
        |
RuntimeServer
        |
mlx-lm BatchGenerator / generate_step
        |
V42RuntimeModel(inputs, cache)
        |
+-------------------+-------------------+
|                   |                   |
V42Cache          V42Block            V42MoE
|                   |                   |
Token history     HyperConnection     QuantizedSwitchLinear
Window KV         Sparse attention    sqrtsoftplus router
Compressed KV     Compressor          Shared BF16 expert
Index state       Engram injection
        |
WeightStore
        |
BF16 critical weights + Q8/Q4 routed experts + SSD Engram
```

## 5. 文件边界

新目录：`v41f/mlx/runtime/`

| 文件 | 职责 |
|---|---|
| `model.py` | `V42RuntimeModel` 与 24 层前向 |
| `cache.py` | Token、窗口 KV、压缩 KV、索引状态 |
| `attention.py` | 窗口与压缩稀疏注意力 |
| `moe.py` | mlx-lm QuantizedSwitchLinear 封装 |
| `engram.py` | n-gram 状态与 SSD/内存查询 |
| `weights.py` | 唯一权重格式和校验 |
| `generate.py` | generate_step 与 BatchGenerator 封装 |
| `server.py` | SSE、OpenAI API、健康检查 |
| `benchmark.py` | 单流和批量性能基准 |
| `verify.py` | 精度、停止符、RSS、HTTP 验收 |

旧文件不再作为生产依赖。它们保留到新后端验收通过。

## 6. 缓存设计

### 6.1 Token 状态

每个请求保存完整 token 序列。Engram 只读取最后 4 个 token 的上下文。

### 6.2 窗口 KV

每层保存一个 MQA KV cache。每个位置只保存 256 维 BF16 KV。缓存使用 mlx-lm `KVCache`。

### 6.3 压缩状态

每个请求保存 3 个压缩阶段：源层 2、8、12。

每个阶段包含：

- `compress_kv`
- `index_k`
- 当前未完成的 ratio=2 token 组
- 当前 query 的 top-k 结果

非索引层不复用上一 query 的 top-k。它只使用当前计算图允许的选择。

### 6.4 批处理

缓存对象必须实现：

- `state`
- `merge`
- `extract`
- `filter`
- `prepare`
- `finalize`
- `nbytes`

这样才能直接使用 mlx-lm `BatchGenerator`。

## 7. 权重格式

只保留一个发布目录：`ckpt_local/v42_runtime/`。

### accurate 档

- 注意力、HyperConnection、Engram 投影：BF16/FP32。
- 路由专家：Q8 affine，group size 64。
- 共享专家：BF16。
- Engram 表：SSD FP8。

### fast 档

- 路由专家：Q4 affine，group size 64。
- 共享专家：Q8。
- 注意力投影：Q8。
- HyperConnection、norm、gate、sink：FP32/BF16。

fast 档必须单独通过 64-token 一致率门。未通过时不得成为默认档。

### manifest

Manifest 必须包含：

- 源 checkpoint SHA256。
- 每个文件的 SHA256。
- shape、dtype、量化位数和 group size。
- 模型结构参数。
- tokenizer SHA256。
- Engram 表 SHA256。

## 8. 生成与服务

### 单请求

使用 mlx-lm `generate_step`。

### 多请求

使用 mlx-lm `BatchGenerator`。

默认参数：

- `completion_batch_size=8`
- `prefill_batch_size=4`
- `prefill_step_size=512`
- 停止 token：`im_end=32764`

服务器提供：

- `GET /`
- `GET /healthz`
- `GET /metrics`
- `POST /api/chat`
- `POST /v1/chat/completions`

服务启动时执行：

1. Manifest 校验。
2. 权重 mmap。
3. MoE 量化权重预热。
4. 1-token 自检。
5. 通过后开放端口。

## 9. 性能路线

按以下顺序实施：

1. 消除 SSD 重复读取。
2. 使用增量窗口 KV。
3. 使用压缩状态缓存。
4. 使用 QuantizedSwitchLinear。
5. 使用异步 `generate_step`。
6. 启用连续批处理。
7. 评估 Q4 fast 档。
8. 最后评估推测解码。

不得先做推测解码。基础模型质量不稳定时，推测解码收益不可靠。

## 10. 验收矩阵

### L0：权重

- 每种保留精度的权重逐元素一致。
- Q8/Q4 记录最大误差和均值误差。
- Manifest 哈希全部通过。

### L1：模块

- RMSNorm。
- HyperConnection。
- Q/KV/O 投影。
- MoE 路由与专家输出。
- Engram hash 与注入。
- Compressor 与 Indexer。
- 稀疏注意力。

### L2：逐层

同一 prompt 比较 24 层 hidden state。首个超门层立即阻断。

### L3：生成

- 算术。
- 中文问答。
- Python 代码。
- 长文本。
- 多轮 ChatML。
- 64-token argmax。
- 256-token 连贯性。

### L4：服务

- SSE 正常结束。
- 客户端主动停止。
- 两并发请求。
- 八并发请求。
- 页面刷新。
- 服务重启。

### L5：性能

分别报告：

- TTFT。
- 单流 decode token/s。
- 8 路 aggregate token/s。
- 16 路 aggregate token/s。
- 峰值 RSS。
- Engram SSD 命中率。

## 11. 切换与回滚

实施期间，当前 8731 服务保持运行。

新服务先运行在 8732。全部验收通过后：

1. 停止 8731。
2. 将新服务切到 8731。
3. 重跑网页验收。
4. 删除旧生产入口。
5. 保留 `legacy/` 只读对照 7 天。

任一硬门失败时保持旧服务，不执行切换。

## 12. 一次性实施批次

实施按一个提交完成：

1. 建 `runtime/`。
2. 导出统一权重。
3. 接 mlx-lm 单流生成。
4. 接连续批处理。
5. 接服务和网页。
6. 跑全部验收。
7. 切换端口。
8. 清理旧入口。

提交前不向生产入口合并中间状态。
