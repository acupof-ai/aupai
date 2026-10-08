# v41f MLX 推理后端

原生 MLX (Metal) 推理后端，适配 Apple Silicon。

## 快速上手

```bash
# 激活 venv
source .venv/bin/activate

# SFT 模型生成（accurate 模式，默认）
python -m v41f.mlx.cli --sft --prompt "The cat" --max-new-tokens 4

# 小模型测试
python -m v41f.mlx.cli --small --prompt "hello" --max-new-tokens 16
```

## 生成模式

| 模式 | 说明 | 精度 | 速度 |
|---|---|---|---|
| `accurate`（默认） | 每步对完整前缀重跑 full forward | 与 full forward 数学一致 | ~6 tok/s |
| `experimental-kv` | KV cache 增量解码 | Layer 1 有 12.9% 发散，argmax 不稳 | 更快 |

experimental-kv 仅用于实验，生产请用 accurate。

## 性能基线

| 指标 | 数值 |
|---|---|
| warm TTFT (prefill) | ~115ms |
| decode 吞吐 (accurate) | ~6 tok/s |
| 峰值 RSS | 3.3–3.9 GB < 6GB memguard |
| SSD Engram 命中率 | 89% |
| SSD 每层加载 | 10KB（非整表 100MB） |

这是原生 MLX 基线，不是充分优化的高性能终态。

## 权重

- 权威目录：`ckpt_local/sft_mlx_v2/`（621 文件，3.50GB，含 manifest.json）
- 源 ckpt：`ckpt_local/ckpt_v42_sft_run.pt`（7.9GB，sha256 2836fa7e...）
- SSD Engram：`ckpt_local/engram_ssd/embed_L{1,5,9,13,17,21}.bin`（6 文件，579MB）

### 注意

导出脚本峰值 RSS 7.1GB，与 memguard 6GB 限制冲突。不要在本机重新运行导出。默认使用已验证的 sft_mlx_v2 分片。

## 已知限制

1. **KV 增量模式数值不稳定**：24 层 bf16 精度累积导致 Layer 1 发散 12.9%。accurate 模式避免此问题。
2. **导出峰值超限**：导出脚本需要 7.1GB RSS，超过 memguard 6GB 限制。
3. **MoE 性能**：使用全专家 einsum（非 grouped GEMM），64 专家计算量较大。
4. **SFT 评测质量**：后端正确性 ≠ 模型能力。SFT 模型本身评测较差。
