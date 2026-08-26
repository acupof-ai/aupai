# Aupai

200M hybrid Gated DeltaNet + Sliding Window Attention 中文推理模型。

## 特性

- **推理快**：Gated DeltaNet 线性注意力 + SWA，推理速度远超同规模 Transformer
- **RL 训练链路**：continuous batching、paged KV cache、train-infer 统一架构，GRPO 直接复用推理引擎

## 训练

```bash
# 预训练 (6×H20; FP8 via torchao, Attention Residuals opt-in)
torchrun --nproc_per_node=6 train.py --fp8 --attn_res --name attnres_2b
# 任何 --flag 覆盖 Cfg.<flag>（--seq/--batch/--accum/--attn_res_blocks/--attn_res_dyn_q/--grad_ckpt ...）
# AttnRes 开/关 500 步对照：NGPU=6 STEPS=500 scripts/run_ablation.sh
# 架构改动 CPU 自检：python scripts/test_arch_compat.py

# SFT
torchrun --nproc_per_node=8 sft.py --sft_path data/sft_mix.jsonl --epochs 2

# RL (GRPO)
torchrun --nproc_per_node=8 rl.py --steps 200 --group_size 8
```

## 加速

- fla chunkwise kernel（Gated DeltaNet）
- FlashAttention-3 (sliding window)
- FP8 前向 + Liger FLCE
- Muon optimizer + torch.compile
- int32 token IDs + NVMe cache
