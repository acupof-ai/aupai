"""Compare FP8 expert dequantization: PyTorch bf16 vs MLX from_fp8.
Checks if fp8 quantization introduces too much error.
"""
import sys
sys.path.insert(0, '/Users/bytedance/code/aupai')
import torch
import numpy as np
import mlx.core as mx

print('=== Loading PyTorch expert weights (mmap) ===')
ckpt = torch.load('ckpt_local/ckpt_v42_sft_run.pt', map_location='cpu', mmap=True, weights_only=False)
sd = ckpt['model']

# Layer 0 expert w1 (routed, fp8 in MLX)
pt_w1 = sd['layers.0.ffn.w1'].float().numpy()  # [64, 640, 1024]
print(f'PT layers.0.ffn.w1: shape={pt_w1.shape}, dtype={sd["layers.0.ffn.w1"].dtype}')
print(f'  min={pt_w1.min():.4f}, max={pt_w1.max():.4f}, mean={pt_w1.mean():.6f}')

# MLX fp8 version
from v41f.mlx.weights_manifest import load_from_manifest
W, v42 = load_from_manifest()
mx_w1_fp8 = W['layers'][0]['ffn']['w1']  # uint8 array
print(f'\nMX layers.0.ffn.w1 fp8: shape={mx_w1_fp8.shape}, dtype={mx_w1_fp8.dtype}')
print(f'  raw uint8 min={int(mx_w1_fp8.min())}, max={int(mx_w1_fp8.max())}')

# Dequantize MLX fp8
mx_w1_deq = np.array(mx.from_fp8(mx_w1_fp8).astype(mx.float32))
print(f'MX dequantized: shape={mx_w1_deq.shape}')
print(f'  min={mx_w1_deq.min():.4f}, max={mx_w1_deq.max():.4f}, mean={mx_w1_deq.mean():.6f}')

# Compare
if pt_w1.shape == mx_w1_deq.shape:
    diff = np.abs(pt_w1 - mx_w1_deq)
    rel = diff / (np.abs(pt_w1) + 1e-8)
    print(f'\n  max abs diff: {diff.max():.6f}')
    print(f'  mean abs diff: {diff.mean():.8f}')
    print(f'  max rel diff: {rel.max():.4f} ({rel.max()*100:.2f}%)')
    print(f'  mean rel diff: {rel.mean():.6f}')
else:
    print(f'SHAPE MISMATCH: pt={pt_w1.shape}, mx={mx_w1_deq.shape}')

# Check another layer
print('\n=== Layer 5 expert w1 ===')
pt_w1_5 = sd['layers.5.ffn.w1'].float().numpy()
mx_w1_5 = np.array(mx.from_fp8(W['layers'][5]['ffn']['w1']).astype(mx.float32))
if pt_w1_5.shape == mx_w1_5.shape:
    diff = np.abs(pt_w1_5 - mx_w1_5)
    rel = diff / (np.abs(pt_w1_5) + 1e-8)
    print(f'  max abs diff: {diff.max():.6f}')
    print(f'  mean rel diff: {rel.mean():.6f}')

# Check shared experts (should be bf16, not fp8)
print('\n=== Layer 0 shared expert w1 ===')
pt_shared = sd['layers.0.ffn.shared_experts.w1.weight'].float().numpy()
mx_shared = np.array(W['layers'][0]['ffn']['shared']['w1']['weight'].astype(mx.float32))
print(f'PT shape: {pt_shared.shape}, MX shape: {mx_shared.shape}')
if pt_shared.shape == mx_shared.shape:
    diff = np.abs(pt_shared - mx_shared)
    print(f'  max diff: {diff.max():.6f}')
