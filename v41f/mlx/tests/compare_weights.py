"""Compare exported MLX weights vs original PyTorch ckpt weights.
Finds first tensor mismatch to locate the bug.
"""
import sys
sys.path.insert(0, '/Users/bytedance/code/aupai')
import torch
import numpy as np
import mlx.core as mx

print('=== Loading PyTorch state dict (mmap) ===')
ckpt = torch.load('ckpt_local/ckpt_v42_sft_run.pt', map_location='cpu', mmap=True, weights_only=False)
sd = ckpt['model']
v42_cfg = ckpt['cfg']['v42_cfg']
print(f'  {len(sd)} keys')

print('\n=== Loading MLX weights manifest ===')
from v41f.mlx.weights_manifest import load_from_manifest
W, v42 = load_from_manifest()

# Build a flat map of MLX weights
def flatten(d, prefix=''):
    items = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            items.update(flatten(v, key))
        elif isinstance(v, mx.array):
            items[key] = v
    return items

flat_mlx = flatten(W)
print(f'  {len(flat_mlx)} MLX tensors')

# Compare key by key
print('\n=== Comparing key tensors ===')
# Check embed
pt_embed = sd['embed.weight'].float().numpy()
mx_embed = np.array(flat_mlx['embed.weight'].astype(mx.float32))
diff = np.abs(pt_embed - mx_embed)
print(f'embed.weight: shape={pt_embed.shape}, max_diff={diff.max():.6f}, mean={diff.mean():.8f}')

# Check head
pt_head = sd['head.weight'].float().numpy()
mx_head = np.array(flat_mlx['head.weight'].astype(mx.float32))
diff = np.abs(pt_head - mx_head)
print(f'head.weight: shape={pt_head.shape}, max_diff={diff.max():.6f}, mean={diff.mean():.8f}')

# Check layer 0 attention
pt_wq = sd['layers.0.attn.wq.weight'].float().numpy()
mx_wq = np.array(flat_mlx['layers.0.attn.wq.weight'].astype(mx.float32))
diff = np.abs(pt_wq - mx_wq)
print(f'layers.0.attn.wq: max_diff={diff.max():.6f}, mean={diff.mean():.8f}')

# Check layer 0 MoE expert (fp8)
pt_w1 = sd['layers.0.ffn.w1.weight'].float().numpy()
mx_w1_fp8 = flat_mlx['layers.0.ffn.w1.weight']
# Dequantize: uint8 -> e4m3 -> float16/float32
mx_w1 = np.array(mx.from_fp8(mx_w1_fp8).astype(mx.float32))
print(f'layers.0.ffn.w1 (fp8): pt shape={pt_w1.shape}, mx shape={mx_w1.shape}')
if pt_w1.shape == mx_w1.shape:
    diff = np.abs(pt_w1 - mx_w1)
    rel = diff.max() / (np.abs(pt_w1).max() + 1e-8)
    print(f'  max_diff={diff.max():.4f}, rel={rel*100:.2f}%')

# Check layer 1 engram
if 'layers.1.engram.wq.weight' in sd:
    pt_ewq = sd['layers.1.engram.wq.weight'].float().numpy()
    print(f'layers.1.engram.wq: pt shape={pt_ewq.shape}')
    if 'layers.1.engram.wq.weight' in flat_mlx:
        mx_ewq = np.array(flat_mlx['layers.1.engram.wq.weight'].astype(mx.float32))
        diff = np.abs(pt_ewq - mx_ewq)
        print(f'  max_diff={diff.max():.6f}, mean={diff.mean():.8f}')

# Check HyperConn
pt_hc_attn_fn = sd['layers.0.hc.attn_fn.weight'].float().numpy()
mx_hc = np.array(flat_mlx['layers.0.hc.attn_fn.weight'].astype(mx.float32))
diff = np.abs(pt_hc_attn_fn - mx_hc)
print(f'layers.0.hc.attn_fn: max_diff={diff.max():.6f}, mean={diff.mean():.8f}')

print('\nDone')
