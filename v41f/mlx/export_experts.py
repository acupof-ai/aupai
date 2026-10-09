"""Export per-expert BF16 weights to separate files.
Each expert has w1/w2/w3, loaded on demand (only top-8 active experts).
"""
import resource, os, json
import torch

OUT_DIR = 'ckpt_local/sft_mlx_bf16/expert'
os.makedirs(OUT_DIR, exist_ok=True)

print('Loading ckpt (mmap)...')
ckpt = torch.load('ckpt_local/ckpt_v42_sft_run.pt', map_location='cpu', mmap=True, weights_only=False)
sd = ckpt['model']
v42_cfg = ckpt['cfg']['v42_cfg']
n_layers = v42_cfg['n_layers']
n_experts = 64

manifest_entries = []
total_bytes = 0

for L in range(n_layers):
    # Routed experts: w1 is [n_experts, moe_inter_dim, dim] stacked
    w1_all = sd[f'layers.{L}.ffn.w1']  # [64, 640, 1024] bf16
    w2_all = sd[f'layers.{L}.ffn.w2']  # [64, 1024, 640] bf16
    w3_all = sd[f'layers.{L}.ffn.w3']  # [64, 640, 1024] bf16
    
    for e in range(n_experts):
        # w1 expert e
        w1_e = w1_all[e].contiguous().view(torch.uint16).numpy().tobytes()
        fname = f'L{L}_exp{e}_w1.bin'
        with open(os.path.join(OUT_DIR, fname), 'wb') as f:
            f.write(w1_e)
        manifest_entries.append({
            'src_key': f'layers.{L}.ffn.w1[{e}]',
            'file': f'expert/{fname}',
            'shape': list(w1_all[e].shape),
            'dst_dtype': 'bf16',
            'bytes': len(w1_e),
        })
        total_bytes += len(w1_e)
        
        # w2 expert e
        w2_e = w2_all[e].contiguous().view(torch.uint16).numpy().tobytes()
        fname = f'L{L}_exp{e}_w2.bin'
        with open(os.path.join(OUT_DIR, fname), 'wb') as f:
            f.write(w2_e)
        manifest_entries.append({
            'src_key': f'layers.{L}.ffn.w2[{e}]',
            'file': f'expert/{fname}',
            'shape': list(w2_all[e].shape),
            'dst_dtype': 'bf16',
            'bytes': len(w2_e),
        })
        total_bytes += len(w2_e)
        
        # w3 expert e
        w3_e = w3_all[e].contiguous().view(torch.uint16).numpy().tobytes()
        fname = f'L{L}_exp{e}_w3.bin'
        with open(os.path.join(OUT_DIR, fname), 'wb') as f:
            f.write(w3_e)
        manifest_entries.append({
            'src_key': f'layers.{L}.ffn.w3[{e}]',
            'file': f'expert/{fname}',
            'shape': list(w3_all[e].shape),
            'dst_dtype': 'bf16',
            'bytes': len(w3_e),
        })
        total_bytes += len(w3_e)
    
    if L % 4 == 0:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(f'Layer {L}: total={total_bytes/1e9:.2f}GB, RSS={rss:.0f}MB')

# Save expert manifest
with open('ckpt_local/sft_mlx_bf16/expert_manifest.json', 'w') as f:
    json.dump({'experts': manifest_entries}, f, indent=2)

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'\nDone: {len(manifest_entries)} expert files, {total_bytes/1e9:.2f}GB')
print(f'Peak RSS: {rss:.0f}MB')
