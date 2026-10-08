"""Export SFT weights as raw bf16/fp32 to ckpt_local/sft_mlx_bf16/.
Layer-by-layer streaming, low peak RSS.
"""
import resource, time, os, json, hashlib
import torch

OUT_DIR = 'ckpt_local/sft_mlx_bf16'
os.makedirs(OUT_DIR, exist_ok=True)

print('Loading ckpt (mmap)...')
t0 = time.time()
ckpt = torch.load('ckpt_local/ckpt_v42_sft_run.pt', map_location='cpu', mmap=True, weights_only=False)
sd = ckpt['model']
v42_cfg = ckpt['cfg']['v42_cfg']
print(f'  {len(sd)} keys, cfg layers={v42_cfg["n_layers"]}')

entries = []
total_bytes = 0

def export_tensor(key, t):
    """Export one tensor to a raw file. Returns entry dict."""
    global total_bytes
    shape = list(t.shape)
    dtype = str(t.dtype).replace('torch.', '')
    
    # Convert to raw bytes
    if 'bf16' in dtype or 'bfloat16' in dtype:
        raw = t.contiguous().view(torch.uint16).numpy().tobytes()
        out_dtype = 'bf16'
    elif 'float32' in dtype or 'fp32' in dtype:
        raw = t.contiguous().numpy().tobytes()
        out_dtype = 'fp32'
    elif 'float16' in dtype:
        raw = t.contiguous().view(torch.uint16).numpy().tobytes()
        out_dtype = 'fp16'
    else:
        raise ValueError(f'Unknown dtype: {dtype}')
    
    fname = key.replace('.', '_') + '.bin'
    fpath = os.path.join(OUT_DIR, fname)
    with open(fpath, 'wb') as f:
        f.write(raw)
    
    sha = hashlib.sha256(raw).hexdigest()[:16]
    entries.append({
        'src_key': key,
        'file': fname,
        'shape': shape,
        'src_dtype': dtype,
        'dst_dtype': out_dtype,
        'bytes': len(raw),
        'sha256_16': sha,
    })
    total_bytes += len(raw)
    return fname

# Export top-level weights
print('Exporting top-level...')
export_tensor('embed.weight', sd['embed.weight'])
export_tensor('norm.weight', sd['norm.weight'])
export_tensor('head.weight', sd['head.weight'])
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'  RSS={rss:.0f}MB')

# Export layer weights
n_layers = v42_cfg['n_layers']
for L in range(n_layers):
    print(f'Layer {L}...')
    p = f'layers.{L}.'
    
    # Attention norm
    export_tensor(p + 'attn_norm.weight', sd[p + 'attn_norm.weight'])
    
    # Q projection
    export_tensor(p + 'attn.qproj.wq_a.weight', sd[p + 'attn.qproj.wq_a.weight'])
    export_tensor(p + 'attn.qproj.wq_b.weight', sd[p + 'attn.qproj.wq_b.weight'])
    export_tensor(p + 'attn.qproj.q_norm.weight', sd[p + 'attn.qproj.q_norm.weight'])
    
    # KV projection
    export_tensor(p + 'attn.kvproj.wkv.weight', sd[p + 'attn.kvproj.wkv.weight'])
    export_tensor(p + 'attn.kvproj.kv_norm.weight', sd[p + 'attn.kvproj.kv_norm.weight'])
    
    # O projection
    export_tensor(p + 'attn.oproj.wo_a', sd[p + 'attn.oproj.wo_a'])
    export_tensor(p + 'attn.oproj.wo_b.weight', sd[p + 'attn.oproj.wo_b.weight'])
    
    # Attn sink
    export_tensor(p + 'attn.attn_sink', sd[p + 'attn.attn_sink'])
    
    # HyperConn
    for k in ['hc_attn_fn', 'hc_attn_scale', 'hc_attn_base', 'hc_ffn_fn', 'hc_ffn_scale', 'hc_ffn_base']:
        export_tensor(p + 'hc.' + k, sd[p + 'hc.' + k])
    
    # FFN norm
    export_tensor(p + 'ffn_norm.weight', sd[p + 'ffn_norm.weight'])
    
    # Routed experts (bf16, not fp8!)
    export_tensor(p + 'ffn.w1', sd[p + 'ffn.w1'])
    export_tensor(p + 'ffn.w3', sd[p + 'ffn.w3'])
    export_tensor(p + 'ffn.w2', sd[p + 'ffn.w2'])
    
    # Gate
    export_tensor(p + 'ffn.gate.weight', sd[p + 'ffn.gate.weight'])
    export_tensor(p + 'ffn.gate.bias', sd[p + 'ffn.gate.bias'])
    
    # Shared experts (bf16)
    export_tensor(p + 'ffn.shared_experts.w1.weight', sd[p + 'ffn.shared_experts.w1.weight'])
    export_tensor(p + 'ffn.shared_experts.w3.weight', sd[p + 'ffn.shared_experts.w3.weight'])
    export_tensor(p + 'ffn.shared_experts.w2.weight', sd[p + 'ffn.shared_experts.w2.weight'])
    
    # Compressor (if exists)
    if p + 'attn.compressor.wkv.weight' in sd:
        export_tensor(p + 'attn.compressor.wkv.weight', sd[p + 'attn.compressor.wkv.weight'])
        export_tensor(p + 'attn.compressor.norm.weight', sd[p + 'attn.compressor.norm.weight'])
    
    # Indexer
    if p + 'attn.index_key.wk.weight' in sd:
        export_tensor(p + 'attn.index_key.wk.weight', sd[p + 'attn.index_key.wk.weight'])
        export_tensor(p + 'attn.index_key.k_norm.weight', sd[p + 'attn.index_key.k_norm.weight'])
    
    if p + 'attn.indexer.wq_b.weight' in sd:
        export_tensor(p + 'attn.indexer.wq_b.weight', sd[p + 'attn.indexer.wq_b.weight'])
        export_tensor(p + 'attn.indexer.weights_proj.weight', sd[p + 'attn.indexer.weights_proj.weight'])
    
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    if L % 4 == 0:
        print(f'  RSS={rss:.0f}MB, total={total_bytes/1e9:.2f}GB')

# Export Engram weights
print('Exporting Engram...')
for L in v42_cfg['engram_layer_ids']:
    p = f'engrams.{L}.'
    export_tensor(p + 'wkv.weight', sd[p + 'wkv.weight'])
    export_tensor(p + 'q_weight', sd[p + 'q_weight'])
    export_tensor(p + 'k_weight', sd[p + 'k_weight'])

# Write manifest
manifest = {
    'source_ckpt': 'ckpt_v42_sft_run.pt',
    'source_sha256': '2836fa7e67c3767757dd1576f2cbfefffacd0b7308c3bc8922e74ef470653d69',
    'v42_cfg': v42_cfg,
    'entries': entries,
}

with open(os.path.join(OUT_DIR, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'\nDone: {len(entries)} files, {total_bytes/1e9:.2f}GB')
print(f'Peak RSS: {rss:.0f}MB')
print(f'Time: {time.time()-t0:.1f}s')
