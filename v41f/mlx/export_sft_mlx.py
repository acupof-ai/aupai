"""Export SFT ckpt to MLX-readable format in ckpt_local/sft_mlx_v2/.

Low memory: loads one tensor at a time via mmap, writes immediately.
Does NOT overwrite old sft_fp8/.
"""
import os, json, hashlib, time, resource
import torch
import numpy as np

SRC = 'ckpt_local/ckpt_v42_sft_run.pt'
DST = 'ckpt_local/sft_mlx_v2'
os.makedirs(DST, exist_ok=True)

print(f'Loading ckpt structure: {SRC}')
ck = torch.load(SRC, mmap=True, weights_only=False, map_location='cpu')
sd = ck['model']
v42 = ck['cfg']['v42_cfg']
n_layers = v42['n_layers']

# ---- Helper ----
def sha256_bytes(b):
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def write_tensor(name, t, target_dtype):
    """Write tensor to DST/name.bin, return manifest entry."""
    path = os.path.join(DST, name + '.bin')
    src_dtype = str(t.dtype)
    
    if target_dtype == 'fp8_e4m3':
        # bf16 -> fp8 uint8
        arr = t.contiguous().to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    elif target_dtype == 'bf16':
        arr = t.contiguous().view(torch.uint16).numpy()
    elif target_dtype == 'fp32':
        arr = t.contiguous().float().numpy()
    else:
        raise ValueError(f'Unknown target dtype: {target_dtype}')
    
    arr = np.ascontiguousarray(arr)
    with open(path, 'wb') as f:
        f.write(arr.tobytes())
    
    return {
        'file': name + '.bin',
        'shape': list(t.shape),
        'src_dtype': src_dtype,
        'dst_dtype': target_dtype,
        'bytes': arr.nbytes,
        'sha256': sha256_bytes(arr.tobytes()),
    }

entries = []
t0 = time.perf_counter()

# ---- Top weights ----
print('Exporting top weights...')
entries.append(('embed.weight', write_tensor('embed', sd['embed.weight'], 'bf16')))
entries.append(('norm.weight', write_tensor('norm', sd['norm.weight'], 'bf16')))
entries.append(('head.weight', write_tensor('head', sd['head.weight'], 'fp32')))

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'  RSS={rss:.0f}MB')

# ---- Per-layer weights ----
print(f'Exporting {n_layers} layers...')
for L in range(n_layers):
    p = f'layers.{L}.'
    
    # Attention norms
    entries.append((p+'attn_norm.weight', write_tensor(f'L{L}_attn_norm', sd[p+'attn_norm.weight'], 'bf16')))
    entries.append((p+'ffn_norm.weight', write_tensor(f'L{L}_ffn_norm', sd[p+'ffn_norm.weight'], 'bf16')))
    
    # Q proj
    entries.append((p+'attn.qproj.wq_a.weight', write_tensor(f'L{L}_wq_a', sd[p+'attn.qproj.wq_a.weight'], 'fp8_e4m3')))
    entries.append((p+'attn.qproj.wq_b.weight', write_tensor(f'L{L}_wq_b', sd[p+'attn.qproj.wq_b.weight'], 'fp8_e4m3')))
    entries.append((p+'attn.qproj.q_norm.weight', write_tensor(f'L{L}_q_norm', sd[p+'attn.qproj.q_norm.weight'], 'bf16')))
    
    # KV proj
    entries.append((p+'attn.kvproj.wkv.weight', write_tensor(f'L{L}_wkv', sd[p+'attn.kvproj.wkv.weight'], 'fp8_e4m3')))
    entries.append((p+'attn.kvproj.kv_norm.weight', write_tensor(f'L{L}_kv_norm', sd[p+'attn.kvproj.kv_norm.weight'], 'bf16')))
    
    # O proj
    entries.append((p+'attn.oproj.wo_a', write_tensor(f'L{L}_wo_a', sd[p+'attn.oproj.wo_a'], 'fp8_e4m3')))
    entries.append((p+'attn.oproj.wo_b.weight', write_tensor(f'L{L}_wo_b', sd[p+'attn.oproj.wo_b.weight'], 'fp8_e4m3')))
    
    # Sink
    entries.append((p+'attn.attn_sink', write_tensor(f'L{L}_sink', sd[p+'attn.attn_sink'], 'fp32')))
    
    # HC
    for k in ['attn_fn', 'attn_scale', 'attn_base', 'ffn_fn', 'ffn_scale', 'ffn_base']:
        entries.append((p+f'hc.hc_{k}', write_tensor(f'L{L}_hc_{k}', sd[p+f'hc.hc_{k}'], 'fp32')))
    
    # MoE experts (bf16 -> fp8)
    entries.append((p+'ffn.w1', write_tensor(f'L{L}_w1', sd[p+'ffn.w1'], 'fp8_e4m3')))
    entries.append((p+'ffn.w3', write_tensor(f'L{L}_w3', sd[p+'ffn.w3'], 'fp8_e4m3')))
    entries.append((p+'ffn.w2', write_tensor(f'L{L}_w2', sd[p+'ffn.w2'], 'fp8_e4m3')))
    
    # Gate
    entries.append((p+'ffn.gate.weight', write_tensor(f'L{L}_gate_w', sd[p+'ffn.gate.weight'], 'fp32')))
    entries.append((p+'ffn.gate.bias', write_tensor(f'L{L}_gate_b', sd[p+'ffn.gate.bias'], 'fp32')))
    
    # Shared experts
    entries.append((p+'ffn.shared_experts.w1.weight', write_tensor(f'L{L}_shared_w1', sd[p+'ffn.shared_experts.w1.weight'], 'fp8_e4m3')))
    entries.append((p+'ffn.shared_experts.w3.weight', write_tensor(f'L{L}_shared_w3', sd[p+'ffn.shared_experts.w3.weight'], 'fp8_e4m3')))
    entries.append((p+'ffn.shared_experts.w2.weight', write_tensor(f'L{L}_shared_w2', sd[p+'ffn.shared_experts.w2.weight'], 'fp8_e4m3')))
    
    # Compressor (if exists)
    cw = p+'attn.compressor.wkv.weight'
    if cw in sd:
        entries.append((cw, write_tensor(f'L{L}_comp_wkv', sd[cw], 'fp8_e4m3')))
        cg = p+'attn.compressor.wgate.weight'
        if cg in sd:
            entries.append((cg, write_tensor(f'L{L}_comp_wgate', sd[cg], 'fp8_e4m3')))
        entries.append((p+'attn.compressor.norm.weight', write_tensor(f'L{L}_comp_norm', sd[p+'attn.compressor.norm.weight'], 'bf16')))
    
    # Index key (if exists)
    ik = p+'attn.index_key.wk.weight'
    if ik in sd:
        entries.append((ik, write_tensor(f'L{L}_index_wk', sd[ik], 'fp8_e4m3')))
        entries.append((p+'attn.index_key.k_norm.weight', write_tensor(f'L{L}_index_knorm', sd[p+'attn.index_key.k_norm.weight'], 'bf16')))
    
    # Indexer (if exists)
    iw = p+'attn.indexer.wq_b.weight'
    if iw in sd:
        entries.append((iw, write_tensor(f'L{L}_index_wqb', sd[iw], 'fp8_e4m3')))
        entries.append((p+'attn.indexer.weights_proj.weight', write_tensor(f'L{L}_index_wp', sd[p+'attn.indexer.weights_proj.weight'], 'fp8_e4m3')))
    
    if L % 6 == 5:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(f'  layer {L}: RSS={rss:.0f}MB, {time.perf_counter()-t0:.1f}s')

# ---- Engram weights ----
print('Exporting Engram weights...')
for L in v42['engram_layer_ids']:
    entries.append((f'engrams.{L}.wkv.weight', write_tensor(f'en{L}_wkv', sd[f'engrams.{L}.wkv.weight'], 'bf16')))
    entries.append((f'engrams.{L}.q_weight', write_tensor(f'en{L}_qw', sd[f'engrams.{L}.q_weight'], 'bf16')))
    entries.append((f'engrams.{L}.k_weight', write_tensor(f'en{L}_kw', sd[f'engrams.{L}.k_weight'], 'bf16')))

# ---- Build manifest ----
total_bytes = sum(e[1]['bytes'] for e in entries)
manifest = {
    'source_ckpt': SRC,
    'source_sha256': '2836fa7e67c3767757dd1576f2cbfefffacd0b7308c3bc8922e74ef470653d69',
    'v42_cfg': v42,
    'quantization': 'fp8_e4m3 for experts/attention projections, bf16 for norms/small, fp32 for head/gate/sink/hc',
    'total_files': len(entries),
    'total_bytes': total_bytes,
    'entries': [{'src_key': k, **v} for k, v in entries],
}

with open(os.path.join(DST, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'\nDone: {len(entries)} files, {total_bytes/1e9:.2f} GB, peak RSS={rss:.0f}MB, {time.perf_counter()-t0:.1f}s')
print(f'Manifest: {DST}/manifest.json')
