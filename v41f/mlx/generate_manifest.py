"""Generate or verify manifest.json for ckpt_local/sft_fp8/ weights.

Usage:
  python v41f/mlx/generate_manifest.py           # generate manifest
  python v41f/mlx/generate_manifest.py --verify   # verify (exit nonzero on mismatch)
"""
import os, json, hashlib, glob, sys

DIR = 'ckpt_local/sft_fp8'
MANIFEST = os.path.join(DIR, 'manifest.json')

# Source ckpt info
SRC_SHA = '2836fa7e67c3767757dd1576f2cbfefffacd0b7308c3bc8922e74ef470653d69'
SRC_NAME = 'ckpt_v42_sft_run.pt'

# Known authoritative files (from run_sft.py loading)
AUTHOR_PREFIXES = ['L*_w1.bin', 'L*_w3.bin', 'L*_w2.bin', 'L*_gate_w.bin', 'L*_gate_b.bin',
                   'L*_wq_a.bin', 'L*_wq_b.bin', 'L*_wkv.bin', 'L*_wo_a.bin', 'L*_wo_b.bin',
                   'L*_comp_wkv.bin', 'L*_index_wk.bin', 'L*_index_wqb.bin', 'L*_index_wp.bin',
                   'embed.bin', 'norm.bin', 'head.bin', 'attn_norm_*.bin', 'ffn_norm_*.bin',
                   'q_norm_*.bin', 'kv_norm_*.bin', 'attn_sink_*.bin', 'hc_*.bin',
                   'shared_w1.bin', 'shared_w3.bin', 'shared_w2.bin', 'comp_norm_*.bin',
                   'index_knorm_*.bin']

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

# Collect all files
all_files = sorted(glob.glob(os.path.join(DIR, '*.bin')))
print(f'Found {len(all_files)} .bin files in {DIR}/')

# Classify
entries = []
total_bytes = 0
orphans = []

for path in all_files:
    name = os.path.basename(path)
    sz = os.path.getsize(path)
    total_bytes += sz
    # Determine category
    if name.startswith('L') and '_w' in name and '_gate' not in name and '_wq' not in name and '_wkv' not in name and '_wo' not in name and '_comp' not in name and '_index' not in name:
        cat = 'expert_fp8'
    elif 'gate' in name:
        cat = 'gate_fp8'
    elif 'shared' in name:
        cat = 'shared_fp8'
    elif name in ('embed.bin', 'norm.bin', 'head.bin'):
        cat = 'top_bf16'
    else:
        cat = 'other'
    
    entries.append({
        'file': name,
        'bytes': sz,
        'category': cat,
        'sha256': sha256_file(path),
    })

# Orphans: files not matching expected patterns (real duplicates/unused)
for e in entries:
    name = e['file']
    is_expected = False
    # Expert files
    if name.startswith('L') and ('_w1.bin' in name or '_w3.bin' in name or '_w2.bin' in name) and 'shared' not in name:
        is_expected = True
    # Shared expert files (per-layer)
    elif 'shared_experts' in name:
        is_expected = True
    # Attention projection files
    elif name.startswith('L') and ('_wq' in name or '_wkv' in name or '_wo' in name):
        is_expected = True
    # Gate files
    elif name.startswith('L') and '_gate_' in name:
        is_expected = True
    # Compressor/index files
    elif name.startswith('L') and ('compressor' in name or 'index_' in name):
        is_expected = True
    # Norm/sink/hc small files
    elif name.startswith('L') and ('norm' in name or 'sink' in name or 'hc_' in name):
        is_expected = True
    # Top weights
    elif name in ('embed_weight.bin', 'norm_weight.bin', 'head_weight.bin'):
        is_expected = True
    # Engram wkv/q/k weights (en1_*.bin format)
    elif name.startswith('en') and ('_wkv_' in name or '_q_' in name or '_k_' in name):
        is_expected = True
    if not is_expected:
        orphans.append(name)
    elif name.startswith('L') and ('_gate' in name or '_hc' in name or '_norm' in name or '_sink' in name):
        is_expected = True
    elif name.startswith('L') and ('_comp' in name or '_index' in name):
        is_expected = True
    if not is_expected:
        orphans.append(name)

manifest = {
    'source_ckpt': SRC_NAME,
    'source_sha256': SRC_SHA,
    'quantization': 'fp8_e4m3 for experts and linears, bf16 for norms/small weights',
    'total_files': len(entries),
    'total_bytes': total_bytes,
    'entries': entries,
    'orphans': orphans,
}

with open(MANIFEST, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f'Manifest: {MANIFEST}')
print(f'Total: {len(entries)} files, {total_bytes/1e9:.2f} GB')
print(f'Orphans: {len(orphans)} files')
for o in orphans[:10]:
    print(f'  {o}')

# Verify mode
if '--verify' in sys.argv:
    print('\n=== Verify ===')
    with open(MANIFEST) as f:
        old = json.load(f)
    mismatches = 0
    for e in entries:
        old_e = next((x for x in old['entries'] if x['file'] == e['file']), None)
        if old_e is None:
            print(f'  MISSING: {e["file"]}')
            mismatches += 1
        elif old_e['sha256'] != e['sha256']:
            print(f'  HASH MISMATCH: {e["file"]}')
            mismatches += 1
    if mismatches:
        print(f'FAIL: {mismatches} mismatches')
        sys.exit(1)
    print('PASS: all hashes match')
    sys.exit(0)
