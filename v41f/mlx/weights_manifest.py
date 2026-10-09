"""Load SFT model from manifest.json in ckpt_local/sft_mlx_v2/."""
import os, json
import numpy as np
import mlx.core as mx

DIR = 'ckpt_local/sft_mlx_v2'

def load_from_manifest():
    """Load all weights from manifest-based directory. Returns (W, v42)."""
    with open(os.path.join(DIR, 'manifest.json')) as f:
        m = json.load(f)
    
    v42 = m['v42_cfg']
    n = v42['n_layers']
    entries = {e['src_key']: e for e in m['entries']}
    
    def load(key):
        e = entries[key]
        path = os.path.join(DIR, e['file'])
        shape = tuple(e['shape'])
        if e['dst_dtype'] == 'fp8_e4m3':
            return mx.array(np.fromfile(path, dtype=np.uint8).reshape(shape), dtype=mx.uint8)
        elif e['dst_dtype'] == 'bf16':
            return mx.array(np.fromfile(path, dtype=np.uint16).reshape(shape), dtype=mx.uint16).view(mx.bfloat16)
        elif e['dst_dtype'] == 'fp32':
            return mx.array(np.fromfile(path, dtype=np.float32).reshape(shape), dtype=mx.float32)
    
    W = {'embed': {'weight': load('embed.weight')},
         'norm': {'weight': load('norm.weight')},
         'head': {'weight': load('head.weight')},
         'layers': {}, 'engrams': {}}
    
    for L in range(n):
        p = f'layers.{L}.'
        d = {}
        d['attn_norm'] = {'weight': load(p+'attn_norm.weight')}
        d['ffn_norm'] = {'weight': load(p+'ffn_norm.weight')}
        d['qproj'] = {'wq_a': {'weight': load(p+'attn.qproj.wq_a.weight')},
                      'wq_b': {'weight': load(p+'attn.qproj.wq_b.weight')},
                      'q_norm': {'weight': load(p+'attn.qproj.q_norm.weight')}}
        d['kvproj'] = {'wkv': {'weight': load(p+'attn.kvproj.wkv.weight')},
                       'kv_norm': {'weight': load(p+'attn.kvproj.kv_norm.weight')}}
        d['oproj'] = {'wo_a': load(p+'attn.oproj.wo_a'),
                      'wo_b': {'weight': load(p+'attn.oproj.wo_b.weight')}}
        d['attn_sink'] = load(p+'attn.attn_sink')
        d['hc'] = {k: load(p+f'hc.hc_{k}') for k in ['attn_fn','attn_scale','attn_base','ffn_fn','ffn_scale','ffn_base']}
        d['ffn'] = {'w1': load(p+'ffn.w1'),
                    'w3': load(p+'ffn.w3'),
                    'w2': load(p+'ffn.w2'),
                    'gate': {'weight': load(p+'ffn.gate.weight'),
                             'bias': load(p+'ffn.gate.bias')},
                    'shared': {'w1': {'weight': load(p+'ffn.shared_experts.w1.weight')},
                               'w3': {'weight': load(p+'ffn.shared_experts.w3.weight')},
                               'w2': {'weight': load(p+'ffn.shared_experts.w2.weight')}}}
        if p+'attn.compressor.wkv.weight' in entries:
            comp = {'wkv': {'weight': load(p+'attn.compressor.wkv.weight')},
                    'norm': {'weight': load(p+'attn.compressor.norm.weight')}}
            if p+'attn.compressor.wgate.weight' in entries:
                comp['wgate'] = {'weight': load(p+'attn.compressor.wgate.weight')}
            d['compressor'] = comp
        if p+'attn.index_key.wk.weight' in entries:
            d['index_key'] = {'wk': {'weight': load(p+'attn.index_key.wk.weight')},
                              'k_norm': {'weight': load(p+'attn.index_key.k_norm.weight')}}
        if p+'attn.indexer.wq_b.weight' in entries:
            d['indexer'] = {'wq_b': {'weight': load(p+'attn.indexer.wq_b.weight')},
                            'weights_proj': {'weight': load(p+'attn.indexer.weights_proj.weight')}}
        W['layers'][L] = d
    
    for L in v42['engram_layer_ids']:
        W['engrams'][L] = {'wkv': {'weight': load(f'engrams.{L}.wkv.weight')},
                           'q_weight': load(f'engrams.{L}.q_weight'),
                           'k_weight': load(f'engrams.{L}.k_weight')}
    
    return W, v42
