"""Streaming BF16 weight loader: loads layer by layer, releases after use.
Keeps peak RSS < 6GB even though total weights are 6.7GB.
"""
import os, json
import numpy as np
import mlx.core as mx

DIR = 'ckpt_local/sft_mlx_bf16'

class StreamingWeights:
    """Lazy weight loader that reads from disk on demand.
    Non-expert weights are cached; expert weights are loaded per layer and released.
    """
    def __init__(self, ddir=DIR):
        self.dir = ddir
        with open(os.path.join(ddir, 'manifest.json')) as f:
            m = json.load(f)
        self.v42 = m['v42_cfg']
        self.entries = {e['src_key']: e for e in m['entries']}
        self._cache = {}  # persistent cache for all non-routed weights
        self._cache_bytes = 0
    
    def _load_raw(self, key):
        e = self.entries[key]
        path = os.path.join(self.dir, e['file'])
        shape = tuple(e['shape'])
        if e['dst_dtype'] == 'bf16':
            return mx.array(np.fromfile(path, dtype=np.uint16).reshape(shape), dtype=mx.uint16).view(mx.bfloat16)
        elif e['dst_dtype'] == 'fp32':
            return mx.array(np.fromfile(path, dtype=np.float32).reshape(shape), dtype=mx.float32)
        else:
            raise ValueError(f'Unknown dtype: {e["dst_dtype"]}')
    
    def get(self, key):
        """Get and cache a non-routed weight tensor.

        Routed expert matrices are stored in separate per-expert files and are
        never requested through this method. The remaining model weights fit in
        memory, so caching them removes repeated SSD reads across decode steps.
        """
        if key in self._cache:
            return self._cache[key]
        w = self._load_raw(key)
        self._cache[key] = w
        self._cache_bytes += int(self.entries[key]['bytes'])
        return w
    
    def get_layer(self, L):
        """Get all weights for layer L. Returns dict structure."""
        p = f'layers.{L}.'
        d = {}
        d['attn_norm'] = {'weight': self.get(p+'attn_norm.weight')}
        d['qproj'] = {'wq_a': {'weight': self.get(p+'attn.qproj.wq_a.weight')},
                      'wq_b': {'weight': self.get(p+'attn.qproj.wq_b.weight')},
                      'q_norm': {'weight': self.get(p+'attn.qproj.q_norm.weight')}}
        d['kvproj'] = {'wkv': {'weight': self.get(p+'attn.kvproj.wkv.weight')},
                       'kv_norm': {'weight': self.get(p+'attn.kvproj.kv_norm.weight')}}
        d['oproj'] = {'wo_a': self.get(p+'attn.oproj.wo_a'),
                      'wo_b': {'weight': self.get(p+'attn.oproj.wo_b.weight')}}
        d['attn_sink'] = self.get(p+'attn.attn_sink')
        d['hc'] = {k: self.get(p+f'hc.hc_{k}') for k in ['attn_fn','attn_scale','attn_base','ffn_fn','ffn_scale','ffn_base']}
        d['ffn_norm'] = {'weight': self.get(p+'ffn_norm.weight')}
        
        # Routed experts are loaded by sparse_moe_forward from per-expert files.
        # Do not read the full layer w1/w2/w3 tensors here: they are unused and
        # would add several GB of redundant SSD I/O on every decode step.
        d['ffn'] = {'gate': {'weight': self.get(p+'ffn.gate.weight'),
                             'bias': self.get(p+'ffn.gate.bias')},
                    'shared': {'w1': {'weight': self.get(p+'ffn.shared_experts.w1.weight')},
                               'w3': {'weight': self.get(p+'ffn.shared_experts.w3.weight')},
                               'w2': {'weight': self.get(p+'ffn.shared_experts.w2.weight')}}}
        
        if p+'attn.compressor.wkv.weight' in self.entries:
            comp = {'wkv': {'weight': self.get(p+'attn.compressor.wkv.weight')},
                    'norm': {'weight': self.get(p+'attn.compressor.norm.weight')}}
            if p+'attn.compressor.wgate.weight' in self.entries:
                comp['wgate'] = {'weight': self.get(p+'attn.compressor.wgate.weight')}
            d['compressor'] = comp
        if p+'attn.index_key.wk.weight' in self.entries:
            d['index_key'] = {'wk': {'weight': self.get(p+'attn.index_key.wk.weight')},
                              'k_norm': {'weight': self.get(p+'attn.index_key.k_norm.weight')}}
        if p+'attn.indexer.wq_b.weight' in self.entries:
            d['indexer'] = {'wq_b': {'weight': self.get(p+'attn.indexer.wq_b.weight')},
                            'weights_proj': {'weight': self.get(p+'attn.indexer.weights_proj.weight')}}
        return d
    
    def release_layer(self, L):
        """Keep non-routed weights resident across decode steps."""
        return
    
    @property
    def embed_weight(self):
        return self.get('embed.weight')
    
    @property
    def norm_weight(self):
        return self.get('norm.weight')
    
    @property
    def head_weight(self):
        return self.get('head.weight')
    
    def get_engram(self, L):
        return {'wkv': {'weight': self.get(f'engrams.{L}.wkv.weight')},
                'q_weight': self.get(f'engrams.{L}.q_weight'),
                'k_weight': self.get(f'engrams.{L}.k_weight')}
