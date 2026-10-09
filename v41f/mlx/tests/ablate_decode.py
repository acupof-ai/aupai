"""Systematic ablation: find first divergence between incremental decode and full forward.

Tests:
1. Full model (all components)
2. No Engram
3. No compressor/indexer
4. No Engram + no compressor/indexer
"""
import sys
sys.path.insert(0, '.')
import mlx.core as mx
from v41f.mlx.weights_manifest import load_from_manifest
from v41f.mlx.config import MLXV42Config
from v41f.mlx.model import MLXV42Model
from v41f.mlx.generator import MLXGenerator
from v41f.mlx.engram_hash import MLXNgramHash
from v41f.mlx.engram_ssd import EngramSSDLookup
from tokenizers import Tokenizer

def setup_model(no_engram=False, no_comp=False):
    W, v42 = load_from_manifest()
    mcfg = MLXV42Config.from_v42_cfg(v42)
    model = MLXV42Model(mcfg, W)
    tok = Tokenizer.from_file('ckpt_local/tok/tokenizer.json')
    
    if not no_engram:
        model.engram_hash = MLXNgramHash(tok, layer_ids=tuple(v42['engram_layer_ids']),
                                         max_ngram_size=v42['engram_max_ngram_size'],
                                         n_heads=v42['engram_n_heads'],
                                         engram_vocab_size=v42['engram_compressed_vocab_size'],
                                         pad_id=v42.get('engram_pad_id', 2))
        engram_sizes = {1:786862, 5:788118, 9:789492, 13:791110, 17:792776, 21:794672}
        for L in v42['engram_layer_ids']:
            model.ssd_lookups[L] = EngramSSDLookup(f'ckpt_local/engram_ssd/embed_L{L}.bin', engram_sizes[L], head_dim=128, cache_rows=4096)
    else:
        # Remove engram weights
        model.w['engrams'] = {}
    
    if no_comp:
        # Remove compressor/indexer weights
        for L in model.w['layers']:
            if 'compressor' in model.w['layers'][L]:
                del model.w['layers'][L]['compressor']
            if 'indexer' in model.w['layers'][L]:
                del model.w['layers'][L]['indexer']
            if 'index_key' in model.w['layers'][L]:
                del model.w['layers'][L]['index_key']
    
    return model, v42

def compare(model, prompt, next_tok):
    """Compare full forward on prompt+[next_tok] vs decode_step."""
    # Full forward
    full = mx.array([prompt + [next_tok]], dtype=mx.int32)
    fl = model.forward(full)
    mx.eval(fl)
    full_last = fl[0, -1, :]
    
    # Prefill + decode
    gen = MLXGenerator(model)
    pl = gen.prefill(mx.array([prompt], dtype=mx.int32))
    mx.eval(pl)
    sl = gen.decode_step(mx.array([next_tok], dtype=mx.int32), pos=len(prompt))
    mx.eval(sl)
    
    diff = mx.abs(sl[0].astype(mx.float32) - full_last.astype(mx.float32))
    return {
        'max_diff': float(mx.max(diff)),
        'mean_diff': float(mx.mean(diff)),
        'dec_arg': int(mx.argmax(sl[0])),
        'full_arg': int(mx.argmax(full_last)),
        'dec_max': float(mx.max(sl[0])),
        'full_max': float(mx.max(full_last)),
    }

prompt = [1, 5, 9]
next_tok = 13

tests = [
    ('1. Full model', False, False),
    ('2. No Engram', True, False),
    ('3. No Comp/Index', False, True),
    ('4. No Engram + No Comp', True, True),
]

print('=== Ablation: incremental decode vs full-prefix forward ===')
print(f'prompt={prompt}, next_tok={next_tok}')
print()

for name, no_en, no_comp in tests:
    model, v42 = setup_model(no_engram=no_en, no_comp=no_comp)
    # Warm up
    _ = model.forward(mx.array([prompt], dtype=mx.int32)); mx.eval(_)
    r = compare(model, prompt, next_tok)
    match = '✓' if r['dec_arg'] == r['full_arg'] else '✗'
    rel = r['max_diff'] / (abs(r['full_max']) + 1e-8) * 100
    print(f'{name}:')
    print(f'  dec_arg={r["dec_arg"]}, full_arg={r["full_arg"]} {match}')
    print(f'  max_diff={r["max_diff"]:.2f} ({rel:.1f}% of full_max)')
    print(f'  mean_diff={r["mean_diff"]:.4f}')
    print()
    del model
