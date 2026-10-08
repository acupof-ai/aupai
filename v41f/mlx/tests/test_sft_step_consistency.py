"""Step-by-step decode vs full-prefix forward consistency test (real SFT).

Prompt: [1,5,9,13], generate 4 tokens.
Each step: incremental decode_step vs full prefix model.forward.
"""
import resource, sys
sys.path.insert(0, '.')
import mlx.core as mx
from v41f.mlx.weights_manifest import load_from_manifest
from v41f.mlx.config import MLXV42Config
from v41f.mlx.model import MLXV42Model
from v41f.mlx.generator import MLXGenerator
from v41f.mlx.engram_hash import MLXNgramHash
from v41f.mlx.engram_ssd import EngramSSDLookup
from tokenizers import Tokenizer

print('Loading SFT model...')
W, v42 = load_from_manifest()
mcfg = MLXV42Config.from_v42_cfg(v42)
model = MLXV42Model(mcfg, W)
tok = Tokenizer.from_file('ckpt_local/tok/tokenizer.json')
model.engram_hash = MLXNgramHash(tok, layer_ids=tuple(v42['engram_layer_ids']), max_ngram_size=v42['engram_max_ngram_size'], n_heads=v42['engram_n_heads'], engram_vocab_size=v42['engram_compressed_vocab_size'], pad_id=v42.get('engram_pad_id', 2))
engram_sizes = {1:786862, 5:788118, 9:789492, 13:791110, 17:792776, 21:794672}
for L in v42['engram_layer_ids']:
    model.ssd_lookups[L] = EngramSSDLookup(f'ckpt_local/engram_ssd/embed_L{L}.bin', engram_sizes[L], head_dim=128, cache_rows=4096)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'Model built, RSS={rss:.0f}MB')

# Prompt
prompt = [1, 5, 9, 13]
tokens = mx.array([prompt], dtype=mx.int32)

# Warm up
_ = model.forward(tokens); mx.eval(_)
gen = MLXGenerator(model)
_ = gen.prefill(tokens); mx.eval(_)
print()

# Step-by-step test
print('=== Step-by-step decode vs full-prefix forward ===')
gen = MLXGenerator(model)
logits0 = gen.prefill(tokens)
mx.eval(logits0)
next_tok = int(mx.argmax(logits0[0]))
print(f'step 0 (prefill): argmax={next_tok}')

prefix = list(prompt)
all_pass = True

for step in range(4):
    prefix.append(next_tok)
    
    # Full forward on complete prefix
    full = mx.array([prefix], dtype=mx.int32)
    fl = model.forward(full)
    mx.eval(fl)
    full_argmax = int(mx.argmax(fl[0, -1, :]))
    
    # Incremental decode step
    sl = gen.decode_step(mx.array([next_tok], dtype=mx.int32), pos=len(prefix)-1)
    mx.eval(sl)
    dec_argmax = int(mx.argmax(sl[0]))
    
    diff = mx.abs(sl[0].astype(mx.float32) - fl[0, -1, :].astype(mx.float32))
    max_diff = float(mx.max(diff))
    mean_diff = float(mx.mean(diff))
    finite = bool(mx.all(mx.isfinite(sl[0])))
    
    ok = finite and (max_diff < 1.0) and (dec_argmax == full_argmax)
    status = 'PASS' if ok else 'FAIL'
    print(f'step {step+1}: max_diff={max_diff:.4f}, mean={mean_diff:.6f}, dec_arg={dec_argmax}, full_arg={full_argmax}, finite={finite} [{status}]')
    
    if not ok:
        all_pass = False
    next_tok = dec_argmax

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'\nPeak RSS={rss:.0f}MB')
print('\nSSD stats:')
for L in [1,5,9,13,17,21]:
    s = model.ssd_lookups[L].stats
    print(f'  L{L}: lookups={s.lookups} hits={s.hits} bytes={s.bytes_loaded}')

print(f'\n{"ALL PASS" if all_pass else "FAIL"}')
sys.exit(0 if all_pass else 1)
