"""Layer-by-layer trace: incremental decode vs full-prefix forward.
Find the first layer/module where they diverge."""
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

print('Loading...')
W, v42 = load_from_manifest()
mcfg = MLXV42Config.from_v42_cfg(v42)
model = MLXV42Model(mcfg, W)
tok = Tokenizer.from_file('ckpt_local/tok/tokenizer.json')
model.engram_hash = MLXNgramHash(tok, layer_ids=tuple(v42['engram_layer_ids']), max_ngram_size=v42['engram_max_ngram_size'], n_heads=v42['engram_n_heads'], engram_vocab_size=v42['engram_compressed_vocab_size'], pad_id=v42.get('engram_pad_id', 2))
engram_sizes = {1:786862, 5:788118, 9:789492, 13:791110, 17:792776, 21:794672}
for L in v42['engram_layer_ids']:
    model.ssd_lookups[L] = EngramSSDLookup(f'ckpt_local/engram_ssd/embed_L{L}.bin', engram_sizes[L], head_dim=128, cache_rows=4096)

# Use short prefix: [1,5,9] then decode [13]
prompt = [1, 5, 9]
next_tok = 13
full_prefix = [1, 5, 9, 13]

# Warm up
tokens = mx.array([prompt], dtype=mx.int32)
_ = model.forward(tokens); mx.eval(_)
gen = MLXGenerator(model)
_ = gen.prefill(tokens); mx.eval(_)

# Now do full forward on [1,5,9,13] and get hidden state per layer
# We need to modify forward to return intermediate states
# Instead, compare prefill output vs decode output directly
print('\n=== Compare prefill last token vs decode step ===')

# Prefill on full [1,5,9,13]
full_toks = mx.array([full_prefix], dtype=mx.int32)
full_logits = model.forward(full_toks)
mx.eval(full_logits)
full_last = full_logits[0, -1, :]
print(f'Full forward [1,5,9,13]: max={float(mx.max(full_last)):.2f}, argmax={int(mx.argmax(full_last))}')

# Prefill on [1,5,9] then decode_step(13)
gen2 = MLXGenerator(model)
pl = gen2.prefill(mx.array([prompt], dtype=mx.int32))
mx.eval(pl)
sl = gen2.decode_step(mx.array([next_tok], dtype=mx.int32), pos=3)
mx.eval(sl)
print(f'Decode step(13): max={float(mx.max(sl[0])):.2f}, argmax={int(mx.argmax(sl[0]))}')

diff = mx.abs(sl[0].astype(mx.float32) - full_last.astype(mx.float32))
print(f'max diff: {float(mx.max(diff)):.4f}')
print(f'mean diff: {float(mx.mean(diff)):.6f}')

# Check if it's just a scale difference (argmax same but magnitude different)
sl_f = sl[0].astype(mx.float32)
fl_f = full_last.astype(mx.float32)
# Check correlation
sl_norm = sl_f / (mx.norm(sl_f) + 1e-8)
fl_norm = fl_f / (mx.norm(fl_f) + 1e-8)
cos = float(mx.sum(sl_norm * fl_norm))
print(f'cosine similarity: {cos:.6f}')
print(f'sl norm: {float(mx.norm(sl_f)):.2f}, fl norm: {float(mx.norm(fl_f)):.2f}')
print(f'ratio: {float(mx.norm(sl_f)) / (float(mx.norm(fl_f)) + 1e-8):.4f}')
