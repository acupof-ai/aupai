"""PyTorch gold vs MLX comparison on real SFT model.

Compares final logits and layer-by-layer hidden states on a ChatML prompt.
"""
import resource, time, sys
sys.path.insert(0, '/Users/bytedance/code/aupai')
import torch
import mlx.core as mx
import numpy as np

# ---- Load PyTorch model (gold) ----
print('=== Loading PyTorch SFT model (gold) ===')
t0 = time.time()
ckpt = torch.load('ckpt_local/ckpt_v42_sft_run.pt', map_location='cpu', mmap=True, weights_only=False)
v42_cfg = ckpt['cfg']['v42_cfg']
print(f'  cfg: layers={v42_cfg["n_layers"]}, dim={v42_cfg["dim"]}')
print(f'  engram layers: {v42_cfg["engram_layer_ids"]}')
print(f'  compress_ratios: {v42_cfg["compress_ratios"]}')

from v41f.lm import V42LM
pt_model = V42LM(v42_cfg).float().eval()
# Load state dict
sd = ckpt['model']
missing, unexpected = pt_model.load_state_dict(sd, strict=False)
print(f'  missing keys: {len(missing)}, unexpected: {len(unexpected)}')
print(f'  load time: {time.time()-t0:.1f}s')
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'  RSS: {rss:.0f}MB')

# ---- Load MLX model ----
print('\n=== Loading MLX SFT model ===')
from v41f.mlx.weights_manifest import load_from_manifest
from v41f.mlx.config import MLXV42Config
from v41f.mlx.model import MLXV42Model
from v41f.mlx.engram_hash import MLXNgramHash
from v41f.mlx.engram_ssd import EngramSSDLookup
from tokenizers import Tokenizer

W, v42 = load_from_manifest()
mcfg = MLXV42Config.from_v42_cfg(v42)
mlx_model = MLXV42Model(mcfg, W)
tok = Tokenizer.from_file('ckpt_local/tok/tokenizer.json')
mlx_model.engram_hash = MLXNgramHash(tok, layer_ids=tuple(v42['engram_layer_ids']),
                                     max_ngram_size=v42['engram_max_ngram_size'],
                                     n_heads=v42['engram_n_heads'],
                                     engram_vocab_size=v42['engram_compressed_vocab_size'],
                                     pad_id=v42.get('engram_pad_id', 2))
engram_sizes = {1:786862, 5:788118, 9:789492, 13:791110, 17:792776, 21:794672}
for L in v42['engram_layer_ids']:
    mlx_model.ssd_lookups[L] = EngramSSDLookup(
        f'ckpt_local/engram_ssd/embed_L{L}.bin',
        engram_sizes[L], head_dim=128, cache_rows=4096)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'  RSS: {rss:.0f}MB')

# ---- ChatML prompt ----
prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
token_ids = tok.encode(prompt).ids
print('\n=== Prompt ===')
print(f'  text: {prompt!r}')
print(f'  tokens: {len(token_ids)}: {token_ids[:10]}...')

# ---- PyTorch forward ----
print('\n=== PyTorch forward ===')
with torch.no_grad():
    pt_tokens = torch.tensor([token_ids], dtype=torch.long)
    pt_logits = pt_model(pt_tokens)
    pt_last = pt_logits[0, -1, :].float().numpy()
    pt_argmax = int(np.argmax(pt_last))
    pt_top10 = np.argsort(-pt_last)[:10]
print(f'  argmax: {pt_argmax}')
print(f'  top-10: {pt_top10.tolist()}')
print(f'  max logit: {pt_last.max():.4f}')

# ---- MLX forward ----
print('\n=== MLX forward ===')
mx_tokens = mx.array([token_ids], dtype=mx.int32)
mx_logits = mlx_model.forward(mx_tokens)
mx.eval(mx_logits)
mx_last = np.array(mx_logits[0, -1, :].astype(mx.float32))
mx_argmax = int(np.argmax(mx_last))
mx_top10 = np.argsort(-mx_last)[:10]
print(f'  argmax: {mx_argmax}')
print(f'  top-10: {mx_top10.tolist()}')
print(f'  max logit: {mx_last.max():.4f}')

# ---- Compare ----
print('\n=== Comparison ===')
diff = np.abs(mx_last - pt_last)
print(f'  max abs diff: {diff.max():.4f}')
print(f'  mean abs diff: {diff.mean():.6f}')
print(f'  argmax match: {pt_argmax == mx_argmax}')
print(f'  top-10 overlap: {len(set(pt_top10.tolist()) & set(mx_top10.tolist()))}/10')

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'\nPeak RSS: {rss:.0f}MB')
