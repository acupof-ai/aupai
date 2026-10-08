"""DEPRECATED: Load SFT from sft_fp8 (old path).
Use weights_manifest.load_from_manifest() for sft_mlx_v2 instead.
Kept for reference only."""
import resource, time, sys
sys.path.insert(0, '/Users/bytedance/code/aupai')
import numpy as np, torch
import mlx.core as mx

t0 = time.perf_counter()
d = 'ckpt_local/sft_fp8'

# ---- Top weights ----
embed = mx.array(np.fromfile(f'{d}/embed_weight.bin', dtype=np.uint16).reshape(32768, 1024), dtype=mx.bfloat16)
norm = mx.array(np.fromfile(f'{d}/norm_weight.bin', dtype=np.uint16).reshape(1024), dtype=mx.bfloat16)
# Head is fp32 in ckpt, load directly (not from wrong bf16 .bin)
ck_tmp = torch.load('ckpt_local/ckpt_v42_sft_run.pt', map_location='cpu', mmap=True, weights_only=False)
head = mx.array(ck_tmp['model']['head.weight'].numpy(), dtype=mx.float32)
del ck_tmp
mx.eval(embed, norm, head)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'[1/4] top weights: RSS={rss:.0f}MB, {time.perf_counter()-t0:.1f}s')

# ---- Load all weights from SFT ckpt via mmap ----
# We load directly from the ckpt (not .bin) for non-expert small weights,
# and use .bin for experts (already quantized)
ck = torch.load('ckpt_local/ckpt_v42_sft_run.pt', map_location='cpu', mmap=True, weights_only=False)
sd = ck['model']
v42 = ck['cfg']['v42_cfg']
n_layers = v42['n_layers']

def u8_fp8(t):
    return mx.array(t.contiguous().to(torch.float8_e4m3fn).view(torch.uint8).numpy(), dtype=mx.uint8)

def u16_bf16(t):
    return mx.array(t.contiguous().view(torch.uint16).numpy(), dtype=mx.bfloat16)

def f32(t):
    return mx.array(t.contiguous().float().numpy())

W = {'embed': {'weight': embed}, 'norm': {'weight': norm}, 'head': {'weight': head},
     'layers': [], 'engrams': {}}

t1 = time.perf_counter()
for L in range(n_layers):
    WL = {}
    # Attention projections (bf16 -> fp8)
    WL['qproj'] = {
        'wq_a': {'weight': u8_fp8(sd[f'layers.{L}.attn.qproj.wq_a.weight'])},
        'q_norm': {'weight': u16_bf16(sd[f'layers.{L}.attn.qproj.q_norm.weight'])},
        'wq_b': {'weight': u8_fp8(sd[f'layers.{L}.attn.qproj.wq_b.weight'])},
    }
    WL['kvproj'] = {
        'wkv': {'weight': u8_fp8(sd[f'layers.{L}.attn.kvproj.wkv.weight'])},
        'kv_norm': {'weight': u16_bf16(sd[f'layers.{L}.attn.kvproj.kv_norm.weight'])},
    }
    WL['oproj'] = {
        'wo_a': u8_fp8(sd[f'layers.{L}.attn.oproj.wo_a']),
        'wo_b': {'weight': u8_fp8(sd[f'layers.{L}.attn.oproj.wo_b.weight'])},
    }
    WL['attn_sink'] = f32(sd[f'layers.{L}.attn.attn_sink'])
    WL['attn_norm'] = {'weight': u16_bf16(sd[f'layers.{L}.attn_norm.weight'])}

    # Compressor
    cw = f'layers.{L}.attn.compressor.wkv.weight'
    if cw in sd:
        comp = {'wkv': {'weight': u8_fp8(sd[cw])}}
        cg = f'layers.{L}.attn.compressor.wgate.weight'
        if cg in sd:
            comp['wgate'] = {'weight': u8_fp8(sd[cg])}
        comp['norm'] = {'weight': u16_bf16(sd[f'layers.{L}.attn.compressor.norm.weight'])}
        WL['compressor'] = comp

    # Index key
    ik = f'layers.{L}.attn.index_key.wk.weight'
    if ik in sd:
        WL['index_key'] = {
            'wk': {'weight': u8_fp8(sd[ik])},
            'k_norm': {'weight': u16_bf16(sd[f'layers.{L}.attn.index_key.k_norm.weight'])},
        }

    # Indexer
    iw = f'layers.{L}.attn.indexer.wq_b.weight'
    if iw in sd:
        WL['indexer'] = {
            'wq_b': {'weight': u8_fp8(sd[iw])},
            'weights_proj': {'weight': u8_fp8(sd[f'layers.{L}.attn.indexer.weights_proj.weight'])},
        }

    # MoE: experts from .bin (pre-quantized fp8)
    WL['ffn'] = {
        'w1': mx.array(np.memmap(f'{d}/L{L}_w1.bin', dtype=np.uint8, mode='r').reshape(64,640,1024), dtype=mx.uint8),
        'w3': mx.array(np.memmap(f'{d}/L{L}_w3.bin', dtype=np.uint8, mode='r').reshape(64,640,1024), dtype=mx.uint8),
        'w2': mx.array(np.memmap(f'{d}/L{L}_w2.bin', dtype=np.uint8, mode='r').reshape(64,1024,640), dtype=mx.uint8),
        'gate': {
            'weight': f32(sd[f'layers.{L}.ffn.gate.weight']),
            'bias': f32(sd[f'layers.{L}.ffn.gate.bias']),
        },
        'shared': {
            'w1': {'weight': u8_fp8(sd[f'layers.{L}.ffn.shared_experts.w1.weight'])},
            'w3': {'weight': u8_fp8(sd[f'layers.{L}.ffn.shared_experts.w3.weight'])},
            'w2': {'weight': u8_fp8(sd[f'layers.{L}.ffn.shared_experts.w2.weight'])},
        },
    }
    WL['ffn_norm'] = {'weight': u16_bf16(sd[f'layers.{L}.ffn_norm.weight'])}

    # HyperConn (fp32 -> bf16)
    WL['hc'] = {
        'attn_fn': mx.array(sd[f'layers.{L}.hc.hc_attn_fn'].bfloat16().contiguous().view(torch.uint16).numpy(), dtype=mx.bfloat16),
        'ffn_fn': mx.array(sd[f'layers.{L}.hc.hc_ffn_fn'].bfloat16().contiguous().view(torch.uint16).numpy(), dtype=mx.bfloat16),
        'attn_base': f32(sd[f'layers.{L}.hc.hc_attn_base']),
        'ffn_base': f32(sd[f'layers.{L}.hc.hc_ffn_base']),
        'attn_scale': f32(sd[f'layers.{L}.hc.hc_attn_scale']),
        'ffn_scale': f32(sd[f'layers.{L}.hc.hc_ffn_scale']),
    }
    W['layers'].append(WL)
    if L % 6 == 5:
        mx.eval(*[v for layer in W['layers'] for v in layer.values() if isinstance(v, mx.array)])
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(f'  layer {L}: RSS={rss:.0f}MB, {time.perf_counter()-t1:.1f}s')

# Engram small weights (embed via SSD, not whole-table)
engram_sizes = {1:786862, 5:788118, 9:789492, 13:791110, 17:792776, 21:794672}
for L in v42['engram_layer_ids']:
    e = {
        'wkv': {'weight': u16_bf16(sd[f'engrams.{L}.wkv.weight'])},
        'q_weight': u16_bf16(sd[f'engrams.{L}.q_weight']),
        'k_weight': u16_bf16(sd[f'engrams.{L}.k_weight']),
    }
    W['engrams'][L] = e

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'[2/4] all {n_layers} layers loaded: RSS={rss:.0f}MB, {time.perf_counter()-t1:.1f}s')

# ---- Build model and attach real Engram hash + SSD lookup ----
from v41f.mlx.config import MLXV42Config
from v41f.mlx.model import MLXV42Model
from v41f.mlx.engram_hash import MLXNgramHash
from v41f.mlx.engram_ssd import EngramSSDLookup
from tokenizers import Tokenizer

mcfg = MLXV42Config.from_v42_cfg(v42)
model = MLXV42Model(mcfg, W)

# Real n-gram hash from tokenizer
tok = Tokenizer.from_file('ckpt_local/tok/tokenizer.json')
model.engram_hash = MLXNgramHash(
    tok, layer_ids=tuple(v42['engram_layer_ids']),
    max_ngram_size=v42['engram_max_ngram_size'],
    n_heads=v42['engram_n_heads'],
    engram_vocab_size=v42['engram_compressed_vocab_size'],
    pad_id=v42.get('engram_pad_id', 2),
)

# SSD on-demand lookup (LRU cache, no whole-table load)
engram_sizes = {1:786862, 5:788118, 9:789492, 13:791110, 17:792776, 21:794672}
for L in v42['engram_layer_ids']:
    model.ssd_lookups[L] = EngramSSDLookup(
        f'ckpt_local/engram_ssd/embed_L{L}.bin',
        engram_sizes[L], head_dim=128, cache_rows=4096,
    )

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f'[3/4] model built + Engram hash/SSD attached: RSS={rss:.0f}MB')

# Real prefill
tokens = mx.array([[1, 5, 9, 13]], dtype=mx.int32)
t2 = time.perf_counter()
logits = model.forward(tokens)
mx.eval(logits)
ttft = time.perf_counter() - t2
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
lg = np.array(logits[0, -1].astype(mx.float32))
print(f'[4/4] prefill 4 tokens: TTFT={ttft*1000:.1f}ms ({4/ttft:.1f} tok/s)')
print(f'  argmax={lg.argmax()}, top5={lg.argsort()[-5:][::-1]}')
print(f'  peak RSS={rss:.0f}MB')
print(f'  total time={time.perf_counter()-t0:.1f}s')
