"""Layer-by-layer trace: compare full forward vs incremental decode at each layer.
Saves pre/post attention and FFN states for comparison."""
import sys
sys.path.insert(0, '.')
import mlx.core as mx
from v41f.mlx.weights_manifest import load_from_manifest
from v41f.mlx.config import MLXV42Config
from v41f.mlx.model import MLXV42Model, _rms_norm, hc_mixes, hc_pre, hc_post, apply_rope_real, sparse_attention
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

prompt = [1, 5, 9]
next_tok = 13
full_prompt = [1, 5, 9, 13]
cfg = model.cfg

# === Full forward: trace layer by layer ===
print('\n=== Full forward trace [1,5,9,13] ===')
tokens = mx.array([full_prompt], dtype=mx.int32)
b, s = tokens.shape

h = mx.take(W['embed']['weight'].astype(mx.bfloat16), tokens.astype(mx.int32), axis=0)
h = mx.broadcast_to(h[:, :, None, :], (b, s, cfg.hc_mult, cfg.dim))
pre_mix = mx.concatenate([mx.ones((b, s, 1), dtype=mx.float32), mx.zeros((b, s, cfg.hc_mult - 1), dtype=mx.float32)], axis=-1)

attn_cos = model.attn_cos[:s][None]
attn_sin = model.attn_sin[:s][None]

full_states = {}

for L in range(cfg.n_layers):
    WL = W['layers'][L]
    
    # Engram
    if L in W['engrams']:
        hash_ids = model._engram_hash(L, tokens)
        eW = W['engrams'][L]
        if L in model.ssd_lookups:
            ssd_emb = model.ssd_lookups[L].lookup(hash_ids)
            from v41f.mlx.model import engram_forward_with_embed
            h = engram_forward_with_embed(h, hash_ids, ssd_emb, eW, cfg)
    
    residual = h
    attn_pre, attn_post, attn_comb = hc_mixes(
        h, WL['hc']['attn_fn'], WL['hc']['attn_scale'], WL['hc']['attn_base'],
        cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
    a = hc_pre(h, pre_mix)
    a = _rms_norm(a, WL['attn_norm']['weight'], cfg.norm_eps)
    
    q, qr = model._qproj(a, WL)
    q = apply_rope_real(q, attn_cos[0], attn_sin[0])
    kv = model._kvproj(a, WL)
    kv = apply_rope_real(kv, attn_cos[0], attn_sin[0])
    
    # Save layer 0 attention inputs
    if L == 0:
        full_states['q_L0'] = q
        full_states['kv_L0'] = kv
        full_states['a_L0'] = a
        full_states['h_in_L0'] = h
    
    # Window attention
    win_idx = model._window_idx(s)
    o = sparse_attention(q, kv, WL['attn_sink'], win_idx, None,
                          cfg.head_dim ** -0.5, cfg.attn_logit_softcap)
    o = apply_rope_real(o, attn_cos[0], attn_sin[0], inverse=True)
    a_out = model._oproj(o, WL)
    h = hc_post(a_out, residual, attn_post, attn_comb)
    
    if L == 0:
        full_states['h_attn_L0'] = h
    
    # FFN
    residual = h
    ffn_pre, ffn_post, ffn_comb = hc_mixes(
        h, WL['hc']['ffn_fn'], WL['hc']['ffn_scale'], WL['hc']['ffn_base'],
        cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
    f = hc_pre(h, attn_pre)
    f = _rms_norm(f, WL['ffn_norm']['weight'], cfg.norm_eps)
    from v41f.mlx.model import moe_forward
    f = moe_forward(f, WL['ffn'], cfg)
    h = hc_post(f, residual, ffn_post, ffn_comb)
    pre_mix = ffn_pre
    
    if L == 0:
        full_states['h_out_L0'] = h
        print(f'Layer 0: h_in max={float(mx.max(full_states["h_in_L0"])):.2f}')
        print(f'  q max={float(mx.max(q)):.4f}, kv max={float(mx.max(kv)):.4f}')
        print(f'  h_attn max={float(mx.max(full_states["h_attn_L0"])):.2f}')
        print(f'  h_out max={float(mx.max(h)):.2f}')

# === Incremental decode: trace layer by layer ===
print('\n=== Incremental decode trace (after prefill [1,5,9]) ===')
from v41f.mlx.generator import MLXGenerator
gen = MLXGenerator(model)
pl = gen.prefill(mx.array([prompt], dtype=mx.int32))
mx.eval(pl)

# Now manually do decode_step(13) and trace layer 0
token = mx.array([next_tok], dtype=mx.int32)
pos = 3
b = 1

h = mx.take(W['embed']['weight'].astype(mx.bfloat16), token.astype(mx.int32), axis=0)
h = mx.broadcast_to(h[:, None, None, :], (b, 1, cfg.hc_mult, cfg.dim))
pre_mix_dec = mx.concatenate([mx.ones((b, 1, 1)), mx.zeros((b, 1, cfg.hc_mult - 1))], axis=-1)

attn_cos_d = model.attn_cos[pos:pos+1][None]
attn_sin_d = model.attn_sin[pos:pos+1][None]

WL = W['layers'][0]

# Engram
if 0 in W['engrams']:
    pass  # layer 0 has no engram

residual = h
attn_pre, attn_post, attn_comb = hc_mixes(
    h, WL['hc']['attn_fn'], WL['hc']['attn_scale'], WL['hc']['attn_base'],
    cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
a = hc_pre(h, pre_mix_dec)
a = _rms_norm(a, WL['attn_norm']['weight'], cfg.norm_eps)

q, qr = model._qproj(a, WL)
q = apply_rope_real(q, attn_cos_d[0], attn_sin_d[0])
kv = model._kvproj(a, WL)
kv = apply_rope_real(kv, attn_cos_d[0], attn_sin_d[0])

print(f'Decode layer 0: h_in max={float(mx.max(h)):.2f}')
print(f'  q max={float(mx.max(q)):.4f}, kv max={float(mx.max(kv)):.4f}')

# Compare with full forward last position
full_q_last = full_states['q_L0'][0, -1, :, :]  # [h, d]
full_kv_last = full_states['kv_L0'][0, -1, :]  # [d]
dec_q = q[0, 0, :, :]  # [h, d]
dec_kv = kv[0, 0, :]  # [d]

q_diff = mx.abs(dec_q.astype(mx.float32) - full_q_last.astype(mx.float32))
kv_diff = mx.abs(dec_kv.astype(mx.float32) - full_kv_last.astype(mx.float32))
print(f'\nLayer 0 Q diff: max={float(mx.max(q_diff)):.6f}, mean={float(mx.mean(q_diff)):.6f}')
print(f'Layer 0 KV diff: max={float(mx.max(kv_diff)):.6f}, mean={float(mx.mean(kv_diff)):.6f}')

# KV cache lookup
cached_kv = gen._kv_cache[0][:, :pos+1, :]  # [b, pos+1, d]
print(f'\nKV cache shape: {cached_kv.shape}')
print(f'Cached KV last: max={float(mx.max(cached_kv[0, -1])):.4f}')
print(f'New KV: max={float(mx.max(kv[0, 0])):.4f}')
cache_diff = mx.abs(cached_kv[0, -1].astype(mx.float32) - kv[0, 0].astype(mx.float32))
print(f'KV cache vs new KV diff: max={float(mx.max(cache_diff)):.6f}')
