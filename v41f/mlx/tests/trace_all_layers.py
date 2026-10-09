"""Full per-layer trace: compare full forward vs incremental decode.
Saves post-layer hidden state for all 24 layers, then compares."""
import sys
sys.path.insert(0, '.')
import mlx.core as mx
from v41f.mlx.weights_manifest import load_from_manifest
from v41f.mlx.config import MLXV42Config
from v41f.mlx.model import (MLXV42Model, _rms_norm, hc_mixes, hc_pre, hc_post,
                             apply_rope_real, sparse_attention, moe_forward,
                             engram_forward_with_embed)
from v41f.mlx.engram_hash import MLXNgramHash
from v41f.mlx.engram_ssd import EngramSSDLookup
from v41f.mlx.generator import MLXGenerator
from tokenizers import Tokenizer

print('Loading...')
W, v42 = load_from_manifest()
mcfg = MLXV42Config.from_v42_cfg(v42)
model = MLXV42Model(mcfg, W)
tok = Tokenizer.from_file('ckpt_local/tok/tokenizer.json')
model.engram_hash = MLXNgramHash(tok, layer_ids=tuple(v42['engram_layer_ids']),
                                 max_ngram_size=v42['engram_max_ngram_size'],
                                 n_heads=v42['engram_n_heads'],
                                 engram_vocab_size=v42['engram_compressed_vocab_size'],
                                 pad_id=v42.get('engram_pad_id', 2))
engram_sizes = {1:786862, 5:788118, 9:789492, 13:791110, 17:792776, 21:794672}
for L in v42['engram_layer_ids']:
    model.ssd_lookups[L] = EngramSSDLookup(f'ckpt_local/engram_ssd/embed_L{L}.bin', engram_sizes[L], head_dim=128, cache_rows=4096)

cfg = model.cfg
prompt = [1, 5, 9]
next_tok = 13
full_prompt = [1, 5, 9, 13]

# === Full forward: save post-layer hidden states ===
print('\n=== Full forward [1,5,9,13] ===')
tokens = mx.array([full_prompt], dtype=mx.int32)
b, s = tokens.shape

h = mx.take(W['embed']['weight'].astype(mx.bfloat16), tokens.astype(mx.int32), axis=0)
h = mx.broadcast_to(h[:, :, None, :], (b, s, cfg.hc_mult, cfg.dim))
pre_mix = mx.concatenate([mx.ones((b, s, 1), dtype=mx.float32),
                           mx.zeros((b, s, cfg.hc_mult - 1), dtype=mx.float32)], axis=-1)

attn_cos = model.attn_cos[:s][None]
attn_sin = model.attn_sin[:s][None]
win_idx = model._window_idx(s)

full_layer_out = []

for L in range(cfg.n_layers):
    WL = W['layers'][L]
    
    if L in W['engrams']:
        hash_ids = model._engram_hash(L, tokens)
        eW = W['engrams'][L]
        if L in model.ssd_lookups:
            ssd_emb = model.ssd_lookups[L].lookup(hash_ids)
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
    
    o = sparse_attention(q, kv, WL['attn_sink'], win_idx, None,
                          cfg.head_dim ** -0.5, cfg.attn_logit_softcap)
    o = apply_rope_real(o, attn_cos[0], attn_sin[0], inverse=True)
    a_out = model._oproj(o, WL)
    h = hc_post(a_out, residual, attn_post, attn_comb)
    
    residual = h
    ffn_pre, ffn_post, ffn_comb = hc_mixes(
        h, WL['hc']['ffn_fn'], WL['hc']['ffn_scale'], WL['hc']['ffn_base'],
        cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
    f = hc_pre(h, attn_pre)
    f = _rms_norm(f, WL['ffn_norm']['weight'], cfg.norm_eps)
    f = moe_forward(f, WL['ffn'], cfg)
    h = hc_post(f, residual, ffn_post, ffn_comb)
    pre_mix = ffn_pre
    
    mx.eval(h)
    full_layer_out.append(h[0, -1].astype(mx.float32))  # last position

# === Incremental decode: save post-layer hidden states ===
print('=== Incremental decode (prefill [1,5,9] + step 13) ===')
gen = MLXGenerator(model)
pl = gen.prefill(mx.array([prompt], dtype=mx.int32))
mx.eval(pl)

token = mx.array([next_tok], dtype=mx.int32)
pos = len(prompt)
b = 1

h = mx.take(W['embed']['weight'].astype(mx.bfloat16), token.astype(mx.int32), axis=0)
h = mx.broadcast_to(h[:, None, None, :], (b, 1, cfg.hc_mult, cfg.dim))
pre_mix = mx.concatenate([mx.ones((b, 1, 1)), mx.zeros((b, 1, cfg.hc_mult - 1))], axis=-1)

attn_cos_d = model.attn_cos[pos:pos+1][None]
attn_sin_d = model.attn_sin[pos:pos+1][None]

dec_layer_out = []

for L in range(cfg.n_layers):
    WL = W['layers'][L]
    
    if L in W['engrams']:
        toks = mx.array([prompt + [next_tok]], dtype=mx.int32)
        hash_ids = model._engram_hash(L, toks)
        hash_ids_last = hash_ids[:, -1:]
        eW = W['engrams'][L]
        if L in model.ssd_lookups:
            ssd_emb = model.ssd_lookups[L].lookup(hash_ids_last)
            h = engram_forward_with_embed(h, hash_ids_last, ssd_emb, eW, cfg)
    
    residual = h
    attn_pre, attn_post, attn_comb = hc_mixes(
        h, WL['hc']['attn_fn'], WL['hc']['attn_scale'], WL['hc']['attn_base'],
        cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
    a = hc_pre(h, pre_mix)
    a = _rms_norm(a, WL['attn_norm']['weight'], cfg.norm_eps)
    
    q, qr = model._qproj(a, WL)
    q = apply_rope_real(q, attn_cos_d[0], attn_sin_d[0])
    kv = model._kvproj(a, WL)
    kv = apply_rope_real(kv, attn_cos_d[0], attn_sin_d[0])
    
    # KV cache: append new kv
    gen._kv_cache[L] = gen._kv_cache[L].at[:, pos:pos+1].add(kv)
    cached_kv = gen._kv_cache[L][:, :pos+1, :]  # [1, pos+1, head_dim] (MQA)
    
    # Window for current position
    w = cfg.window_size
    lo = max(0, pos - w + 1)
    win_idx_d = mx.arange(lo, lo + w, dtype=mx.int32)[None, None, :]
    win_idx_d = mx.where(win_idx_d > pos, -1, win_idx_d)
    
    o = sparse_attention(q, cached_kv, WL['attn_sink'], win_idx_d, None,
                          cfg.head_dim ** -0.5, cfg.attn_logit_softcap)
    o = apply_rope_real(o, attn_cos_d[0], attn_sin_d[0], inverse=True)
    a_out = model._oproj(o, WL)
    h = hc_post(a_out, residual, attn_post, attn_comb)
    
    residual = h
    ffn_pre, ffn_post, ffn_comb = hc_mixes(
        h, WL['hc']['ffn_fn'], WL['hc']['ffn_scale'], WL['hc']['ffn_base'],
        cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.norm_eps)
    f = hc_pre(h, attn_pre)
    f = _rms_norm(f, WL['ffn_norm']['weight'], cfg.norm_eps)
    f = moe_forward(f, WL['ffn'], cfg)
    h = hc_post(f, residual, ffn_post, ffn_comb)
    pre_mix = ffn_pre
    
    mx.eval(h)
    dec_layer_out.append(h[0, 0].astype(mx.float32))  # single position

# === Compare layer by layer ===
print('\n=== Per-layer comparison (post-layer hidden state) ===')
print(f'{"Layer":>6} {"max_diff":>12} {"mean_diff":>12} {"dec_max":>12} {"full_max":>12} {"rel%":>8}')
first_diverge = None
for L in range(cfg.n_layers):
    diff = mx.abs(dec_layer_out[L] - full_layer_out[L])
    md = float(mx.max(diff))
    mn = float(mx.mean(diff))
    dm = float(mx.max(mx.abs(dec_layer_out[L])))
    fm = float(mx.max(mx.abs(full_layer_out[L])))
    rel = md / (fm + 1e-8) * 100
    marker = ''
    if md > 0.01 and first_diverge is None:
        first_diverge = L
        marker = ' <-- FIRST DIVERGE'
    print(f'{L:>6} {md:>12.4f} {mn:>12.6f} {dm:>12.2f} {fm:>12.2f} {rel:>7.2f}%{marker}')

print(f'\nFirst divergence at layer: {first_diverge}')
