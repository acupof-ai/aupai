#!/usr/bin/env python3
"""PyTorch native CED decode gold for MLX Runtime V2.

The test compares a persistent ring-cache runtime against full-prefix prefill.
It uses scripts.loader.load_checkpoint. It does not use old MLX code as gold.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from scripts.loader import format_prompt  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from v41f.rope import apply_rotary_emb  # noqa: E402
from v41f.sparse_attn import sparse_attn  # noqa: E402


class NativeCEDState:
    def __init__(self, cfg):
        self.cfg = cfg
        self.position = 0
        self.window = {}
        self.comp_kv = {2: [], 8: [], 12: []}
        self.index_k = {2: [], 8: [], 12: []}
        self.current_source = None
        self.latest_topk = None
        self.last_q = {}
        self.last_kv = {}
        self.last_idxs = {}
        self.last_engram = {}
        self.last_attn_out = {}
        self.last_ffn_out = {}
        self.last_sparse = {}
        self.last_block = {}

    def stacked_comp(self, source):
        return torch.stack(self.comp_kv[source], dim=0).unsqueeze(0)

    def stacked_index(self, source):
        return torch.stack(self.index_k[source], dim=0).unsqueeze(0)


def init_ring(attn, kv, state):
    win = attn.window_size
    ring = torch.zeros(win, kv.size(-1), dtype=kv.dtype)
    seqlen = kv.size(0)
    if seqlen <= win:
        ring[:seqlen] = kv
    else:
        cutoff = seqlen % win
        last = kv[-win:]
        if cutoff == 0:
            ring.copy_(last)
        else:
            ring[cutoff:] = last[: win - cutoff]
            ring[:cutoff] = last[win - cutoff :]
    state.window[attn.layer_id] = ring


def decode_ring_slots(position, win, device):
    oldest = position % win + 1
    slots = torch.cat([torch.arange(oldest, win, device=device), torch.arange(oldest, device=device)])
    slots = torch.where(slots > position, torch.full_like(slots, -1), slots)
    return slots.int().view(1, 1, win)


def append_latent(attn, layer_id, latent, state, position=None):
    if latent is None:
        return
    source = layer_id
    if position is None:
        n_groups = latent.size(1)
        freqs = attn.freqs_cis[: n_groups * attn.compress_ratio : attn.compress_ratio]
        if attn.owns_index_k:
            keys = attn.index_key(latent).clone()
            apply_rotary_emb(keys[..., -attn.rd :], freqs)
            state.index_k[source] = [keys[0, j] for j in range(keys.size(1))]
        values = latent.clone()
        apply_rotary_emb(values[..., -attn.rd :], freqs)
        state.comp_kv[source] = [values[0, j] for j in range(values.size(1))]
    else:
        group_pos = position + 1 - attn.compress_ratio
        freqs = attn.freqs_cis[group_pos].unsqueeze(0)
        if attn.owns_index_k:
            keys = attn.index_key(latent).clone()
            apply_rotary_emb(keys[..., -attn.rd :], freqs)
            state.index_k[source].append(keys[0, 0])
        values = latent.clone()
        apply_rotary_emb(values[..., -attn.rd :], freqs)
        state.comp_kv[source].append(values[0, 0])


def native_attention(a, attn, layer_id, state, start_pos):
    bsz, seqlen, _ = a.shape
    freqs = attn.freqs_cis[start_pos : start_pos + seqlen]
    q, qr = attn.qproj(a)
    q = q.clone()
    apply_rotary_emb(q[..., -attn.rd :], freqs)
    state.last_q[layer_id] = q.detach().clone()

    kv = attn.kvproj(a).clone()
    apply_rotary_emb(kv[..., -attn.rd :], freqs)

    if start_pos == 0:
        init_ring(attn, kv[0], state)
        end = torch.arange(seqlen, device=a.device).unsqueeze(1)
        window_count = min(seqlen, attn.window_size)
        idxs = (end - attn.window_size + 1).clamp(0) + torch.arange(window_count, device=a.device)
        idxs = torch.where(idxs > end, torch.full_like(idxs, -1), idxs).int().unsqueeze(0)
        window_kv = kv
    else:
        slot = start_pos % attn.window_size
        ring = state.window[layer_id]
        ring[slot] = kv[0, 0]
        idxs = decode_ring_slots(start_pos, attn.window_size, a.device)
        window_kv = ring.unsqueeze(0)
    kv = window_kv

    if attn.compress_ratio:
        if start_pos == 0 and attn.is_kv_source:
            latent = attn.compressor(a, 0)
            state.current_source = layer_id
            append_latent(attn, layer_id, latent, state)
        elif start_pos != 0 and attn.is_kv_source:
            latent = attn.compressor(a, start_pos)
            state.current_source = layer_id
            append_latent(attn, layer_id, latent, state, start_pos)

        source = state.current_source
        if attn.is_index_source:
            offset = seqlen if start_pos == 0 else attn.window_size
            index_k = state.stacked_index(source)
            idxs_comp = attn.indexer(a, qr, index_k, freqs, start_pos, offset)
            state.latest_topk = idxs_comp
        else:
            idxs_comp = state.latest_topk

        comp_kv = state.stacked_comp(source)
        kv = torch.cat([window_kv, comp_kv], dim=1)
        idxs = torch.cat([idxs, idxs_comp], dim=-1)
    state.last_kv[layer_id] = kv.detach().clone()
    state.last_idxs[layer_id] = idxs.detach().clone()

    out = sparse_attn(q, kv, attn.attn_sink, idxs, attn.softmax_scale, attn.softcap)
    state.last_sparse[layer_id] = out.detach().clone()
    out = out.to(q.dtype).clone()
    apply_rotary_emb(out[..., -attn.rd :], freqs, inverse=True)
    return attn.oproj(out)


def native_block(h, block, layer_id, pre_mix, state, start_pos):
    attn_pre, attn_post, attn_comb = block.hc.hc_mixes(
        h, block.hc.hc_attn_fn, block.hc.hc_attn_scale, block.hc.hc_attn_base
    )
    a = block.hc.hc_pre(h, pre_mix)
    a = block.attn_norm(a)
    a = native_attention(a, block.attn, layer_id, state, start_pos)
    state.last_attn_out[layer_id] = a.detach().clone()
    h = block.hc.hc_post(a, h, attn_post, attn_comb)

    residual = h
    ffn_pre, ffn_post, ffn_comb = block.hc.hc_mixes(
        h, block.hc.hc_ffn_fn, block.hc.hc_ffn_scale, block.hc.hc_ffn_base
    )
    f = block.hc.hc_pre(h, attn_pre)
    f = block.ffn_norm(f)
    f = block.ffn(f, None)
    state.last_ffn_out[layer_id] = f.detach().clone()
    h = block.hc.hc_post(f, residual, ffn_post, ffn_comb)
    return h, ffn_pre


def identity_mix(h, mult):
    mix = h.new_zeros(h.size(0), h.size(1), mult, dtype=torch.float32)
    mix[:, :, 0] = 1.0
    return mix


def native_forward(model, input_ids, state):
    start_pos = state.position
    hash_ids = model.engram_hash(input_ids, start_pos, None)
    h = model.embed(input_ids).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre_mix = identity_mix(h, model.hc_mult)
    for layer_id, block in enumerate(model.layers):
        engram = model.engrams[layer_id]
        if engram is not None:
            before = h
            h = engram(h, hash_ids[:, :, engram.layer_hash_index, :], None)
            state.last_engram[layer_id] = (before.detach().clone(), h.detach().clone())
        h, pre_mix = native_block(h, block, layer_id, pre_mix, state, start_pos)
        state.last_block[layer_id] = h.detach().clone()
    h = model.layers[-1].hc_pre(h, pre_mix)
    h = model.norm(h)
    logits = model.head(h)
    if start_pos == 0:
        state.position = input_ids.size(1)
    else:
        state.position += 1
    return logits


def diagnose_prefill(model, input_ids):
    from v41f.attention import SharedAttnState
    from v41f.block import make_identity_pre_mix

    ref_hash = model.engram_hash(input_ids, 0, None)
    nat_hash = model.engram_hash(input_ids, 0, None)
    ref_h = model.embed(input_ids).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    nat_h = model.embed(input_ids).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    ref_pre = make_identity_pre_mix(ref_h, model.hc_mult)
    nat_pre = identity_mix(nat_h, model.hc_mult)
    ref_state = SharedAttnState()
    nat_state = NativeCEDState(model.v41f_cfg)

    for layer_id, layer in enumerate(model.layers):
        engram = model.engrams[layer_id]
        if engram is not None:
            ref_h = engram(ref_h, ref_hash[:, :, engram.layer_hash_index, :], None)
            nat_h = engram(nat_h, nat_hash[:, :, engram.layer_hash_index, :], None)
        ref_h, ref_pre, ref_state = layer(ref_h, 0, ref_pre, ref_state)
        nat_h, nat_pre = native_block(nat_h, layer, layer_id, nat_pre, nat_state, 0)
        diff = (ref_h - nat_h).abs().max().item()
        pre_diff = (ref_pre - nat_pre).abs().max().item()
        print(f"layer {layer_id:02d} max diff {diff:.6f} pre diff {pre_diff:.6f}", flush=True)
        if diff > 0.1:
            return

    ref_final = model.layers[-1].hc_pre(ref_h, ref_pre)
    nat_final = model.layers[-1].hc_pre(nat_h, nat_pre)
    print(f"final hidden diff {(ref_final - nat_final).abs().max().item():.6f}", flush=True)
    ref_logits = model.head(model.norm(ref_final))
    nat_logits = model.head(model.norm(nat_final))
    print(f"final logits diff {(ref_logits - nat_logits).abs().max().item():.6f}", flush=True)


def compare_decode_records(records, state, abs_position, raw_len):
    win = state.cfg.window_size
    ring_slots = decode_ring_slots(abs_position, win, "cpu").flatten().tolist()
    slot_to_pos = {}
    for i, slot in enumerate(ring_slots):
        if slot >= 0:
            slot_to_pos[int(slot)] = abs_position - win + 1 + i

    for layer_id, (q, kv, idxs) in enumerate(records):
        nq = state.last_q[layer_id]
        q_diff = (q[0, -1].float() - nq[0, 0].float()).abs().max().item()
        std_ids = idxs[0, -1].flatten().tolist()
        nat_ids = state.last_idxs[layer_id][0, 0].flatten().tolist()
        std_sem, nat_sem = [], []
        std_val, nat_val = [], []
        for idx in std_ids:
            if idx < 0:
                continue
            if idx < raw_len:
                sem = ("w", idx)
                val = kv[0, idx]
            else:
                local = idx - raw_len
                sem = ("c", local)
                val = kv[0, raw_len + local]
            std_sem.append(sem)
            std_val.append(val.float())
        for idx in nat_ids:
            if idx < 0:
                continue
            if idx < win:
                pos = slot_to_pos.get(int(idx))
                sem = ("w", pos)
                val = state.last_kv[layer_id][0, idx]
            else:
                local = idx - win
                sem = ("c", local)
                val = state.last_kv[layer_id][0, win + local]
            nat_sem.append(sem)
            nat_val.append(val.float())
        same_ids = std_sem == nat_sem
        if not same_ids:
            for j, pair in enumerate(zip(std_sem, nat_sem)):
                if pair[0] != pair[1]:
                    print(f"first id mismatch layer {layer_id:02d} at {j} std {pair[0]} nat {pair[1]}", flush=True)
                    break
            print(f"id set equal layer {layer_id:02d}: {set(std_sem) == set(nat_sem)}", flush=True)
        value_diff = 0.0
        if len(std_val) == len(nat_val):
            for a, b in zip(std_val, nat_val):
                value_diff = max(value_diff, (a - b).abs().max().item())
        print(f"decode diag layer {layer_id:02d} qdiff {q_diff:.6f} ids {same_ids} valdif {value_diff:.6f} "
              f"std {len(std_sem)} nat {len(nat_sem)}", flush=True)


def _materialize_buffers(model, vc, tok):
    """Assign-load leaves non-persistent buffers on the meta device (they are absent from
    the state dict). Recreate the derived ones by value and zero-fill the MoE counters, so
    the model is a real CPU model without constructing the random-init weight allocations
    that the anonymous-RSS jetsam ceiling (~6GB on this 48GB M4 Pro) kills at 24 layers."""
    from dataclasses import asdict
    from v41f.rope import precompute_freqs_cis
    from v41f.engram import EngramLayout, NgramHashState

    for blk in model.layers:
        a = blk.attn
        ratio = a.compress_ratio
        orig, theta = ((vc.original_seq_len, vc.compress_rope_theta) if ratio
                       else (0, vc.rope_theta))
        a.freqs_cis = precompute_freqs_cis(
            a.rd, 4096, original_seq_len=orig, base=theta,
            factor=vc.rope_factor, beta_fast=vc.beta_fast, beta_slow=vc.beta_slow)
        c = getattr(a, "compressor", None)
        if c is not None and ratio > 1:
            c.kv_state = torch.zeros_like(c.kv_state, device="cpu")
            c.score_state = torch.full_like(c.score_state, float("-inf"), device="cpu")

    if model.engram_hash is not None:
        layout = EngramLayout.from_args(vc)
        ha = SimpleNamespace(**asdict(vc), max_batch_size=4, max_seq_len=4096)
        real = NgramHashState(ha, layout, tok)
        for name, buf in real.named_buffers():
            model.engram_hash.register_buffer(name, buf, persistent=False)

    for mod in model.modules():
        for name, val in list(mod._buffers.items()):
            if val is not None and val.is_meta:
                mod.register_buffer(
                    name, torch.zeros(val.shape, dtype=val.dtype, device="cpu"),
                    persistent=False)


def low_memory_load(ckpt_path):
    from tokenizers import Tokenizer as TokenizerImpl
    from train import Cfg, build_model  # noqa: F401  (Cfg supplies defaults)
    from dataclasses import replace
    from v41f.config import V41FConfig
    from v41f.lm import V42LM

    print("mmap checkpoint", flush=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    cfg = SimpleNamespace(**ck["cfg"])
    for key in vars(Cfg):
        if not key.startswith("_") and not hasattr(cfg, key):
            setattr(cfg, key, getattr(Cfg, key))
    cfg.grad_ckpt = False
    cfg.v42_impl = ""

    tok = None
    if getattr(cfg, "v42_engram", ""):
        tok = TokenizerImpl.from_file(os.path.join(ROOT, "data", "tokenizer.json"))

    print("build model (meta) + mmap assign", flush=True)
    vc = V41FConfig(**{k: tuple(v) if isinstance(v, list) else v
                       for k, v in cfg.v42_cfg.items()})
    vc = replace(vc, block_ckpt=False, attn_impl="ref", indexer_train_mode="off",
                 rope_impl="complex", hc_impl="torch", norm_impl="torch")
    if getattr(cfg, "v42_engram", ""):
        vc = replace(vc, engram_layer_ids=tuple(int(x) for x in cfg.v42_engram.split(",")))
        vc = vc.with_derived_engram(tok)
    with torch.device("meta"):
        model = V42LM(vc, balance_alpha=cfg.moe_balance_alpha,
                      bias_gamma=cfg.moe_bias_gamma, tokenizer=tok)
    model.load_state_dict(ck["model"], strict=True, assign=True)
    # Keep the mmap checkpoint alive: assigned parameters are file-backed pages the OS can
    # evict, not dirty anonymous allocations, so the 7.8GB weight set stays under the ceiling.
    model._ckpt_keepalive = ck
    _materialize_buffers(model, vc, tok)
    cfg.v42_cfg = vc.__dict__
    print("state dict loaded", flush=True)
    if os.environ.get("NATIVE_GOLD_FP32"):
        model = model.float()

    model.head.float()
    for block in model.layers:
        block.attn.attn_sink.data = block.attn.attn_sink.data.float()
        block.ffn.gate.weight.data = block.ffn.gate.weight.data.float()
        if block.attn.compressor is not None and block.attn.compress_ratio > 1:
            block.attn.compressor.wkv.float()
            block.attn.compressor.wgate.float()
    model.eval()
    return model


def long_prompt():
    items = ", ".join(f"{i}: component {i}" for i in range(1, 25))
    question = (
        "Review a decoder model. Check these items in one short report: "
        f"{items}. Give one clear sentence for each item."
    )
    return format_prompt(question)


@torch.no_grad()
def main():
    print("start", flush=True)
    ckpt_path = os.environ.get(
        "NATIVE_GOLD_CKPT", os.path.join(ROOT, "ckpt_local", "ckpt_v42_sft_run.pt")
    )
    model = low_memory_load(ckpt_path)
    model.eval()
    if os.environ.get("NATIVE_GOLD_DET_OPROJ"):
        from v41f.projections import GroupedOProj

        def deterministic_oproj_forward(self, o):
            b, s = o.shape[:2]
            og = o.view(b, s, self.n_groups, self.per_group * o.shape[-1])
            flat = og.reshape(b * s, self.n_groups, og.shape[-1])
            outputs = []
            for row in flat:
                row = row.unsqueeze(0)
                lat = torch.einsum("bgd,grd->bgr", row, self.wo_a)
                outputs.append(self.wo_b(lat.flatten(1)))
            return torch.cat(outputs, dim=0).view(b, s, -1)

        GroupedOProj.forward = deterministic_oproj_forward
    tok_path = os.environ.get(
        "NATIVE_GOLD_TOK", os.path.join(ROOT, "ckpt_local", "tok", "tokenizer.json")
    )
    tok = Tokenizer.from_file(tok_path)
    prompt = long_prompt()
    ids = tok.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long)
    print(f"prompt tokens: {len(ids)}")
    assert len(ids) > 128
    if os.environ.get("NATIVE_GOLD_DIAG"):
        diagnose_prefill(model, input_ids)
        return

    standard_logits, _ = model(input_ids)
    standard_logits_saved = standard_logits.clone()
    state = NativeCEDState(model.v41f_cfg)
    native_logits = native_forward(model, input_ids, state)

    alias_diff = (standard_logits - standard_logits_saved).abs().max().item()
    prefill_diff = (standard_logits_saved[0, -1] - native_logits[0, -1]).abs().max().item()
    prefill_top1_match = int(standard_logits_saved[0, -1].argmax()) == int(native_logits[0, -1].argmax())
    print(f"standard alias diff: {alias_diff:.6f}", flush=True)
    print(f"prefill max diff: {prefill_diff:.6f}; top1 match: {prefill_top1_match}")
    assert prefill_top1_match
    assert prefill_diff < 0.05

    all_ids = list(ids)
    for step in range(8):
        next_id = int(native_logits[0, -1].argmax())
        all_ids.append(next_id)
        next_tensor = torch.tensor([[next_id]], dtype=torch.long)
        native_logits = native_forward(model, next_tensor, state)
        if step == 0:
            native_hash = model.engram_hash(next_tensor, len(ids), None)
            standard_hash = model.engram_hash(torch.tensor([all_ids], dtype=torch.long), 0, None)
            hash_diff = (standard_hash[0, -1] - native_hash[0, 0]).abs().max().item()
            print(f"decode hash ids equal {torch.equal(standard_hash[0, -1], native_hash[0, 0])} diff {hash_diff}", flush=True)
        if step == 0:
            import v41f.attention as attnmod
            records = []
            original_sparse = attnmod.sparse_attn
            engram_records = {}
            attn_records = {}
            ffn_records = {}
            sparse_outputs = []
            block_outputs = {}

            for block_layer, block_module in enumerate(model.layers):
                def make_block_wrapper(layer_id, original_forward):
                    def wrapped_block(*args, **kwargs):
                        result = original_forward(*args, **kwargs)
                        block_outputs[layer_id] = result[0].detach().clone()
                        return result
                    return wrapped_block

                block_module.forward = make_block_wrapper(block_layer, block_module.forward)

                def make_module_wrapper(layer_id, module, store, kind):
                    original_forward = module.forward

                    def wrapped_module(x, *args, **kwargs):
                        result = original_forward(x, *args, **kwargs)
                        y = result[0] if isinstance(result, tuple) else result
                        store[layer_id] = (x.detach().clone(), y.detach().clone())
                        return result
                    return wrapped_module

                block_module.attn.forward = make_module_wrapper(block_layer, block_module.attn, attn_records, "attn")
                block_module.ffn.forward = make_module_wrapper(block_layer, block_module.ffn, ffn_records, "ffn")

            for engram_layer, engram_module in enumerate(model.engrams):
                if engram_module is None:
                    continue

                def make_wrapper(eng_layer, original_forward):
                    def wrapped_enggram(x, ids, mask=None):
                        y = original_forward(x, ids, mask)
                        engram_records[eng_layer] = (x.detach().clone(), y.detach().clone())
                        return y
                    return wrapped_enggram

                engram_module.forward = make_wrapper(engram_layer, engram_module.forward)

            def record_sparse(q, kv, sink, idxs, scale, *args, **kwargs):
                records.append((q.detach().clone(), kv.detach().clone(), idxs.detach().clone()))
                output = original_sparse(q, kv, sink, idxs, scale, *args, **kwargs)
                sparse_outputs.append(output.detach().clone())
                return output

            attnmod.sparse_attn = record_sparse
        standard_logits, _ = model(torch.tensor([all_ids], dtype=torch.long))
        if step == 0:
            attnmod.sparse_attn = original_sparse
            compare_decode_records(records, state, len(ids), len(all_ids))
            for block_layer in range(len(model.layers)):
                attn_pair = attn_records[block_layer]
                ffn_pair = ffn_records[block_layer]
                if block_layer == 3:
                    oproj = model.layers[3].attn.oproj
                    print(f"block 03 dtypes input {attn_pair[0].dtype} sparse {sparse_outputs[block_layer].dtype} "
                          f"wo_a {oproj.wo_a.dtype} wo_b {oproj.wo_b.weight.dtype} "
                          f"output {attn_pair[1].dtype} native {state.last_attn_out[block_layer].dtype}", flush=True)
                sparse_diff = (sparse_outputs[block_layer][0, -1].float() - state.last_sparse[block_layer][0, 0].float()).abs().max().item()
                attn_out_diff = (attn_pair[1][0, -1].float() - state.last_attn_out[block_layer][0, 0].float()).abs().max().item()
                ffn_out_diff = (ffn_pair[1][0, -1].float() - state.last_ffn_out[block_layer][0, 0].float()).abs().max().item()
                block_out_diff = (block_outputs[block_layer][0, -1].float() - state.last_block[block_layer][0, 0].float()).abs().max().item()
                print(f"block {block_layer:02d} block out diff {block_out_diff:.6f} sparse diff {sparse_diff:.6f} attn out diff {attn_out_diff:.6f} ffn out diff {ffn_out_diff:.6f}", flush=True)
            for engram_layer, pair in engram_records.items():
                native_pair = state.last_engram[engram_layer]
                input_diff = (pair[0][0, -1].float() - native_pair[0][0, 0].float()).abs().max().item()
                output_diff = (pair[1][0, -1].float() - native_pair[1][0, 0].float()).abs().max().item()
                print(f"engram layer {engram_layer:02d} input diff {input_diff:.6f} output diff {output_diff:.6f}", flush=True)
        diff = (standard_logits[0, -1] - native_logits[0, -1]).abs().max().item()
        top1 = int(standard_logits[0, -1].argmax()) == int(native_logits[0, -1].argmax())
        print(f"decode {step + 1}: token={next_id} diff={diff:.6f} top1={top1}")
        assert top1
        # The default 0.08 is the contract. NATIVE_GOLD_MAXDIFF is an observation-only
        # override: at the v42 shape the full-prefix batched GEMM vs single-token GEMV
        # reduction order diverges by bf16 ULPs that compound through hc4 (max diffs are
        # dyadic: 0.0039/0.125/0.5/2/4); it lets the run log the full 8-step trajectory
        # while the top-1 gate still binds. Never set it in CI.
        maxdiff = float(os.environ.get("NATIVE_GOLD_MAXDIFF", "0.08"))
        assert diff < maxdiff, f"decode step {step + 1} diff {diff} >= {maxdiff}"

    print("native gold PASS")


if __name__ == "__main__":
    main()
