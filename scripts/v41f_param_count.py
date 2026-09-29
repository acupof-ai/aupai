#!/usr/bin/env python3
"""V41F-S parameter count: total and per-token ACTIVE params, by config formula.

Pure formulas over v41f/config.py field shapes -- it never instantiates the model
(the MoE/Attention modules are still landing), so every term is keyed to the exact
weight shape in the vendored reference third_party/deepseek_v41_ref/model_ref.py.ref.

Counts are PARAMETERS (scalar weights), not bytes; fp8/fp4 storage does not change a
parameter count. ACTIVE means parameters touched on the compute path of one token:

- Attention is dense per token: the low-rank Q (all heads), one MQA K/V, and the
  grouped output projection all run for every token, so they count in full.
- MoE is the only sparse-at-parameter level: only `n_activated_experts` routed
  experts plus the single shared expert run; the gate still scores ALL experts.
- Hyper-connections run their pre/post/comb projection for every token, in full.
- The LM head and embedding are counted separately; tie_word_embeddings collapses
  them (official V4.1-Flash config has tie_word_embeddings=false).

Two fields of the prod shape are not yet fully specified and are reported honestly, not
silently zeroed (this counts V41FConfig() shape; v41f_s() would require the tokenizer):
- engram_num_embeddings=() : the n-gram table rows are filled in from the real
  tokenizer at build; until then the engram dense projection is countable but the
  embedding TABLE is reported as 0/uncounted (--engram-rows supplies it).
- n_mtp_layers=1 but compress_ratios has only n_layers entries, so the DSpark
  draft layer's attention ratio is unspecified; --include-mtp counts it assuming
  the ratio-0 (window-only) shape its forward asserts.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from v41f.config import V41FConfig  # noqa: E402


def _expert(dim: int, inter: int) -> int:
    """One SwiGLU expert: w1(dim,inter)+w3(dim,inter)+w2(inter,dim)."""
    return 3 * dim * inter


def attention_params(c: V41FConfig) -> dict:
    """Dense per-token attention core, identical in every backbone layer."""
    H, hd, g = c.n_heads, c.head_dim, c.o_groups
    ql, ol = c.q_lora_rank, c.o_lora_rank
    q_a = c.dim * ql  # down-project to q latent
    q_norm = ql
    q_b = ql * (H * hd)  # up-project Q for every head
    wkv = c.dim * hd  # ONE MQA K/V head
    kv_norm = hd
    wo_a = (H * hd) * ol  # (H*hd/g) x (g*ol) collapses to H*hd*ol
    wo_b = (g * ol) * c.dim
    sink = H
    base = q_a + q_norm + q_b + wkv + kv_norm + wo_a + wo_b + sink
    return {
        "q_lora_down": q_a,
        "q_norm": q_norm,
        "q_up_all_heads": q_b,
        "mqa_kv": wkv,
        "kv_norm": kv_norm,
        "grouped_out_a": wo_a,
        "grouped_out_b": wo_b,
        "attn_sink": sink,
        "total": base,
    }


def indexer_params(c: V41FConfig, layer_id: int) -> dict:
    """Side index attention on index_source layers; owns_k only on the kv source."""
    ih, ihd = c.index_n_heads, c.index_head_dim
    q = c.q_lora_rank * (ih * ihd)
    wproj = c.dim * ih
    owns = layer_id in c.kv_source_layers
    wk = c.head_dim * ihd if owns else 0
    k_norm = ihd if owns else 0
    return {
        "indexer_q": q,
        "indexer_weights_proj": wproj,
        "indexer_wk_owner": wk,
        "indexer_k_norm_owner": k_norm,
        "total": q + wproj + wk + k_norm,
    }


def compressor_params(c: V41FConfig, ratio: int) -> dict:
    """KV compressor on a kv-source layer (r>1 adds the pooling score gate)."""
    wkv = c.dim * c.head_dim
    wgate = c.dim * c.head_dim if ratio > 1 else 0
    norm = c.head_dim
    return {
        "compressor_wkv": wkv,
        "compressor_wgate": wgate,
        "compressor_norm": norm,
        "total": wkv + wgate + norm,
    }


def moe_params(c: V41FConfig, n_routed: int, n_act: int) -> dict:
    """One MoE sublayer: routed experts + 1 shared + gate. total vs per-token active."""
    pe = _expert(c.dim, c.moe_inter_dim)
    routed_total = n_routed * pe
    shared = pe  # exactly one shared expert, config asserts it
    gate_w = n_routed * c.dim
    gate_bias = n_routed  # no vision -> no bias_vl
    gate = gate_w + gate_bias
    total = routed_total + shared + gate
    active = n_act * pe + shared + gate
    return {
        "routed_experts_total": routed_total,
        "shared_expert": shared,
        "gate": gate,
        "total": total,
        "active": active,
        "n_routed": n_routed,
        "n_active": n_act,
        "per_expert": pe,
    }


def hyperconn_params(c: V41FConfig) -> dict:
    """Two mix projections (attn + ffn), each [mix_hc, hc_dim] + base + scale."""
    m = c.hc_mult
    mix_hc = (2 + m) * m
    hc_dim = m * c.dim
    one_fn = mix_hc * hc_dim
    one_base = mix_hc
    one_scale = 3
    per_layer = 2 * (one_fn + one_base + one_scale)
    return {"mix_proj": 2 * one_fn, "bases": 2 * one_base, "scales": 2 * one_scale, "total": per_layer}


def count(
    c: V41FConfig, engram_rows: int = 0, include_mtp: bool = False, tied_embeddings: bool = False
) -> dict:
    attn = attention_params(c)
    hc = hyperconn_params(c)

    layers = []
    for L in range(c.n_layers):
        ratio = c.compress_ratios[L]
        a = dict(attn)
        extra = 0
        if L in c.index_source_layers and ratio > 0:
            ix = indexer_params(c, L)
            extra += ix["total"]
            a.update({k: ix[k] for k in ix if k != "total"})
        if L in c.kv_source_layers and ratio > 0:
            cp = compressor_params(c, ratio)
            extra += cp["total"]
            a.update({k: cp[k] for k in cp if k != "total"})
        a["attn_plus_side_total"] = attn["total"] + extra
        moe = moe_params(c, c.n_routed_experts, c.n_activated_experts)
        norms = 2 * c.dim  # attn_norm + ffn_norm
        layer_total = a["attn_plus_side_total"] + moe["total"] + hc["total"] + norms
        layer_active = attn["total"] + extra + moe["active"] + hc["total"] + norms
        layers.append(
            {
                "layer": L,
                "ratio": ratio,
                "is_kv_source": L in c.kv_source_layers,
                "is_index_source": L in c.index_source_layers,
                "attn": a["attn_plus_side_total"],
                "moe_total": moe["total"],
                "moe_active": moe["active"],
                "hyperconn": hc["total"],
                "norms": norms,
                "total": layer_total,
                "active": layer_active,
            }
        )

    backbone_total = sum(x["total"] for x in layers)
    backbone_active = sum(x["active"] for x in layers)

    # engram (one module per engram layer): dense projection countable; the hash
    # embedding table needs rows that v41f_s leaves for the tokenizer build.
    engram_dense = engram_table = 0
    if c.engram_layer_ids:
        n_cols = (c.engram_max_ngram_size - 1) * c.engram_n_heads
        engram_dense = n_cols * c.engram_head_dim * (c.dim * (c.hc_mult + 1)) + 2 * c.hc_mult * c.dim
        engram_table = len(c.engram_layer_ids) * engram_rows * c.engram_head_dim

    # embeddings + head
    embed = c.vocab_size * c.dim
    head = 0 if tied_embeddings else c.vocab_size * c.dim
    final_norm = c.dim

    mtp_total = mtp_active = 0
    mtp_detail = None
    if include_mtp:
        # DSpark draft layer: window-only attention (no indexer/compressor), a
        # backbone-shaped MoE, HC, norms, plus main_proj/main_norm and a confidence
        # head (markov_rank=0 -> zero-sized markov embed/head). Embedding/head reused.
        dspark_routed = c.n_routed_experts
        dspark_act = c.n_activated_experts
        m_moe = moe_params(c, dspark_routed, dspark_act)
        main_proj = c.dim * len(c.dspark_target_layer_ids) * c.dim
        conf = (c.dim + c.dspark_markov_rank) * 1
        mtp_total = attn["total"] + m_moe["total"] + hc["total"] + 3 * c.dim + main_proj + conf
        mtp_active = attn["total"] + m_moe["active"] + hc["total"] + 3 * c.dim + main_proj + conf
        mtp_detail = {
            "moe_total": m_moe["total"],
            "moe_active": m_moe["active"],
            "main_proj": main_proj,
            "confidence_head": conf,
        }

    total = backbone_total + engram_dense + engram_table + embed + head + final_norm + mtp_total
    active = backbone_active + engram_dense + engram_table + embed + head + final_norm + mtp_active

    return {
        "config": {
            k: getattr(c, k)
            for k in (
                "dim",
                "n_layers",
                "n_heads",
                "head_dim",
                "rope_head_dim",
                "q_lora_rank",
                "o_lora_rank",
                "o_groups",
                "vocab_size",
                "n_routed_experts",
                "n_activated_experts",
                "n_shared_experts",
                "moe_inter_dim",
                "hc_mult",
                "index_n_heads",
                "index_head_dim",
                "index_topk",
                "n_mtp_layers",
            )
        },
        "module_templates": {
            "attention_per_layer": attn,
            "hyperconn_per_layer": hc,
            "moe_per_layer": moe_params(c, c.n_routed_experts, c.n_activated_experts),
            "compressor_r_gt_1": compressor_params(c, 2),
        },
        "layers": layers,
        "backbone_total": backbone_total,
        "backbone_active": backbone_active,
        "engram_dense": engram_dense,
        "engram_table_rows_assumed": engram_rows,
        "engram_table_params": engram_table,
        "embedding": embed,
        "lm_head": head,
        "tied_embeddings": tied_embeddings,
        "final_norm": final_norm,
        "mtp_included": include_mtp,
        "mtp": mtp_detail,
        "total_params": total,
        "active_params_per_token": active,
        "active_fraction": active / total if total else 0.0,
        "moe_total_all_layers": sum(x["moe_total"] for x in layers),
        "moe_active_all_layers": sum(x["moe_active"] for x in layers),
    }


def _print_human(r: dict) -> None:
    cfg = r["config"]
    print(
        f"V41F-S  dim={cfg['dim']} layers={cfg['n_layers']} heads={cfg['n_heads']} "
        f"head_dim={cfg['head_dim']} experts={cfg['n_routed_experts']} "
        f"top{cfg['n_activated_experts']}+{cfg['n_shared_experts']}shared "
        f"inter={cfg['moe_inter_dim']} hc_mult={cfg['hc_mult']} "
        f"vocab={cfg['vocab_size']}"
    )
    print("-" * 72)
    for x in r["layers"]:
        tag = []
        if x["is_kv_source"]:
            tag.append("kv-src")
        if x["is_index_source"]:
            tag.append("idx-src")
        print(
            f"  L{x['layer']:<2} r={x['ratio']} {' '.join(tag):<14} "
            f"attn={x['attn']:>11,} moe_tot={x['moe_total']:>11,} "
            f"moe_act={x['moe_active']:>10,} hc={x['hyperconn']:>9,} "
            f"tot={x['total']:>12,} act={x['active']:>12,}"
        )
    print("-" * 72)
    print(
        f"  backbone (n_layers)         total={r['backbone_total']:>14,} active={r['backbone_active']:>14,}"
    )
    print(f"  engram dense projection           {r['engram_dense']:>12,}")
    print(
        f"  engram table (rows={r['engram_table_rows_assumed']})          "
        f"{r['engram_table_params']:>12,}  (0 = rows pending tokenizer)"
    )
    print(f"  embedding                         {r['embedding']:>12,}")
    print(f"  lm_head ({'tied' if r['tied_embeddings'] else 'untied'})            {r['lm_head']:>12,}")
    print(f"  final norm                        {r['final_norm']:>12,}")
    if r["mtp_included"]:
        print(
            f"  MTP/DSpark draft layer            {r['mtp'] and sum(v for v in r['mtp'].values() if isinstance(v, int)):>12,}"
        )
    print("=" * 72)
    print(f"  TOTAL PARAMS                 {r['total_params']:>16,} ({r['total_params'] / 1e9:.4f} B)")
    print(
        f"  ACTIVE PARAMS / TOKEN        {r['active_params_per_token']:>16,} "
        f"({r['active_params_per_token'] / 1e6:.3f} M)"
    )
    print(f"  active fraction              {r['active_fraction'] * 100:>15.2f}%")
    print(f"  MoE total across layers      {r['moe_total_all_layers']:>16,}")
    print(f"  MoE active across layers     {r['moe_active_all_layers']:>16,}")


def _selftest() -> int:
    # Known-answer tiny config, hand-computed:
    # dim=8 inter=4 -> one expert = 3*8*4 = 96.
    # 2 routed experts top1: routed 192; 1 shared 96; gate w=2*8=16 + bias 2 = 18.
    # MoE total 192+96+18 = 306; active 1*96+96+18 = 210.
    c = V41FConfig(
        dim=8,
        n_layers=1,
        n_heads=2,
        head_dim=8,
        rope_head_dim=4,
        q_lora_rank=4,
        o_lora_rank=4,
        o_groups=2,
        vocab_size=10,
        n_routed_experts=2,
        n_activated_experts=1,
        moe_inter_dim=4,
        hc_mult=1,
        compress_ratios=(0,),
        kv_source_layers=(),
        index_source_layers=(),
        index_n_heads=1,
        index_head_dim=4,
        engram_layer_ids=(),
        n_mtp_layers=0,
        dspark_block_size=0,
        dspark_target_layer_ids=(),
        candidate_source_layer=-1,
    )
    assert _expert(8, 4) == 96
    m = moe_params(c, 2, 1)
    assert m["per_expert"] == 96 and m["routed_experts_total"] == 192
    assert m["shared_expert"] == 96 and m["gate"] == 18
    assert m["total"] == 306 and m["active"] == 210, (m["total"], m["active"])
    r = count(c, tied_embeddings=True)
    # tied: embed counted once, no head
    assert r["embedding"] == 10 * 8 and r["lm_head"] == 0
    # total == sum of module pieces (self-consistency identity)
    pieces = (
        r["backbone_total"]
        + r["engram_dense"]
        + r["engram_table_params"]
        + r["embedding"]
        + r["lm_head"]
        + r["final_norm"]
    )
    assert r["total_params"] == pieces
    # hc_mult=1 -> mix_hc=(2+1)*1=3, hc_dim=8: per layer 2*(3*8 + 3 + 3)=60
    assert hyperconn_params(c)["total"] == 60
    print("selftest ok: expert=96, moe total=306 active=210, tied embed=80, hc=60")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--engram-rows",
        type=int,
        default=0,
        help="rows per engram hash table (tokenizer-derived; 0=uncounted)",
    )
    ap.add_argument(
        "--include-mtp", action="store_true", help="count the DSpark draft layer assuming ratio-0 attention"
    )
    ap.add_argument("--tied", action="store_true", help="tie embedding and LM head")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    # V41FConfig(), not v41f_s(): this counts SHAPE, building nothing, and the engram ON
    # default cannot be validated without a tokenizer to measure the compressed vocab from.
    # The bare name says "unvalidated shape"; v41f_s would refuse exactly as intended.
    r = count(V41FConfig(), engram_rows=a.engram_rows, include_mtp=a.include_mtp, tied_embeddings=a.tied)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        _print_human(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
