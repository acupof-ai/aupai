"""v42 trainer gates, CPU: the V4.1 optimizer (v41f/optim.py) and a 3-step smoke through train.py's
own pieces (build_model, set_schedule, the balancer, clip) on a tiny v42-shaped V42LM.

    python3 tests/v41f/test_v42_train.py --selftest

The smoke replaces only the Liger fused CE (GPU/Triton) with F.cross_entropy over lm_logits. With
CUDA present it also checks the grouped-GEMM MoE dispatch against the reference loop.
"""

import argparse
import os
import sys
from dataclasses import asdict

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from v41f.config import v42_s24  # noqa: E402
from v41f.lm import V42LM  # noqa: E402
from v41f.optim import (  # noqa: E402
    V42Muon,
    build_v42_optimizers,
    orthogonalize,
    sinkhorn_balance,
    v42_param_groups,
)


def tiny_cfg():
    return v42_s24(
        vocab_size=512, dim=128, n_layers=4, n_heads=4, head_dim=64, rope_head_dim=32,
        q_lora_rank=64, o_groups=2, o_lora_rank=64, window_size=16,
        compress_ratios=(0, 0, 2, 1), kv_source_layers=(2, 3), index_source_layers=(2, 3),
        index_n_heads=2, index_head_dim=64, index_topk=16,
        n_routed_experts=8, n_activated_experts=2, moe_inter_dim=64)


def test_sinkhorn_row_col_rms():
    torch.manual_seed(0)
    g = torch.randn(4096, 256) * (torch.rand(4096, 1) * 3 + 0.1)
    g[:7] *= 1e-6  # near-zero rows: masked (tau = 1e-3 of the mean row norm)
    d = sinkhorn_balance(g)
    row = d.square().mean(1).sqrt()
    col = d[7:].square().mean(0).sqrt()
    assert torch.all(d[:7] == 0), "near-zero rows must be masked to zero"
    assert torch.allclose(row[7:], torch.ones(4089), atol=1e-4), f"row RMS {row[7:].min():.4f}..{row[7:].max():.4f}"
    assert (col - 1).abs().max() < 0.05, f"column RMS off by {(col - 1).abs().max():.4f}"
    # negative control: the raw gradient's rows are nowhere near unit RMS
    assert (g[7:].square().mean(1).sqrt() - 1).abs().max() > 0.5


def test_grad_norm_report_names_the_planted_param_first():
    """A large grad planted on one parameter puts its module group first and inflates only its
    optimizer group; the other groups read their small background norm."""
    from v41f.optim import grad_norm_report

    torch.manual_seed(0)
    m = V42LM(tiny_cfg(), max_batch_size=1)
    for p in m.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p) * 1e-3
    target = m.layers[2].attn.indexer.wq_b.weight
    target.grad = torch.full_like(target, 10.0)
    line = grad_norm_report(m)
    first = line.split("gradnorm top5 ")[1].split(" ")[0]
    assert first.startswith("layers.2.attn.indexer="), line
    planted = 10.0 * target.numel() ** 0.5
    assert abs(float(first.split("=")[1]) - planted) / planted < 1e-2, line
    groups = dict(kv.split("=") for kv in line.split("| groups ")[1].split(" "))
    assert float(groups["muon"]) > 0.99 * planted and float(groups["sinkhorn"]) < 1.0, line
    # negative control: without the plant, the indexer is not first
    target.grad = torch.randn_like(target) * 1e-3
    assert not grad_norm_report(m).split("gradnorm top5 ")[1].startswith("layers.2.attn.indexer="), line
    print(f"  {line[:120]}")


def test_headwise_muon_equals_per_head_ns():
    torch.manual_seed(1)
    h, hd, c = 4, 32, 48
    w0 = torch.randn(h * hd, c)
    g = torch.randn(h * hd, c)

    def run(heads):
        p = torch.nn.Parameter(w0.clone())
        p.grad = g.clone()
        V42Muon([{"params": [p], "heads": [heads]}], lr=1.0, momentum=0.0, weight_decay=0.0).step()
        return w0 - p.detach()

    got = run(h)
    want = []
    for gh in g.view(h, hd, c):
        o = orthogonalize(gh[None])[0]
        want.append(o * (0.18 / o.square().mean().sqrt()))
    want = torch.cat(want)
    assert torch.allclose(got, want, atol=1e-5), f"head-wise max err {(got - want).abs().max():.2e}"
    assert abs(float(got.square().mean().sqrt()) - 0.18) < 1e-4, "update RMS must be 0.18 * lr"
    whole = run(1)
    assert (whole - want).abs().max() > 1e-2, "whole-matrix NS must differ from head-wise (control)"


def test_census_one_group_each():
    m = V42LM(tiny_cfg())
    g = v42_param_groups(m)
    seen = {}
    for name, items in g.items():
        for n, _ in items:
            assert n not in seen, f"{n} in both {seen[n]} and {name}"
            seen[n] = name
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    assert set(seen) == trainable, f"missing {sorted(trainable - set(seen))[:5]}"
    assert not frozen & set(seen)
    want = {"embed.weight": "sinkhorn", "head.weight": "sinkhorn", "norm.weight": "adamw_decay",
            "layers.0.ffn.gate.weight": "adamw_decay", "layers.0.attn.attn_sink": "adamw_nodecay",
            "layers.0.hc.hc_attn_base": "adamw_nodecay", "layers.0.hc.hc_attn_fn": "muon",
            "layers.2.attn.compressor.wgate.weight": "muon", "layers.0.attn.oproj.wo_a": "muon",
            "layers.0.ffn.experts.3.w2.weight": "muon", "layers.0.attn_norm.weight": "adamw_decay"}
    for n, grp in want.items():
        assert seen.get(n) == grp, f"{n}: {seen.get(n)} != {grp}"


def test_smoke_three_steps():
    import train

    torch.manual_seed(0)
    train.Cfg.arch = "v42"
    train.Cfg.v42_cfg = asdict(tiny_cfg())
    m = train.build_model(train.Cfg)
    assert isinstance(m, V42LM) and m.v41f_cfg.n_layers == 4
    m.to(torch.bfloat16)  # train.py refuses v42 without --fp8/--bf16, both of which cast
    opts = build_v42_optimizers(m, m.v41f_cfg, lr=3e-3)
    b, t, steps = 2, 64, 3
    x = (torch.arange(b * (t + 1)).view(b, t + 1) * 7) % 97
    xb, yb = x[:, :-1], x[:, 1:]
    layers = [m.blocks[i].ffn for i in m.moe_layers]
    losses = []
    for step in range(steps):
        train.set_schedule(opts, step, 100, train.Cfg)
        hidden, _ = m(xb, yb, None, None)
        ce = F.cross_entropy(m.lm_logits(hidden).reshape(-1, 512), yb.reshape(-1))
        loss = ce + m.aux_loss()
        loss.backward()
        assert m.commit_moe_token_counts() == len(layers)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        for layer in layers:
            layer.update_bias(layer.step_tokens_per_expert)
            layer.step_tokens_per_expert.zero_()
        assert torch.isfinite(loss), f"step {step}: loss {loss}"
        losses.append(float(ce))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    assert all(float(layer.gate.bias.abs().sum()) > 0 for layer in layers), "bias balancer never moved"
    assert abs(float(layers[0].gate.bias.mean())) < 1e-6, "bias must stay zero-mean"
    d = layers[0].diagnostics()
    assert d["tokens"] == b * t * 2 * steps and d["n_routed"] == 8, d
    print(f"  smoke losses {[round(v, 4) for v in losses]}")


def test_grouped_moe_matches_loop_cuda():
    if not torch.cuda.is_available():
        print("  SKIP grouped MoE parity: no CUDA (runs in the GPU window)")
        return
    torch.manual_seed(0)
    m = V42LM(tiny_cfg()).cuda().to(torch.bfloat16)
    moe = m.layers[0].ffn
    x = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)
    y_grouped = moe(x)
    moe_cpu_x = x.float().cpu().to(torch.bfloat16)
    moe.cpu()
    y_loop = moe(moe_cpu_x)
    err = (y_grouped.cpu().float() - y_loop.float()).abs().max()
    assert err < 3e-2, f"grouped vs loop max abs {err:.3e}"


def test_packed_rows_train_the_indexer():
    """The track A/B join: packed rows through V42LM, aux_loss carries the indexer KL, and every
    trainable leaf is in a group and (routed experts aside) gets a gradient. A forward that drops
    the indexer loss leaves wq_b/weights_proj/index_key gradient-less, which is this test's red."""
    import train

    torch.manual_seed(0)
    train.Cfg.arch = "v42"
    train.Cfg.v42_cfg = asdict(tiny_cfg())
    m = train.build_model(train.Cfg).to(torch.bfloat16)
    assert m.v41f_cfg.indexer_train_mode == "kl" and m.v41f_cfg.attn_impl == "chunked"
    b, t, eos = 2, 64, 1
    x = (torch.arange(b * (t + 1)).view(b, t + 1) * 7) % 97 + 2
    x[0, 20] = eos
    x[1, 41] = eos
    xb, yb = x[:, :-1], x[:, 1:]
    cu = train.doc_cu_seqlens(xb, eos)
    assert cu.numel() == 5, cu
    hidden, _ = m(xb, yb, cu, None)
    assert m.indexer_loss is not None and float(m.indexer_loss) > 0
    ce = F.cross_entropy(m.lm_logits(hidden).float().reshape(-1, 512), yb.reshape(-1))
    (ce + m.aux_loss()).backward()
    groups = {n for g in v42_param_groups(m).values() for n, _ in g}
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert groups == trainable, sorted(groups ^ trainable)
    idx = [n for n in trainable if ".indexer." in n or ".index_key." in n]
    assert idx, "kl mode must leave the indexer trainable"
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and ".ffn.experts." not in n and (p.grad is None or not p.grad.abs().sum())]
    assert not dead, dead


TESTS = [test_packed_rows_train_the_indexer, test_sinkhorn_row_col_rms, test_headwise_muon_equals_per_head_ns, test_census_one_group_each,
         test_smoke_three_steps, test_grouped_moe_matches_loop_cuda, test_grad_norm_report_names_the_planted_param_first]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    for t in TESTS:
        t()
        print(f"ok   {t.__name__}")
    print(f"v42 train gates: {len(TESTS)}/{len(TESTS)} passed")
