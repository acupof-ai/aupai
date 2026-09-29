"""CED incremental cache parity: tiny random CED model, CPU.

exact mode must match a whole-sequence forward's last-position logits at every prefill
length; bounded mode runs too. A deliberately corrupted SWA cache row must move the
logits (negative control: a cache the attention does not actually read would pass
whatever the join did).

Run: CUDA_VISIBLE_DEVICES= python3 scripts/test_ced_cache.py [--selftest]
"""
import os
import sys

import torch

if len(sys.argv) > 1 and sys.argv[1] != "--selftest":
    sys.exit(f"usage: {os.path.basename(__file__)} [--selftest]  (got {sys.argv[1:]})")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import HybridLM  # noqa: E402
from scripts.ced_cache import CEDCache  # noqa: E402


def tiny_cfg():
    class C:
        pass

    c = C()
    c.d = 64
    c.dim = 64
    c.layers = 6
    c.heads = 4
    c.ffn_hidden = 144
    c.vocab = 256
    c.vocab_real = 256
    c.seq = 256
    c.seed = 0
    c.grad_ckpt = False
    c.attn_every = 1
    c.attn_res = False
    c.attn_res_blocks = 0
    c.attn_hybrid = True
    c.n_swa_only_layers = 2
    c.csa = True
    c.csa2 = True
    c.csa2_m = 4
    c.csa2_top_k = 2
    c.csa2_n_win = 8
    c.csa2_indexer_dim = 8
    c.csa2_indexer_heads = 2
    c.csa2_modes = "F,F,F,F"
    c.csa2_joint = False
    c.csa2_win_flash = True
    c.rope_dims = 8
    c.ced = 1
    c.ced_enc_layers = 3
    c.ced_kc_norm = False
    c.swa = False
    c.hca = False
    c.moe_experts = 8
    c.moe_top_k = 2
    c.moe_shared = 1
    c.moe_expert_ffn = 48
    c.moe_layers = "0-5"
    c.router_score = "softmax"
    c.router_logit_cap = 0.0
    c.moe_bias_gamma = 0.001
    c.moe_balance_alpha = 1e-4
    c.untie_head = False
    c.value_embed = False
    c.fone = False
    c.head_mixed = 0
    c.mem_values = 0
    c.moe_arm = None
    return c


def full_last_logits(model, ids):
    with torch.no_grad():
        out = model(ids.view(1, -1))[0][0, -1]
    return out.float()


def main():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = HybridLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab, (40,))

    failures = []

    # ---- exact parity over several prefill lengths ----
    for P in (13, 25, 40):
        eng = CEDCache(model, mode="exact")
        got = eng.prefill(ids[:P])
        ref = full_last_logits(model, ids[:P])
        md = (got.float() - ref).abs().max().item()
        agree = got.argmax().item() == ref.argmax().item()
        print(f"prefill P={P}: max|d|={md:.5f} argmax_agree={agree}")
        if md > 0.05 or not agree:
            failures.append(f"prefill P={P} md={md}")

    # ---- exact greedy steps vs whole-sequence forward ----
    # prefill logits predict token P; after choosing it, one cached step must match a
    # whole-sequence forward over prompt+[token] at its last position.
    eng = CEDCache(model, mode="exact")
    cur = ids[:25].clone()
    nxt = eng.prefill(cur)
    step_md = []
    for st in range(6):
        tok = nxt.argmax().item()
        cur = torch.cat([cur, torch.tensor([tok])])
        nxt = eng.step(tok)
        ref = full_last_logits(model, cur)
        md = (nxt.float() - ref).abs().max().item()
        agree = nxt.argmax().item() == ref.argmax().item()
        step_md.append(md)
        print(f"step {st}: max|d|={md:.5f} argmax_agree={agree} tok={tok}")
        if md > 0.05 or not agree:
            failures.append(f"decode step {st} md={md}")

    # ---- negative control: corrupting a SWA window key/value must move a decode step ----
    # Every decode query reads its SWA ring, so perturbing a ring row cannot be invisible.
    def clean_step():
        c = CEDCache(model, mode="exact")
        c.prefill(ids[:25])
        return c.step(int(ids[-1].item()))
    cb = CEDCache(model, mode="exact")
    cb.prefill(ids[:25])
    with torch.no_grad():
        cb.swa_k[0][-1].add_(50.0)
        cb.swa_v[0][-1].add_(50.0)
    moved = (cb.step(int(ids[-1].item())) - clean_step()).abs().max().item()
    print(f"negative control: max|d| after SWA-ring corruption = {moved:.4f}")
    if moved < 0.1:
        failures.append(f"corruption not detected ({moved})")

    # ---- bounded runs and reports a gap vs exact ----
    eb = CEDCache(model, mode="bounded")
    lb = eb.prefill(ids[:40]).float()
    ee = CEDCache(model, mode="exact")
    le = ee.prefill(ids[:40]).float()
    p = torch.softmax(le, -1)
    q = torch.softmax(lb, -1)
    kl = (p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum().item()
    print(f"bounded vs exact prefill: KL={kl:.5f} top1_agree={lb.argmax().item()==le.argmax().item()}")

    if failures:
        print("FAIL:", "; ".join(failures))
        sys.exit(1)
    print("ALL PARITY CHECKS PASS")


if __name__ == "__main__":
    main()
