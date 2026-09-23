#!/usr/bin/env python3
"""Step-3 acceptance for the flat-branch cut (2026-09-23): CED is unchanged.

The cut deleted the flat single-pass body and the Cfg.ced flag so the two-pass CED body is the
only non-AttnRes path. A deletion refactor must change no arithmetic, so this builds the SAME
gate-shaped csa2/CED config twice in ONE process -- old model.py read from git (PRE_CUT_SHA,
which still carries `if self.ced:` and the flat branch), new model.py from the worktree, each
with an identical seed -- and requires:

  1. total parameter count equal (nothing added or dropped)
  2. state_dict KEY SETS equal and numel equal per key (w_kv/w_z on the same decoder layers)
  3. fixed-input INFERENCE logits torch.equal -- not allclose, a reorder is a different change
  4. fixed-input TRAIN logits and every parameter GRADIENT torch.equal

ONE PROCESS, as in test_split_bitwise.py: across processes cuBLAS workspace/autotune differ
and bitwise comparison goes false red. Small gate-shaped config (12 layers, 2 SWA, CSA2 F/R
modes, split 6) so it runs CPU in seconds; the gate-shape absolute count is the separately
recorded S0 number 3,221,975,040.

    python3 scripts/test_flat_cut_bitwise.py

# restartable: a seconds-long pure-CPU comparison with no side effects and no shards; an
# interrupt loses nothing but the rerun, which rebuilds both models from fixed seeds.
"""
import hashlib
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
PRE_CUT_SHA = "c26daf67"   # last main commit carrying the flat branch and Cfg.ced


def _load_old():
    src = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-p", f"{PRE_CUT_SHA}:model.py"],
                         capture_output=True, text=True, check=True).stdout
    assert "if self.ced:" in src, "PRE_CUT_SHA model.py must still contain the flat/ced branch"
    td = tempfile.mkdtemp(prefix="flatcut_old_")
    path = Path(td) / "model_precut.py"
    path.write_text(src)
    sys.path.insert(0, td)
    sys.path.insert(0, str(ROOT))   # sibling imports (fone, ...) resolve to today's modules
    spec = importlib.util.spec_from_file_location("model_precut", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg():
    common = dict(
        d=128, layers=12, heads=4, ffn_hidden=256, vocab=1024, vocab_real=1000, seq=64,
        attn_every=1, n_swa_only_layers=2, attn_res=False, attn_res_blocks=0,
        attn_res_dyn_q=False, csa=True, csa2=True, csa2_win_flash=False,
        csa2_m=8, csa2_top_k=64, csa2_n_win=128, csa2_indexer_heads=4, csa2_indexer_dim=64,
        moe_experts=0, moe_layers="", moe_top_k=3, moe_shared=1, moe_expert_ffn=256,
        rope_dims=32, grad_ckpt=False, ced_enc_layers=6, attn_hybrid=False,
        csa2_modes="F,R,R,R,X,R,R,R,R,R")
    return SimpleNamespace(**common, ced=1), SimpleNamespace(**common)


def main():
    old = _load_old()
    import model as new  # noqa: PLC0415

    co, cn = _cfg()
    torch.manual_seed(11)
    m_old = old.HybridLM(co).eval()
    torch.manual_seed(11)
    m_new = new.HybridLM(cn).eval()

    n_old = sum(p.numel() for p in m_old.parameters())
    n_new = sum(p.numel() for p in m_new.parameters())
    assert n_old == n_new, f"total params differ: old {n_old} new {n_new}"

    k_old, k_new = set(m_old.state_dict()), set(m_new.state_dict())
    assert k_old == k_new, (f"state_dict key sets differ:\n only old: "
                            f"{sorted(k_old - k_new)[:5]}\n only new: {sorted(k_new - k_old)[:5]}")
    for k in sorted(k_old):
        a, b = m_old.state_dict()[k].numel(), m_new.state_dict()[k].numel()
        assert a == b, f"{k}: numel {a} vs {b}"

    def dig(y):
        return hashlib.sha256(y.float().numpy().tobytes()).hexdigest()[:16]

    g = torch.Generator().manual_seed(7)
    x = torch.randint(0, 1024, (2, 64), generator=g)
    torch.manual_seed(12)
    with torch.no_grad():
        y_old = m_old(x)[0]
    torch.manual_seed(12)
    with torch.no_grad():
        y_new = m_new(x)[0]
    assert torch.equal(y_old, y_new), (
        f"inference logits differ (max {((y_old - y_new).abs().max().item()):.3e}); a flat-cut "
        "refactor may not reorder arithmetic")

    m_old.train(); m_new.train()
    xb = torch.randint(0, 1024, (2, 64), generator=torch.Generator().manual_seed(21))
    yo = m_old(xb)[0]
    yn = m_new(xb)[0]
    assert torch.equal(yo, yn), "train-mode logits diverged before the gradient comparison"
    go = torch.randn_like(yo, generator=torch.Generator().manual_seed(22))
    yo.backward(go)
    yn.backward(go)
    bad = []
    for (na, pa), (nb, pb) in zip(m_old.named_parameters(), m_new.named_parameters()):
        assert na == nb
        if pa.grad is None and pb.grad is None:
            continue
        if pa.grad is None or pb.grad is None or not torch.equal(pa.grad, pb.grad):
            bad.append(na)
    assert not bad, f"gradients differ on {len(bad)} params, e.g. {bad[:3]}"

    print(f"flat-cut bitwise OK: {n_old} params, {len(k_old)} identical keys, inference logits "
          f"and every gradient torch.equal (digest {dig(y_new)})")


if __name__ == "__main__":
    main()
