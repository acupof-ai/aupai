"""de-108 default-path identity gate, environment-independent.

A fixed-seed forward+backward of one CSA2 layer with cfg.csa2_joint UNSET must be
byte-identical between the tree under test and a baseline model.py. The two are hashed in
ONE process on the CURRENT device and compared; no hash constant is baked in. A constant
was wrong: float64 reduction order differs across BLAS/device, so a laptop hash does not
match a pod hash even for identical math. Same-process comparison removes the environment
from the predicate.

Both paths must be readable on the machine (the pod /work/aupai is not a git repo, so
obtain a baseline there by copying a pre-flag model.py to a temp path; on a git box
`git show <pre-flag-sha>:model.py > /tmp/ref_model.py`).

Run:
    python3 scripts/de108/check_default_identical.py <candidate_model.py> \
        --ref <baseline_model.py> [--device cpu]
"""
import argparse
import hashlib
import importlib.util
from types import SimpleNamespace

import torch


def hash_model(model_path, device):
    spec = importlib.util.spec_from_file_location(f"mm_idpin_{abs(hash(model_path))}", model_path)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    B, T, H, hd, d = 2, 24, 4, 8, 16
    cfg = SimpleNamespace(
        csa=True, csa2=True, csa2_m=4, csa2_top_k=3, csa2_n_win=6,
        csa2_indexer_dim=8, csa2_indexer_heads=2, d=16)
    torch.manual_seed(123)
    m = M.CompressedSparseAttention(cfg, h=4, hd=8).to(torch.float64).to(device)
    q = torch.randn(B, T, H, hd, dtype=torch.float64, device=device)
    k = torch.randn(B, T, H, hd, dtype=torch.float64, device=device)
    v = torch.randn(B, T, H, hd, dtype=torch.float64, device=device)
    x = torch.randn(B, T, d, dtype=torch.float64, device=device)
    cu = torch.tensor([0, T // 2, T, 2 * T], dtype=torch.int32, device=device)
    q, k, v, x = (t.requires_grad_() for t in (q, k, v, x))
    gout = torch.randn(B, T, H, hd, dtype=torch.float64, device=device)
    y = m(q, k, v, cu=cu, x=x)
    y.backward(gout)
    h = hashlib.sha256()
    tensors = [y, q.grad, k.grad, v.grad, x.grad]
    tensors += [p.grad for p in m.parameters() if p.grad is not None]
    for t in tensors:
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", help="model.py under test (default path)")
    ap.add_argument("--ref", required=True, help="baseline model.py to match")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    href = hash_model(a.ref, a.device)
    hgot = hash_model(a.candidate, a.device)
    print(f"ref {a.ref}: {href}")
    print(f"new {a.candidate}: {hgot}")
    assert href == hgot, (
        f"default path changed on {a.device}: {hgot} != {href} "
        "(candidate and baseline model.py differ when csa2_joint is unset)")
    print("DEFAULT IDENTICAL")


if __name__ == "__main__":
    main()
