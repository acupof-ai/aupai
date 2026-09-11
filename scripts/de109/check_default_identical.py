"""de-109 default-path identity gate.

With cfg.csa2_win_flash unset (False) the CSA2 layer is byte-identical to the
materialized implementation: the flash-window path is additive behind a guard. Pins the
fixed-seed fwd+bwd hash of the output and every q/k/v/x + parameter gradient to the
origin/main value, so a default-path change fails.

Run: python3 scripts/de109/check_default_identical.py [model_path]
"""
import hashlib
import importlib.util
import sys
from types import SimpleNamespace

import torch

EXPECTED = "f84c27ff0b30954dc6d5998662413ae73b3623b5a35b8fda62800650bb0cb390"


def run(model_path):
    spec = importlib.util.spec_from_file_location("mm_under_test", model_path)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    B, T, H, hd, d = 2, 24, 4, 8, 16
    cfg = SimpleNamespace(
        csa=True, csa2=True, csa2_m=4, csa2_top_k=3, csa2_n_win=6,
        csa2_indexer_dim=8, csa2_indexer_heads=2, d=16)
    torch.manual_seed(123)
    m = M.CompressedSparseAttention(cfg, h=4, hd=8).to(torch.float64)
    q = torch.randn(B, T, H, hd, dtype=torch.float64)
    k = torch.randn(B, T, H, hd, dtype=torch.float64)
    v = torch.randn(B, T, H, hd, dtype=torch.float64)
    x = torch.randn(B, T, d, dtype=torch.float64)
    cu = torch.tensor([0, T // 2, T, 2 * T], dtype=torch.int32)
    q, k, v, x = (t.requires_grad_() for t in (q, k, v, x))
    gout = torch.randn(B, T, H, hd, dtype=torch.float64)
    y = m(q, k, v, cu=cu, x=x)
    y.backward(gout)
    h = hashlib.sha256()
    tensors = [y, q.grad, k.grad, v.grad, x.grad]
    tensors += [p.grad for p in m.parameters() if p.grad is not None]
    for t in tensors:
        h.update(t.detach().numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "model.py"
    got = run(path)
    print(got)
    assert got == EXPECTED, f"default path changed: {got} != {EXPECTED}"
    print("DEFAULT IDENTICAL")
