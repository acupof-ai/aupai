"""fp8 head: does a planted outlier destroy the capped logits sharing its chunk?

The absmax fact (p50 48 / p99 62 / max 82 vs e4m3's 448) is used to justify PER-TENSOR scaling.
That argument is only safe if a single large element cannot crush the elements that matter. This
plants one element far above the rest and checks the ordinary ones survive the round trip --
testing the mechanism the range fact is used to dismiss, rather than the comfortable case.
"""
import sys

import torch

sys.argv = ["x"]
sys.path.insert(0, "/work/aupai/scripts")
sys.path.insert(0, "/work/aupai")
import train_t as t  # noqa: E402

M, N = 2048, 32784
for outlier in (82.0, 448.0, 4480.0):
    a = torch.full((M, 1024), 0.0, device="cuda", dtype=torch.bfloat16)
    a.normal_(0, 0.5)
    b = torch.empty(1024, N, device="cuda", dtype=torch.bfloat16).normal_(0, 0.02)
    # plant: one row of a scaled so its product reaches `outlier`, rest near p50
    a[0] *= outlier / max(a[0].abs().max().item(), 1e-9)
    ref = (a.float() @ b.float())
    got = t._fp8_mm(a, b, torch.float32)
    # the ORDINARY rows are what must survive; row 0 is the planted outlier
    ord_ref, ord_got = ref[1:], got[1:]
    rel = ((ord_got - ord_ref).abs() / ord_ref.abs().clamp(min=1e-6))
    cos = torch.nn.functional.cosine_similarity(ord_ref.flatten(), ord_got.flatten(), dim=0)
    scale = a.detach().abs().amax() / 448.0
    print(f"outlier {outlier:7.1f} -> a_scale {scale:.4f} | ordinary rows: "
          f"cosine {cos:.6f} median rel {rel.median() * 100:.3f}% | "
          f"finite {torch.isfinite(got).all().item()}")
