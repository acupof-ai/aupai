"""Consistency test: incremental decode with KV cache vs full prefill.

Compares decode_step logits against full forward at the same position.
Both should agree within bf16 tolerance (~0.5 max diff, argmax match).

Run: .venv/bin/python -m v41f.mlx.tests.test_decode_consistency
Exit code: 0 = pass, 1 = fail.
"""
from __future__ import annotations

import os, sys
import numpy as np
import mlx.core as mx

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from v41f.config import v41f_small
from v41f.lm import V42LM
from v41f.mlx.tests.compare_small import convert_to_mlx, _seed_all
from v41f.mlx.config import MLXV42Config
from v41f.mlx.model import MLXV42Model
from v41f.mlx.generator import MLXGenerator


def main():
    _seed_all(0)
    cfg = v41f_small()
    pt = V42LM(cfg).float().eval()
    W = convert_to_mlx(pt, cfg)
    mcfg = MLXV42Config.from_v41f_config(cfg)
    model = MLXV42Model(mcfg, W)

    prompt = [1, 5, 9, 13, 2, 7, 3, 0]
    s = len(prompt)
    x = mx.array(np.array([prompt], dtype=np.int32))

    # Full forward on all tokens
    full_logits = np.array(model.forward(x).astype(mx.float32))  # [1, s, vocab]

    # Incremental: prefill on prompt[:-1], then decode last token at pos s-1
    gen = MLXGenerator(model)
    gen.prefill(mx.array(np.array([prompt[:-1]], dtype=np.int32)))
    decode_logits = np.array(
        gen.decode_step(mx.array([prompt[-1]], dtype=mx.int32), pos=s - 1).astype(mx.float32)
    )  # [1, vocab]

    expected = full_logits[0, s - 1]  # last position
    diff = np.abs(expected - decode_logits[0])

    print("=== Decode vs Full Forward consistency ===")
    print(f"  prompt: {prompt}")
    print(f"  position: {s-1}")
    print(f"  full argmax:   {expected.argmax()}")
    print(f"  decode argmax: {decode_logits[0].argmax()}")
    print(f"  max abs diff:  {diff.max():.6f}")
    print(f"  mean abs diff: {diff.mean():.6f}")

    # Tolerance: bf16 has ~3 decimal digits. Argmax must match.
    ok_argmax = expected.argmax() == decode_logits[0].argmax()
    ok_diff = diff.max() < 0.5  # bf16 tolerance
    passed = ok_argmax and ok_diff

    if passed:
        print("  PASS")
        return 0
    else:
        print(f"  FAIL (argmax_match={ok_argmax}, diff<0.5={ok_diff})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
