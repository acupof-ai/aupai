#!/usr/bin/env python3
"""b0-8 condition 2: the split changed no bit of the forward pass.

Builds the SAME architecture twice in ONE process -- once from the merge-base
train.py (the file before model.py existed), once from today's model.py -- loads
one set of weights into both, and requires `torch.equal` on the logits.

WHY torch.equal AND NOT allclose: moving code must not change any arithmetic. If
only allclose passes, the move reordered a computation, and that is a different
change needing its own ruling. `test_arch_compat.py:119` uses allclose, but it
is checking legacy KEY REMAPPING (w1|w3 -> w13), where a tolerance is right --
it never compares pre-split against post-split code, so condition 2 was not
covered by it. Existence of a passing test is not coverage of the claim.

WHY ONE PROCESS: across two processes, cuBLAS workspace and autotune caches
differ, so bitwise comparison goes FALSE RED -- worse than no check, because it
sends someone hunting a bug that is not there (tilerl's constraint, 2026-09-02).

    python3 scripts/test_split_bitwise.py            # exit 1 on any difference
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

# The last commit where train.py still held the model: the parent of the split.
SPLIT_COMMIT = "be845ec"


def load_pre_split_module(tmpdir):
    """Import the pre-split train.py under a private name, from git, not the worktree."""
    base = subprocess.run(["git", "rev-parse", f"{SPLIT_COMMIT}^"],
                          capture_output=True, text=True, check=True).stdout.strip()
    src = subprocess.run(["git", "cat-file", "-p", f"{base}:train.py"],
                         capture_output=True, text=True, check=True).stdout
    # `train.py` at that commit defines the model itself and imports no `model`
    # module, which is the whole point of the comparison.
    assert "from model import" not in src, (
        f"{base[:12]}:train.py already imports model -- SPLIT_COMMIT is wrong, so this would "
        f"compare the split against itself and pass for the wrong reason")
    assert "class HybridLM" in src, f"{base[:12]}:train.py has no HybridLM"
    path = Path(tmpdir) / "train_presplit.py"
    path.write_text(src)
    # tmpdir FIRST so `train_presplit` resolves, and the repo root SECOND so its
    # sibling imports (`fone`, ...) resolve to the same modules today's model.py
    # gets. Without the repo root the import dies on `No module named 'fone'`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, tmpdir)
    import train_presplit  # noqa: PLC0415

    return train_presplit


def main():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        pre = load_pre_split_module(tmp)
        import model as post  # noqa: PLC0415

        # No fla on this machine (needs triton): the real chunk_kda cannot run on CPU. Patch a
        # shape-preserving stand-in into BOTH modules, the same way test_arch_compat.py:33 does.
        # It must be set on each module SEPARATELY: train.py's `chunk_kda` is a re-exported
        # binding, and rebinding the name in `model` does not rebind train's copy.
        if getattr(post, "chunk_kda", None) is None or getattr(pre, "chunk_kda", None) is None:
            stub = lambda q, k, v, **kw: (q * 0 + v, None)  # noqa: E731
            pre.chunk_kda = post.chunk_kda = stub
            print("note: fla unavailable -- chunk_kda stubbed in BOTH modules. The comparison "
                  "still covers every line the split moved (norms, gating, projections, "
                  "AttnRes, residual wiring); it does NOT cover the fla call itself, which "
                  "the split did not touch.")

        cfg = pre.Cfg()
        # Small enough to build on CPU, deep enough to include both block kinds and
        # the MLA positions the real config uses (every 4th block).
        cfg.d, cfg.layers, cfg.heads, cfg.ffn_hidden = 128, 4, 4, 256
        cfg.vocab = 512
        cfg.vocab_real = 500

        a = pre.HybridLM(cfg).eval()
        b = post.HybridLM(cfg).eval()

        # RANDOMIZE EVERY PARAMETER BEFORE COMPARING. At init every norm gain is exactly 1.0,
        # and 1.0 makes whole classes of arithmetic change invisible: `x * r * g` and
        # `x * (r * g)` are bitwise equal when g is ones, so a reassociation introduced during
        # the move passes. Verified -- that exact broken world exited 0 against the earlier
        # version of this script, which is why the randomization is here and not a nicety.
        # A test whose inputs sit on the identity element cannot see a change in the operation.
        torch.manual_seed(1234)
        with torch.no_grad():
            for q in a.parameters():
                q.normal_(mean=0.7, std=0.5)  # off zero AND off one
        assert not any(torch.equal(q, torch.ones_like(q)) for q in a.parameters()), \
            "a parameter is still all-ones, so degenerate arithmetic stays invisible"

        # ONE set of weights, strict -- a missing key here would silently leave half
        # the second model at its init and still produce plausible logits.
        r = b.load_state_dict(a.state_dict(), strict=True)
        assert not r.missing_keys and not r.unexpected_keys, r

        x = torch.randint(0, cfg.vocab_real, (2, 16))
        with torch.no_grad():
            la = a(x)
            lb = b(x)
        la = la[0] if isinstance(la, tuple) else la
        lb = lb[0] if isinstance(lb, tuple) else lb

        if not torch.equal(la, lb):
            d = (la - lb).abs().max().item()
            close = torch.allclose(la, lb, atol=1e-5)
            print(f"FAIL logits differ: max |delta| {d:.3e}, allclose={close}", file=sys.stderr)
            print("  allclose but not equal means the move REORDERED arithmetic -- that is a "
                  "behaviour change, not a move, and needs its own ruling.", file=sys.stderr)
            return 1

        n = sum(p.numel() for p in a.parameters())
        print(f"OK logits bitwise identical (torch.equal) across the split: "
              f"{tuple(la.shape)} logits, {n:,} params, {cfg.layers} blocks, one process, "
              f"pre-split source read from {SPLIT_COMMIT}^:train.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
