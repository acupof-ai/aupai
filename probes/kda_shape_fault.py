#!/usr/bin/env python3
"""Is the KDA misaligned-address crash a SHAPE fault or a model fault?

The card-7 arm-correlation run died with

    torch.AcceleratorError: CUDA error: misaligned address
      triton/runtime/autotuner.py:202 check_disk_cache -> bench_fn
      triton/testing.py:150 do_bench -> torch.cuda.synchronize()

inside fla's Triton AUTOTUNER while benchmarking configs, after both checkpoints had loaded --
not in the forward proper. That leaves two candidates and this probe separates them:

  SHAPE  the per-rank shape the probe used (batch 4, seq 4096, world 1) is one the arms never
         ran, so the autotuner benchmarked configs for a shape with no cache entry. The arms
         trained at batch 16, accum 2, world 2, i.e. a per-rank batch of 16.
  MODEL  something about the checkpointed weights or the model wiring faults regardless of shape.

NO CHECKPOINT IS LOADED. chunk_kda is called directly on random tensors, which is what makes this
cheap (seconds, ~1 GB) and what makes it decisive: if random tensors at the probe's shape crash
and the arms' shape does not, the weights are exonerated without ever reading one.

THE PRIOR IS NOT THE CONCLUSION. facts/efficiency.json#eff.flash_attn_cute_mask_mod_backward_wrong_sm90
records a shape/arch-specific kernel fault on this hardware, which is why SHAPE is the first
hypothesis rather than a guess. It is a prior about the family of bug, not evidence about this one.

Needs a GPU: facts/efficiency.json#eff.model_cannot_forward_on_cpu -- chunk_kda is Triton-only.
Costs one card for under two minutes. Refuses cuda without --allow_cuda, for the same reason
arm_token_corr does: the cards are assigned in runs/card_assignment.json, not by nvidia-smi.
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# The arms' geometry, from their launch lines in runs/experiments.jsonl:
# --dim 1024 --layers 12 --heads 8, world 2, batch 16, accum 2, seq 4096.
HEADS = 8
HEAD_DIM = 1024 // HEADS  # 128
SEQ = 4096

# (label, batch, seq, chunk_size). The two cells 4c asked for, plus the reduction between them
# so a crash boundary can be located rather than just bracketed.
CELLS = [
    ("minimal_b1_t128", 1, 128, 32),
    ("minimal_b1_full_seq", 1, SEQ, 32),
    ("probe_shape_b4", 4, SEQ, 32),          # the shape that crashed
    ("arms_per_rank_b16", 16, SEQ, 32),      # the shape the arms actually trained at
    ("probe_shape_b4_chunk64", 4, SEQ, 64),  # chunk_size is an autotune key
]


def run_cell(batch, seq, chunk, device, dtype=torch.bfloat16):
    """One chunk_kda call on random tensors. Returns (ok, detail).

    torch.cuda.synchronize() after the call, because a CUDA fault is reported
    asynchronously: without it a later unrelated call inherits the error and the cell that
    actually faulted reads as clean.
    """
    from fla.ops.kda import chunk_kda

    try:
        g = torch.Generator(device="cpu").manual_seed(20260905)
        mk = lambda: torch.randn(batch, seq, HEADS, HEAD_DIM, generator=g).to(  # noqa: E731
            device=device, dtype=dtype
        )
        q, k, v, gate = mk(), mk(), mk(), mk()
        beta = torch.randn(batch, seq, HEADS, generator=g).to(device=device, dtype=dtype)
        A_log = torch.randn(HEADS, generator=g).to(device=device, dtype=torch.float32)
        dt_bias = torch.randn(HEADS, generator=g).to(device=device, dtype=torch.float32)
        out, _ = chunk_kda(
            q, k, v, g=gate, beta=beta, A_log=A_log, dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=-5.0,
            state_v_first=True, disable_recompute=True, chunk_size=chunk,
        )
        torch.cuda.synchronize()
        return True, f"ok, out {tuple(out.shape)}"
    except Exception as e:  # noqa: BLE001 -- the crash IS the measurement
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"


def _attr_chain(node):
    """Yield the Attribute/Name nodes of a dotted call target, innermost last.

    `torch.cuda.synchronize` parses as Attribute(Attribute(Name(torch), cuda), synchronize), and
    the selftest needs the dotted string to tell that call from any other `synchronize`.
    """
    import ast

    while isinstance(node, ast.Attribute):
        yield node
        node = node.value
    if isinstance(node, ast.Name):
        yield node


def verdict_for(passed, failed):
    """The verdict, as a function so the selftest can EXERCISE it rather than read main()'s text.

    It lived inline in main() and a mutant that changed `if failed and passed` to `if failed`
    stayed green: the selftest was comparing the ORDER of three string literals in the source,
    which the mutation did not disturb. A verdict that can say SHAPE when nothing passed is the
    one wrong answer that matters here -- it would exonerate the weights on no evidence.
    """
    if failed and passed:
        return "SHAPE: the crashing shape failed while another shape passed in the same process"
    if failed:
        return "MODEL-OR-ENVIRONMENT: the minimal cell itself failed, so no shape is safe here"
    return ("NOT REPRODUCED: every cell passed, so the fault needs something this probe omits "
            "(loaded weights, a warmed autotune cache, or the full model's call sequence)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow_cuda", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if "cuda" in a.device and not a.allow_cuda:
        sys.exit("REFUSING --device cuda without --allow_cuda: cards are assigned in "
                 "runs/card_assignment.json, not by an idle nvidia-smi row.")
    if "cuda" not in a.device:
        sys.exit("This probe needs a GPU: chunk_kda is Triton-only with no CPU fallback "
                 "(facts/efficiency.json#eff.model_cannot_forward_on_cpu). Pass --device cuda:0 "
                 "--allow_cuda on a granted card.")

    rows = []
    for label, batch, seq, chunk in CELLS:
        # EACH CELL IN ITS OWN SUBPROCESS would be cleaner still, but a CUDA fault poisons the
        # context: after a misaligned address every later call in this process raises the same
        # error. So the loop STOPS at the first failure and says so, rather than reporting four
        # crashes that are one crash.
        ok, detail = run_cell(batch, seq, chunk, a.device)
        print(f"  {label:26s} batch={batch:3d} seq={seq:5d} chunk={chunk:3d}  "
              f"{'PASS' if ok else 'FAIL'}  {detail}", flush=True)
        rows.append({"cell": label, "batch": batch, "seq": seq, "chunk": chunk,
                     "ok": ok, "detail": detail})
        if not ok:
            print("  STOPPING: a CUDA fault poisons the context, so every later cell in this "
                  "process would report the same error whether or not it is faulty. Re-run the "
                  "remaining cells in a fresh process.", flush=True)
            break

    passed = [r["cell"] for r in rows if r["ok"]]
    failed = [r["cell"] for r in rows if not r["ok"]]
    verdict = verdict_for(passed, failed)
    print(f"\n  verdict: {verdict}")
    out = {"cells": rows, "passed": passed, "failed": failed, "verdict": verdict,
           "device": a.device, "heads": HEADS, "head_dim": HEAD_DIM}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
    print(json.dumps(out))
    return 0


def _selftest():
    # 1. THE CELL TABLE CONTAINS BOTH SHAPES THE RULING ASKED FOR, and the crashing one. A table
    #    that lost the arms' shape could only ever return "everything crashes".
    labels = {c[0] for c in CELLS}
    assert "minimal_b1_t128" in labels and "arms_per_rank_b16" in labels, labels
    probe = next(c for c in CELLS if c[0] == "probe_shape_b4")
    arms = next(c for c in CELLS if c[0] == "arms_per_rank_b16")
    assert probe[1] == 4 and arms[1] == 16, (probe, arms)
    assert probe[2] == arms[2] == SEQ, (
        f"the crashing cell and the arms' cell must differ in BATCH ONLY, or a difference in "
        f"verdict cannot be attributed to batch: seq {probe[2]} vs {arms[2]}"
    )
    assert probe[3] == arms[3], (
        f"chunk_size differs between the two comparison cells ({probe[3]} vs {arms[3]}); it is "
        f"an autotune key, so that confounds the comparison"
    )
    assert HEAD_DIM == 128, HEAD_DIM

    # 2. THE LOOP STOPS AT THE FIRST FAILURE. A CUDA fault poisons the context, so continuing
    #    would report every later cell as broken and manufacture a MODEL verdict out of one
    #    SHAPE crash -- the exact wrong conclusion.
    import inspect

    src = inspect.getsource(main)
    assert "if not ok:" in src and "break" in src, (
        "the cell loop no longer stops at the first failure; a poisoned CUDA context would then "
        "report every later cell as failing and turn one crash into a MODEL verdict"
    )
    # THE SYNCHRONIZE IS CHECKED AT ITS CALL SITE, not by grepping the function. My first
    # version asserted `"torch.cuda.synchronize()" in getsource(run_cell)`, and deleting the
    # call left that assertion green -- run_cell's own DOCSTRING mentions synchronize, so the
    # substring survived the mutation. A text search over a function that documents itself
    # matches its prose.
    import ast

    _tree = ast.parse(inspect.getsource(run_cell).lstrip())
    _calls = {
        ".".join(
            part.id if isinstance(part, ast.Name) else part.attr
            for part in reversed(list(_attr_chain(n.func)))
        )
        for n in ast.walk(_tree)
        if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))
    }
    assert "torch.cuda.synchronize" in _calls, (
        f"run_cell does not CALL torch.cuda.synchronize (calls found: {sorted(_calls)}). A CUDA "
        f"fault is reported asynchronously, so without it the faulting cell reads clean and a "
        f"later innocent cell inherits the error."
    )

    # 3. THE VERDICT LOGIC, exercised on synthetic outcomes. It is a function precisely so this
    #    can call it: the previous inline version was only checked by the ORDER of three string
    #    literals in main()'s source, and a mutant that let SHAPE fire with nothing passing kept
    #    that order intact and stayed green.
    assert verdict_for(["a"], ["b"]).startswith("SHAPE")
    assert verdict_for([], ["a"]).startswith("MODEL-OR-ENVIRONMENT")
    assert verdict_for(["a", "b"], []).startswith("NOT REPRODUCED")
    # The one wrong answer that matters: SHAPE exonerates the weights, so it must be unreachable
    # when nothing passed.
    for _p, _f in (([], ["x"]), ([], ["x", "y"]), ([], [])):
        assert not verdict_for(_p, _f).startswith("SHAPE"), (_p, _f, verdict_for(_p, _f))

    # 4. CPU IS REFUSED WITH THE REASON, not attempted. chunk_kda on CPU raises "0 active
    #    drivers" from inside Triton, which reads like a broken environment rather than an
    #    architectural fact, and that misreading already cost one debugging cycle.
    assert "eff.model_cannot_forward_on_cpu" in src, (
        "the CPU refusal no longer cites the fact; the raw Triton error reads as a broken "
        "environment and sends the next reader after the wrong thing"
    )
    assert 'if "cuda" in a.device and not a.allow_cuda' in src

    print("kda_shape_fault selftest OK: the cell table keeps both the crashing shape (batch 4) "
          "and the arms' per-rank shape (batch 16) differing in batch only, the loop stops at "
          "the first failure so a poisoned CUDA context cannot manufacture a MODEL verdict, the "
          "call synchronizes so an async fault lands on the cell that caused it, the verdict "
          "cannot read SHAPE unless something passed, and CPU is refused citing the fact rather "
          "than raising Triton's misleading '0 active drivers'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
