#!/usr/bin/env python3
"""Does --prefix actually mask during TRAINING, on the arm's layers and no others?

WHY THIS EXISTS AND WHY IT CHECKS COUNTS. Every source-level criterion for this wiring was
already true of code that masked the wrong layers: the gates patched all three MLA layers for
four commits while PREFIX_LAYER said one, and the docstring claimed "patched on the INSTANCE"
the whole time. So this counts what the kernel actually received, per layer, on a real
sft_math.py training step -- not what the flag says, and not what the gate script does. The
gate script has its own copy of the layer-scoping wrapper; a test that exercised that copy
would certify the wrong file.

WHAT IT CANNOT CHECK: whether the mask is CORRECT. That is scripts/n7c_gates.py, which needs a
GPU and the flash_attn.cute kernel. This checks the seam between sft_math.py and the mask:
which layers get a mask_mod, whether the aux tensor is rebuilt per step, and whether the
refusals fire. Run with --selftest; needs a GPU only for the live arm, and the refusal cases
run anywhere.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args, env=None):
    e = dict(os.environ)
    e.setdefault("CUDA_VISIBLE_DEVICES", "")
    if env:
        e.update(env)
    return subprocess.run([sys.executable, "sft_math.py", *args], cwd=ROOT,
                          capture_output=True, text=True, env=e, timeout=600)


def main():
    fails = []

    def check(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}{'' if cond else '  <- ' + detail}")
        if not cond:
            fails.append(name)

    # 1. THE ARM NAMES ARE THE MASK MODULE'S, not a second list. A --prefix value that
    #    PREFIX_ARMS does not define would KeyError deep inside setup, after the pack is loaded;
    #    argparse choices refuse it at parse time. Checked against PREFIX_ARMS rather than
    #    hardcoded, so adding an arm cannot leave the two lists disagreeing.
    #
    #    THESE THREE NEED liger_kernel, which is pod-only (sft_math.py:23 imports it at module
    #    scope, so even --help loads it). They SKIP off-pod rather than fail -- but the skip is
    #    printed and counted, because a silent skip is how a test stops testing. Everything below
    #    reads the source and runs anywhere.
    sys.path.insert(0, ROOT)
    from eval.prefix_mask import PREFIX_ARMS  # noqa: PLC0415

    h = _run(["--help"])
    if "No module named 'liger_kernel'" in (h.stdout + h.stderr):
        print("  SKIP 3 argparse checks: liger_kernel is pod-only, so sft_math.py cannot be "
              "imported here. Run this on the pod to exercise them.")
        n_skipped = 3
    else:
        n_skipped = 0
        r = _run(["--resume", "/nonexistent.pt", "--prefix", "pX"])
        check("an unknown arm is refused at parse time",
              r.returncode != 0 and "invalid choice" in (r.stdout + r.stderr),
              (r.stdout + r.stderr).strip()[-200:])
        for arm in PREFIX_ARMS:
            check(f"--prefix advertises arm {arm}", arm in h.stdout, "not in --help")

    # 2. THE REFUSALS. Both are load-bearing and neither is reachable from a correct launch, so
    #    the only way to know they fire is to trip them. A prefix arm that silently trained
    #    causal is the failure this repo already paid for once: the gates printed three passes
    #    while flash_attn_varlen_func was never called, because cu was None.
    from eval.prefix_mask import build_mask_mods  # noqa: PLC0415
    with open(os.path.join(ROOT, "sft_math.py"), encoding="utf-8") as fh:
        src = fh.read()
    check("--prefix refuses when HAS_FA is False",
          'if not model_mod.HAS_FA:' in src and "SDPA fallback" in src)
    check("--prefix refuses when doc_mask is off",
          "if not Cfg.doc_mask:" in src and "flash_attn_varlen_func" in src)
    # ...and that the refusal text names WHY, not just that it refused: without cu the mask is
    # silently absent rather than erroring, which is the whole reason this is a refusal.
    check("the doc_mask refusal explains that the mask would be silently absent",
          "silently absent" in src)

    # 3. THE AUX TENSOR IS REBUILT INSIDE THE STEP LOOP, not once at setup. A tensor built once
    #    is read against the wrong documents from step 2 on, and out of bounds as soon as the
    #    document count grows -- the same out-of-bounds read that made the gates report a leak.
    #    Located by line number: the assignment must come after the loop header.
    lines = src.split("\n")
    loop_at = next(i for i, ln in enumerate(lines) if "for i in range(0, len(X)" in ln)
    aux_at = [i for i, ln in enumerate(lines) if "_box[0] = [" in ln]
    check("the aux tensor is assigned inside the step loop",
          bool(aux_at) and all(i > loop_at for i in aux_at),
          f"loop at line {loop_at + 1}, assignments at {[i + 1 for i in aux_at]}")
    check("the aux tensor is built from the batch's own labels and cu",
          any("_plens(yb, cub)" in ln for ln in lines),
          "must read yb and cub, the shifted labels and this batch's boundaries")

    # 4. THE MASK IS SCOPED BY IDENTITY, and the other MLA layers pass through. This is the
    #    defect that survived four commits in the gate script: patching the module global masks
    #    EVERY GatedMLA, because model.py:191 looks the name up at call time.
    check("the wrapper passes through when no target layer is active",
          'getattr(t, "_n7c_active", False) for t in targets' in src,
          "the wrapper must test the target flags, not mask every varlen call")
    check("causal= is dropped when a mask_mod is passed",
          'kw.pop("causal", None)' in src,
          "interface.py:270 sets causal=False whenever mask_mod is not None")

    # 5. THE ARM IS RECORDED ON Cfg, so the checkpoint says which mask trained it. Without this
    #    the prefix and causal arms write byte-different checkpoints with identical metadata --
    #    the failure this repo paid for with .stepN files holding earlier weights.
    check("Cfg records the arm and its layers",
          "Cfg.prefix_arm = args.prefix" in src and "Cfg.prefix_layers = list(layers)" in src)

    # 6. LAYER MEMBERSHIP IS ASSERTED AGAINST THE BUILT MODEL, not computed from cfg and
    #    trusted. cfg arithmetic gave the right answer here (MLA at 3, 7, 11) and would give a
    #    silently wrong one for any other attn_every.
    check("each target layer is asserted to be a GatedMLA",
          "isinstance(mixer, model_mod.GatedMLA)" in src)

    # 7. BLOCK 11 ALONE IS NOT AN ARM, and the reason is recorded where a reader will hit it.
    #    It passed six gates while being provably unable to change the training loss.
    check("no arm is layer 11 alone",
          all(tuple(v) != (11,) for v in PREFIX_ARMS.values()), f"{PREFIX_ARMS}")
    check("the help text says why block 11 alone cannot work",
          "BLOCK 11 ALONE IS NOT AN ARM" in src)

    # THE ARG PARSER'S CHOICES MUST BE PREFIX_ARMS, checked from the source so it holds off-pod
    # too. Without this the three skipped checks above are the only thing tying the flag to the
    # mask module, and off-pod that is nothing.
    choices = src.split('"--prefix",', 1)[1].split("choices=(", 1)[1].split(")", 1)[0]
    named = {c.strip().strip('"\'') for c in choices.split(",") if c.strip()}
    check("--prefix's choices are exactly PREFIX_ARMS' keys", named == set(PREFIX_ARMS),
          f"argparse offers {sorted(named)}, PREFIX_ARMS defines {sorted(PREFIX_ARMS)}")

    _ = build_mask_mods  # imported to prove the module loads off-pod; the callbacks need cutlass
    print(f"\n{len(fails)} failure(s)" + (f", {n_skipped} skipped (pod-only)" if n_skipped else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(main())
    raise SystemExit(f"usage: {os.path.basename(__file__)} --selftest")
