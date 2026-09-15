#!/usr/bin/env python3
"""v42_phi_sft.sh must launch torchrun WITHOUT a pre-training card_claim acquire.

The defect (fb, measured 2026-09-15): the launcher called
`card_claim.py acquire --wait-for-device 300` BEFORE torchrun. At that instant the
shell has no GPU-holding descendant, so _resolve_to_device_holder polls an empty tree
for the full deadline, exp is marked fail, and the script exits having never trained.
The outer acquire is also redundant: sft_math self-claims each card per rank through
load_checkpoint -> claim_my_cards (loader.py:87), and those claims lapse on rank exit.

The gate is checked by RUNNING the launcher with torchrun stubbed (not by grepping it):
  * the torchrun stub must run (training is reached) and record its argv;
  * the card_claim CLI stub writes a marker if ever invoked with `acquire` -- that
    marker must be ABSENT (old launcher called acquire, it returned nonzero, the
    launcher REFUSED and torchrun never ran, so the old script fails this test);
  * wall time is small -- the bug path blocks for the device-wait deadline, the fixed
    path reaches the stub immediately.

The read-only live-claim preflight still runs: the stub card_claim MODULE exposes
claims()=([],{}) so the inline preflight passes. exp.py and sft_math.py are stubbed.

    python3 scripts/test_phisft_launcher.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER = os.path.join(ROOT, "runs", "v42_phi_sft.sh")
MARKER_ACQUIRE = "acquire_was_called"
MARKER_TORCHRUN = "torchrun_ran"


def _build_tree():
    d = tempfile.mkdtemp(prefix="phisft_")
    for sub in ("runs", "scripts", "eval", "bin", "data/sft"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    shutil.copy(LAUNCHER, os.path.join(d, "runs", "v42_phi_sft.sh"))
    os.chmod(os.path.join(d, "runs", "v42_phi_sft.sh"), 0o755)
    shutil.copy(os.path.join(ROOT, "eval", "_devs.sh"), os.path.join(d, "eval", "_devs.sh"))

    # Minimal real pack the NOTES inline block can torch.load(weights_only=True), and a
    # resume ckpt whose existence the launcher checks.
    import torch
    torch.save({"input_ids": torch.zeros(4, 2, dtype=torch.long),
                "labels": torch.zeros(4, 2, dtype=torch.long),
                "vocab_id": "stub-vocab"},
               os.path.join(d, "data/sft/sft_phi_codeexercises_v42_65m_0914.pt"))
    open(os.path.join(d, "ckpt_v41_r3_0914.pt"), "w").close()

    # sft_math.py: --check_pack (cardless gate) exits 0; any training argv also 0.
    open(os.path.join(d, "sft_math.py"), "w").write(
        "import sys\nsys.exit(0)\n")

    # exp.py: accept start/done no matter the flags.
    open(os.path.join(d, "scripts", "exp.py"), "w").write(
        "import sys\nsys.exit(0)\n")

    # card_claim: MODULE used by the read-only preflight (claims returns nothing held);
    # CLI records and refuses any `acquire` so the old launcher cannot reach torchrun.
    stub = os.path.join(d, "scripts", "card_claim.py")
    open(stub, "w").write(
        "def claims():\n"
        "    return ([], {})\n"
        "if __name__ == '__main__':\n"
        "    import os, sys\n"
        "    if 'acquire' in sys.argv:\n"
        "        open(os.environ['CARD_CLAIM_MARKER'], 'w').close()\n"
        "        sys.exit(3)\n"
        "    sys.exit(0)\n")

    # torchrun stub: record that launch reached training, then exit 42 so the launcher
    # takes its training-fail path and never reaches the (absent) post-eval step.
    tr = os.path.join(d, "bin", "torchrun")
    open(tr, "w").write(
        "#!/bin/bash\n"
        'echo "$@" > "$TREE/markers/torchrun_argv"\n'
        'touch "$TREE/markers/' + MARKER_TORCHRUN + '"\n'
        "exit 42\n")
    os.chmod(tr, 0o755)
    return d


def main():
    d = _build_tree()
    markers = os.path.join(d, "markers")
    os.makedirs(markers)
    env = dict(os.environ, PATH=os.path.join(d, "bin") + os.pathsep + os.environ["PATH"],
               TREE=d, HYPOTHESIS="selftest world",
               CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
               CARD_CLAIM_MARKER=os.path.join(markers, MARKER_ACQUIRE))

    import time
    t0 = time.time()
    p = subprocess.run(["bash", os.path.join(d, "runs", "v42_phi_sft.sh")],
                       cwd=d, env=env, capture_output=True, text=True, timeout=60)
    dt = time.time() - t0

    tr_ran = os.path.exists(os.path.join(markers, MARKER_TORCHRUN))
    acquired = os.path.exists(os.path.join(markers, MARKER_ACQUIRE))

    assert tr_ran, ("launcher never reached torchrun; stdout/stderr:\n"
                    + p.stdout[-800:] + p.stderr[-800:])
    assert not acquired, "launcher called card_claim acquire before torchrun"
    assert dt < 60, f"launcher blocked {dt:.1f}s -- device-wait deadline path still live"
    # torchrun got the 8-rank SFT argv.
    argv = open(os.path.join(markers, "torchrun_argv")).read()
    assert "--nproc_per_node=8" in argv and "sft_math.py" in argv, argv
    # Source-level belt: no card_claim acquire invocation survives in the launcher.
    src = open(LAUNCHER, encoding="utf-8").read()
    assert "card_claim.py acquire" not in src, "shell still invokes card_claim acquire"

    print(f"phi SFT launcher OK: reached torchrun (8-rank sft_math) with no pre-training "
          f"card_claim acquire in {dt:.1f}s; ranks self-claim via loader")
    shutil.rmtree(d, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
