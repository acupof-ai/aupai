#!/usr/bin/env python3
"""load_checkpoint claims the card when it loads to cuda, and claims nothing when it loads to cpu.

de-55 step 2. MEASURED 2026-09-05: 31 files load a checkpoint and move it to a device; 23 of them
never called card_claim, so runs/claims/ sat empty on the pod while lane jobs held cards all day and
card_held_without_claim -- which detects exactly that -- reports SKIP off-pod, so nobody saw it. 19 of
the 23 reach the card through scripts.loader.load_checkpoint, which is why one acquire there covers
the most of them.

The row said 10, not 23. Its predicate was "loads a checkpoint and NAMES cuda", which matches 53
files including every one that mentions the word in prose. The right predicate is "loads a checkpoint
AND moves it to a device": .to(dev), device="cuda", or .cuda().

WHAT IS TESTED, and world 2 is the load-bearing one:
  1 device="cuda..." acquires, once, under a name a human can read in `card_claim.py status`.
  2 device="cpu" acquires NOTHING. claim_my_cards' docstring rejected an acquire on this path partly
    because its device defaults to cpu, and that objection is correct for an UNGATED call: CI runs
    test_arch_compat on a machine with no fla where DEV == "cpu", and every scoring pass that loads
    to cpu and moves later would refuse on a box with no CUDA_VISIBLE_DEVICES. The gate is what makes
    the placement safe, so a test that only checked world 1 would pass for the version that bricks CI.
  3 claim=False acquires nothing even on cuda -- for a caller already holding the card under a
    launcher's name, where a second acquire under a different name would clash with the first.
  4 TWO loads in one process do not refuse the second. eval/domain_loss.py, probes/arm_token_corr.py
    and scripts/e1_n8_row_edge_probe.py call load_checkpoint 2, 2 and 3 times, so without step 1's
    idempotency this placement would make three working scripts refuse themselves, naming their own
    pid as the occupant.

NO GPU AND NO CHECKPOINT ARE NEEDED, and that is deliberate rather than a compromise: the subject is
which acquires happen, not what torch does afterwards. torch.load is stubbed to raise a sentinel, so
the call dies immediately AFTER the claim decision -- which also proves the claim happens BEFORE the
load rather than after it, the ordering that matters when the load is what takes the memory.

restartable: yes -- temp CLAIM_DIR removed in a finally; CUDA_VISIBLE_DEVICES, card_claim.CLAIM_DIR
and torch.load are all restored. Nothing reads or writes the repository's real runs/claims/.
"""
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
# ROOT too: load_checkpoint does `from train import Cfg, HybridLM` before the claim block, and
# without this the whole function dies in that import and every world reports "failed before
# torch.load" -- a test that cannot reach its subject. The import order is correct as it stands:
# `import torch` and the train import allocate nothing on a card, and the claim still precedes
# torch.load, which is where the memory is taken.
sys.path.insert(0, ROOT)


class _Sentinel(Exception):
    """Raised by the stubbed torch.load: the claim decision is already made by then."""


def main():
    fails = []
    import card_claim
    import torch

    import loader

    d = tempfile.mkdtemp(prefix="loaderclaim_")
    saved_dir = card_claim.CLAIM_DIR
    saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    saved_load = torch.load
    try:
        card_claim.CLAIM_DIR = d
        os.environ["AUPAI_CLAIM_DIR"] = d
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"

        def _stub(*a, **k):
            raise _Sentinel("torch.load reached")

        torch.load = _stub

        def _attempt(**kw):
            """Call load_checkpoint and report the claims it left behind.

            SystemExit is caught EXPLICITLY, and that is not defensive breadth. claim_my_cards
            refuses with `raise SystemExit(...)`, and SystemExit derives from BaseException, not
            Exception -- so an `except Exception` handler lets it through and the test process dies
            with the refusal on stderr instead of recording it as a failed world. Measured while
            mutation-testing this file: reverting step 1's idempotency made world 4's refusal escape
            as an uncaught SystemExit, so the mutation was caught by a crash rather than by the
            assertion that names it, which proves nothing about the assertion.
            """
            try:
                loader.load_checkpoint("/nonexistent/ckpt.pt", **kw)
            except _Sentinel:
                pass
            except SystemExit as e:
                return f"claim refused: {str(e)[:150]}", sorted(os.listdir(d))
            except Exception as e:  # noqa: BLE001
                return f"{type(e).__name__}: {str(e)[:120]}", sorted(os.listdir(d))
            files = sorted(os.listdir(d))
            return None, files

        # WORLD 1: a cuda load claims.
        err, files = _attempt(device="cuda:0")
        if err:
            fails.append(f"1: load_checkpoint(device='cuda:0') failed before torch.load: {err}. The "
                         f"claim decision is upstream of the load, so this is the claim path itself "
                         f"raising, not a missing checkpoint.")
        elif len(files) != 1:
            fails.append(f"1: a cuda load left {files}, expected exactly one claim file. Empty means "
                         f"the acquire did not run BEFORE torch.load -- either it is not on this "
                         f"path at all (19 of the 23 uncovered entry points reach a card through "
                         f"here, which is the whole gap) or it sits after the load, where the memory "
                         f"is already taken. The stub raises at torch.load, so both read the same "
                         f"here, and both are the defect.")
        else:
            # The name must be readable: a card in `status` labelled after the tool, not "load".
            if "test_loader_claim" not in files[0]:
                fails.append(f"1: the claim is named {files[0]!r}. It should carry the invoking "
                             f"script's stem, so two tools on one card are distinguishable in "
                             f"card_claim.py status.")
        for f in os.listdir(d):
            os.unlink(os.path.join(d, f))

        # WORLD 2, THE LOAD-BEARING ONE: a cpu load claims nothing.
        err, files = _attempt(device="cpu")
        if err:
            fails.append(f"2: a CPU load raised before torch.load: {err}. A cpu load must not touch "
                         f"the claim path at all -- CI runs test_arch_compat where DEV == 'cpu' with "
                         f"no CUDA_VISIBLE_DEVICES, and an ungated acquire refuses there.")
        if files:
            fails.append(f"2: a CPU load left claims {files}. This is the objection claim_my_cards' "
                         f"docstring recorded against putting the acquire here, and it is correct "
                         f"for an UNGATED call: 7 of the 19 callers load to cpu and move later, CI "
                         f"loads on cpu with no cards, and every one of them would refuse. The gate "
                         f"on the resolved device is what makes the placement safe.")
        for f in os.listdir(d):
            os.unlink(os.path.join(d, f))

        # WORLD 3: claim=False opts out even on cuda.
        err, files = _attempt(device="cuda:0", claim=False)
        if files:
            fails.append(f"3: claim=False still claimed {files}. A caller already holding the card "
                         f"under a launcher's name needs the opt-out: a second acquire under a "
                         f"different name clashes with the first.")
        for f in os.listdir(d):
            os.unlink(os.path.join(d, f))

        # WORLD 4: two loads in one process. Without step 1's idempotency the second refuses.
        err1, _ = _attempt(device="cuda:0")
        err2, files2 = _attempt(device="cuda:0")
        if err2:
            fails.append(f"4: the SECOND load in the same process failed: {err2}. "
                         f"eval/domain_loss.py, probes/arm_token_corr.py and "
                         f"scripts/e1_n8_row_edge_probe.py call load_checkpoint 2, 2 and 3 times, so "
                         f"a refusal here breaks three working scripts -- each naming its own pid as "
                         f"the occupant of the card it already holds.")
        elif len(files2) != 1:
            fails.append(f"4: two loads left {files2}, expected one claim file. A second file is a "
                         f"second claim on a card already held.")
    finally:
        torch.load = saved_load
        card_claim.CLAIM_DIR = saved_dir
        os.environ.pop("AUPAI_CLAIM_DIR", None)
        if saved_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd
        shutil.rmtree(d, ignore_errors=True)

    if fails:
        print("test_loader_claim FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("test_loader_claim ok: load_checkpoint claims the card it is about to load onto when the "
          "device is cuda and claims NOTHING on a cpu load (so CI and the 7 load-to-cpu callers are "
          "unaffected), claim=False opts out, and a second load in the same process reuses the claim "
          "instead of refusing itself")
    return 0


if __name__ == "__main__":
    sys.exit(main())
