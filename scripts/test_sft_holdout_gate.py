#!/usr/bin/env python3
"""Both directions of sft_math.py's unstamped-pack refusal, by RUNNING the real script.

Not a copy of the comparison: the gate is four lines of argparse-plus-dict logic, and a test that
re-implemented them would pass while the shipped file said something else. Each case builds a
minimal pack, invokes sft_math.py, and reads what it printed.

BOTH SIDES OF THE RUN ARE REAL AS OF 2026-09-19. The first version passed `--resume
/nonexistent_ckpt.pt` and reasoned that "the holdout gate sits BEFORE the model loads, so
reaching the checkpoint error is itself proof the gate let the pack through". Both halves of
that were wrong:

  - the gate sits AFTER the checkpoint load (load was :203, gate :268), so a missing checkpoint
    aborted before the gate ever ran -- every absence-asserting case below passed on a run that
    never reached the code under test;
  - and the one presence-asserting case ("gate is upstream of the ckpt load", asserting the
    checkpoint error's TEXT) was green in both orderings, and greener in the broken one, because
    the checkpoint error appears sooner when the gate is below it.

sft_math.py now loads the pack and runs the holdout/vocab gate BEFORE the checkpoint, so the
gate's own wording is what these cases read, and the checkpoint is a real (minimal) file. The
ordering is asserted directly below, with the corrupt-checkpoint case, and it is the one
assertion that reds if the gate is moved back below the load.
"""
import hashlib
import os
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDOUT = os.path.join(ROOT, "data", "eval", "holdout_hashes.txt")
live = hashlib.sha256(open(HOLDOUT, "rb").read()).hexdigest()[:16]

#: A checkpoint that EXISTS and cannot be unpickled. The gate must refuse an unstamped pack
#: before the loader is ever reached, so the loader's failure is what the ordering case reads.
BROKEN_CKPT_BODY = "this is deliberately not a torch checkpoint\n"


def _tmp(suffix, body=None):
    p = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
    if body is not None:
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(body)
    return p


def ckpt():
    """A minimal but REAL checkpoint: argparse-valid, `ck.get("cfg", {})` empty."""
    p = _tmp(".pt")
    torch.save({"cfg": {}, "model": {}}, p)
    return p


def pack(fp):
    """A pack carrying holdout_fp=fp, or none at all when fp is None."""
    d = {"input_ids": torch.zeros((4, 8), dtype=torch.long),
         "labels": torch.zeros((4, 8), dtype=torch.long), "vocab_id": "x"}
    if fp is not None:
        d["holdout_fp"] = fp
    p = _tmp(".pt")
    torch.save(d, p)
    return p


def run(pack_path, extra=(), ckpt_path=None):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "sft_math.py"),
                        "--resume", ckpt_path or ckpt(), "--sft_path", pack_path,
                        "--out", "/tmp/x.pt", *extra],
                       capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "FLA_FLASH_KDA": "0"})
    out = r.stdout + r.stderr
    # AN IMPORT CRASH MUST NOT COUNT AS EITHER ANSWER. Several checks below assert a string is
    # ABSENT, and sft_math.py failing at an import satisfies that trivially -- the first version
    # of this test printed two "ok" lines on a run that never reached the gate. A test whose
    # passing case is indistinguishable from "the script did not start" measures nothing.
    if "ModuleNotFoundError" in out or "ImportError" in out:
        raise SystemExit(
            "REFUSING: sft_math.py could not import its dependencies here, so the holdout gate "
            "was never reached and an absent-string check would pass vacuously. sft_math.py "
            "guards liger_kernel itself; if this fires, the missing module is a different one.\n"
            + out.strip()[-400:])
    return out

fails = []
def check(name, cond, out):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        fails.append(name)
        print("       ---- output tail ----")
        for l in out.strip().splitlines()[-6:]:
            print("      ", l[:150])

# 1. UNSTAMPED REFUSES, and the message names the flag rather than just complaining.
o = run(pack(None))
check("unstamped pack refuses", "carries NO holdout_fp" in o, o)
check("refusal names --allow_unstamped_pack", "--allow_unstamped_pack" in o, o)

# 2. UNSTAMPED + THE FLAG PROCEEDS, and says so loudly.
o = run(pack(None), ["--allow_unstamped_pack"])
check("flag lets it through", "carries NO holdout_fp" not in o, o)
check("flag prints holdout status unknown", "holdout status unknown" in o, o)

# 3. STALE STILL REFUSES, and the flag must NOT rescue it -- a stale stamp is a KNOWN mismatch,
#    not an unknown, so the escape hatch has no business covering it.
o = run(pack("deadbeefdeadbeef"))
check("stale pack refuses", "was packed against holdout set" in o, o)
o = run(pack("deadbeefdeadbeef"), ["--allow_unstamped_pack"])
check("the flag does NOT rescue a stale stamp", "was packed against holdout set" in o, o)

# 4. A CORRECTLY STAMPED PACK passes the gate with no warning about holdout at all.
o = run(pack(live))
check("live stamp passes", "carries NO holdout_fp" not in o and "was packed against" not in o, o)
check("live stamp prints no unknown-status line", "holdout status unknown" not in o, o)

# 5. THE GATE RUNS BEFORE THE CHECKPOINT IS LOADED. This is the assertion the original file
#    claimed to make and could not: it read the checkpoint error's text, which appears in both
#    orderings. Run the gate's own refusal against a checkpoint that EXISTS but cannot be
#    unpickled -- if the gate is below the load, the loader dies first and the gate's wording is
#    absent. MUTATION: move the holdout block back after `ck = torch.load(args.resume)` and this
#    case reds (measured 2026-09-19: gate text absent, rc=1 from the loader).
o = run(pack(None), ckpt_path=_tmp(".pt", BROKEN_CKPT_BODY))
check("gate refuses before the checkpoint is loaded", "carries NO holdout_fp" in o, o)
check("the broken checkpoint was never reached", "UnpicklingError" not in o and "could not find MARK" not in o, o)

print(f"\n{'ALL OK' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
