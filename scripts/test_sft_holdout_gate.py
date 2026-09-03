#!/usr/bin/env python3
"""Both directions of sft_math.py's unstamped-pack refusal, by RUNNING the real script.

Not a copy of the comparison: the gate is four lines of argparse-plus-dict logic, and a test that
re-implemented them would pass while the shipped file said something else. Each case builds a
minimal pack, invokes sft_math.py, and reads what it printed -- the runs die later on a missing
checkpoint, which is fine: the holdout gate sits BEFORE the model loads, so reaching the
checkpoint error is itself proof the gate let the pack through.
"""
import hashlib, os, subprocess, sys, tempfile, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDOUT = os.path.join(ROOT, "data", "eval", "holdout_hashes.txt")
live = hashlib.sha256(open(HOLDOUT, "rb").read()).hexdigest()[:16]

def pack(fp):
    """A pack carrying holdout_fp=fp, or none at all when fp is None."""
    d = {"input_ids": torch.zeros((4, 8), dtype=torch.long),
         "labels": torch.zeros((4, 8), dtype=torch.long), "vocab_id": "x"}
    if fp is not None:
        d["holdout_fp"] = fp
    p = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
    torch.save(d, p)
    return p

def run(pack_path, extra=()):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "sft_math.py"),
                        "--resume", "/nonexistent_ckpt.pt", "--sft_path", pack_path,
                        "--out", "/tmp/x.pt", *extra],
                       capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "FLA_FLASH_KDA": "0"})
    out = r.stdout + r.stderr
    # AN IMPORT CRASH MUST NOT COUNT AS EITHER ANSWER. Two of the checks below assert a string is
    # ABSENT, and sft_math.py failing at `import liger_kernel` satisfies that trivially -- the
    # first version of this test printed two "ok" lines on a run that never reached the gate.
    # A test whose passing case is indistinguishable from "the script did not start" measures
    # nothing, so refuse loudly instead.
    if "ModuleNotFoundError" in out or "ImportError" in out:
        raise SystemExit(
            "REFUSING: sft_math.py could not import its dependencies here, so the holdout gate "
            "was never reached and an absent-string check would pass vacuously. Run this on the "
            "pod, where liger_kernel and fla are installed.\n" + out.strip()[-400:])
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

# 2. UNSTAMPED + THE FLAG PROCEEDS, and says so loudly. Reaching the checkpoint error proves the
#    gate passed -- the model load is downstream of it.
o = run(pack(None), ["--allow_unstamped_pack"])
check("flag lets it through", "carries NO holdout_fp" not in o, o)
check("flag prints holdout status unknown", "holdout status unknown" in o, o)
check("gate is upstream of the ckpt load", "nonexistent_ckpt" in o or "No such file" in o, o)

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

print(f"\n{'ALL OK' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
