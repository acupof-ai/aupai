#!/usr/bin/env python3
"""sft_math.py records what it was asked to do, and --stop_after does not touch the schedule.

Both properties exist because of one hole: ckpt_control_ours.pt's lr_scale is unrecoverable.
train.py:848 applies the scale inside set_schedule (initial_lr * lr_scale * m), so it reached
neither Cfg nor any log, and the checkpoint whose held-out loss divides every number in
docs/audits/control_pythia160m_vs_ours.md cannot say what lr produced it. The argparse default
being 0.1 is not evidence of what ran.

Each case here is a MUTATION test: it states the wrong version and proves this file's version
disagrees with it. A test that only asserts the current behaviour passes just as happily when
the behaviour is wrong.

    python3 scripts/test_sft_lr_provenance.py
"""
import ast
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "sft_math.py")


# Cfg's CLASS defaults. NOT what an SFT run uses: sft_math.py setattrs every key of the resumed
# checkpoint's cfg onto Cfg, and ckpt_p200m_4b_0902.pt carries warmup 300 / warmdown 0.1. Under
# those, step 40 is still inside warmup and total=1024 vs total=40 give the SAME multiplier --
# so --max_steps would have been harmless for a 40-step prefix of that particular run. I first
# justified the flag with the 18.7x figure below and called it this run's hazard; it is the
# hazard at Cfg's defaults, which is a different claim. The flag is still right (a prefix must
# not depend on the reader knowing which cfg was resumed) but the number is not universal.
CFG_DEFAULT_WARMUP, CFG_DEFAULT_WARMDOWN = 20, 0.65
# What ckpt_p200m_4b_0902.pt actually carries, read from its cfg.
RESUMED_WARMUP, RESUMED_WARMDOWN = 300, 0.1


def lr_mult(step, total, warmup=CFG_DEFAULT_WARMUP, warmdown=CFG_DEFAULT_WARMDOWN,
            final_lr_frac=0.05):
    """train.py:1823 restated. Restated deliberately: importing train.py pulls torch and CUDA
    into a source-level test, and the point here is the ARITHMETIC of total, which is stable."""
    if step < warmup:
        return (step + 1) / warmup
    wd_steps = max(1, int(warmdown * total))
    wd_start = total - wd_steps
    if step < wd_start:
        return 1.0
    progress = min(1.0, (step - wd_start) / wd_steps)
    return final_lr_frac + (1 - final_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    fails, skips = [], []
    src = open(SRC).read()
    tree = ast.parse(src)

    # 1. THE MOTIVATING ARITHMETIC. --max_steps N cannot stand in for "the first N steps",
    #    because lr_mult reads total. If this ever became false the flag would be harmless and
    #    --stop_after unnecessary -- so the test asserts the DIFFERENCE, not the workaround.
    at_full = lr_mult(39, 1024)
    at_short = lr_mult(39, 40)
    if not (at_full > 0.9 and at_short < 0.2):
        fails.append(f"step 40 lr multiplier at Cfg's defaults: total=1024 gives {at_full:.4f}, "
                     f"total=40 gives {at_short:.4f} -- the premise for --stop_after no longer "
                     f"holds, so either the schedule changed or this test is stale")
    if abs(at_short - at_full) < 1e-9:
        fails.append("shortening total does not change the schedule at all -- impossible unless "
                     "lr_mult stopped reading total")

    # 1b. AND THE HAZARD'S SIZE IS cfg-DEPENDENT, which is the correction to my own first
    #     justification. Asserted so nobody (me included) re-reads the 18.7x as universal:
    #     under the RESUMED cfg a 40-step prefix is unaffected, and the flag is justified by
    #     "a prefix must not depend on which cfg was resumed", not by a number.
    r_full = lr_mult(39, 1024, RESUMED_WARMUP, RESUMED_WARMDOWN)
    r_short = lr_mult(39, 40, RESUMED_WARMUP, RESUMED_WARMDOWN)
    if abs(r_full - r_short) > 1e-9:
        fails.append(f"under warmup {RESUMED_WARMUP} / warmdown {RESUMED_WARMDOWN}, step 40 was "
                     f"expected to sit inside warmup so both totals agree, but they read "
                     f"{r_full:.4f} and {r_short:.4f} -- re-derive before quoting either figure")
    if RESUMED_WARMUP <= 40:
        fails.append(f"step 40 is no longer inside warmup ({RESUMED_WARMUP}), so the comment "
                     f"explaining why --max_steps was harmless for THIS base is now wrong")

    # 2. --stop_after EXISTS AND total_steps DOES NOT DEPEND ON IT. The mutation this catches is
    #    the obvious lazy implementation: reusing the max_steps line for stop_after, which would
    #    silently reintroduce the very bug the flag exists to avoid.
    if "--stop_after" not in src:
        fails.append("--stop_after is gone; a prefix run can only be spelled --max_steps, which "
                     "shortens the schedule")
    else:
        # Find the statement that assigns total_steps from max_steps and check stop_after is
        # NOT in the same neighbourhood.
        bad = [n for n in ast.walk(tree)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "total_steps" for t in n.targets)
               and "stop_after" in ast.dump(n.value)]
        if bad:
            fails.append("total_steps is computed from stop_after -- that is exactly what "
                         "--max_steps does and it defeats the flag's purpose")

    # 3. THE TWO FLAGS TOGETHER ARE REFUSED, AND REFUSED BY THE REAL PARSER BEFORE ANY LOAD.
    #    One shortens the schedule and one does not, so a run passing both has no defined
    #    meaning; silent precedence would be a coin flip recorded as a measurement.
    #
    #    Checked by INVOKING the CLI, not by grepping for the check. My source-level version
    #    passed while the refusal sat after torch.load of a 1.6 GB checkpoint -- and it passed
    #    just as happily when I had not verified the flag reached the parser at all. Locally
    #    the module needs liger_kernel to import, so an ImportError here is a SKIP: a check
    #    that cannot run must say so, never report a pass.
    if "--stop_after" in src:
        r = subprocess.run([sys.executable, SRC, "--resume", os.devnull,
                            "--stop_after", "40", "--max_steps", "40"],
                           capture_output=True, text=True, cwd=ROOT,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        blob = r.stdout + r.stderr
        if "ModuleNotFoundError" in blob or "ImportError" in blob:
            skips.append("case 3 (CLI refuses both flags): sft_math.py's imports are "
                         "unavailable here, so the real parser cannot be invoked")
        elif "ambiguous" not in blob:
            fails.append(f"the CLI does not refuse --stop_after together with --max_steps "
                         f"(rc={r.returncode}); tail: {blob.strip().splitlines()[-1][:120] if blob.strip() else '(no output)'}")
        elif r.returncode == 0:
            fails.append("the CLI printed the ambiguity message but exited 0, so a driver "
                         "checking $? would proceed")
        # And the refusal must precede the checkpoint load, or a contradictory launch pays
        # minutes before dying. parse_args is the boundary.
        pa = src.index("args = parser.parse_args()")
        tl = src.index("torch.load(args.resume")
        chk = src.find("--stop_after and --max_steps together are ambiguous")
        if chk == -1 or not (pa < chk < tl):
            fails.append("the both-flags refusal is not between parse_args and the checkpoint "
                         "load: a contradictory launch would load the model first")

    # 4. THE RUN RECORDS ITS ARGV AND ITS REALISED LR. This is the hole itself. Assert the
    #    substance, not the phrasing -- and not mere PRESENCE of a token: my first version
    #    grepped for "initial_lr", which survives a mutation that drops the value from the
    #    format string while the name still appears elsewhere in the file. So the check has to
    #    look at the logging call itself.
    for needle, why in (
        ("sys.argv", "argv is not logged, so a log cannot say what the run was asked to do"),
        ("lr_scale {args.lr_scale}", "the lr_scale value is not logged"),
    ):
        if needle not in src:
            fails.append(why)

    # The per-group line must actually INTERPOLATE both the configured and the realised lr.
    # Find every runlog/print call mentioning lr and require one that formats g['lr'].
    lr_calls = [n for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id in ("runlog", "print")
                and "lr[" in ast.dump(n)]
    if not lr_calls:
        fails.append("no log line reports a per-optimizer-group lr, so a reader cannot tell "
                     "which groups exist or what lr each got")
    else:
        dumped = " ".join(ast.dump(c) for c in lr_calls)
        # A FormattedValue over g['lr'] / g['initial_lr'] leaves the subscript in the dump;
        # a bare literal label does not.
        if "'lr'" not in dumped:
            fails.append("the per-group log line names groups but never interpolates g['lr'], "
                         "so it reports labels rather than the realised learning rate")
        if "'initial_lr'" not in dumped:
            fails.append("the per-group log line does not report initial_lr, so the realised lr "
                         "cannot be checked against the base it was scaled from")

    # 4b. THE SHIPPED BLOCK ACTUALLY EMITS. Cases 4/4a read source, and source-reading has a
    #     floor: mutating the loop to `for opt in []` leaves every string intact and emits
    #     nothing, and no ast inspection separates that from working code.
    #
    #     My first version of this case RE-TYPED the loop here and executed the copy. That
    #     passes with the real loop mutated to `for opt in []` -- it tested my transcription,
    #     not the file. Same defect as eval_heldout.py once carrying its own copy of the chat
    #     template: a duplicate is not a witness. So extract the REAL block's source out of
    #     sft_math.py and exec THAT.
    lr_block = None
    for node in ast.walk(tree):
        # The `if is_main:` block that contains the per-group lr logging.
        if isinstance(node, ast.If) and "lr[" in ast.dump(node) and "initial_lr" in ast.dump(node):
            lr_block = node
            break
    if lr_block is None:
        fails.append("could not locate the step-0 logging block in sft_math.py, so it cannot be "
                     "executed -- either it is gone or this test can no longer find it")
    else:
        try:
            sys.path.insert(0, ROOT)
            from train import set_schedule as _sched  # noqa: E402

            class _FakeOpt:
                def __init__(self, groups):
                    self.param_groups = groups

            class _Cfg2:
                warmup, warmdown, final_lr_frac = 20, 0.65, 0.05
                seed, batch, epochs = 1, 8, 1

            lines = []
            SCALE, BASE = 0.1, 0.01
            ns = {
                "is_main": True,
                "runlog": lambda m: lines.append(str(m)),
                "json": __import__("json"),
                "sys": sys,
                "args": type("A", (), {"lr_scale": SCALE, "stop_after": 40})(),
                "total_steps": 1024,
                "Cfg": _Cfg2,
                "set_schedule": _sched,
                "optimizers": [_FakeOpt([{"initial_lr": BASE, "initial_wd": 0.1, "lr": 0.0,
                                          "momentum": 0.9, "weight_decay": 0.0, "params": []}])],
            }
            exec(compile(ast.Module(body=[lr_block], type_ignores=[]), "<block>", "exec"), ns)
            per_group = [l for l in lines if "lr[" in l]
            if not per_group:
                fails.append("executing sft_math.py's own step-0 block produced no per-group lr "
                             "line -- the code is present but emits nothing")
            else:
                want = BASE * SCALE * lr_mult(0, 1024)
                if f"{want:.3g}" not in per_group[0]:
                    fails.append(f"the realised lr printed does not reflect lr_scale: expected "
                                 f"{want:.3g} in {per_group[0]!r}")
            if not any("argv" in l for l in lines):
                fails.append("executing the block emitted no argv line")
            if not any("lr_scale" in l for l in lines):
                fails.append("executing the block emitted no lr_scale line")
        except ImportError as e:
            skips.append(f"case 4b (execute the shipped block): {type(e).__name__}: {e}")

    #    final save_checkpoint would leave every .stepN intermediate without it -- and an
    #    interrupted run's last .stepN is exactly the file someone must identify later.
    if "Cfg.lr_scale" not in src:
        fails.append("lr_scale never reaches Cfg, so save_checkpoint cannot write it and the "
                     "next checkpoint is as unidentifiable as ckpt_control_ours.pt")
    else:
        set_line = min(i for i, l in enumerate(src.splitlines()) if "Cfg.lr_scale" in l)
        save_lines = [i for i, l in enumerate(src.splitlines()) if "save_checkpoint(" in l]
        if save_lines and set_line > min(save_lines):
            fails.append(f"Cfg.lr_scale is set at line {set_line+1}, AFTER the first "
                         f"save_checkpoint at line {min(save_lines)+1}: intermediate .stepN "
                         f"checkpoints would carry no lr_scale")

    # 6. save_checkpoint ACTUALLY PERSISTS cfg KEYS. Cases 4-5 are about this file; if the
    #    writer dropped cfg, they would all still pass while nothing was recorded. Verified
    #    against the real function with a fake state, no CUDA.
    try:
        sys.path.insert(0, ROOT)
        import tempfile, torch  # noqa: E402
        from train import save_checkpoint  # noqa: E402

        class _C:
            lr_scale = 0.37
            seed = 1
            mix = None
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "t.pt")
            save_checkpoint(p, {"w": torch.zeros(1)}, _C, "vfp")
            got = torch.load(p, map_location="cpu", weights_only=False)
        if got.get("cfg", {}).get("lr_scale") != 0.37:
            fails.append(f"save_checkpoint did not persist cfg['lr_scale']: got "
                         f"{got.get('cfg', {}).get('lr_scale')!r} -- setting Cfg.lr_scale "
                         f"records nothing")
    except ImportError as e:
        skips.append(f"case 6 (save_checkpoint round-trip): {type(e).__name__}: {e}")

    # 7. THE ORIGINAL RUN IS STILL UNIDENTIFIABLE, and that stays true no matter what this file
    #    does now. Recorded as a check so nobody later reads the new logging as retroactive.
    for f in ("runs/control_ours.log",):
        path = os.path.join(ROOT, f)
        if os.path.isfile(path) and "lr_scale" in open(path, encoding="utf-8", errors="replace").read():
            fails.append(f"{f} contains lr_scale -- then it WAS recoverable and e1-27's step 0 "
                         f"was unnecessary; re-read it before trusting this test")

    for s in skips:
        print(f"SKIP: {s}", file=sys.stderr)
    for f in fails:
        print(f"FAIL: {f}", file=sys.stderr)
    if fails:
        return 1
    # The summary names only what actually ran. A line that claims "the two flags cannot
    # combine" while case 3 skipped is the same defect as a harness reporting a check that
    # never executed -- and this file's whole subject is a claim nobody verified.
    ran = [f"--stop_after keeps total_steps (at Cfg's defaults the step-40 multiplier is "
           f"{at_full:.3f} full vs {at_short:.3f} shortened, {at_full / at_short:.1f}x apart; "
           f"under the resumed warmup {RESUMED_WARMUP} both read {r_full:.4f}, so the hazard's "
           f"size depends on the cfg and the flag is justified by not needing to know it)"]
    if not any("case 3" in s for s in skips):
        ran.append("the CLI refuses both flags before loading the checkpoint")
    if not any("case 4b" in s for s in skips):
        ran.append("sft_math.py's own step-0 block emits argv and the scaled per-group lr")
    else:
        ran.append("argv and the per-group lr are logged (source only -- not executed)")
    if not any("case 6" in s for s in skips):
        ran.append("save_checkpoint persists cfg['lr_scale']")
    ran.append("lr_scale reaches cfg before the first save")
    print("sft_lr_provenance OK: " + "; ".join(ran)
          + (f" [{len(skips)} case(s) SKIPPED, listed above]" if skips else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
