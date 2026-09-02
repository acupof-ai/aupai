#!/usr/bin/env python3
# restartable: one card, one checkpoint per invocation; --steps resumes by skipping done rows.
"""b0-16: does layer 9's branch split have a measured consequence, or is it bounded below the bar?

    python3 scripts/l9_branch_probe.py --steps 832,1192,2500,3000,3500 --out runs/b0_16_l9.json
    python3 scripts/l9_branch_probe.py --rescale --step 3500 --out runs/b0_16_l9_rescale.json
    python3 scripts/l9_branch_probe.py --selftest

TWO HALVES, per b0-16's pre-registered reading:

  (a) the table -- layer-9 branch ratio against per-domain domain_loss across the five
      checkpoints. Answers "does the split coincide with anything".
  (b) the rescale -- multiply layer 9's KDA output by the factor that would put its branch ratio
      at the 12-layer median, score domain_loss, compare to the same checkpoint unscaled. This is
      the half that can show CONSEQUENCE: if forcing the branch back to the median moves
      domain_loss by more than 0.24 nat, the split matters; if it does not, the split is bounded
      below the bar and the finding is closed as a correlate.

WHAT IS ALREADY MEASURED (zero card, weights only), so the eval is not asked to rediscover it:

  - Only mixer.o is anomalous. Growth step832->3500: layer 9's mixer.o 1.2037 against the other
    eight KDA layers' median 1.7334 (MAD sigma 0.0241, z -21.9), while its ffn.w2 grows 1.6440
    against their 1.6417 (z +0.18). The ffn is normal; the KDA output projection is not.
  - It is not a dead branch. All twelve Muon momenta are live (4.8e-04 .. 2.1e-02).
  - It is not weight decay. Muon's step is w -= lr*NS(m) + lr*wd*w*mask; ||NS(m)||_F ~ sqrt(1024)
    = 32 by construction, so the push is lr*32 = 0.32 per step against lr*wd*|w| = 0.0051, i.e.
    63x weaker. Decay cannot hold layer 9 at 46.6 while peers reach 64. (My first hypothesis was
    a decay equilibrium; its own arithmetic refuted it -- the implied equilibrium norm is ~2900,
    far above both.)
  - What IS different is direction consistency. Displacement over step3000->3500 against the
    fully-aligned budget lr*32*n = 160: layer 9 realizes 6.85%, the other eight KDA layers
    13.29% (MAD sigma 0.33pp, z -19.3). Same statistic as the embedding same-direction rate, and
    here it is comparable across layers because n, lr and shape are identical -- the control the
    retracted embedding decomposition lacked (see scripts/embed_norm_sdr.py).

So the open question is consequence, not mechanism, which is what b0-16 asks for.
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("FLA_FLASH_KDA", "0")

CKPT = "ckpt_p200m_4b_0902.pt"
BAR = 0.24  # ds.seed_variance_0p2b readable move on domain_loss unweighted_mean


def ckpt_path(step):
    """The checkpoint for a step -- .step<N> or .interrupt.step<N>, whichever exists.

    Both spellings are real in this run (832 and 1192 are interrupts), and guessing one silently
    skips two of the five points.
    """
    for suf in (f".step{step}", f".interrupt.step{step}"):
        p = os.path.join(ROOT, CKPT + suf)
        if os.path.exists(p):
            return p
    return None


def branch_ratios(path):
    """Per-layer mixer.o/ffn.w2 norm ratio, and which layers are KDA.

    Returns (ratios, kda_layers). KDA membership is read from the presence of A_log rather than
    assumed from attn_every: a stale attn_every in cfg would mislabel every layer.
    """
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False, mmap=True)["model"]
    kda = sorted(int(k.split(".")[1]) for k in sd if k.endswith("mixer.A_log"))
    out = {}
    for L in range(64):
        km, kf = f"blocks.{L}.mixer.o.weight", f"blocks.{L}.ffn.w2.weight"
        if km in sd and kf in sd:
            out[L] = (sd[km].float().norm() / sd[kf].float().norm()).item()
    return out, kda


def median(vals):
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def rescaled_checkpoint(path, layer, factor, dest):
    """Write `path` to `dest` with layer's mixer.o scaled by factor, everything else untouched.

    Split out of score() so the selftest exercises THIS code rather than a copy of it: a check
    that re-implements the rescale passes while the shipped path silently no-ops, which would
    report "no consequence" for every factor -- the failure that looks like a finding.
    """
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    # No explicit `if k not in ck["model"]: raise` here: the very next line's dict lookup raises
    # KeyError with the same key name, so the guard was unreachable-equivalent and NO fixture
    # could tell the two apart -- verified by deleting it and watching the selftest stay green.
    # A check that cannot fail is worse than no check, so the guard is gone and the selftest
    # asserts the BEHAVIOUR (KeyError, nothing written) rather than the line.
    k = f"blocks.{layer}.mixer.o.weight"
    w = ck["model"][k]
    ck["model"][k] = (w.float() * factor).to(w.dtype)
    ck.pop("opt", None)   # 60% of the file; the eval never reads it
    torch.save(ck, dest)
    return dest


def score(path, out_json, rescale_layer=None, rescale_factor=None):
    """domain_loss for one checkpoint, optionally with one layer's KDA output rescaled.

    The rescale is applied by editing mixer.o's WEIGHT, not by patching forward: mixer.o is the
    branch's output projection, so scaling its weight scales the branch output exactly, and it
    needs no hook, no monkey-patch, and no model-code change that a later run could inherit. The
    edited checkpoint is written to a temp file and scored by the ordinary eval path, so the
    number is comparable to every other domain_loss row.
    """
    if rescale_layer is None:
        target = path
        tmp = None
    else:
        tmp = os.path.join("/tmp", f"_l9rescale_{rescale_layer}_{rescale_factor:.4f}.pt")
        target = rescaled_checkpoint(path, rescale_layer, rescale_factor, tmp)
    cmd = [sys.executable, os.path.join(ROOT, "eval", "domain_loss.py"), "--ckpt", target]
    if out_json:
        cmd += ["--json", out_json]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if tmp and os.path.exists(tmp):
        os.remove(tmp)
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-8:]
        sys.exit(f"domain_loss failed on {target}:\n" + "\n".join(tail))
    return r.stdout


def _selftest():
    fails = []

    # 1. The rescale must actually change the weight, by exactly the factor, and touch NOTHING
    #    else. A rescale that silently no-ops would make the probe report "no consequence" for
    #    every factor -- the failure that looks like a finding.
    import tempfile

    import torch
    v, d = 8, 4
    sd = {
        "blocks.9.mixer.o.weight": torch.randn(d, d),
        "blocks.9.ffn.w2.weight": torch.randn(d, d),
        "blocks.8.mixer.o.weight": torch.randn(d, d),
        "tok.weight": torch.randn(v, d),
    }
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.pt")
        torch.save({"model": sd, "cfg": {"layers": 12}, "opt": [{"state": {}}]}, p)
        before = {k: t.clone() for k, t in sd.items()}
        f = 1.3333
        k = "blocks.9.mixer.o.weight"
        out = rescaled_checkpoint(p, 9, f, os.path.join(td, "r.pt"))   # the SHIPPED path
        got_sd = torch.load(out, map_location="cpu", weights_only=False)["model"]
        got = (got_sd[k].norm() / before[k].norm()).item()
        if abs(got - f) > 1e-4:
            fails.append(f"rescaled_checkpoint changed the norm by {got:.4f}, expected {f} -- a "
                         f"no-op here reports 'no consequence' for every factor")
        for kk in before:
            if kk == k:
                continue
            if not torch.equal(got_sd[kk], before[kk]):
                fails.append(f"rescaled_checkpoint also modified {kk}")
        if "opt" in torch.load(out, map_location="cpu", weights_only=False):
            fails.append("rescaled_checkpoint kept optimizer state; the temp file is 2.5x bigger "
                         "than it needs to be and /tmp is shared")
        # A layer with no mixer.o must REFUSE with KeyError and leave no file. This asserts
        # BEHAVIOUR, not a line: an explicit `if k not in ...: raise KeyError` used to sit in
        # rescaled_checkpoint, and deleting it left the selftest green because the next line's
        # dict lookup raises the same KeyError. The guard could not fail, so it was removed --
        # what matters is that a typo'd --layer cannot produce a scored, unmodified checkpoint
        # reading as "rescaling changes nothing".
        xp = os.path.join(td, "x.pt")
        try:
            rescaled_checkpoint(p, 7, f, xp)
        except KeyError:
            pass
        except Exception as e:  # noqa: BLE001
            fails.append(f"a missing mixer.o raised {type(e).__name__} instead of KeyError "
                         f"({e}); the deliberate refusal is gone and this passed only because "
                         f"something else happened to fail")
        else:
            fails.append("rescaled_checkpoint accepted a layer with no mixer.o, so a typo'd "
                         "--layer would score an UNMODIFIED checkpoint and read as 'rescaling "
                         "changes nothing'")
        if os.path.exists(xp):
            fails.append("a refused rescale still wrote a checkpoint file")

    # 2. ckpt_path must find BOTH spellings. Two of the five points are `.interrupt.step<N>`, so
    #    a probe that only tries `.step<N>` silently drops step832 and step1192 -- and the table
    #    would still print, three rows short, reading as a complete result.
    with tempfile.TemporaryDirectory() as td:
        # This module, however it was loaded: `import scripts.l9_branch_probe` fails when the
        # file is run as a script (no `scripts` package on sys.path), which is how the hook and
        # every human runs it -- the selftest would die before reaching check 3.
        me = sys.modules[__name__]
        old = me.ROOT
        try:
            me.ROOT = td
            open(os.path.join(td, CKPT + ".step3500"), "wb").close()
            open(os.path.join(td, CKPT + ".interrupt.step832"), "wb").close()
            if me.ckpt_path(3500) is None:
                fails.append("ckpt_path missed a plain .step spelling")
            if me.ckpt_path(832) is None:
                fails.append("ckpt_path missed the .interrupt.step spelling, which is how two of "
                             "the five b0-16 points are named -- the table would silently be "
                             "short two rows and still look complete")
            if me.ckpt_path(9999) is not None:
                fails.append("ckpt_path invented a checkpoint that does not exist")
        finally:
            me.ROOT = old

    # 3. The ratio must be computed per layer and KDA membership read from A_log, not assumed.
    #    A version keyed on attn_every would mislabel every layer if cfg were stale.
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.pt")
        sd2 = dict(sd)
        sd2["blocks.9.mixer.A_log"] = torch.zeros(4)      # 9 is KDA
        sd2["blocks.8.ffn.w2.weight"] = torch.randn(d, d)  # 8 has both halves, no A_log -> MLA
        torch.save({"model": sd2}, p)
        ratios, kda = branch_ratios(p)
        if kda != [9]:
            fails.append(f"KDA layers read as {kda}, expected [9] from A_log presence")
        if set(ratios) != {8, 9}:
            fails.append(f"ratios computed for {sorted(ratios)}, expected layers 8 and 9")
        want = (sd2["blocks.9.mixer.o.weight"].norm()
                / sd2["blocks.9.ffn.w2.weight"].norm()).item()
        if abs(ratios[9] - want) > 1e-5:
            fails.append(f"layer 9 ratio {ratios[9]} != {want}")

    # 4. median() on an even-length list must average the middle two, not pick one. The rescale
    #    factor is median/ratio, so an off-by-one median silently rescales to the wrong target --
    #    a probe that answers a slightly different question than the one asked.
    if median([1.0, 2.0, 3.0, 4.0]) != 2.5:
        fails.append(f"median of even-length list is {median([1.0, 2.0, 3.0, 4.0])}, expected 2.5")
    if median([3.0, 1.0, 2.0]) != 2.0:
        fails.append("median does not sort its input")

    if fails:
        for f in fails:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("l9_branch_probe selftest OK: rescaled_checkpoint (the SHIPPED path, not a copy) "
          "changes exactly one tensor by exactly the "
          "requested factor (a silent no-op would report 'no consequence' for every factor, the "
          "failure that looks like a finding); ckpt_path finds both .step and .interrupt.step "
          "spellings, which is how two of b0-16's five points are named; KDA membership is read "
          "from A_log presence rather than assumed from attn_every; and median averages the "
          "middle two on an even-length list, since the rescale factor is median/ratio.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", help="comma-separated steps for the table")
    ap.add_argument("--step", type=int, help="single step, for --rescale")
    ap.add_argument("--rescale", action="store_true",
                    help="scale layer 9's mixer.o to put its branch ratio at the median")
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--out", help="json for the eval rows")
    ap.add_argument("--claim", action="store_true", help="claim the lane card for this pid")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()

    if a.claim:
        # Claim with THIS process's pid: card_claim refuses a shell's pid by design (a shell
        # claim reads ORPHAN when the shell exits, or held after the job is gone).
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "card_claim.py"),
                        "acquire", "--name", "b0_16_l9", "--cards", "7",
                        "--note", "b0-16 layer-9 branch probe", "--pid", str(os.getpid())],
                       cwd=ROOT, check=False)

    if a.rescale:
        if not a.step:
            ap.error("--rescale needs --step")
        p = ckpt_path(a.step)
        if p is None:
            sys.exit(f"no checkpoint for step {a.step}")
        ratios, kda = branch_ratios(p)
        med = median(ratios.values())
        f = med / ratios[a.layer]
        print(f"step {a.step}: layer {a.layer} ratio {ratios[a.layer]:.4f}, 12-layer median "
              f"{med:.4f} -> rescale mixer.o by {f:.4f}", flush=True)
        print("=== unscaled ===", flush=True)
        print(score(p, a.out), flush=True)
        print(f"=== layer {a.layer} rescaled x{f:.4f} ===", flush=True)
        print(score(p, a.out, rescale_layer=a.layer, rescale_factor=f), flush=True)
        print(f"Compare the two unweighted_mean values against the {BAR} nat bar.", flush=True)
        return 0

    if not a.steps:
        ap.error("--steps, or --rescale with --step (or --selftest)")
    for s in [int(x) for x in a.steps.split(",")]:
        p = ckpt_path(s)
        if p is None:
            print(f"step {s}: NO CHECKPOINT", flush=True)
            continue
        ratios, kda = branch_ratios(p)
        med = median(ratios.values())
        print(f"\n=== step {s}  layer {a.layer} ratio {ratios[a.layer]:.4f}  median {med:.4f}  "
              f"({ratios[a.layer] / med:.3f}x) ===", flush=True)
        print(score(p, a.out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
