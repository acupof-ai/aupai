#!/usr/bin/env python3
# restartable: one card for --steps/--rescale, ZERO cards for --weights; one checkpoint per
# invocation; --steps resumes by skipping done rows.
"""b0-16: does layer 9's branch split have a measured consequence, or is it bounded below the bar?

    python3 scripts/l9_branch_probe.py --steps 832,1192,2500,3000,3500 --out runs/b0_16_l9.json
    python3 scripts/l9_branch_probe.py --rescale --step 3500 --out runs/b0_16_l9_rescale.json
    CUDA_VISIBLE_DEVICES= python3 scripts/l9_branch_probe.py --weights \
        --from_step 832 --to_step 3500 --layer 9 --out runs/b0_16_weights.json
    python3 scripts/l9_branch_probe.py --selftest

--weights TAKES NO CARD. It is torch.load(map_location="cpu", mmap=True) plus tensor norms, so
it must not queue behind a lane job -- on 2026-09-03 it would have waited ~95 min for a card it
never touches. The other two modes score domain_loss and do need one.

THREE READINGS, of which b0-16 pre-registered the first two:

  (a) the table -- layer-9 branch ratio against per-domain domain_loss across the five
      checkpoints. Answers "does the split coincide with anything".
  (b) the rescale -- multiply layer 9's KDA output by the factor that would put its branch ratio
      at the 12-layer median, score domain_loss, compare to the same checkpoint unscaled. This is
      the half that can show CONSEQUENCE: if forcing the branch back to the median moves
      domain_loss by more than 0.24 nat, the split matters; if it does not, the split is bounded
      below the bar and the finding is closed as a correlate.

  (c) --weights, added after review: the growth and direction-consistency numbers below, as a
      JSONL artifact (runs/b0_16_weights.json). They were quoted in a fact, this docstring and a
      review row with nothing behind them -- e1 caught it -- and a number that lives only in
      prose cannot be recomputed by a reader or checked against a later run.

WHAT THE WEIGHTS SAY, now with an artifact behind every figure:

  - Only mixer.o is anomalous. Growth step832->3500: layer 9's mixer.o 1.2037 against the other
    eight KDA layers' median 1.7334 (MAD sigma 0.0241, z -21.9), while its ffn.w2 grows 1.6440
    against their 1.6417 (z +0.2). The ffn is normal; the KDA output projection is not.
  - It is not a dead branch. All twelve Muon momenta are live (4.8e-04 .. 2.1e-02).
  - It is not weight decay. Muon's step is w -= lr*NS(m) + lr*wd*w*mask; ||NS(m)||_F ~ sqrt(1024)
    = 32 by construction, so the push is lr*32 = 0.32 per step against lr*wd*|w| = 0.0051, i.e.
    63x weaker. Decay cannot hold layer 9 at 46.6 while peers reach 64. (My first hypothesis was
    a decay equilibrium; its own arithmetic refuted it -- the implied equilibrium norm is ~2900,
    far above both.)
  - What IS different is direction consistency. Displacement over step3000->3500 against the
    fully-aligned budget lr*32*n = 160: layer 9 realizes 6.85%, the other eight KDA layers
    13.29% (MAD sigma 0.33pp -- 0.0033 as the fraction the artifact stores, same number in the
    units of its neighbours; z -19.3), while ffn.w2's consistency reads z -1.5. Same statistic
    as the embedding same-direction rate, and comparable across layers here because n, lr and
    shape are identical -- the control the retracted embedding decomposition lacked (see
    scripts/embed_norm_sdr.py).

  EVERY z DEPENDS ON THE WINDOW, AND NEITHER WINDOW IS UNIFORMLY BETTER. Both statistics were run
  on both intervals (both rows are in the artifact):

                          step832->3500 (n=2668)   step3000->3500 (n=500)
      mixer.o growth              z  -21.9                z -177.4
      mixer.o consistency         z   -7.9                z  -19.3
      ffn.w2  growth              z   +0.2                z   -1.1
      ffn.w2  consistency         z   -0.1                z   -1.5

  Growth is 8x stronger at equal n, because over 500 steps the peers cluster to MAD sigma 0.0004
  while layer 9 sits at 1.0047; the wide window lets the peers spread. So the -21.9 the fact
  leads with paid twice: it used the window where the cross-layer comparison is NOT legitimate
  (unequal n) AND threw away 8x of resolution, both losses in the same direction. Consistency
  goes the other way -- equal n is stronger there too (-19.3 vs -7.9), but for the opposite
  reason: the budget lr*32*n grows linearly while realized displacement does not, so the fraction
  compresses as n grows. NO WINDOW IS CONSISTENTLY STRONGER; the two statistics prefer opposite
  ends for opposite reasons, which is why a z quoted without its window is unreadable rather than
  merely imprecise. Same interval-length dependence that retracted the 1.71x depth term.

  ffn.w2 SITS AT ZERO IN BOTH WINDOWS, on both statistics, and THAT IS THE NEGATIVE CONTROL THIS
  READING CARRIES. All eight z values, in one place:

                          step832->3500 (n=2668)   step3000->3500 (n=500)
      mixer.o growth              -21.93                  -177.43
      mixer.o consistency          -7.88                   -19.33
      ffn.w2  growth               +0.18                    -1.09
      ffn.w2  consistency          -0.09                    -1.54

  Four extreme, four indistinguishable from zero, split exactly along the tensor rather than along
  the window. If layer 9 were simply an anomalous BLOCK, ffn.w2 would move too; it does not, in
  either window. That is stronger evidence than any single large z, because it is an alternative
  explanation eliminated rather than an existing one amplified -- and one window could not do it,
  since a window is exactly what a confound would be free to pick. e1 raised both halves of this
  during review (the both-windows requirement, then the ffn contrast as a control).

So the open question is consequence, not mechanism, which is what b0-16 asks for.
"""

import argparse
import json
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


def mad_sigma(vals, med=None):
    """MAD-based sigma: 1.4826 * median(|x - median|), the normal-consistent scaling.

    MAD and not sd because n=8 and the question is whether ONE layer is an outlier -- an
    outlier inflates sd, so a z against sd is shrunk by the very point being tested.

    THE 1.4826 IS PART OF THE STATISTIC'S NAME HERE, NOT AN IMPLEMENTATION DETAIL. "MAD sigma"
    without it is a different number by a factor of 1.48, and every z in this file and in
    eff.l9_branch_split_p200m is divided by it. Concretely, on the wide window's growth: raw MAD
    0.016288, times 1.4826 gives sigma 0.024148 and z -21.93; WITHOUT the constant the same data
    gives z -32.52, i.e. 48% larger. Two numbers that both print as "sigma", one 48% off. e1 recomputed all six figures independently and
    matched to the digit -- with the same constant, which it noted was luck rather than a
    convention we had agreed. Two people each implementing a named statistic and happening to
    pick the same convention is the first half of eff.vocab_padding_softmax_defect's sibling
    failure, where `eval_loss` meant two different things in two arms for a whole round. So the
    constant is stated wherever the number is quoted.
    """
    m = median(vals) if med is None else med
    return 1.4826 * median([abs(x - m) for x in vals])


def weights_reading(step_a, step_b, layer, lr, ns_norm=None):
    """The zero-card half of b0-16: growth and direction consistency, per KDA layer.

    THIS EXISTS BECAUSE THE NUMBERS DID NOT. z -21.9, the 1.7334 peer median, the 0.0241 MAD
    sigma and the 6.85%/13.29% pair were quoted in a fact, a docstring and a review row with NO
    artifact behind any of them (e1 caught it during review). A number that lives only in prose
    cannot be recomputed by a reader and cannot be checked against a later run -- and these are
    the numbers the fact leads with.

    Two statistics, and they answer different questions:

      growth        |w_b| / |w_a| per tensor. Says the norm stopped rising.
      consistency   ||w_b - w_a||_F / (lr * ns_norm * n), the realized displacement over the
                    budget if every Muon step had pushed the same direction. Says WHY: a layer
                    can hold its norm because it stopped moving, or because it moves and cancels.

    The consistency denominator uses ||NS(m)||_F ~ sqrt(d) = 32, which is Newton-Schulz's design
    property (the whole point of the orthogonalization), not a measurement of this run's updates.
    That is the reading's main soft spot and the fact's uncertainty field says so.

    COMPARABLE ACROSS LAYERS ONLY AT EQUAL n. The same-direction rate is a within-interval-length
    statistic -- `predicted` is linear in n while real growth compounds -- so it moves ~2000x
    between n=10 and n=1000, and comparing two layers read over different intervals produced the
    retracted 1.71x depth term (scripts/embed_norm_sdr.py). Here n, lr and shape are identical
    for every layer by construction, which is what makes the cross-layer z legitimate.
    """
    import torch
    if ns_norm is None:
        raise ValueError("ns_norm must be passed explicitly: it is sqrt(d) for the tensor's "
                         "output dim, and defaulting it hides which d the budget assumed")
    pa, pb = ckpt_path(step_a), ckpt_path(step_b)
    if pa is None or pb is None:
        raise FileNotFoundError(f"need checkpoints for both step {step_a} and step {step_b}; "
                                f"got {pa} and {pb}")
    n = step_b - step_a
    sda = torch.load(pa, map_location="cpu", weights_only=False, mmap=True)["model"]
    sdb = torch.load(pb, map_location="cpu", weights_only=False, mmap=True)["model"]
    kda = sorted(int(k.split(".")[1]) for k in sda if k.endswith("mixer.A_log"))
    rows = {}
    for L in kda:
        r = {}
        for tag, key in (("mixer_o", f"blocks.{L}.mixer.o.weight"),
                         ("ffn_w2", f"blocks.{L}.ffn.w2.weight")):
            wa, wb = sda[key].float(), sdb[key].float()
            r[tag] = {
                "norm_a": wa.norm().item(),
                "norm_b": wb.norm().item(),
                "growth": (wb.norm() / wa.norm()).item(),
                # Realized displacement over the fully-aligned budget. lr*ns_norm is one step's
                # push; times n is the most a monotone walk could travel.
                "consistency": (wb - wa).norm().item() / (lr * ns_norm * n),
            }
        rows[L] = r
    peers = [L for L in kda if L != layer]
    out = {"step_a": step_a, "step_b": step_b, "n": n, "layer": layer, "lr": lr,
           "ns_norm": ns_norm, "kda_layers": kda, "per_layer": rows}
    for tag in ("mixer_o", "ffn_w2"):
        for stat in ("growth", "consistency"):
            pv = [rows[L][tag][stat] for L in peers]
            med, sig = median(pv), mad_sigma(pv)
            lv = rows[layer][tag][stat]
            out[f"{tag}_{stat}"] = {
                "layer": lv, "peer_median": med, "peer_mad_sigma": sig, "peer_n": len(pv),
                # z is against the WITHIN-RUN CROSS-LAYER spread, not a seed distribution. One
                # seed, one run: the run-to-run spread of a per-layer ratio is unmeasured.
                "z": (lv - med) / sig if sig else None,
            }
    return out


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


def pick_control(ratios, kda, layer, med):
    """A KDA layer near the median, to receive the SAME rescale factor as `layer`.

    Split out to be testable and to FAIL LOUDLY. Returning None here would crash inside the
    third eval arm, after arm A has already been paid for -- and the run would look like an
    infrastructure error rather than "this checkpoint has no valid control".
    """
    cands = [L for L in kda if L != layer
             and abs(ratios[L] - med) < abs(ratios[layer] - med) / 4]
    if not cands:
        raise ValueError(
            f"no KDA layer is near the median: layer {layer} is {abs(ratios[layer] - med):.4f} "
            f"from the median {med:.4f} and no other KDA layer is within a quarter of that. "
            f"Ratios: { {L: round(ratios[L], 4) for L in kda} }. Without a control a null "
            f"result cannot be told apart from the probe having no resolution -- pass "
            f"--control explicitly and say in the writeup why that layer stands in.")
    # The closest to the median, so the control is the most 'normal' layer available rather than
    # whichever one the dict happened to yield first.
    return min(cands, key=lambda L: abs(ratios[L] - med))


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

    # 3b. pick_control must return the layer CLOSEST to the median (not merely a passing one),
    #     and must RAISE when none qualifies. A None return would crash inside the third eval arm
    #     after arm A was already paid for, reading as infrastructure failure rather than "this
    #     checkpoint has no valid control".
    # The closest layer is deliberately NOT the first candidate: with layer 0 at 0.900 and layer 2
    # at 0.867, any-passing-candidate returns 0 while closest returns 2. My first fixture had the
    # closest layer first, so `return cands[0]` passed it -- the check could not fail.
    rr = {0: 0.900, 1: 0.880, 2: 0.867, 9: 0.605}
    kk = [0, 1, 2, 9]
    got = pick_control(rr, kk, 9, 0.8673)
    if got != 2:
        fails.append(f"pick_control chose layer {got}; layer 2 (0.867) is closest to the median "
                     f"0.8673 while layer 0 (0.900) merely passes the threshold. 'A layer that "
                     f"qualifies' is not 'the most normal layer available', and the control has "
                     f"to be the latter for a null to mean anything.")
    # The refusal must be the DELIBERATE ValueError carrying the explanation, not whatever the
    # code throws next. Measured: replacing the guarded return with `cands[0]` raises IndexError
    # on the empty list, and a bare `except ValueError` let that escape as a crash -- the selftest
    # reported an error rather than a FAIL, which reads as a broken test instead of a broken
    # guard. Same shape as the KeyError guard deleted from rescaled_checkpoint: an alternative
    # path raising a DIFFERENT exception hides the missing guard, and one raising the SAME
    # exception makes it unfalsifiable.
    try:
        pick_control({9: 0.605, 8: 0.700}, [8, 9], 9, 0.8673)
    except ValueError as e:
        if "no KDA layer is near the median" not in str(e):
            fails.append(f"pick_control raised ValueError without its explanation: {e}")
    except Exception as e:  # noqa: BLE001
        fails.append(f"pick_control raised {type(e).__name__} ({e}) instead of the deliberate "
                     f"ValueError -- the guard is gone and something else failed in its place")
    else:
        fails.append("pick_control accepted layer 8 at 0.700 as 'near' the median 0.8673 when "
                     "layer 9 is 0.262 away -- 0.167 is not within a quarter of that, so a "
                     "not-normal layer would stand in as the control and a null would be "
                     "unreadable in a way nothing reports")

    # 4. median() on an even-length list must average the middle two, not pick one. The rescale
    #    factor is median/ratio, so an off-by-one median silently rescales to the wrong target --
    #    a probe that answers a slightly different question than the one asked.
    if median([1.0, 2.0, 3.0, 4.0]) != 2.5:
        fails.append(f"median of even-length list is {median([1.0, 2.0, 3.0, 4.0])}, expected 2.5")
    if median([3.0, 1.0, 2.0]) != 2.0:
        fails.append("median does not sort its input")

    # 5. mad_sigma must NOT be shrunk by the outlier it is measuring. The whole reason the fact
    #    quotes z against MAD and not sd: with n=8 peers plus one outlier, sd absorbs the
    #    outlier and divides the z by it, so the more extreme the layer the smaller its z. MAD
    #    is computed from the PEERS only here, but the property is what makes that choice
    #    matter, so it is asserted rather than assumed.
    tight = [1.70, 1.72, 1.73, 1.735, 1.74, 1.75, 1.76, 1.78]
    sig = mad_sigma(tight)
    if not (0.005 < sig < 0.06):
        fails.append(f"mad_sigma of a tight peer cluster is {sig:.4f}; the fact quotes 0.0241 "
                     f"for exactly this shape, so a scaling error here rescales every z")
    import statistics
    with_outlier = tight + [1.20]
    if mad_sigma(with_outlier) > 2 * sig:
        fails.append(f"mad_sigma more than doubled when one outlier was added "
                     f"({sig:.4f} -> {mad_sigma(with_outlier):.4f}) -- it is behaving like sd, "
                     f"and a z computed with it shrinks as the outlier gets more extreme")
    if statistics.stdev(with_outlier) <= statistics.stdev(tight) * 2:
        fails.append("the fixture's outlier does not even inflate sd, so this check cannot "
                     "show MAD's advantage -- make the outlier more extreme")

    # 6. THE CONSISTENCY STATISTIC MUST DISTINGUISH 'STOPPED MOVING' FROM 'MOVED AND CANCELLED'.
    #    This is the whole reason the fact reports it alongside growth: both a frozen layer and a
    #    thrashing layer hold their norm, and only this statistic tells them apart. A version
    #    that divided by ||w_b - w_a|| itself, or normalized by the realized displacement, would
    #    return ~1 for both and the reading would say "layer 9 stopped moving" when what it does
    #    is move and cancel. Two synthetic layers, same norm growth, opposite mechanism.
    with tempfile.TemporaryDirectory() as td:
        torch.manual_seed(0)
        d, n, lr, nsn = 8, 100, 0.01, 8.0
        base = torch.randn(d, d)
        step = lr * nsn                       # ONE Muon step's displacement: w -= lr*NS(m), and
        #                                       ||NS(m)||_F is ns_norm, so this is already the
        #                                       whole tensor's move. Dividing by sqrt(d) here (my
        #                                       first draft) made the fixture walk 0.32 of the
        #                                       budget while claiming 0.9, and check 6's second
        #                                       assertion is what caught it.
        # layer 0: FROZEN. Ends where it started, having barely moved.
        # layer 1: THRASHING. Takes n full-sized steps that cancel, ending at the same place.
        walk = torch.randn(d, d)
        walk = walk / walk.norm() * step * n * 0.9
        sda = {"blocks.0.mixer.A_log": torch.zeros(d), "blocks.1.mixer.A_log": torch.zeros(d),
               "blocks.0.mixer.o.weight": base.clone(), "blocks.0.ffn.w2.weight": base.clone(),
               "blocks.1.mixer.o.weight": base.clone(), "blocks.1.ffn.w2.weight": base.clone()}
        sdb = {k: v.clone() for k, v in sda.items()}
        sdb["blocks.0.mixer.o.weight"] = base + walk * 0.02     # small net move
        sdb["blocks.1.mixer.o.weight"] = base + walk            # large net move, same n
        pa, pb = os.path.join(td, "c.pt.step100"), os.path.join(td, "c.pt.step200")
        torch.save({"model": sda}, pa)
        torch.save({"model": sdb}, pb)
        _orig_ckpt = globals()["CKPT"]
        globals()["CKPT"] = os.path.join(td, "c.pt")
        try:
            r = weights_reading(100, 200, 0, lr, ns_norm=nsn)
            c0 = r["per_layer"][0]["mixer_o"]["consistency"]
            c1 = r["per_layer"][1]["mixer_o"]["consistency"]
            if not (c1 > 10 * c0):
                fails.append(f"consistency does not separate a frozen layer from a moving one "
                             f"({c0:.4f} vs {c1:.4f}) -- both hold their norm, and if this "
                             f"statistic cannot tell them apart the reading's 'why' is guesswork")
            # And it must be a FRACTION OF THE BUDGET, not a normalized direction: the moving
            # layer took 0.9 of a fully-aligned walk, so it must read near 0.9, not near 1.0.
            if not (0.5 < c1 < 1.2):
                fails.append(f"the moving layer reads {c1:.4f} of its budget; the fixture walks "
                             f"0.9 of it, so the denominator is not lr*ns_norm*n")
            if r["mixer_o_growth"]["z"] is not None and r["mixer_o_growth"]["peer_n"] != 1:
                fails.append(f"peer_n is {r['mixer_o_growth']['peer_n']}, expected 1 for a "
                             f"two-KDA-layer fixture -- the layer under test is in its own peers")
        except Exception as e:                                    # noqa: BLE001
            fails.append(f"weights_reading raised on a real two-checkpoint fixture: {e!r}")
        finally:
            globals()["CKPT"] = _orig_ckpt

    # 7. ns_norm has NO DEFAULT inside weights_reading: the budget depends on sqrt(d), and a
    #    silently-defaulted 32 would report a d1024 budget for any width.
    try:
        weights_reading(100, 200, 0, 0.01)
        fails.append("weights_reading accepted a missing ns_norm; the consistency denominator "
                     "would then assume d1024 for any model")
    except ValueError:
        pass
    except Exception:                                            # noqa: BLE001
        # FileNotFoundError etc. means it got past the ns_norm check -- which is the bug.
        fails.append("weights_reading reached checkpoint loading with ns_norm unset, so the "
                     "guard is downstream of the work and a wrong budget is only caught by luck")

    if fails:
        for f in fails:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("l9_branch_probe selftest OK: rescaled_checkpoint (the SHIPPED path, not a copy) "
          "changes exactly one tensor by exactly the "
          "requested factor (a silent no-op would report 'no consequence' for every factor, the "
          "failure that looks like a finding); ckpt_path finds both .step and .interrupt.step "
          "spellings, which is how two of b0-16's five points are named; KDA membership is read "
          "from A_log presence rather than assumed from attn_every; median averages the "
          "middle two on an even-length list, since the rescale factor is median/ratio; and "
          "pick_control returns the layer CLOSEST to the median and RAISES when none qualifies, "
          "because the control arm is what separates a real null from the probe having no "
          "resolution at all.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", help="comma-separated steps for the table")
    ap.add_argument("--step", type=int, help="single step, for --rescale")
    ap.add_argument("--rescale", action="store_true",
                    help="scale layer 9's mixer.o to put its branch ratio at the median")
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--control", type=int, default=None,
                    help="layer for the same-factor control arm (default: a near-median KDA layer)")
    ap.add_argument("--out", help="json for the eval rows")
    ap.add_argument("--claim", action="store_true", help="claim the lane card for this pid")
    ap.add_argument("--weights", action="store_true",
                    help="the zero-card reading: growth + direction consistency per KDA layer, "
                         "with the peer median, MAD sigma and z that the fact quotes. NEEDS NO "
                         "CARD -- it is torch.load(map_location='cpu', mmap=True) and tensor "
                         "norms, nothing else, so run it with CUDA_VISIBLE_DEVICES= rather than "
                         "queueing behind a lane job (2026-09-03: it would have waited ~95min "
                         "for a card it never touches)")
    ap.add_argument("--from_step", type=int, help="--weights interval start")
    ap.add_argument("--to_step", type=int, help="--weights interval end")
    ap.add_argument("--lr", type=float, default=0.01, help="Muon lr for the consistency budget")
    ap.add_argument("--ns_norm", type=float, default=32.0,
                    help="||NS(m)||_F, sqrt(d)=32 at d1024 -- Newton-Schulz's design property")
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

    if a.weights:
        if not (a.from_step and a.to_step):
            ap.error("--weights needs --from_step and --to_step")
        r = weights_reading(a.from_step, a.to_step, a.layer, a.lr, ns_norm=a.ns_norm)
        print(f"=== weights-only reading, step {r['step_a']} -> {r['step_b']} (n={r['n']}), "
              f"layer {r['layer']} against {len(r['kda_layers']) - 1} peer KDA layers ===")
        print(f"{'L':>3} {'mixer.o growth':>15} {'consistency':>12} {'ffn.w2 growth':>14} "
              f"{'consistency':>12}")
        for L in r["kda_layers"]:
            p = r["per_layer"][L]
            mark = "  <-- " if L == r["layer"] else ""
            print(f"{L:>3} {p['mixer_o']['growth']:>15.4f} "
                  f"{100 * p['mixer_o']['consistency']:>11.2f}% "
                  f"{p['ffn_w2']['growth']:>14.4f} "
                  f"{100 * p['ffn_w2']['consistency']:>11.2f}%{mark}")
        for k in ("mixer_o_growth", "mixer_o_consistency", "ffn_w2_growth", "ffn_w2_consistency"):
            s = r[k]
            z = f"{s['z']:+.1f}" if s["z"] is not None else "n/a (zero spread)"
            print(f"{k:22s} layer {s['layer']:.4f}  peer median {s['peer_median']:.4f}  "
                  f"MAD sigma {s['peer_mad_sigma']:.4f} (1.4826*median|x-med|)  z {z}")
        print(f"\nz is against the WITHIN-RUN CROSS-LAYER spread over n="
              f"{r['mixer_o_growth']['peer_n']} peers, NOT a seed distribution -- one seed, "
              f"one run. It is also SPECIFIC TO THIS WINDOW (n={r['n']}): growth's z grows as n "
              f"shrinks (peers cluster) while consistency's shrinks as n grows (the budget "
              f"lr*ns_norm*n is linear, realized displacement is not), so no window is "
              f"uniformly stronger and a z quoted without its n is unreadable.")
        # THE CONTROL TENSOR'S VERDICT, stated rather than left for a reader to assemble from two
        # runs. "Only mixer.o is involved" is the claim that rules out 'the whole block is moving
        # and mixer.o is merely the visible part' -- and a single window cannot rule that out,
        # because a window is exactly what such a confound would be free to pick.
        ffn_flat = all(abs(r[f"ffn_w2_{k}"]["z"] or 0) < 3 for k in ("growth", "consistency"))
        print(f"ffn.w2 in THIS window: growth z {r['ffn_w2_growth']['z']:+.1f}, consistency z "
              f"{r['ffn_w2_consistency']['z']:+.1f} -- "
              + ("flat, so the anomaly is confined to mixer.o here. Run the OTHER window too: "
                 "one window cannot exclude 'the whole block is changing'."
                 if ffn_flat else
                 "NOT flat. The ffn is moving too, so 'only mixer.o is involved' does not hold "
                 "in this window and the branch-ratio reading needs restating."))
        if a.out:
            with open(a.out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")   # JSONL, like the other two artifacts
            print(f"appended to {a.out}")
        return 0

    if a.rescale:
        if not a.step:
            ap.error("--rescale needs --step")
        p = ckpt_path(a.step)
        if p is None:
            sys.exit(f"no checkpoint for step {a.step}")
        ratios, kda = branch_ratios(p)
        med = median(ratios.values())
        f = med / ratios[a.layer]
        # THE CONTROL, and without it a null is unreadable. Rescaling layer 9 to the median makes
        # its branch 43% louder; if domain_loss does not move, that has two explanations -- layer
        # 9's branch does not matter, OR the network is insensitive to ANY branch gain at this
        # magnitude. Applying the SAME factor to a normal KDA layer separates them: if the control
        # also does not move, the probe has no resolution and the null says nothing about layer 9.
        ctrl = a.control if a.control is not None else pick_control(ratios, kda, a.layer, med)
        print(f"step {a.step}: layer {a.layer} ratio {ratios[a.layer]:.4f}, 12-layer median "
              f"{med:.4f} -> rescale mixer.o by {f:.4f}", flush=True)
        print(f"control: layer {ctrl} (ratio {ratios[ctrl]:.4f}, near the median) gets the SAME "
              f"x{f:.4f}, so a null can be told apart from no resolution", flush=True)
        print("=== A: unscaled ===", flush=True)
        print(score(p, a.out), flush=True)
        print(f"=== B: layer {a.layer} (the split) rescaled x{f:.4f} ===", flush=True)
        print(score(p, a.out, rescale_layer=a.layer, rescale_factor=f), flush=True)
        print(f"=== C: layer {ctrl} (control, normal) rescaled x{f:.4f} ===", flush=True)
        print(score(p, a.out, rescale_layer=ctrl, rescale_factor=f), flush=True)
        print(f"\nREAD, pre-registered before these numbers existed:\n"
              f"  |B-A| > {BAR}                -> layer {a.layer}'s split has a measured consequence\n"
              f"  |B-A| <= {BAR} and |C-A| > {BAR} -> bounded below the bar; the probe HAS "
              f"resolution, so this is a real null and b0-16 closes as a correlate\n"
              f"  |B-A| <= {BAR} and |C-A| <= {BAR} -> NO RESOLUTION: a 43% branch-gain change is "
              f"invisible to domain_loss anywhere, so the null says nothing about layer {a.layer}\n"
              f"  B WORSE than A does NOT mean the split is harmless -- it means the trained value "
              f"beats the median, which is a different claim.", flush=True)
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
