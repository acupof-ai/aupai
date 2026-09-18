"""Step-D long-horizon numeric-stability probe for the v41f training path.

Runs `v41f.train.train_step` with a `v41f.master.TrainState` (bf16 forward / fp32 master
AdamW) for N steps on each indexer/engram FORM, and reports, per form:

  layer 1 (the comparable learning signal): per-pool-cycle mean CE over a FIXED pool of
      random batches, first/last cycle delta, and same-batch wins;
  layer 2: per-point single-batch CE min/max/range (noise, not learning);
      master-vs-bf16 max/mean gap first and last; AdamW exp_avg / exp_avg_sq health;
      bit-invariance of every frozen leaf; which SIX indexer leaves moved; NaN/Inf.

Why a tiny synthetic tokenizer: uniform-random ids over the full 12800 vocab are i.i.d. and
UNLEARNABLE -- CE just random-walks near ln12800=9.46, so that setup cannot test "loss keeps
descending" (measured 2026-09-18). The 12-piece tokenizer makes the fixed pool small enough
to be memorized, so a healthy training step drives cycle-mean CE down. That MEMORIZATION is
the boundary: this probe establishes optimizer/numeric hygiene over a long CPU horizon, not
convergence or generalization, and it does not exercise checkpoint/resume (that is the
bit-exact gate in tests/v41f/test_p1_train_ckpt.py).

Usage:
    python probes/v41f_stepd_longrun.py [STEPS=160] [BATCH=1] [OUTDIR]
Writes OUTDIR/v41f_longrun_{form}.json (OUTDIR defaults to ./runs/longrun). The three forms
run as separate invocations in the shell wrapper; call one form directly:
    python probes/v41f_stepd_longrun.py 160 1 runs/longrun off
Memory: each form is one process building an 180M model + fp32 master + AdamW; on a shared
laptop gate launch on >=9 GB free (the ste differentiable attention graph peaks higher and a
memory-starved run is SIGKILLed with rc137, not a Python traceback).
"""

import dataclasses
import gc
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "tests", "v41f"))  # ref_oracle sibling
sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from ref_oracle import synthetic_tokenizer  # noqa: E402
from v41f.config import v41f_small  # noqa: E402
from v41f.master import TrainState  # noqa: E402
from v41f.model import V41FModel  # noqa: E402
from v41f.train import train_step  # noqa: E402

PROBE_BF16 = "layers.0.attn.qproj.wq_b.weight"
POOL = 32
SEQ = 64
SEED = 20260918
LR = 3e-3


def cfg_for(form, tok):
    if form == "engram_on":
        c = dataclasses.replace(
            v41f_small(),
            indexer_train_mode="off",
            vocab_size=len(tok),
            engram_layer_ids=(1,),
            engram_n_heads=2,
            engram_head_dim=8,
            engram_vocab_size=20,
            engram_pad_id=2,
        )
        # with_derived_engram measures engram_compressed_vocab_size and the table rows from
        # the tokenizer itself; passing them by hand drifts from that contract.
        return c.with_derived_engram(tok)
    return v41f_small(indexer_train_mode="ste" if form == "ste" else "off", vocab_size=len(tok))


def build(cfg, tok, batch):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        m = V41FModel(cfg, max_batch_size=batch, max_seq_len=SEQ, tokenizer=tok)
    finally:
        torch.set_default_dtype(prev)
    return m


def run(form, steps, batch):
    torch.manual_seed(SEED)
    tok = synthetic_tokenizer()
    cfg = cfg_for(form, tok)
    model = build(cfg, tok, batch)
    st = TrainState(model, lr=LR)
    named = dict(model.named_parameters())

    g = torch.Generator().manual_seed(7)
    pool = [torch.randint(0, cfg.vocab_size, (batch, SEQ), generator=g) for _ in range(POOL)]

    frozen = sorted(n for n, p in named.items() if not p.requires_grad)
    frozen_sum0 = {n: float(named[n].float().sum()) for n in frozen}
    six = {
        f"layers.{l}.attn.indexer.{w}.weight"
        for l in cfg.index_source_layers
        for w in ("wq_b", "weights_proj")
    }
    six0 = {n: named[n].detach().float().clone() for n in six}

    losses, gaps, mv, all_loss = [], [], [], []
    t0 = time.time()
    for i in range(steps):
        loss = train_step(model, pool[i % POOL], None, state=st)
        st.refresh_bf16()
        if (i + 1) % POOL == 0:
            # ste keeps differentiable attention activations live; collect each cycle so peak
            # does not creep into jetsam on a shared laptop.
            gc.collect()
        v = float(loss)
        if not (v == v) or v in (float("inf"), float("-inf")):
            print(json.dumps({"form": form, "died_at": i, "loss": v}))
            sys.exit(1)
        all_loss.append(v)
        if i < 5 or (i + 1) % 10 == 0:
            losses.append([i + 1, v])
            with torch.no_grad():
                d = (named[PROBE_BF16].float() - st.master[PROBE_BF16].float()).abs()
                gaps.append([i + 1, float(d.max()), float(d.mean())])
        p = st.master[PROBE_BF16]
        s = st.optimizer.state.get(p)
        if s is not None and ((i + 1) % 10 == 0 or i == steps - 1):
            m_, v_ = s["exp_avg"], s["exp_avg_sq"]
            mv.append(
                [
                    i + 1,
                    float(m_.abs().max()),
                    float(v_.max()),
                    float((v_ > 0).float().mean()),
                    bool(torch.isfinite(m_).all() and torch.isfinite(v_).all()),
                ]
            )

    ncycle = steps // POOL
    cyc = [sum(all_loss[c * POOL : (c + 1) * POOL]) / POOL for c in range(ncycle)]
    first_c, last_c = all_loss[:POOL], all_loss[(ncycle - 1) * POOL : ncycle * POOL]
    wins = sum(1 for a, b in zip(first_c, last_c) if b < a)
    frozen_moved = [n for n in frozen if float(named[n].float().sum()) != frozen_sum0[n]]
    six_drift = {n: float((named[n].float() - six0[n]).abs().max()) for n in six}
    win = 8
    backbone_probe = round(float(st.master[PROBE_BF16].float().sum()), 2)
    return {
        "form": form,
        "steps": steps,
        "batch": batch,
        "seq": SEQ,
        "pool": POOL,
        "lr": LR,
        "seed": SEED,
        "vocab_size": cfg.vocab_size,
        "sec": round(time.time() - t0, 1),
        "loss_first": losses[0][1],
        "loss_last": losses[-1][1],
        "loss_min": min(v for _, v in losses),
        "win_first8": round(sum(all_loss[:win]) / win, 4),
        "win_last8": round(sum(all_loss[-win:]) / win, 4),
        "cycle_means": [round(x, 4) for x in cyc],
        "cycle_first_mean": round(cyc[0], 4) if cyc else None,
        "cycle_last_mean": round(cyc[-1], 4) if cyc else None,
        "cycle_delta": round(cyc[-1] - cyc[0], 4) if cyc else None,
        "same_batch_wins": f"{wins}/{POOL}",
        "loss_curve": losses,
        "point_min": round(min(v for _, v in losses), 4),
        "point_max": round(max(v for _, v in losses), 4),
        "gap_first_max": gaps[0][1],
        "gap_last_max": gaps[-1][1],
        "gap_last_mean": gaps[-1][2],
        "gap_series": gaps[:: max(1, len(gaps) // 6)],
        "mv_last": mv[-1] if mv else None,
        "frozen_moved": frozen_moved,
        "n_frozen": len(frozen),
        "six_drift": {k: round(v, 8) for k, v in sorted(six_drift.items())},
        "bad_master": [n for n, p in st.master.items() if not torch.isfinite(p).all()],
        "bad_run": [n for n, p in model.named_parameters() if not torch.isfinite(p).all()],
        "backbone_probe": backbone_probe,
    }


# Structural + learning invariants asserted on EVERY run, so a JSON nobody reads is not the
# point: a broken freeze, a non-finite weight, a diverged bf16 gap, or no descent exits
# nonzero instead of printing a file. n_frozen and SIX are form-specific: off/engram freeze
# F+SIX (8); ste freezes only F (4) and the four SIX indexer leaves must actually update.
EXPECT_FROZEN = {"off": 8, "ste": 4, "engram_on": 8}
_SIX_LEAF_EPS = 1e-6
_GAP_BOUND = 2e-3
_DESCENT_MARGIN = 1.0


def _fin(form, name, x):
    # Explicit finite gate: a NaN slips through every NaN-aware-looking comparison
    # (nan > bound, nan <= eps, nan < margin are all False), so a validator that only
    # compares would print PASS on a NaN field. Check the number itself first.
    if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise AssertionError(f"{form}: {name} is not finite: {x!r}")
    return float(x)


def validate(d):
    form = d["form"]
    if d["bad_master"] or d["bad_run"]:
        raise AssertionError(
            f"{form}: non-finite master/run weights: {d['bad_master'][:3]} {d['bad_run'][:3]}"
        )
    if d["frozen_moved"]:
        raise AssertionError(f"{form}: frozen leaves moved: {d['frozen_moved'][:3]}")
    if d["n_frozen"] != EXPECT_FROZEN[form]:
        raise AssertionError(f"{form}: n_frozen {d['n_frozen']} != {EXPECT_FROZEN[form]}")
    # every SIX drift is a real number; ste must move all four, the others must freeze all.
    for k, v in d["six_drift"].items():
        vv = _fin(form, f"six_drift[{k}]", v)
        if form == "ste" and vv <= _SIX_LEAF_EPS:
            raise AssertionError(f"ste: SIX leaf did not update: {k}={vv}")
        if form != "ste" and vv != 0.0:
            raise AssertionError(f"{form}: SIX leaf must stay frozen: {k}={vv}")
    mv = d["mv_last"]
    if not mv:
        raise AssertionError(f"{form}: missing optimizer m/v record")
    _fin(form, "exp_avg_max", mv[1])
    _fin(form, "exp_avg_sq_max", mv[2])
    vpos = _fin(form, "exp_avg_sq_positive_fraction", mv[3])
    if not mv[4] or vpos < 1.0:
        raise AssertionError(f"{form}: optimizer m/v not finite/positive")
    gap = _fin(form, "gap_last_max", d["gap_last_max"])
    if gap > _GAP_BOUND:
        raise AssertionError(f"{form}: master-bf16 gap {gap} > {_GAP_BOUND}")
    # descent: 40-step selftest records first/last 8-step windows; the committed 160-step
    # evidence records full pool-cycle delta. Either schema (unknown keys are ignored, so the
    # committed evidence keeps its legacy nan_or_inf field untouched).
    if "win_first8" in d:
        a, b = _fin(form, "win_first8", d["win_first8"]), _fin(form, "win_last8", d["win_last8"])
        margin, label = a - b, "first8->last8"
    else:
        if d.get("cycle_delta") is None:
            raise AssertionError(f"{form}: no descent signal (windows or cycle_delta)")
        margin, label = -_fin(form, "cycle_delta", d["cycle_delta"]), "cycle_first->last"
    if margin < _DESCENT_MARGIN:
        raise AssertionError(f"{form}: weak/no descent {label} margin {margin:.3f}")
    return True


def main(argv):
    steps = int(argv[1]) if len(argv) > 1 else 160
    batch = int(argv[2]) if len(argv) > 2 else 1
    outdir = argv[3] if len(argv) > 3 else os.path.join("runs", "longrun")
    form = argv[4] if len(argv) > 4 else "off"
    os.makedirs(outdir, exist_ok=True)
    out = run(form, steps, batch)
    validate(out)
    path = os.path.join(outdir, f"v41f_longrun_{form}.json")
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "form",
                    "cycle_first_mean",
                    "cycle_last_mean",
                    "cycle_delta",
                    "same_batch_wins",
                    "gap_last_max",
                    "frozen_moved",
                    "n_frozen",
                )
            }
        )
    )
    print(path)


# -- selftest -------------------------------------------------------------------------------
# Two forms at 40 steps (off and ste cover the n_frozen 8/4 difference and the SIX off=0 /
# ste>0 dispatch; engram_on is the same requires_grad set as off and already carried by its
# committed 160-step JSON). Each form is its own subprocess: one 180M model + fp32 master +
# AdamW is ~2.5 GB, so two in one process OOMs a laptop. The parent additionally proves the
# STE/off BACKBONE identity across processes (bit-identical backbone probe), then mutates one
# record and asserts validate() goes red -- a self-assertion nobody mutates would be blind.
SELFTEST_STEPS = 40
_SELFTEST_FORMS = ("off", "ste")


def _selftest():
    import shutil
    import subprocess
    import tempfile

    tmp = tempfile.mkdtemp(prefix="plr_selftest_")  # process-private; removed in finally
    env = dict(os.environ, OMP_NUM_THREADS="2")
    records = {}
    try:
        for form in _SELFTEST_FORMS:
            r = subprocess.run(
                [sys.executable, __file__, str(SELFTEST_STEPS), "1", tmp, form],
                capture_output=True,
                text=True,
                env=env,
            )
            if r.returncode != 0:
                print(r.stdout)
                print(r.stderr)
                raise AssertionError(f"selftest form {form} failed (rc={r.returncode})")
            records[form] = json.load(open(os.path.join(tmp, f"v41f_longrun_{form}.json")))
        if records["off"]["backbone_probe"] != records["ste"]["backbone_probe"]:
            raise AssertionError("off/ste backbone must be bit-identical (STE forward identity)")

        # mutation: every guard must catch its own broken record.
        mutants = (
            ({"n_frozen": 7}, "frozen count"),
            ({"frozen_moved": ["x"]}, "moved frozen leaf"),
            ({"bad_master": ["x"]}, "non-finite master"),
            ({"gap_last_max": _GAP_BOUND * 10}, "diverged gap"),
            ({"win_first8": 0.0, "win_last8": 0.0}, "no descent"),
            # NaN compares False to every bound, so without the explicit finite gate these
            # PASS -- genB measured a NaN slipping end to end. Both descent schemas covered.
            ({"win_first8": float("nan"), "win_last8": float("nan")}, "NaN windows"),
        )
        base = dict(records["off"])
        for patch, label in mutants:
            broken = dict(base)
            broken.update(patch)
            try:
                validate(broken)
            except AssertionError:
                continue
            raise AssertionError(f"validate() failed to catch mutant: {label}")
        # the cycle-delta schema (160-step evidence) takes a different branch; mutate it on a
        # record with the window keys removed so the NaN cannot be shadowed by windows.
        cycle_broken = dict(base)
        cycle_broken.pop("win_first8", None)
        cycle_broken.pop("win_last8", None)
        cycle_broken["cycle_delta"] = float("nan")
        try:
            validate(cycle_broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("validate() failed to catch mutant: NaN cycle delta")
        # NaN in the non-descent numeric fields must also be caught.
        for patch, label in (
            ({"gap_last_max": float("nan")}, "NaN gap"),
            ({"mv_last": [40, float("nan"), 1.0, 1.0, True]}, "NaN exp_avg"),
            ({"mv_last": [40, 1.0, float("nan"), 1.0, True]}, "NaN exp_avg_sq"),
            ({"six_drift": {k: float("nan") for k in base["six_drift"]}}, "NaN six drift"),
        ):
            broken = dict(base)
            broken.update(patch)
            try:
                validate(broken)
            except AssertionError:
                continue
            raise AssertionError(f"validate() failed to catch mutant: {label}")
        ste_broken = dict(records["ste"])
        ste_broken["six_drift"] = {k: 0.0 for k in ste_broken["six_drift"]}
        try:
            validate(ste_broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("validate() failed to catch a frozen SIX leaf in ste mode")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # The same flag also guards the committed 160-step evidence (cheap, no model), so the
    # evidence files are not unguarded. cwd is repo_root under the hook and for direct runs.
    ev = os.path.join(_ROOT, "runs", "longrun")
    if os.path.isdir(ev):
        validate_committed(ev)
    print("v41f step-D longrun selftest OK")


def validate_committed(directory):
    """Run the SAME validate() over the committed 160-step evidence JSONs, so the evidence
    files have a guard (a probe --selftest alone only guards freshly generated records).
    Unknown keys are ignored: the evidence predates a schema field removal and must keep its
    bytes. Refuses if the directory holds none of the expected forms."""
    import glob

    paths = sorted(glob.glob(os.path.join(directory, "v41f_longrun_*.json")))
    if not paths:
        raise AssertionError(f"no v41f_longrun_*.json under {directory}")
    seen = []
    for path in paths:
        d = json.load(open(path))
        validate(d)
        seen.append(d["form"])
    print(f"committed longrun evidence OK: {', '.join(seen)} ({len(paths)} file(s))")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--validate-committed" in sys.argv:
        i = sys.argv.index("--validate-committed")
        directory = sys.argv[i + 1] if i + 1 < len(sys.argv) else os.path.join("runs", "longrun")
        validate_committed(directory)
    else:
        main(sys.argv)
