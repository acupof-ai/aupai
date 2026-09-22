#!/usr/bin/env python3
# restartable: read-only collector; spawns the real gate's CPU workers, writes only the JSON
# records under --out and removes each pair dump after reading it. Re-running just resamples.
"""Calibration collector for gate_resume_equivalent_to_uninterrupted (K6, 2026-09-22).

PASSIVE, MANUAL, CPU-ONLY. This file is invoked by the workflow_dispatch `calibrate-resume`
job (and a developer). It is not a required check, does not run on push/PR, and sets NO
gate bound: it only measures, on the environment that actually flakes (the ubuntu x86 CI
runner, torch CPU), the healthy and mutated distributions the future geometric bound must
separate. See docs/standards/resume_gate_divergence_instrumentation.md (K6 section).

Two families, the same split the replacement gate uses:
  det         bit-exact: optimizer AdamW triple control-optK vs restart-loadK
              (torch.equal), populated-but-dropped leaf set, strict-load missing/unexpected.
  geometric   whole-tensor on the gate's own probe leaf (layers.0.attn.qproj.wq_b.weight)
              for BOTH fp32 master and bf16 run weight: rel-L2, cosine, max|d|/rms.
              Elementwise equality is deliberately NOT a verdict (the rejected oracle).

Modes:
  --healthy N   N independent control/restart pairs, ZERO injection, reusing the REAL gate
                worker (test_p1_train_ckpt.py --run control|restart) with GATE_DUMP_DIR so
                the worker's own optK/loadK/ckpt_identity instrumentation lands per pair.
  --mutants     drop_leaf / rot_triple / block_x1.1 / reinit restart arms; each must be
                caught -- the first two by the DET family, the last two by GEOMETRIC --
                proving the eventual bound separates a real save/load harm from healthy noise.

Injections happen on the SAVED BLOB between save and load (a one-process calibration
worker), never in v41f/ training code or the required test. The healthy path runs the
untouched gate worker byte-for-byte, including two separate processes per pair -- the
cross-process CPU bf16 reduction nondeterminism is the quantity being sampled, so inlining
the two trajectories in one process would measure the wrong thing.
"""
import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # tests/v41 -> repo root (v41f package)
sys.path.insert(0, _HERE)        # ref_oracle sibling
sys.path.insert(0, _ROOT)        # v41f package

_GATE_TEST = os.path.join(_HERE, "test_p1_train_ckpt.py")
PROBE = "layers.0.attn.qproj.wq_b.weight"
B, T, K, N, SEED, LR = 2, 16, 2, 4, 123, 1e-2


# ---------------------------------------------------------------------------- metrics

def tensor_metrics(want: torch.Tensor, got: torch.Tensor) -> dict:
    w = want.detach().cpu().float().flatten()
    g = got.detach().cpu().float().flatten()
    diff = w - g
    nw = float(torch.linalg.norm(w))
    ng = float(torch.linalg.norm(g))
    rms = float(torch.sqrt(torch.mean(w ** 2)))
    rmse = float(torch.sqrt(torch.mean(diff ** 2)))
    return {
        "n": int(w.numel()),
        "n_nan": int(torch.isnan(diff).sum()),
        "n_inf": int(torch.isinf(diff).sum()),
        "rel_l2": float(torch.linalg.norm(diff) / nw) if nw > 0 else None,
        "cosine": float(torch.dot(w, g) / (nw * ng)) if nw > 0 and ng > 0 else None,
        "max_abs": float(diff.abs().max()),
        "rms_want": rms,
        # AGGREGATE near-zero-robust HARD metric: RMSE/ref-RMS is a whole-tensor ratio and
        # discriminates on measured data (healthy runner ~0.02, block x1.1 ~0.1). max|d|/rms
        # is a single-coordinate statistic dominated by the largest coordinate (healthy
        # ~0.33, x3 margin exceeds 1) -> diagnostic only, never a verdict.
        "rmse": rmse,
        "rmse_over_rms": float(rmse / rms) if rms > 0 else None,
        "max_abs_over_rms": float(diff.abs().max() / rms) if rms > 0 else None,
        "frac_element_diff": int((~torch.eq(w, g)).sum()) / int(w.numel()),
    }


def _dist(rows, key):
    xs = sorted(r[key] for r in rows if r[key] is not None)
    return {"min": xs[0], "median": statistics.median(xs), "max": xs[-1]} if xs else {}


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def env_header():
    return {
        "torch": torch.__version__, "platform": platform.platform(), "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "affinity_cpus": (str(len(os.sched_getaffinity(0)))
                          if hasattr(os, "sched_getaffinity") else "n/a"),
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "GATE_ONEDNN": os.environ.get("GATE_ONEDNN"),
    }


# ---------------------------------------------------------------------------- healthy pairs

def _run_gate_worker(kind: str, dump_dir: str, env: dict):
    """The REAL gate worker; returns (fp32_master, bf16_run_weight) probe tensors."""
    r = subprocess.run([sys.executable, _GATE_TEST, "--run", kind],
                       capture_output=True, env=env, check=True,
                       cwd=_ROOT)
    a, b = r.stdout.split(b"\x00SEP\x00")
    return (torch.frombuffer(bytearray(a), dtype=torch.float32).clone(),
            torch.frombuffer(bytearray(b), dtype=torch.float32).clone())


def _det_from_dump(pair_dir: str):
    """Bit-exact det family from the worker's own GATE_DUMP_DIR instrumentation."""
    import diag_resume_bimodal as diag
    cdir, rdir = os.path.join(pair_dir, "control"), os.path.join(pair_dir, "restart")
    with open(os.path.join(rdir, "leaves.json")) as fh:
        leaves = json.load(fh)
    rec = {"triple_mismatch": [], "triple_checked": 0, "populated_but_dropped": [],
           "model_missing_keys": None, "model_unexpected_keys": None, "pass": True}
    for j, n in enumerate(leaves):
        for part in ("exp_avg", "exp_avg_sq", "step"):
            c = diag._load(cdir, f"optK.l{j}.{part}")
            l = diag._load(rdir, f"loadK.l{j}.{part}")
            if c is None and l is None:
                continue
            rec["triple_checked"] += 1
            if c is None or l is None or not torch.equal(c, l):
                rec["triple_mismatch"].append(f"{n}.{part}")
    ident_path = os.path.join(rdir, "ckpt_identity.json")
    if os.path.exists(ident_path):
        with open(ident_path) as fh:
            ident = json.load(fh)
        rec["populated_but_dropped"] = ident.get("populated_but_dropped", [])
        rec["model_missing_keys"] = ident.get("model_missing_keys")
        rec["model_unexpected_keys"] = ident.get("model_unexpected_keys")
    rec["pass"] = (rec["triple_checked"] > 0
                   and not rec["triple_mismatch"] and not rec["populated_but_dropped"]
                   and not rec["model_missing_keys"] and not rec["model_unexpected_keys"])
    # triple_checked > 0 is load-bearing: every predicate above is a negation, so a dump with
    # no optK/loadK tensors at all (a picked leaf that had no AdamW state at K, or a broken
    # dump path) would otherwise pass vacuously and silently void the drop_leaf/rot_triple
    # discrimination. One-sided absence is already a mismatch; this closes both-sides-absent.
    return rec


# The healthy distribution is sampled UNDER LOAD, not only idle. The D17 red fires under
# scheduling contention on a shared runner; an idle runner produces a zero-variance bf16
# quantization gap (batch 1 measured rel-L2 0.0161556 constant across 10 pairs), and a margin
# on a constant has no statistical basis. These arms mirror diag_resume_bimodal's matrix:
# stress = outer-sibling CPU burners (diag._start_burn); GATE_OMP/GATE_ONEDNN are read by the
# gate worker itself (_apply_diag_thread_env). `load` marks the arms the bound must cover;
# idle is the baseline only.
HEALTHY_ARMS = [
    {"name": "A_idle_t2", "stress": 0, "env": {"GATE_OMP": "2"}},
    {"name": "B_load_t2", "stress": 4, "env": {"GATE_OMP": "2"}},
    {"name": "C_load_t2_nodnn", "stress": 4, "env": {"GATE_OMP": "2", "GATE_ONEDNN": "0"}},
    {"name": "D_load_t1", "stress": 4, "env": {"GATE_OMP": "1"}},
]

# Whole-tensor metrics a bound is written from. max_abs_over_rms is deliberately excluded:
# it is a single-coordinate statistic (healthy ~0.33) with no headroom for a margin.
HARD_METRICS = ("rel_l2", "cosine", "rmse_over_rms")


def healthy(pairs_per_arm: int, out: str, env: dict) -> dict:
    import diag_resume_bimodal as diag

    rows = []
    for arm in HEALTHY_ARMS:
        aenv = dict(env)
        aenv.update({k: v for k, v in arm["env"].items() if v is not None})
        aout = os.path.join(out, arm["name"])
        os.makedirs(aout, exist_ok=True)
        # burners live for the whole arm: build + both workers per pair, same budget the diag
        # colocate matrix uses (~60 s/pair + startup margin).
        burns = diag._start_burn(arm["stress"], pairs_per_arm * 60 + 120) if arm["stress"] else []
        try:
            for i in range(pairs_per_arm):
                pair_dir = os.path.join(aout, f"pair{i}")
                os.makedirs(pair_dir, exist_ok=True)
                penv = dict(aenv, GATE_DUMP_DIR=pair_dir)
                cm, cb = _run_gate_worker("control", pair_dir, penv)
                rm, rb = _run_gate_worker("restart", pair_dir, penv)
                det = _det_from_dump(pair_dir)
                row = {"arm": arm["name"], "load_arm": bool(arm["stress"]), "pair": i,
                       "det": det,
                       "master_fp32": tensor_metrics(cm, rm),
                       "bf16_weight": tensor_metrics(cb, rb)}
                rows.append(row)
                shutil.rmtree(pair_dir, ignore_errors=True)
                mm = row["master_fp32"]
                print(f"{arm['name']:16s} pair {i:2d}: det={'PASS' if det['pass'] else 'FAIL'} | "
                      f"master rel-L2 {mm['rel_l2']:.4e} cos {mm['cosine']:.6f} "
                      f"rmse/rms {mm['rmse_over_rms']:.4e} | bf16 rel-L2 "
                      f"{row['bf16_weight']['rel_l2']:.4e} cos {row['bf16_weight']['cosine']:.6f} "
                      f"rmse/rms {row['bf16_weight']['rmse_over_rms']:.4e}", flush=True)
        finally:
            for bp in burns:
                bp.terminate()

    def fam_dist(subset):
        return {fam: {metric: _dist([r[fam] for r in subset], metric)
                      for metric in HARD_METRICS}
                for fam in ("master_fp32", "bf16_weight")}

    per_arm = {arm["name"]: fam_dist([r for r in rows if r["arm"] == arm["name"]])
               for arm in HEALTHY_ARMS}
    load_rows = [r for r in rows if r["load_arm"]]
    # The bound inputs over the arms the red actually occurs on: maxima for the upper-bounded
    # metrics, minimum cosine. Recorded per family so a later PR cannot silently use idle.
    load_extreme = {}
    for fam in ("master_fp32", "bf16_weight"):
        rel = [r[fam]["rel_l2"] for r in load_rows if r[fam]["rel_l2"] is not None]
        rmsf = [r[fam]["rmse_over_rms"] for r in load_rows
                if r[fam]["rmse_over_rms"] is not None]
        cos = [r[fam]["cosine"] for r in load_rows if r[fam]["cosine"] is not None]
        load_extreme[fam] = {
            "rel_l2_max": max(rel, default=None),
            "rmse_over_rms_max": max(rmsf, default=None),
            "cosine_min": min(cos, default=None),
        }
    return {"schema": "v41f-resume-healthy-distribution/v2", "calibrated": False,
            "kind": "healthy", "probe": PROBE, "pairs_per_arm": pairs_per_arm,
            "arms": [a["name"] for a in HEALTHY_ARMS],
            "env": env_header(), "pairs": rows,
            "distribution_per_arm": per_arm,
            "distribution_pooled": fam_dist(rows),
            "load_arm_extremes": load_extreme}


# ---------------------------------------------------------------------------- mutation worker

def _ids(cfg, s):
    g = torch.Generator().manual_seed(s)
    return torch.randint(0, cfg.vocab_size, (B, T), generator=g)


def _build(cfg):
    from v41f.model import V41FModel
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return V41FModel(cfg, max_batch_size=B)
    finally:
        torch.set_default_dtype(prev)


def _inject(blob: dict, kind: str):
    """Mutate the SAVED blob between save and load. Returns the names touched."""
    sbn = blob["optim_named"]["state_by_name"]
    if kind == "drop_leaf":
        # The master.py `if not st: continue` silent-fresh-optimizer shape: a populated name
        # simply absent from state_by_name. Pick the probe leaf when populated, else first.
        name = PROBE if PROBE in sbn else sorted(sbn)[0]
        del sbn[name]
        return [name]
    if kind == "rot_triple":
        name = PROBE if PROBE in sbn else sorted(sbn)[0]
        # fill_, not negate: exp_avg can be ~0 at K and a negation leaves zeros bit-identical.
        sbn[name]["exp_avg"].fill_(1.0)
        return [name]
    touched = []
    prefix = "layers.0." if kind == "block_x1.1" else None
    for n in list(blob["master_fp32"]):
        if kind == "block_x1.1":
            if not n.startswith(prefix):
                continue
            blob["master_fp32"][n] = blob["master_fp32"][n] * 1.1
            if n in blob["model"] and isinstance(blob["model"][n], torch.Tensor):
                t = blob["model"][n]
                blob["model"][n] = (t.float() * 1.1).to(t.dtype)
            touched.append(n)
        elif kind == "reinit" and n == PROBE:
            g = torch.Generator().manual_seed(999)
            t = blob["master_fp32"][n]
            blob["master_fp32"][n] = torch.randn(t.shape, generator=g) * 0.02
            if n in blob["model"]:
                tt = blob["model"][n]
                blob["model"][n] = (blob["master_fp32"][n]).to(tt.dtype)
            touched.append(n)
    if not touched:
        raise RuntimeError(f"mutation {kind} touched no tensor")
    return touched


def mutant_arm(kind: str, out: str):
    """One restart trajectory with a blob injected at K. Emits the gate worker's exact stdout
    framing (master NULSEP bf16) so the parent compares it like any restart arm."""
    import gc

    import diag_resume_bimodal as diag
    from ref_oracle import synthetic_tokenizer

    from v41f.config import v41f_small
    from v41f.master import TrainState, load_train_checkpoint, save_train_checkpoint
    from v41f.train import train_step

    cfg = v41f_small(indexer_train_mode="off")
    batches = [_ids(cfg, s) for s in range(N)]
    torch.manual_seed(SEED)
    m = _build(cfg)
    st = TrainState(m, lr=LR)
    leaves = diag._pick_leaves(st)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "leaves.json"), "w") as fh:
        json.dump(leaves, fh)
    tok = synthetic_tokenizer()
    for i in range(K):
        train_step(m, batches[i], None, state=st)
    f = os.path.join(out, "c.pt")
    save_train_checkpoint(f, model=m, cfg=cfg, state=st, tokenizer=tok, step=K)
    # Free the live trajectory BEFORE reading the blob: the blob is ~2.5 GB and holding model
    # + fp32 master + AdamW moments alongside it peaks ~2x and gets the arm OOM-killed
    # (SIGKILL/137). The real restart worker deletes its model before load for the same reason.
    del m, st
    gc.collect()
    blob = torch.load(f, map_location="cpu", weights_only=False)
    touched = _inject(blob, kind)
    dropped = [n for n in touched if n not in blob["optim_named"]["state_by_name"]]
    torch.save(blob, f)
    del blob
    gc.collect()
    m, st, _, _ = load_train_checkpoint(f, tokenizer=tok, max_batch_size=B)
    diag._dump_opt_triple(out, "loadK", st, leaves)
    ident = diag._ckpt_identity(f, live_model_keys=set(m.state_dict().keys()))
    # The dropped set is exactly the touched names absent from the saved state_by_name; the
    # field mirrors the real worker's so the parent's det reader handles mutants unchanged.
    ident["populated_but_dropped"] = dropped
    with open(os.path.join(out, "ckpt_identity.json"), "w") as fh:
        json.dump(ident, fh)
    # The blob has served its only purpose (load + loadK triple + identity sha/keys). Keep it
    # out of the uploaded artifact: each arm's blob is ~2.5 GB and five of them made batch 1's
    # artifact 18 GB. loadK/identity/probe are already in hand/JSON before this point.
    os.remove(f)
    for i in range(K, N):
        train_step(m, batches[i], None, state=st)
    st.refresh_bf16()
    named = dict(m.named_parameters())
    sys.stdout.buffer.write(st.master[PROBE].detach().cpu().float().numpy().tobytes())
    sys.stdout.buffer.write(b"\x00SEP\x00")
    sys.stdout.buffer.write(named[PROBE].detach().cpu().float().numpy().tobytes())


def _run_mutant_worker(kind: str, out: str, env: dict):
    r = subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--mutant-arm", kind, out],
                       capture_output=True, env=env, cwd=_ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"mutant arm {kind} crashed:\n{r.stderr.decode()[-2000:]}")
    a, b = r.stdout.split(b"\x00SEP\x00")
    return (torch.frombuffer(bytearray(a), dtype=torch.float32).clone(),
            torch.frombuffer(bytearray(b), dtype=torch.float32).clone())


MUTANT_EXPECT = {
    "drop_leaf": "det", "rot_triple": "det",
    "block_x1.1": "geometric", "reinit": "geometric",
}


# Geometric separation cutoff when NO healthy distribution is supplied (a standalone
# --mutants run). This is a DISCRIMINATOR, not a calibrated bound: it is deliberately far
# above the one measured red-run rel-L2 (1.6e-2) and far below the mildest mutant the
# mutation arm injects (block x1.1 ~0.3), only to name caught/not-caught in isolation. The
# CI path always passes the in-run healthy MAX, which replaces this with a measured number;
# the published gate bound comes later from healthy_distribution.json, never from here.
GEOM_CUTOFF_FALLBACK = 0.05


def mutants(out: str, env: dict, load_extreme: dict | None = None) -> dict:
    # one shared healthy control across mutant arms; its dump provides control optK. The
    # control runs IDLE: the mutant harms are O(0.2..1.0), far above even loaded healthy
    # noise, so idle control measures the same separation and avoids holding burners for the
    # whole mutant sweep. Build the mutant root fresh so re-running is a clean resample.
    shutil.rmtree(out, ignore_errors=True)
    cdir = os.path.join(out, "mut_control", "control")
    os.makedirs(cdir, exist_ok=True)
    cenv = dict(env, GATE_DUMP_DIR=os.path.join(out, "mut_control"))
    cm, cb = _run_gate_worker("control", os.path.join(out, "mut_control"), cenv)
    rows = []
    for kind, expect_family in MUTANT_EXPECT.items():
        rdir = os.path.join(out, f"mut_{kind}")
        os.makedirs(rdir, exist_ok=True)
        # mutant arm writes into rdir directly; give _det_from_dump a pair-shaped dir by
        # symlinking the shared control dump as its "control". Remove a stale link first.
        pair_dir = os.path.join(out, f"pair_{kind}")
        os.makedirs(pair_dir, exist_ok=True)
        for link_name, target in (("control", cdir), ("restart", rdir)):
            link = os.path.join(pair_dir, link_name)
            if os.path.islink(link) or os.path.exists(link):
                os.unlink(link)
            os.symlink(target, link)
        rm, rb = _run_mutant_worker(kind, rdir, env)
        det = _det_from_dump(pair_dir)
        row = {"kind": kind, "expected_family": expect_family, "det": det,
               "master_fp32": tensor_metrics(cm, rm),
               "bf16_weight": tensor_metrics(cb, rb)}
        # Geometric cutoff is the LOAD-arm healthy rel-L2 MAX (the environment the red occurs
        # on) when the healthy arm ran this invocation; a mutant counts as geometrically
        # caught only strictly above that measured noise, or outright on NaN/Inf. Standalone
        # --mutants falls back to the labelled constant; neither is a published bound.
        cutoff = (load_extreme["master_fp32"]["rel_l2_max"]
                  if load_extreme is not None else GEOM_CUTOFF_FALLBACK)
        mm = row["master_fp32"]
        det_caught = not det["pass"]
        nonfinite = (mm["n_nan"] > 0 or mm["n_inf"] > 0)
        geom_caught = nonfinite or (mm["rel_l2"] is not None and mm["rel_l2"] > cutoff)
        caught_family = "det" if det_caught else ("geometric" if geom_caught else "NONE")
        row["geom_cutoff_used"] = cutoff
        row["cutoff_source"] = ("load_arm healthy rel_l2_max (this run)"
                                if load_extreme is not None
                                else "GEOM_CUTOFF_FALLBACK (no healthy arm; not a bound)")
        row["caught_by"] = caught_family
        row["separates"] = caught_family == expect_family
        rows.append(row)

        def f(v, spec=".4e"):
            return format(v, spec) if isinstance(v, (int, float)) else str(v)

        print(f"mutant {kind:11s}: expect {expect_family:9s} caught_by {caught_family:9s} "
              f"{'OK' if row['separates'] else 'FAILURE'} | master rel-L2 "
              f"{f(mm['rel_l2'])} cos {f(mm['cosine'], '.6f')} "
              f"rmse/rms {f(mm['rmse_over_rms'])} (cutoff {f(cutoff)})", flush=True)
    geom_rows = [r["master_fp32"]["rel_l2"] for r in rows
                 if r["expected_family"] == "geometric"
                 and r["master_fp32"]["rel_l2"] is not None]
    return {"schema": "v41f-resume-mutation-discrimination/v2", "calibrated": False,
            "kind": "mutants", "probe": PROBE, "env": env_header(),
            "separated": all(r["separates"] for r in rows),
            "load_rel_l2_max_supplied": cutoff,
            "cutoff_source": rows[0]["cutoff_source"] if rows else None,
            "mutant_min_rel_l2": min(geom_rows) if geom_rows else None,
            "mutants": rows}


# ---------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--healthy", type=int, default=0, metavar="N",
                   help="healthy PAIRS PER ARM (4 arms: idle + 3 loaded); >=5 recommended")
    p.add_argument("--mutants", action="store_true")
    p.add_argument("--mutant-arm", choices=list(MUTANT_EXPECT), help=argparse.SUPPRESS)
    p.add_argument("--out", default=None)
    p.add_argument("arm_out", nargs="?")
    args = p.parse_args(argv)

    if args.mutant_arm:
        mutant_arm(args.mutant_arm, args.arm_out)
        return 0
    if not args.out:
        p.error("--out is required")

    os.makedirs(args.out, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("GATE_OMP", "2"))
    if os.environ.get("GATE_MKL"):
        env["MKL_NUM_THREADS"] = os.environ["GATE_MKL"]
    if os.environ.get("GATE_ONEDNN") == "0":
        torch.backends.mkldnn.enabled = False

    healthy_rec = None
    if args.healthy:
        healthy_rec = healthy(args.healthy, os.path.join(args.out, "healthy"), env)
        with open(os.path.join(args.out, "healthy_distribution.json"), "w") as fh:
            json.dump(healthy_rec, fh, indent=2)
        print(f"\nHEALTHY, {args.healthy} pairs each over {len(HEALTHY_ARMS)} arms "
              "(idle + loaded). Bound inputs are the LOAD-arm extremes, not idle:")
        for fam, ex in healthy_rec["load_arm_extremes"].items():
            print(f"  {fam:12s} LOAD rel_l2_max {ex['rel_l2_max']:.4e} "
                  f"rmse/rms_max {ex['rmse_over_rms_max']:.4e} cosine_min "
                  f"{ex['cosine_min']:.6f}")
        print("  per-arm master rel-L2 (min/median/MAX):")
        for arm in HEALTHY_ARMS:
            d = healthy_rec["distribution_per_arm"][arm["name"]]["master_fp32"]["rel_l2"]
            print(f"    {arm['name']:16s} {d['min']:.4e} / {d['median']:.4e} / {d['max']:.4e}")
    if args.mutants:
        # Pass the load-arm extremes so mutant geometric cutoff is measured on the environment
        # the red fires in. Standalone --mutants passes None -> labelled fallback.
        lext = healthy_rec["load_arm_extremes"] if healthy_rec is not None else None
        result = mutants(os.path.join(args.out, "mutants"), env, load_extreme=lext)
        with open(os.path.join(args.out, "mutation_discrimination.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nMUTATION SEPARATION: {'all mutants caught by their family' if result['separated'] else 'A MUTANT WAS NOT CAUGHT'}")
        # non-blocking diagnostic: report status in the record, exit 0 either way
    print(f"wrote records under {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
