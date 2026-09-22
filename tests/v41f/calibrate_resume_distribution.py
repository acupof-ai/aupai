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
    return {
        "n": int(w.numel()),
        "rel_l2": float(torch.linalg.norm(diff) / nw) if nw > 0 else None,
        "cosine": float(torch.dot(w, g) / (nw * ng)) if nw > 0 and ng > 0 else None,
        "max_abs": float(diff.abs().max()),
        "rms_want": rms,
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


def healthy(n_pairs: int, out: str, env: dict) -> dict:
    rows = []
    for i in range(n_pairs):
        pair_dir = os.path.join(out, f"pair{i}")
        os.makedirs(pair_dir, exist_ok=True)
        penv = dict(env, GATE_DUMP_DIR=pair_dir)
        cm, cb = _run_gate_worker("control", pair_dir, penv)
        rm, rb = _run_gate_worker("restart", pair_dir, penv)
        det = _det_from_dump(pair_dir)
        row = {"pair": i,
               "det": det,
               "master_fp32": tensor_metrics(cm, rm),
               "bf16_weight": tensor_metrics(cb, rb)}
        rows.append(row)
        # The durable result is the row above; the pair dump holds ~5-leaf AdamW triples only
        # (no full tensors in the gate worker), but ten pairs still add up on a ~14 GiB runner.
        # Drop it as soon as the det verdict is read, mirroring diag's coloc disk discipline.
        shutil.rmtree(pair_dir, ignore_errors=True)
        print(f"pair {i:2d}: det={'PASS' if det['pass'] else 'FAIL'} | "
              f"master rel-L2 {row['master_fp32']['rel_l2']:.4e} cos "
              f"{row['master_fp32']['cosine']:.6f} max/rms "
              f"{row['master_fp32']['max_abs_over_rms']:.4e} | bf16 rel-L2 "
              f"{row['bf16_weight']['rel_l2']:.4e} cos {row['bf16_weight']['cosine']:.6f} "
              f"max/rms {row['bf16_weight']['max_abs_over_rms']:.4e}", flush=True)
    dist = {}
    for fam in ("master_fp32", "bf16_weight"):
        dist[fam] = {metric: _dist([r[fam] for r in rows], metric)
                     for metric in ("rel_l2", "cosine", "max_abs_over_rms")}
    return {"schema": "v41f-resume-healthy-distribution/v1", "calibrated": False,
            "kind": "healthy", "probe": PROBE, "n_pairs": n_pairs,
            "env": env_header(), "pairs": rows, "distribution": dist}


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


def mutants(out: str, env: dict, healthy_max_rel_l2: float | None = None) -> dict:
    # one shared healthy control across mutant arms; its dump provides control optK
    # Build the whole mutant root fresh so a second `--out` run is a clean resample instead
    # of dying on the symlinks from the previous run (the restartable contract).
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
        # Geometric cutoff is the measured healthy MAX when the healthy arm ran in this same
        # invocation; a mutant is "caught" only if it lands strictly above healthy noise, so
        # the conclusion cannot be a consequence of a typed constant.
        cutoff = (healthy_max_rel_l2 if healthy_max_rel_l2 is not None
                  else GEOM_CUTOFF_FALLBACK)
        det_caught = not det["pass"]
        geom_caught = (row["master_fp32"]["rel_l2"] is not None
                       and row["master_fp32"]["rel_l2"] > cutoff)
        caught_family = "det" if det_caught else ("geometric" if geom_caught else "NONE")
        row["geom_cutoff_used"] = cutoff
        row["cutoff_source"] = ("healthy_max_rel_l2 (this run)"
                                if healthy_max_rel_l2 is not None
                                else "GEOM_CUTOFF_FALLBACK (no healthy arm; not a bound)")
        row["caught_by"] = caught_family
        row["separates"] = caught_family == expect_family
        rows.append(row)

        def f(v, spec=".4e"):
            return format(v, spec) if isinstance(v, (int, float)) else str(v)

        print(f"mutant {kind:11s}: expect {expect_family:9s} caught_by {caught_family:9s} "
              f"{'OK' if row['separates'] else 'FAILURE'} | master rel-L2 "
              f"{f(row['master_fp32']['rel_l2'])} cos {f(row['master_fp32']['cosine'], '.6f')} "
              f"max/rms {f(row['master_fp32']['max_abs_over_rms'])} (cutoff {f(cutoff)})",
              flush=True)
    geom_rows = [r["master_fp32"]["rel_l2"] for r in rows
                 if r["expected_family"] == "geometric"
                 and r["master_fp32"]["rel_l2"] is not None]
    return {"schema": "v41f-resume-mutation-discrimination/v1", "calibrated": False,
            "kind": "mutants", "probe": PROBE, "env": env_header(),
            "separated": all(r["separates"] for r in rows),
            "healthy_max_rel_l2_supplied": healthy_max_rel_l2,
            "mutant_min_rel_l2": min(geom_rows) if geom_rows else None,
            "mutants": rows}


# ---------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--healthy", type=int, default=0, metavar="N")
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
        print("\nHEALTHY distribution (min/median/MAX; bound candidates derive from MAX):")
        for fam in ("master_fp32", "bf16_weight"):
            for metric in ("rel_l2", "cosine", "max_abs_over_rms"):
                d = healthy_rec["distribution"][fam][metric]
                print(f"  {fam:12s} {metric:16s} {d['min']:.4e} / {d['median']:.4e} / {d['max']:.4e}")
    if args.mutants:
        # Pass the in-run healthy master MAX so the mutant geometric cutoff is measured in the
        # same job/environment, not typed. Falls back to the labelled constant only if --mutants
        # ran alone.
        hmax = None
        if healthy_rec is not None:
            hmax = max(r["master_fp32"]["rel_l2"] for r in healthy_rec["pairs"]
                       if r["master_fp32"]["rel_l2"] is not None)
        result = mutants(os.path.join(args.out, "mutants"), env, healthy_max_rel_l2=hmax)
        with open(os.path.join(args.out, "mutation_discrimination.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nMUTATION SEPARATION: {'all mutants caught by their family' if result['separated'] else 'A MUTANT WAS NOT CAUGHT'}")
        # non-blocking diagnostic: report status in the record, exit 0 either way
    print(f"wrote records under {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
