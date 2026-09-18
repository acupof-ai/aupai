#!/usr/bin/env python3
"""Non-blocking CI diagnostic for the gate_resume bimodal red.

The required gate tests/v41f/test_p1_train_ckpt.py::gate_resume_equivalent_to_uninterrupted
is bit-identical on a clean physical host (digest: OMP 1/2/4, many cores and 4-core pinned,
14/14 green) but goes red on GitHub ubuntu runners with a byte-identical signature
(max|delta|=1.334e-2, n_diff=506533/524288, first_flat_idx=0, n_nan=0) on ~half of runs of
the SAME blob. That is a runner-specific divergence, not random noise (the signature is
reproducible bit-for-bit). This script exists to catch one red INSIDE the runner and report
the first microstage that differs plus the environment and the fp32-alias audit, so the
cause is localized on the machine that exhibits it.

It changes no gate and always exits 0: the CI step that calls it is continue-on-error.
Mirrors the gate's build exactly: v41f_small(indexer_train_mode="off"), B=2 T=16, four
batches from per-batch Generators s=0..3, manual_seed(123), lr=1e-2, checkpoint after 2 of
4 steps.

Modes:
  --diag-orch RUNS OUTDIR   spawn control/restart pair RUNS times; report first red; exit 0
  --diag-arm control|restart OUTDIR   one trajectory, dump microstage tensors + env header
"""
import argparse
import gc
import json
import os
import platform
import subprocess
import sys

import torch


def _repo_on_path():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    for p in (root, os.path.join(root, "tests", "v41f")):
        if p not in sys.path:
            sys.path.insert(0, p)


_repo_on_path()
from ref_oracle import synthetic_tokenizer  # noqa: E402
from v41f import train as v41f_train  # noqa: E402
from v41f.config import v41f_small  # noqa: E402
from v41f.loss import shifted_cross_entropy  # noqa: E402
from v41f.model import V41FModel  # noqa: E402
from v41f.master import TrainState, save_train_checkpoint, load_train_checkpoint  # noqa: E402

SEED = 123
B, T = 2, 16
N, K = 4, 2


def _apply_engine_env():
    """Mirror test_p1_train_ckpt._apply_diag_thread_env in the diag arms: GATE_OMP pins the
    intra-op pool, GATE_ONEDNN=0 disables oneDNN before any tensor op. No-op by default."""
    omp = os.environ.get("GATE_OMP")
    if omp:
        torch.set_num_threads(int(omp))
    if os.environ.get("GATE_ONEDNN") == "0":
        torch.backends.mkldnn.enabled = False


_apply_engine_env()

# The loss MUST be the exact function object the gate's train_step calls
# (v41f.train imports shifted_cross_entropy from v41f.loss). A re-inlined
# F.cross_entropy here would run a different computation (no fp32 cast of the
# logits) and compound into a different 4-step trajectory; the selftest pins
# identity, not numeric closeness.
LOSS_FN = shifted_cross_entropy


def _cgroup_cpu_quota():
    """cgroup v2 cpu.max ("quota period" or "max period") then v1 cfs quota; cores or None."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, p = f.read().split()
        if q != "max":
            return round(int(q) / int(p), 3)
    except (OSError, ValueError):
        pass
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0 and p > 0:
            return round(q / p, 3)
    except (OSError, ValueError):
        pass
    return None


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
    aff = "n/a"
    if hasattr(os, "sched_getaffinity"):
        try:
            aff = str(len(os.sched_getaffinity(0)))
        except OSError:
            aff = "err"
    interop = "n/a"
    try:
        interop = str(torch.get_num_interop_threads())
    except RuntimeError:
        interop = "err-before-init"
    return {
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "intra_op_threads": torch.get_num_threads(),
        "interop_threads": interop,
        "cpu_count": os.cpu_count(),
        "affinity_cpus": aff,
        "cgroup_cpu_quota_cores": _cgroup_cpu_quota(),
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "mkldnn_enabled": bool(getattr(torch.backends.mkldnn, "enabled", None)),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "GATE_ONEDNN": os.environ.get("GATE_ONEDNN"),
    }


def _cfg():
    return v41f_small(indexer_train_mode="off")


def _build(cfg):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return V41FModel(cfg, max_batch_size=B)
    finally:
        torch.set_default_dtype(prev)


def _ids(cfg, s):
    g = torch.Generator().manual_seed(s)
    return torch.randint(0, cfg.vocab_size, (B, T), generator=g)


def _save(d, name, t):
    torch.save(t.detach().cpu().float().clone(), os.path.join(d, name + ".pt"))


def _pick_leaves(st):
    names = sorted(st.in_group_names)
    picks = [names[0], names[len(names) // 2], names[-1]]
    for want in ("head.weight", "layers.0.attn.qproj.wq_b.weight"):
        if want in names:
            picks.append(want)
    seen, out = set(), []
    for n in picks:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def arm(kind, out):
    os.makedirs(out, exist_ok=True)
    cfg = _cfg()
    batches = [_ids(cfg, s) for s in range(N)]
    torch.manual_seed(SEED)
    m = _build(cfg)
    named = dict(m.named_parameters())
    st = TrainState(m, lr=1e-2)
    leaves = _pick_leaves(st)
    with open(os.path.join(out, "env.json"), "w") as f:
        json.dump(env_header(), f, indent=2)
    with open(os.path.join(out, "leaves.json"), "w") as f:
        json.dump(leaves, f)
    _save(out, "state.p0", st.master[leaves[0]])

    def alias_audit(tag):
        named_now = dict(m.named_parameters())
        bad = [n for n in st.fp32_native
               if st.master[n].data_ptr() != named_now[n].data_ptr()]
        with open(os.path.join(out, f"alias_{tag}.json"), "w") as f:
            json.dump({"fp32_native": len(st.fp32_native), "broken_alias": bad}, f)

    alias_audit("build")

    def step(i):
        st.zero_model_grads()
        st.refresh_bf16()
        logits, _ = m(batches[i])
        _save(out, f"fwd_s{i}.logits", logits)
        loss = LOSS_FN(logits, batches[i], ignore_index=-100)
        loss.backward()
        for j, n in enumerate(leaves):
            g = named[n].grad
            if g is not None:
                _save(out, f"bwd_s{i}.l{j}", g)
        st.collect_master_grads()
        st.optimizer.step()
        for j, n in enumerate(leaves):
            if n in st.master:
                _save(out, f"opt_s{i}.l{j}", st.master[n])

    tok = synthetic_tokenizer()
    if kind == "control":
        for i in range(N):
            step(i)
    else:
        for i in range(K):
            step(i)
        ck = os.path.join(out, "c.pt")
        save_train_checkpoint(ck, model=m, cfg=cfg, state=st, tokenizer=tok, step=K)
        del m, st
        gc.collect()
        m, st, _, _ = load_train_checkpoint(ck, tokenizer=tok, max_batch_size=B)
        named = dict(m.named_parameters())
        leaves = json.load(open(os.path.join(out, "leaves.json")))
        alias_audit("load")
        for j, n in enumerate(leaves):
            if n in st.master:
                _save(out, f"loadK.l{j}", st.master[n])
        for i in range(K, N):
            step(i)
    print(f"{kind} arm done -> {out}")


def _load(d, name):
    p = os.path.join(d, name + ".pt")
    return torch.load(p, weights_only=False) if os.path.exists(p) else None


def _cmp(tag, a, b, first):
    if a is None or b is None:
        return first
    if a.shape != b.shape:
        print(f"  {tag:16s} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
        return first if first is not None else tag
    if torch.equal(a, b):
        return first
    d = (a - b).abs()
    rel = d.max().item() / (a.abs().max().item() + 1e-30)
    n_diff = int((~torch.eq(a, b)).sum())
    print(f"  {tag:16s} DIFF relmax={rel:.4e} n_diff={n_diff}/{a.numel()} "
          f"({100.0*n_diff/a.numel():.1f}%)")
    return first if first is not None else tag


def compare_iter(cdir, rdir):
    leaves = json.load(open(os.path.join(cdir, "leaves.json")))
    first = None
    ce = json.load(open(os.path.join(cdir, "env.json")))
    re = json.load(open(os.path.join(rdir, "env.json")))
    if ce != re:
        print("  ENV control/restart differ:")
        for k in ce:
            if ce[k] != re[k]:
                print(f"    {k}: control={ce[k]} restart={re[k]}")
    cb = json.load(open(os.path.join(cdir, "alias_build.json")))
    rl = json.load(open(os.path.join(rdir, "alias_load.json")))
    print(f"  alias: control build broken={cb['broken_alias']} "
          f"restart post-load broken={rl['broken_alias']} (fp32_native={cb['fp32_native']})")
    first = _cmp("state.p0", _load(cdir, "state.p0"), _load(rdir, "state.p0"), first)
    for i in range(N):
        first = _cmp(f"fwd_s{i}.logits", _load(cdir, f"fwd_s{i}.logits"),
                     _load(rdir, f"fwd_s{i}.logits"), first)
        for j in range(len(leaves)):
            first = _cmp(f"bwd_s{i}.l{j}", _load(cdir, f"bwd_s{i}.l{j}"),
                         _load(rdir, f"bwd_s{i}.l{j}"), first)
            first = _cmp(f"opt_s{i}.l{j}", _load(cdir, f"opt_s{i}.l{j}"),
                         _load(rdir, f"opt_s{i}.l{j}"), first)
    for j in range(len(leaves)):
        first = _cmp(f"LOADK.l{j}", _load(cdir, f"opt_s{K-1}.l{j}"),
                     _load(rdir, f"loadK.l{j}"), first)
    return first


def _run_pair(cdir, rdir, env):
    """One control + restart pair as two children with a SHARED env; returns (rc, first)."""
    rc = 0
    for kind, d in (("control", cdir), ("restart", rdir)):
        r = subprocess.run([sys.executable, __file__, "--diag-arm", kind, d],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            rc = r.returncode
            print(f"  {kind} arm CRASHED rc={r.returncode}\n{r.stderr[-1200:]}")
    if rc != 0:
        return rc, None
    return 0, compare_iter(cdir, rdir)


def pair_once(out):
    """A single diag control/restart pair (spawned itself by the colocator so it is a sibling
    of the gate child). Prints one RED/GREEN line; non-zero exit never fails CI here."""
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("GATE_OMP", "2"))
    cdir, rdir = os.path.join(out, "c"), os.path.join(out, "r")
    for d in (cdir, rdir):
        os.makedirs(d, exist_ok=True)
    rc, first = _run_pair(cdir, rdir, env)
    if rc:
        print("DIAGPAIR ARM-CRASH")
        return 0
    print("DIAGPAIR RED " + (first or "") if first else "DIAGPAIR GREEN")
    return 0


_GATE_TEST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_p1_train_ckpt.py")


_BURNCpu = ("import os,sys,time\n"
            "# outer-sibling CPU burner: saturate one core to inject scheduling contention\n"
            "t=time.time()+float(sys.argv[1])\n"
            "x=0.0\n"
            "while time.time()<t:\n"
            "    for i in range(200000): x+=i\n"
    )


def _start_burn(n, seconds):
    """Spawn n outer-sibling processes that each pin a core busy for `seconds`. They are
    siblings of the gate/diag children (never in-process), so they inject only scheduler
    contention, matching the hypothesis that a busy shared runner trips a nondeterministic
    bf16 reduction. Returns Popen handles; caller terminates them."""
    return [subprocess.Popen([sys.executable, "-c", _BURNCpu, str(seconds)])
            for _ in range(n)] if n else []


def coloc(runs, out, stress=0):
    """Co-located discriminator: each iteration spawns TWO sibling children with one shared
    env injection path -- the REAL gate (test_p1_train_ckpt.py --gate gate_resume_...) and one
    diag pair. Process tree is symmetric by construction (both are children of this orch;
    neither runs in the orch's own torch context), so a different red rate isolates the test
    BODY, not spawn structure or outer-process torch init. `stress` adds outer-sibling burner
    processes for the whole run (contention arm). Continues even on a mismatch."""
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = os.environ.get("GATE_OMP", "2")
    if os.environ.get("GATE_MKL"):
        env["MKL_NUM_THREADS"] = os.environ["GATE_MKL"]
    print("COLOCATE ENV " + json.dumps(env_header()) + f" stress_outer_siblings={stress}")
    print("process tree: orch -> [outer siblings: burners] + [sibling: real gate subprocess] "
          "and [sibling: diag pair] each iteration; the diag pair then spawns control/restart, "
          "exactly as the real gate's --gate spawns its own control/restart workers.")
    # burners live for the whole sweep; each gate/diag pair takes ~25s, give headroom.
    burn = _start_burn(stress, runs * 60 + 120)
    gate_red = diag_red = 0
    diverge = []
    try:
        for it in range(runs):
            idir = os.path.join(out, f"co{it}")
            os.makedirs(idir, exist_ok=True)
            g = subprocess.run(
                [sys.executable, _GATE_TEST, "--gate",
                 "gate_resume_equivalent_to_uninterrupted"],
                capture_output=True, text=True, env=env)
            g_is_red = g.returncode != 0
            d = subprocess.run([sys.executable, __file__, "--diag-pair",
                                os.path.join(idir, "d")],
                               capture_output=True, text=True, env=env)
            line = next((l for l in d.stdout.splitlines() if l.startswith("DIAGPAIR")),
                        "DIAGPAIR ?")
            d_is_red = line.startswith("DIAGPAIR RED") or line.startswith("DIAGPAIR ARM")
            gate_red += g_is_red
            diag_red += d_is_red
            tag = "" if g_is_red == d_is_red else "  <-- GATE/DIAG DISAGREE"
            if tag:
                sig = (g.stderr or g.stdout)[-600:]
                diverge.append((it, sig, line))
            gsig = "RED" if g_is_red else "green"
            print(f"iter {it}: gate={gsig} {line}{tag}")
    finally:
        for p in burn:
            p.terminate()
    print(f"\nCOLOCATE SUMMARY over {runs} (stress={stress}): gate_red={gate_red} "
          f"diag_red={diag_red} disagreements={len(diverge)}; same trigger only if the two "
          f"rates match. exits 0 regardless.")
    for it, sig, line in diverge[:3]:
        print(f"--- disagreement iter {it}: {line}\n{sig}")


def orch(runs, out):
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("GATE_OMP", "2"))
    print("ENV " + json.dumps(env_header()))
    reds = 0
    first_overall = None
    for it in range(runs):
        idir = os.path.join(out, f"iter{it}")
        cdir, rdir = os.path.join(idir, "c"), os.path.join(idir, "r")
        for d in (cdir, rdir):
            os.makedirs(d, exist_ok=True)
        rc, first = _run_pair(cdir, rdir, env)
        if rc != 0:
            print(f"iter {it}: ARM CRASH (IO/infra, not a tensor divergence)")
            reds += 1
            continue
        print(f"iter {it}:")
        if first is None:
            print("  GREEN (all microstages bit-identical)")
        else:
            print(f"  RED — FIRST DIVERGENT MICROSTAGE: {first}")
            reds += 1
            first_overall = first_overall or first
            break
    print(f"\nDIAG SUMMARY: {reds}/{runs} non-green; first divergent stage={first_overall}; "
          f"diagnostic exits 0 regardless")


def _write_arm_dir(d, env, leaves, stages, *, broken_build=(), broken_load=()):
    os.makedirs(d, exist_ok=True)
    json.dump(env, open(os.path.join(d, "env.json"), "w"))
    json.dump(leaves, open(os.path.join(d, "leaves.json"), "w"))
    json.dump({"fp32_native": 3, "broken_alias": list(broken_build)},
              open(os.path.join(d, "alias_build.json"), "w"))
    json.dump({"fp32_native": 3, "broken_alias": list(broken_load)},
              open(os.path.join(d, "alias_load.json"), "w"))
    for name, t in stages.items():
        _save(d, name, t)


def selftest():
    """The diagnostic must not be an empty watcher: green world reports no divergence, a
    planted divergence is localized to exactly its microstage, and env/alias fields exist."""
    import inspect
    import tempfile
    import v41f.loss as _vloss
    # The diag's loss must be the SAME FUNCTION OBJECT the gate's train_step calls (identity,
    # not numeric closeness: the bimodal signature is itself a small per-step drift that a
    # tolerance would admit). A mutant that re-inlines F.cross_entropy reds here by identity.
    assert LOSS_FN is _vloss.shifted_cross_entropy is v41f_train.shifted_cross_entropy, (
        "diag loss is not v41f.loss.shifted_cross_entropy — trajectory would differ from the "
        "gate's; import and call that function, do not reimplement shift+CE")
    src = inspect.getsource(arm)
    assert "LOSS_FN(" in src and "F.cross_entropy" not in src, (
        "arm must compute its loss through LOSS_FN (the gate's function), not an inlined "
        "F.cross_entropy (which skips shifted_cross_entropy's fp32 logits cast)")
    root = tempfile.mkdtemp(prefix="diag_selftest_")
    leaves = ["a", "b", "c"]
    env = env_header()
    assert env["torch"] and env["platform"] and env["intra_op_threads"] is not None
    assert "cgroup_cpu_quota_cores" in env and "affinity_cpus" in env

    x = torch.arange(8, dtype=torch.float32) / 4.0

    def full_stages(mult=1.0):
        st = {"state.p0": x * mult}
        for i in range(N):
            st[f"fwd_s{i}.logits"] = x * mult
            for j in range(len(leaves)):
                st[f"bwd_s{i}.l{j}"] = x * mult
                st[f"opt_s{i}.l{j}"] = x * mult
        for j in range(len(leaves)):
            st[f"loadK.l{j}"] = x * mult
        return st

    # green world: identical everywhere
    g = os.path.join(root, "g")
    c0, r0 = os.path.join(g, "c"), os.path.join(g, "r")
    _write_arm_dir(c0, env, leaves, full_stages())
    _write_arm_dir(r0, env, leaves, full_stages())
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        first = compare_iter(c0, r0)
    assert first is None, f"green world falsely reported divergence at {first}"
    assert "broken=[]" in buf.getvalue(), "alias audit line missing"

    # mutant world: perturb only opt step 3 leaf 1 by 1.3%; must localize there and nowhere else
    mdir = os.path.join(root, "m")
    c1, r1 = os.path.join(mdir, "c"), os.path.join(mdir, "r")
    mut = full_stages()
    mut["opt_s3.l1"] = mut["opt_s3.l1"] * 1.013
    _write_arm_dir(c1, env, leaves, full_stages())
    _write_arm_dir(r1, env, leaves, mut)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        first1 = compare_iter(c1, r1)
    assert first1 == "opt_s3.l1", f"expected localization opt_s3.l1, got {first1}"
    assert buf.getvalue().count("DIFF") == 1, "exactly one stage must flag for one mutated tensor"

    # alias break must be visible in the audit (the storage-sharing check is real)
    ab = os.path.join(root, "ab")
    c2, r2 = os.path.join(ab, "c"), os.path.join(ab, "r")
    _write_arm_dir(c2, env, leaves, full_stages())
    _write_arm_dir(r2, env, leaves, full_stages(), broken_load=["head.weight"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        compare_iter(c2, r2)
    assert "post-load broken=['head.weight']" in buf.getvalue(), "broken alias not reported"
    print("diag_resume_bimodal selftest OK: green world clean, 1.3% mutant localized to its "
          "exact microstage alone, env/alias fields present and a broken alias is reported.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-orch", type=int, metavar="RUNS")
    ap.add_argument("--diag-colocate", type=int, metavar="RUNS")
    ap.add_argument("--stress", type=int, default=0, metavar="CORES")
    ap.add_argument("--diag-pair", action="store_true")
    ap.add_argument("--diag-arm", choices=["control", "restart"])
    ap.add_argument("--diag-selftest", action="store_true")
    ap.add_argument("out", nargs="?", default=None)
    a = ap.parse_args()
    if a.diag_selftest:
        return selftest()
    if a.diag_arm:
        arm(a.diag_arm, a.out)
    elif a.diag_pair:
        pair_once(a.out or os.path.join("/tmp", "diag_pair_out"))
    elif a.diag_colocate:
        coloc(a.diag_colocate, a.out or os.path.join("/tmp", "diag_colocate_out"),
              stress=a.stress)
    elif a.diag_orch:
        orch(a.diag_orch, a.out or os.path.join("/tmp", "diag_resume_out"))
    else:
        ap.error("need --diag-colocate/--diag-orch RUNS, --diag-pair, --diag-arm, or --selftest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
