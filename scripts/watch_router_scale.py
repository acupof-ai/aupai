#!/usr/bin/env python3
"""Watch a live training run's router-logit scale and stop on a 3-point monotonic rise.

Background monitor for the v41_ced_0926 30B retrain (de, 1e order 2026-09-26). The fixprobe
showed the router logit scale can run away with lr + weight_decay 0; the retrain dropped the
router lr to 0.001. The open question is whether the cross-expert logit std re-grows over the
38K-step run. Every --step-every (default 2000) checkpoint this runs the CPU-ONLY
scripts/router_logit_stats.py (nice/ionice, no GPU, safe beside the live world-8 run), appends
one JSON line per checkpoint, and exits nonzero the moment std_c_median rises over THREE
consecutive measured points -- the agreed signal to stop and add a scale bound (router
weight_decay / z-loss / --router_logit_cap).

It only ever inspects checkpoints and appends to a jsonl. It never signals or touches the run.
Bring the jsonl back to runs/ in the repo after the run (pod-only artifacts do not count).

    # run on a laptop that has the `pod` wrapper; it shells out to pod automatically:
    python3 scripts/watch_router_scale.py --name v41_ced_0926 \
        --ckpt-glob '/work/aupai/ckpt_v41_ced_0926.pt.step*' \
        --out runs/router_scale_0926.jsonl --max-iters 400 --poll-s 120

    python3 scripts/watch_router_scale.py --selftest     # monotonic/parse logic, no pod
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

STEP_RE = re.compile(r"(\d+)$")
LAYER_RE = re.compile(r"^L\s*(\d+)\s+std_c\s+([-\d.]+)\s+dead\s+([-\d.]+)\s+gateT1\s+([-\d.]+)", re.M)


def parse_probe_text(txt, expect_layers=12):
    """Parse router_logit_stats stdout into the per-layer dict. Raises if a layer is missing --
    a truncated CPU forward must not be appended as a real point."""
    layers = {}
    for li, sc, dead, g in LAYER_RE.findall(txt):
        layers[int(li)] = {"std_c": float(sc), "dead_frac": float(dead), "gateT1": float(g)}
    if len(layers) < expect_layers:
        raise ValueError(f"only {len(layers)}/{expect_layers} layers parsed")
    return layers


def record_from_text(base, step, txt, expect_layers=12):
    layers = parse_probe_text(txt, expect_layers)
    std = [layers[k]["std_c"] for k in sorted(layers)]
    return {
        "checkpoint": base,
        "step": int(step),
        "n_layers": len(layers),
        "std_c_median": round(sorted(std)[len(std) // 2], 4),
        "std_c_min": round(min(std), 4),
        "std_c_max": round(max(std), 4),
        "layers": {f"L{k}": layers[k] for k in sorted(layers)},
    }


def monotonic_rise(points):
    """True iff the last three points' std_c_median are strictly increasing. Pure; selftest."""
    m = [p["std_c_median"] for p in points]
    return len(m) >= 3 and m[-1] > m[-2] > m[-3]


def load_points(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    out.sort(key=lambda r: r["step"])
    return out


# ---- pod plumbing (thin; all decisions are the pure functions above) ----------------------

def pod(cmd, timeout=None, pod_bin="pod"):
    # The pod wrapper takes ONE command string (`user_cmd="${*}"`) and sends it verbatim to
    # crictl exec; passing ["bash","-lc",cmd] adds an inner shell that mangles setsid/long env
    # lines (the monitor's detached probe never started). So call it as a single argv element.
    return subprocess.run([pod_bin, cmd], capture_output=True, text=True, timeout=timeout)


def pod_bash(script, timeout=None, pod_bin="pod"):
    r = pod(script, timeout=timeout, pod_bin=pod_bin)
    return r.returncode, r.stdout, r.stderr


def ckpt_ready_pod(path, pod_bin):
    """A checkpoint is only probed once its zip is complete (a save in progress truncated
    step452 once; do not read a half-written file)."""
    rc, out, _ = pod_bash(
        f"python3 -c \"import zipfile;zipfile.ZipFile({path!r}).testzip();print('ok')\"",
        pod_bin=pod_bin)
    return rc == 0 and out.strip().endswith("ok")


PROBE_REMOTE = "/tmp/router_logit_stats.py"
PROBE_WRAPPER = "/tmp/run_router_scale_probe.sh"


def push_file_pod(local_path, remote_path, pod_bin):
    """Write a local file onto the pod through the single-string pod contract using base64.
    /tmp is wiped on container restart (measured: the probe vanished between 30B relaunches), so
    the monitor must re-push the probe every cycle rather than assume it persists."""
    import base64
    with open(local_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    rc, out, err = pod_bash(
        f"echo {b64} | base64 -d > {remote_path} && chmod +x {remote_path} "
        f"&& wc -c < {remote_path}", pod_bin=pod_bin)
    if rc != 0 or not out.strip().isdigit():
        raise RuntimeError(f"push {local_path} -> {remote_path} failed: {err or out}")
    return int(out.strip())


def run_probe_pod(ckpt, n_seq, work_txt, pod_bin, probe_local):
    """Re-push the probe (pod /tmp is not durable), write a quote-free wrapper, launch it
    detached with setsid. Returns (pid, work_txt); the caller polls work_txt for the layers."""
    push_file_pod(probe_local, PROBE_REMOTE, pod_bin)
    wrapper = (
        "#!/bin/bash\nset -e\ncd /work/aupai\n"
        "exec env PYTHONPATH=/work/aupai:/work/aupai/eval "
        "CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 AUPAI_ALLOW_CORESIDENT_CACHE=1 "
        f"nice -n19 ionice -c3 python3 {PROBE_REMOTE} --ckpt {ckpt} "
        f"--n_seq {int(n_seq)}\n"
    )
    import base64
    b64 = base64.b64encode(wrapper.encode()).decode()
    rc, out, err = pod_bash(
        f"echo {b64} | base64 -d > {PROBE_WRAPPER} && chmod +x {PROBE_WRAPPER} && echo wrote",
        pod_bin=pod_bin)
    if rc != 0 or "wrote" not in out:
        raise RuntimeError(f"wrapper write failed: {err or out}")
    rc, out, err = pod_bash(f"setsid bash {PROBE_WRAPPER} > {work_txt} 2>&1 </dev/null & echo $!",
                            pod_bin=pod_bin)
    if rc != 0:
        raise RuntimeError(f"probe launch failed: {err or out}")
    pid = out.strip().split()[-1]
    return pid, work_txt


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="v41_ced_0926")
    ap.add_argument("--ckpt-glob", default="/work/aupai/ckpt_v41_ced_0926.pt.step*")
    ap.add_argument("--out", default="runs/router_scale_0926.jsonl")
    ap.add_argument("--step-every", type=int, default=2000)
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--poll-s", type=float, default=120.0)
    ap.add_argument("--probe-script", default=os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "scripts", "router_logit_stats.py"),
                    help="local path of the probe to push to the pod /tmp each cycle")
    ap.add_argument("--max-iters", type=int, default=400,
                    help="hard iteration cap so a background loop cannot run forever")
    ap.add_argument("--local-dir", default="",
                    help="local mode: probe checkpoints already copied under this dir (no pod)")
    ap.add_argument("--pod-bin", default=os.environ.get("POD_BIN", "") or
                    os.path.expanduser("~/bin/pod"),
                    help="path to the pod wrapper (default ~/bin/pod); used unless --local-dir")
    ap.add_argument("--once", action="store_true", help="one scan pass, no poll loop")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = {p["step"] for p in load_points(a.out)}

    def candidates():
        if a.local_dir:
            paths = glob.glob(os.path.join(a.local_dir, "ckpt_*.pt.step*"))
        else:
            rc, out, _ = pod_bash(f"ls -1 {a.ckpt_glob} 2>/dev/null || true", pod_bin=a.pod_bin)
            paths = out.split()
        out_steps = []
        for p in paths:
            m = STEP_RE.search(os.path.basename(p))
            if m and int(m.group(1)) % a.step_every == 0:
                out_steps.append((int(m.group(1)), p))
        return sorted(set(out_steps))

    def collect_done(step, path):
        """If the detached probe's work file has all 12 layers, append the record. Returns the
        record, None if not ready yet. Raises on a probe Traceback (so a missing script after a
        restart surfaces and the cycle re-pushes on the next launch)."""
        base = os.path.basename(path)
        if a.local_dir:
            r = subprocess.run([sys.executable, a.probe_script,
                                "--ckpt", path, "--n_seq", str(a.n_seq)],
                               capture_output=True, text=True)
            txt = r.stdout + r.stderr
            if r.returncode != 0:
                raise RuntimeError(f"local probe step {step}:\n{txt[-800:]}")
        else:
            work = f"/tmp/rls_scale_{a.name}_{step}.txt"
            rc, out, _ = pod_bash(f"cat {work} 2>/dev/null || true", pod_bin=a.pod_bin)
            txt = out
            if "Traceback" in txt:
                raise RuntimeError(f"probe error at step {step}:\n{txt[-800:]}")
            try:
                parse_probe_text(txt)
            except ValueError:
                return None  # still building (or not launched yet)
        rec = record_from_text(base, step, txt)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"step {step}: std_c med {rec['std_c_median']} "
              f"[{rec['std_c_min']},{rec['std_c_max']}] -> {a.out}", flush=True)
        return rec

    launching = set()  # steps whose probe we started this session (don't double-launch)

    it = 0
    while it < a.max_iters:
        it += 1
        for step, path in candidates():
            if step in done:
                continue
            if not a.local_dir and not ckpt_ready_pod(path, a.pod_bin):
                continue
            work = f"/tmp/rls_scale_{a.name}_{step}.txt"
            # Fire-and-forget: launch the detached probe once, then on later polls merely collect
            # the work file. A slow probe (up to ~an hour under contention) never trips a timeout
            # or blocks the loop. Re-push the probe each launch because pod /tmp is wiped on
            # container restart.
            if not a.local_dir and step not in launching:
                # Skip launch if a probe for this step is ALREADY running or already producing
                # layers (started by hand or an earlier monitor run). Two identical probes write
                # the same work file and corrupt each other.
                _, probe_out, _ = pod_bash(
                    f"grep -q '^L' {work} 2>/dev/null && echo yes || echo no",
                    pod_bin=a.pod_bin)
                has_layers = probe_out.strip().endswith("yes")
                _, run_out, _ = pod_bash(
                    f"ps -eo args | grep '[r]outer_logit_stats.py.*step{step}' | wc -l",
                    pod_bin=a.pod_bin)
                running = int(run_out.strip() or "0") > 0
                already = has_layers or running
                if not already:
                    try:
                        run_probe_pod(path, a.n_seq, work, a.pod_bin, a.probe_script)
                        print(f"step {step}: probe launched", flush=True)
                    except Exception as e:  # transient pod/push error: retry next poll, don't die
                        print(f"step {step}: launch failed ({e}); will retry", flush=True)
                        continue
                launching.add(step)
            try:
                rec = collect_done(step, path)
            except Exception as e:
                # probe errored (e.g. stale script after a restart): drop the work file and the
                # launch marker so the next cycle re-pushes and retries instead of wedging.
                print(f"step {step}: probe error ({e}); resetting for retry", flush=True)
                if not a.local_dir:
                    pod_bash(f"rm -f {work}", pod_bin=a.pod_bin)
                launching.discard(step)
                continue
            if rec is None:
                continue  # probe still running; check again next poll
            done.add(step)
            pts = load_points(a.out)
            if monotonic_rise(pts):
                m = [(p["step"], p["std_c_median"]) for p in pts[-3:]]
                print(f"MONOTONIC_RISE over {m}: stop and add a router scale bound", flush=True)
                return 2
        if a.once:
            return 0
        time.sleep(a.poll_s)
    print(f"reached --max-iters {a.max_iters}; loop stopped", flush=True)
    return 1


def _selftest():
    # monotonic rise detection
    base = [{"step": s, "std_c_median": v} for s, v in ((2000, 3.0), (4000, 4.0), (6000, 5.0))]
    assert monotonic_rise(base)
    assert not monotonic_rise(base[:2])
    assert not monotonic_rise([{"step": s, "std_c_median": v}
                               for s, v in ((2000, 5.0), (4000, 4.0), (6000, 5.0))])
    assert not monotonic_rise([{"step": s, "std_c_median": v}
                               for s, v in ((2000, 3.0), (4000, 3.0), (6000, 3.0))])
    # parser: exact probe text -> 12 layers, median, and a truncated text must raise
    txt = "".join(
        f"L{li:2d} std_c {3.0 + 0.1 * li:8.2f} dead 0.800 gateT1 0.6500\n" for li in range(12))
    rec = record_from_text("ckpt.step2000", 2000, txt)
    assert rec["step"] == 2000 and rec["n_layers"] == 12
    assert abs(rec["std_c_median"] - 3.6) < 1e-9, rec  # lower-middle index 6 of sorted 12
    try:
        parse_probe_text(txt.splitlines()[0] + "\n")
    except ValueError:
        pass
    else:
        raise AssertionError("truncated probe text must raise, not append a partial point")
    # wrapper encoding roundtrips the exact launch (the nested-shell quoting bug regressed to
    # "setsid: no command specified" when the long env line was inlined).
    import base64
    w = ("#!/bin/bash\nset -e\ncd /work/aupai\nexec env PYTHONPATH=/work/aupai:/work/aupai/eval "
         "CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 AUPAI_ALLOW_CORESIDENT_CACHE=1 "
         "nice -n19 ionice -c3 python3 /tmp/router_logit_stats.py --ckpt /w/x.pt --n_seq 24\n")
    assert base64.b64decode(base64.b64encode(w.encode())).decode() == w
    assert "set -e" in w and "/work/aupai" in w and "--ckpt /w/x.pt" in w
    print("watch_router_scale selftest OK: 3-point rise, equal/down no-rise, parse + truncation, "
          "wrapper roundtrip")


if __name__ == "__main__":
    sys.exit(main())
