#!/usr/bin/env python3
"""Training observability snapshot: one self-contained HTML page on stdout.

Every number comes from a harness artifact — this is a view over the existing
sources of truth, never a second one:

  runs/experiments.jsonl            exp.py rows: running jobs and their results
  runs/score_matrix.jsonl           per-checkpoint panel metrics
  facts/*.json                      measured facts
  scripts/harness.py check          the invariants (subprocess, parsed)
  docs/standards/0830v1_gates.md    the gate table and the six-point plan
  data/corpus/*/build_corpus_stats.json + live content fingerprint
  runs/<name>.log                   live step/loss/tok-s/ETA of running jobs
  nvidia-smi                        GPU allocation

Read-only: file reads and nvidia-smi. No GPU compute, no profiler, nothing that
slows training. Runs on the pod (live data) and on a local checkout; the header
states which side the snapshot reads — local green is not pod green.

Usage: python3 scripts/dashboard.py > dash.html
"""
import datetime
import glob
import html
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
GATES_DOC = os.path.join(ROOT, "docs", "standards", "0830v1_gates.md")

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from corpus_fingerprint import fp_dir  # noqa: E402  (stdlib-only)


def side():
    """Which filesystem this snapshot reads. The pod checkout has no .git."""
    if os.path.isdir(os.path.join(ROOT, ".git")):
        try:
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        except Exception:
            commit = "unknown"
        return "LOCAL", ROOT, commit
    return "POD", ROOT, "not a git repo"


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return rows


def experiments():
    rows = read_jsonl(os.path.join(RUNS, "experiments.jsonl"))
    running = [r for r in rows if r.get("status") == "running"]
    done = [r for r in rows if r.get("status") in ("ok", "fail")]
    return running, done


def score_rows():
    return {r["ckpt"]: r for r in read_jsonl(os.path.join(RUNS, "score_matrix.jsonl")) if "ckpt" in r}


def facts_feed(limit=10):
    feed = []
    for path in sorted(glob.glob(os.path.join(ROOT, "facts", "*.json"))):
        try:
            d = json.load(open(path))
        except json.JSONDecodeError:
            continue
        for e in d.get("facts", []):
            feed.append({"file": os.path.basename(path), **e})
    feed.sort(key=lambda e: e.get("measured", ""), reverse=True)
    return feed[:limit]


def harness_check():
    """Subprocess, so the dashboard sees exactly what `harness check` prints."""
    try:
        out = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "harness.py"), "check"],
            capture_output=True, text=True, cwd=ROOT, timeout=300,
        ).stdout
    except Exception as e:
        return [("FAIL", "harness_check", f"could not run harness: {e}")]
    rows = []
    for ln in out.splitlines():
        m = re.match(r"\s*\[(PASS|FAIL|SKIP|WARN)\]\s+(\S+)\s+(.*)", ln)
        if m:
            rows.append((m.group(1), m.group(2), m.group(3).strip()))
    return rows


def gates():
    """Parse the gate table and the six-point plan from the gate doc."""
    if not os.path.exists(GATES_DOC):
        return [], []
    text = open(GATES_DOC).read()
    gate_rows, plan_rows = [], []
    in_gates = in_plan = False
    for ln in text.splitlines():
        if ln.startswith("## "):
            in_gates = ln.startswith("## Gates")
            in_plan = ln.startswith("## The six-point")
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if in_gates and len(cells) == 5 and cells[0] not in ("gate", "") and set(cells[0]) != {"-"}:
            gate_rows.append(dict(zip(("gate", "opens", "evidence", "owner", "status"), cells)))
        if in_plan and len(cells) == 4 and re.match(r"[\d.]+[bB]", cells[0]) and cells[1] != "tokens":
            plan_rows.append(dict(zip(("point", "tokens", "steps", "wall"), cells)))
    return gate_rows, plan_rows


_STEP_RE = re.compile(
    r"step (\d+)/(\d+).*?loss ([\d.]+).*?([\d.]+)K tok/s/gpu.*?ETA ([\d.]+)h"
)


def live_log(name):
    """Last progress line of runs/<name>.log, or None."""
    path = os.path.join(RUNS, f"{name}.log")
    if not os.path.exists(path):
        return None
    last = None
    try:
        with open(path, errors="replace") as f:
            for ln in f:
                m = _STEP_RE.search(ln)
                if m:
                    last = m
    except OSError:
        return None
    if not last:
        return None
    step, total, loss, tps, eta = last.groups()
    return {"step": int(step), "total": int(total), "loss": float(loss), "tps": float(tps), "eta": float(eta)}


def gpu_state():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return None
    cards = []
    for ln in out.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) == 3:
            cards.append({"idx": parts[0], "mem": int(float(parts[1])), "util": int(float(parts[2]))})
    try:
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        holders = {}
        for ln in procs.strip().splitlines():
            p = [x.strip() for x in ln.split(",")]
            if len(p) == 2:
                holders[p[0]] = int(float(p[1]))
    except Exception:
        holders = {}
    for c in cards:
        c["held_mb"] = sum(mb for pid, mb in holders.items() if mb and mb > 1000 and abs(mb - c["mem"]) < 2000) or None
    return cards


def corpus_freeze():
    """Stamped fingerprint vs live content fingerprint per corpus domain."""
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "data", "corpus", "*/"))):
        stats = os.path.join(d, "build_corpus_stats.json")
        if not os.path.exists(stats):
            continue
        domain = os.path.basename(d.rstrip("/"))
        try:
            stamped = json.load(open(stats)).get("fingerprint", "")
        except json.JSONDecodeError:
            stamped = ""
        live = fp_dir(d)
        rows.append({"domain": domain, "stamped": stamped, "live": live, "match": stamped == live})
    return rows


def six_point_cards(plan, running, done, scores):
    """Join the plan table with experiments (by mix name in cmd) and score rows (by ckpt name)."""
    cards = []
    for p in plan:
        point = p["point"]
        mix = f"mix_scale_{point}"
        exp = next((r for r in running + done if mix in r.get("cmd", "")), None)
        ckpt = next((c for c in scores if point in c), None)
        score = scores.get(ckpt) if ckpt else None
        card = {"point": point, "tokens": p["tokens"], "wall": p["wall"], "exp": exp, "ckpt": ckpt, "score": score}
        if exp and exp.get("status") == "running":
            card["status"] = "running"
            card["live"] = live_log(exp["name"])
        elif exp and exp.get("status") == "fail":
            card["status"] = "failed"
        elif score or (exp and exp.get("status") == "ok"):
            card["status"] = "done"
        else:
            card["status"] = "pending"
        cards.append(card)
    return cards


def blocker(gate_rows):
    """What blocks the six points (G4's own 'blocked on X, Y' cell, resolved live);
    falling back to the first non-GREEN gate."""
    by_name = {g["gate"]: g for g in gate_rows}
    g4 = by_name.get("G4 six points")
    if g4 and "blocked on" in g4["status"]:
        deps = re.findall(r"G\d+[a-z]?", g4["status"].split("blocked on", 1)[1])
        # gate names are "G3 corpus", "G3b warmup": match dep + space, so "G3"
        # does not match "G3b"
        open_deps = [
            g for g in gate_rows
            if any(g["gate"] == d or g["gate"].startswith(d + " ") for d in deps)
            and not g["status"].startswith("GREEN")
        ]
        if open_deps:
            return open_deps
    return [g for g in gate_rows if not g["status"].startswith("GREEN")][:1]


def esc(s):
    return html.escape(str(s), quote=True)


def badge(status):
    colors = {"done": ("#1a7f37", "#dafbe1"), "running": ("#0969da", "#ddf4ff"),
              "pending": ("#57606a", "#eaeef2"), "failed": ("#cf222e", "#ffebe9"),
              "GREEN": ("#1a7f37", "#dafbe1"), "ok": ("#1a7f37", "#dafbe1"),
              "FAIL": ("#cf222e", "#ffebe9"), "SKIP": ("#9a6700", "#fff8c5"),
              "WARN": ("#9a6700", "#fff8c5"), "PASS": ("#1a7f37", "#dafbe1")}
    fg, bg = colors.get(status, ("#57606a", "#eaeef2"))
    return f'<span style="color:{fg};background:{bg};padding:1px 8px;border-radius:10px;font-size:12px;white-space:nowrap">{esc(status)}</span>'


CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#f6f8fa;color:#1f2328}
.wrap{max-width:1100px;margin:0 auto;padding:16px 20px 60px}
h1{font-size:20px;margin:0}
h2{font-size:15px;margin:24px 0 8px;border-bottom:1px solid #d0d7de;padding-bottom:4px}
header{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:8px}
.src{font-size:12px;padding:2px 10px;border-radius:10px;background:#ddf4ff;color:#0969da;font-weight:600}
.meta{font-size:12px;color:#57606a}
.banner{border-radius:8px;padding:10px 14px;margin:12px 0;font-size:14px;font-weight:600}
.banner.block{background:#ffebe9;color:#cf222e;border:1px solid #ff8182}
.banner.ok{background:#dafbe1;color:#1a7f37;border:1px solid #4ac26b}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:8px}
.card{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:10px 12px}
.card h3{margin:0 0 4px;font-size:16px;display:flex;justify-content:space-between;align-items:center}
.card .kv{font-size:12px;color:#57606a;line-height:1.7}
.card .val{font-size:20px;font-weight:700;color:#1f2328}
.bar{height:6px;background:#eaeef2;border-radius:3px;margin-top:6px;overflow:hidden}
.bar>i{display:block;height:100%;background:#0969da;border-radius:3px}
table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}
th,td{border:1px solid #d0d7de;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#eaeef2;font-weight:600}
tr.fail td{background:#fff5f7}
details{margin-top:6px}
summary{cursor:pointer;font-size:13px;color:#57606a}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.ok2{color:#1a7f37;font-weight:600}.bad{color:#cf222e;font-weight:600}.mut{color:#57606a}
"""


def render():
    side_name, root, commit = side()
    running, done = experiments()
    scores = score_rows()
    checks = harness_check()
    gate_rows, plan = gates()
    feed = facts_feed()
    cards = six_point_cards(plan, running, done, scores)
    freeze = corpus_freeze()
    gpus = gpu_state()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    open_blockers = blocker(gate_rows)
    any_running = bool(running)

    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>aupai training — {esc(side_name)}</title>",
             f"<style>{CSS}</style></head><body><div class='wrap'>"]
    parts.append(
        f"<header><h1>aupai training</h1><span class='src'>{esc(side_name)}</span>"
        f"<span class='meta'>{esc(root)}</span><span class='meta'>{esc(now)}</span>"
        f"<span class='meta mono'>{esc(commit)}</span></header>"
    )

    if open_blockers:
        for b in open_blockers:
            parts.append(
                f"<div class='banner block'>BLOCKED — {esc(b['gate'])}: {esc(b['status'])} "
                f"&middot; owner {esc(b['owner'])}</div>"
            )
    else:
        parts.append("<div class='banner ok'>all gates green</div>")

    # Six budget points — the milestone, always the biggest block.
    parts.append("<h2>six budget points</h2><div class='grid'>")
    for c in cards:
        parts.append("<div class='card'><h3>%s %s</h3>" % (esc(c["point"]), badge(c["status"])))
        parts.append(f"<div class='kv'>{esc(c['tokens'])} tokens &middot; planned wall {esc(c['wall'])}</div>")
        if c["status"] == "running" and c.get("live"):
            lv = c["live"]
            pct = 100 * lv["step"] / max(lv["total"], 1)
            parts.append(f"<div class='kv'>step {lv['step']}/{lv['total']} &middot; loss {lv['loss']:.3f} "
                         f"&middot; {lv['tps']:.0f}K tok/s/gpu &middot; ETA {lv['eta']:.1f}h</div>")
            parts.append(f"<div class='bar'><i style='width:{pct:.0f}%'></i></div>")
        elif c["status"] == "running":
            parts.append("<div class='kv'>starting&hellip;</div>")
        elif c["score"]:
            m = c["score"].get("metrics", {})
            dl = m.get("domain_loss", {}).get("unweighted_mean")
            n_metrics = len(m)
            n_skip = len(c["score"].get("skipped", {}))
            parts.append(f"<div class='val'>{dl:.4f}</div>" if isinstance(dl, (int, float))
                         else "<div class='val mut'>—</div>")
            parts.append(f"<div class='kv'>score matrix: {n_metrics} metrics, {n_skip} skipped"
                         + (f" &middot; {esc(c['ckpt'])}" if c["ckpt"] else "") + "</div>")
        elif c["status"] == "failed":
            parts.append(f"<div class='kv bad'>failed: {esc((c['exp'] or {}).get('result', '')[:120])}</div>")
        else:
            why = f"blocked on {esc(open_blockers[0]['gate'])}" if open_blockers else "queued"
            parts.append(f"<div class='kv mut'>{why}</div>")
        parts.append("</div>")
    parts.append("</div>")

    # Running jobs — live process. Full width only while something runs.
    if running:
        parts.append("<h2>running jobs</h2><table><tr><th>name</th><th>progress</th><th>loss</th>"
                     "<th>tok/s/gpu</th><th>ETA</th><th>started</th><th>hypothesis</th></tr>")
        for r in running:
            lv = live_log(r["name"])
            prog = f"{lv['step']}/{lv['total']}" if lv else "—"
            loss = f"{lv['loss']:.3f}" if lv else "—"
            tps = f"{lv['tps']:.0f}K" if lv else "—"
            eta = f"{lv['eta']:.1f}h" if lv else "—"
            parts.append(f"<tr><td class='mono'>{esc(r['name'])}</td><td>{prog}</td><td>{loss}</td>"
                         f"<td>{tps}</td><td>{eta}</td><td>{esc(r.get('started', ''))}</td>"
                         f"<td>{esc(r.get('hypothesis', '')[:150])}</td></tr>")
        parts.append("</table>")

    # Harness — one line unless red.
    n_fail = sum(1 for s, _, _ in checks if s == "FAIL")
    n_skip = sum(1 for s, _, _ in checks if s == "SKIP")
    n_warn = sum(1 for s, _, _ in checks if s == "WARN")
    summary = f"{len(checks)} checks — {len(checks) - n_fail - n_skip - n_warn} PASS"
    if n_fail:
        summary += f", <span class='bad'>{n_fail} FAIL</span>"
    if n_warn:
        summary += f", {n_warn} WARN"
    if n_skip:
        summary += f", {n_skip} SKIP"
    parts.append(f"<h2>harness <span class='meta'>(<a href='#harness'>red — expand</a> if FAIL, else green)</span></h2>")
    parts.append(f"<div>{summary}</div>")
    parts.append("<details id='harness'" + (" open" if n_fail else "") + "><summary>per-check evidence</summary><table>"
                 "<tr><th>state</th><th>check</th><th>evidence</th></tr>")
    for state, name, ev in checks:
        cls = "bad" if state == "FAIL" else ("mut" if state in ("SKIP", "WARN") else "ok2")
        parts.append(f"<tr><td class='{cls}'>{state}</td><td class='mono'>{esc(name)}</td><td>{esc(ev)}</td></tr>")
    parts.append("</table></details>")

    # Gates — the live blocker table. Expanded when nothing runs (it IS the answer then).
    parts.append(f"<details{' open' if not any_running else ''}><summary class='h2sum'>gates — what blocks the next milestone</summary>"
                 "<table><tr><th>gate</th><th>opens when</th><th>owner</th><th>status</th></tr>")
    for g in gate_rows:
        st = g["status"]
        state = "GREEN" if st.startswith("GREEN") else ("FAIL" if "fail" in st.lower() else "WARN")
        parts.append(f"<tr><td class='mono'>{esc(g['gate'])}</td><td>{esc(g['opens'])}</td>"
                     f"<td>{esc(g['owner'])}</td><td>{badge(state)} {esc(st.split(';')[0])}</td></tr>")
    parts.append("</table></details>")

    # Corpus freeze.
    if freeze:
        parts.append("<h2>corpus freeze</h2><table><tr><th>domain</th><th>stamped</th><th>live</th><th></th></tr>")
        for r in freeze:
            mark = "<span class='ok2'>match</span>" if r["match"] else "<span class='bad'>DRIFT</span>"
            parts.append(f"<tr><td class='mono'>{esc(r['domain'])}</td><td class='mono'>{esc(r['stamped'][:16])}</td>"
                         f"<td class='mono'>{esc(r['live'][:16])}</td><td>{mark}</td></tr>")
        parts.append("</table>")

    # Finished results.
    if done:
        parts.append("<h2>finished runs</h2><table><tr><th>name</th><th>status</th><th>result</th>"
                     "<th>finding</th><th>decision</th></tr>")
        for r in reversed(done[-15:]):
            cls = "fail" if r.get("status") == "fail" else ""
            parts.append(f"<tr class='{cls}'><td class='mono'>{esc(r['name'])}</td><td>{esc(r.get('status', ''))}</td>"
                         f"<td>{esc(r.get('result', '')[:200])}</td><td>{esc(r.get('finding', '')[:200])}</td>"
                         f"<td>{esc(r.get('decision', '')[:200])}</td></tr>")
        parts.append("</table>")

    # Facts feed.
    parts.append("<details><summary>facts — newest 10</summary>")
    for e in feed:
        parts.append(f"<div class='kv' style='margin:4px 0'><span class='mono'>{esc(e.get('id', '?'))}</span> "
                     f"{badge(e.get('status', ''))} <span class='mut'>{esc(e.get('measured', ''))}</span><br>"
                     f"{esc(str(e.get('value', e.get('claim', '')))[:220])}</div>")
    parts.append("</details>")

    # GPU.
    if gpus:
        parts.append("<h2>GPUs</h2><table><tr><th>idx</th><th>mem (MiB)</th><th>util %</th><th>held</th></tr>")
        for c in gpus:
            held = f"{c['held_mb']} MiB" if c["held_mb"] else ("free" if c["mem"] < 100 else f"{c['mem']} MiB")
            parts.append(f"<tr><td>{c['idx']}</td><td>{c['mem']}</td><td>{c['util']}</td>"
                         f"<td class='{'mut' if c['mem'] < 100 else ''}'>{held}</td></tr>")
        parts.append("</table>")

    parts.append("</div></body></html>")
    return "\n".join(parts)


if __name__ == "__main__":
    print(render())
