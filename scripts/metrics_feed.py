#!/usr/bin/env python3
# restartable: one read-only pass over a log, then one page overwritten in place; an interrupt
# costs the next poll's 20 minutes and nothing else.
"""Every metric the training log carries, as curves, refreshed to Feishu (de, user order 2026-10-01).

The user asked for "所有的重要的指标都每个点" on Feishu with simple clear curves, and said a
script, automatic, is fine. Three parts, each replaceable on its own:

  parse   the pod's training log -> one series per metric, EVERY point, no downsampling
  render  a self-contained HTML page of small line charts (Chart.js from cdnjs)
  push    a Feishu group-webhook message: the latest value of each metric plus the page's link

Why a link and not an image: a Feishu group webhook can post text and cards, but an IMAGE needs
an image_key from the open API, which needs app credentials this project does not have. So the
numbers go in the message (readable without clicking) and the curves live on a page. Publish the
page wherever it can be served -- an Artifact URL is what this session used -- and pass it with
--page-url so the message links it.

    python3 scripts/metrics_feed.py --run v42_gate_1001r                  # parse + render
    python3 scripts/metrics_feed.py --run v42_gate_1001r --push           # also post to Feishu
    python3 scripts/metrics_feed.py --selftest                            # no pod, no network

The webhook is read from $AUPAI_FEISHU_WEBHOOK or ~/.aupai_feishu_webhook and is never printed:
it is a credential, and a URL in a log is a URL anyone can post to. Without it --push refuses
loudly rather than skipping quietly, because a metrics feed that silently stops feeding is worse
than one that is obviously broken.

Read-only on the pod: it tails a log through ~/bin/pod. It never touches the run, and it holds
no card, so it is safe beside a live gate run.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every field the [main] progress line carries. Each becomes one curve over step.
PROGRESS = re.compile(
    r"step (?P<step>\d+)/(?P<total>\d+)\s+\d+% \[main\]"
    r" \| loss (?P<loss>[\d.]+)"
    r" \| lr (?P<lr>[\d.e+-]+)"
    r".*?\| gnorm (?P<gnorm>[\d.]+)"
    r" \| (?P<tok>[\d.]+)B tok"
    r" \| (?P<tps>[\d.]+)K tok/s/gpu"
    r" \| MFU (?P<mfu>\d+)%"
    r" \| peak (?P<peak>[\d.]+)GiB"
    r" \| ETA (?P<eta>[\d.]+)h"
)
VAL = re.compile(r"step (?P<step>\d+)\b.*?\bval (?P<val>[\d.]+)")
QK = re.compile(r"step (?P<step>\d+) health qk_scale .*?\| median=(?P<qk_median>[\d.eE+-]+)")
GROUPS = re.compile(r"step (?P<step>\d+) health gradnorm .*?\| groups (?P<rest>.+)")
MOE = re.compile(r"step (?P<step>\d+) health \| moe top1 max (?P<moe_top1>\S+)"
                 r" zero-load (?P<zero_load>\d+) \| opt moved min (?P<opt_moved>[\d.]+)%"
                 r" \| alarms (?P<alarms>\d+)")

#: name -> (unit, "up" if higher is better else "down" if lower is better else None). The
#: direction is only for the arrow in the Feishu line; no threshold is invented here.
METRICS = {
    "loss": ("train loss", "down"), "val": ("val loss", "down"),
    "lr": ("learning rate", None), "gnorm": ("grad norm (pre-clip)", None),
    "tps": ("K tok/s/gpu", "up"), "mfu": ("MFU %", "up"),
    "peak": ("peak GiB/card", None), "eta": ("ETA hours", "down"),
    "qk_median": ("qk_scale median", None), "opt_moved": ("params moved %", None),
    "zero_load": ("experts with no tokens", "down"), "alarms": ("alarms", "down"),
}


def read_log(run, log_path=None, local=None):
    """The whole log, from the pod unless --local-log names a file already on this machine."""
    if local:
        with open(local, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    path = log_path or f"/work/aupai/runs/{run}.log"
    pod = os.path.expanduser("~/bin/pod")
    out = subprocess.run([pod, f"cat {path}"], capture_output=True, text=True, timeout=300)
    if out.returncode != 0 or not out.stdout.strip():
        raise SystemExit(f"could not read {path} on the pod: rc={out.returncode} "
                         f"{out.stderr.strip()[:200]}")
    return out.stdout


def parse(text):
    """{metric: [[step, value], ...]} over EVERY point in the log, plus a `meta` dict.

    Duplicate steps collapse to the last reading: the launcher's own runlog echoes each
    progress line, so counting lines would double every point (the same duplication that
    made a watchdog window span half the steps it claimed).
    """
    series, meta = {}, {"total_steps": None, "run": None}
    seen = {}

    def keep(key, step, raw):
        """float() accepts "nan" and "inf", so a diagnostic that did not compute would enter
        the series as a point and Chart.js would draw a gap that looks like missing data
        rather than a broken instrument. Non-finite is not a reading."""
        try:
            v = float(raw)
        except ValueError:
            return
        if math.isfinite(v):
            seen.setdefault(key, {})[step] = v

    for line in text.splitlines():
        m = PROGRESS.search(line)
        if m:
            d = m.groupdict()
            step = int(d.pop("step"))
            meta["total_steps"] = int(d.pop("total"))
            d.pop("tok", None)
            for k, v in d.items():
                keep(k, step, v)
            continue
        for rx in (VAL, QK, MOE):
            m = rx.search(line)
            if m:
                d = m.groupdict()
                step = int(d.pop("step"))
                for k, v in d.items():
                    keep(k, step, v)
                break
        else:
            m = GROUPS.search(line)
            if m:
                step = int(m.group("step"))
                for part in m.group("rest").split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                        keep(f"grad_{k}", step, v)
    for k, points in seen.items():
        series[k] = [[s, points[s]] for s in sorted(points)]
    return series, meta


def summary(series, meta):
    """The latest reading of each metric, and how it moved since the previous point."""
    rows = []
    for k, points in series.items():
        if not points:
            continue
        step, last = points[-1]
        prev = points[-2][1] if len(points) > 1 else None
        label, direction = METRICS.get(k, (k, None))
        arrow = ""
        if prev is not None and last != prev:
            up = last > prev
            arrow = {"up": "↑" if up else "↓", "down": "↑" if up else "↓", None: "↑" if up else "↓"}[direction]
            if direction:
                arrow += " ok" if (up if direction == "up" else not up) else " worse"
        rows.append({"key": k, "label": label, "step": step, "value": last,
                     "prev": prev, "arrow": arrow.strip(), "n": len(points)})
    rows.sort(key=lambda r: (r["key"] not in METRICS, r["key"]))
    return rows


def render_html(series, meta, rows, title):
    """One small line chart per metric, plain and readable. Self-contained but for Chart.js."""
    charts = [k for k in list(METRICS) if series.get(k)] + \
             [k for k in sorted(series) if k.startswith("grad_")]
    data = {k: series[k] for k in charts}
    head = "".join(
        f'<div class="kv"><span class="k">{METRICS.get(r["key"], (r["key"], None))[0]}</span>'
        f'<span class="v">{r["value"]:g}</span>'
        f'<span class="a">{r["arrow"]}</span></div>' for r in rows if r["key"] in METRICS)
    last_step = max((s[-1][0] for s in series.values() if s), default=0)
    total = meta.get("total_steps") or 0
    pct = f"{100 * last_step / total:.1f}%" if total else "?"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{{--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--acc:#2563eb}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#111418;--fg:#e8e8e8;--mut:#9aa0a6;--line:#2a2f36;--acc:#6ea8fe}}}}
:root[data-theme="dark"]{{--bg:#111418;--fg:#e8e8e8;--mut:#9aa0a6;--line:#2a2f36;--acc:#6ea8fe}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;padding:24px 16px}}
.wrap{{max-width:1100px;margin:0 auto}}
h1{{font-size:20px;margin:0 0 4px}}
.sub{{color:var(--mut);font-size:13px;margin-bottom:18px}}
.kvs{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:22px}}
.kv{{border:1px solid var(--line);border-radius:8px;padding:8px 11px;min-width:120px}}
.k{{display:block;color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em}}
.v{{font-size:19px;font-variant-numeric:tabular-nums}}
.a{{font-size:11px;color:var(--mut);margin-left:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}}
.card{{border:1px solid var(--line);border-radius:10px;padding:12px}}
.card h2{{font-size:13px;margin:0 0 8px;color:var(--mut);font-weight:600}}
canvas{{width:100%!important;height:170px!important}}
</style></head><body><div class="wrap">
<h1>{title}</h1>
<div class="sub">step {last_step} of {total} ({pct}) &middot; every logged point, no downsampling</div>
<div class="kvs">{head}</div>
<div class="grid" id="grid"></div>
</div>
<script>
const DATA = {json.dumps(data)};
const LABEL = {json.dumps({k: METRICS.get(k, (k, None))[0] for k in charts})};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const grid = document.getElementById('grid');
for (const [key, pts] of Object.entries(DATA)) {{
  if (!pts.length) continue;
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `<h2>${{LABEL[key] || key}}</h2><canvas></canvas>`;
  grid.appendChild(card);
  new Chart(card.querySelector('canvas'), {{
    type: 'line',
    data: {{ datasets: [{{
      data: pts.map(([x, y]) => ({{x, y}})),
      borderColor: css('--acc'), borderWidth: 1.6,
      pointRadius: pts.length > 60 ? 0 : 2, tension: 0.15, fill: false }}] }},
    options: {{
      animation: false, parsing: false, normalized: true,
      plugins: {{ legend: {{ display: false }},
        tooltip: {{ callbacks: {{ title: i => 'step ' + i[0].parsed.x }} }} }},
      scales: {{
        x: {{ type: 'linear', grid: {{ color: css('--line') }},
              ticks: {{ color: css('--mut'), maxTicksLimit: 6 }} }},
        y: {{ grid: {{ color: css('--line') }},
              ticks: {{ color: css('--mut'), maxTicksLimit: 5 }} }} }} }}
  }});
}}
</script></body></html>
"""


def webhook():
    """The Feishu group webhook, never echoed. Env first, then the dotfile."""
    url = os.environ.get("AUPAI_FEISHU_WEBHOOK", "").strip()
    if not url:
        p = os.path.expanduser("~/.aupai_feishu_webhook")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                url = fh.read().strip()
    return url


AWB = os.path.expanduser("~/.local/bin/awb")


def _med(vals):
    """The median, printed. Empty is '?', never 0 -- an absent reading is not a zero one."""
    if not vals:
        return "?"
    v = sorted(vals)
    n = len(v)
    return f"{(v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2):.3g}"


def _mx(vals):
    return "?" if not vals else f"{max(vals):.3g}"


def awb_push(series, meta, name, target, awb_dir, state_path, dry=False):
    """The goal's number on the AWB board, by step, plus one news line.

    AWB draws ONE goal series, so only val goes here -- the other fifteen curves are the
    page's job. The series gets its OWN name: the board already carries `v42_val` from the
    retired stage-2 line, and appending this run's readings to it would put two runs in one
    curve with nothing recording where one ends. Two series, no join key, is how a board
    starts lying.

    Only points newer than the last one pushed are sent, so a 20-minute poll does not append
    the same reading 72 times a day. The watermark is per (board, series).
    """
    vals = series.get("val") or []
    if not vals:
        return "no val reading yet"
    try:
        with open(state_path, encoding="utf-8") as fh:
            sent = json.load(fh)
    except (OSError, ValueError):
        sent = {}
    key = f"{awb_dir}::{name}"
    fresh = [(s, v) for s, v in vals if s > sent.get(key, -1)]
    if not fresh:
        return f"{name}: no new point past step {sent[key]}"
    env = dict(os.environ, AWB_DIR=awb_dir)
    last_step = max((p[-1][0] for p in series.values() if p), default=0)
    total = meta.get("total_steps") or 0

    def at(k, step):
        """The reading of k nearest at-or-before `step`; None when the series has none."""
        pts = [v for s, v in (series.get(k) or []) if s <= step]
        return pts[-1] if pts else None

    def g(k, step, fmt="{:g}", dash="?"):
        v = at(k, step)
        return dash if v is None else fmt.format(v)

    cmds = []
    # The train-loss curve as a reference series: --ref draws it BESIDE the goal, so the card
    # shows val against train rather than val alone. Its own name, so it is never the goal.
    for s, _ in fresh:
        tl = at("loss", s)
        if tl is not None:
            cmds.append([AWB, "metric", "--ref", "--step", str(s), f"{name}_train",
                         f"{tl:g}", f"{target:g}"])
    # The NOTE under the bar carries the numbers that do not get their own curve on the card.
    for s, v in fresh:
        cmds.append([AWB, "metric", "--step", str(s), name, f"{v:g}", f"{target:g}",
                     f"训练误差 {g('loss', s)} · 速度 {g('tps', s)}K/卡 · MFU {g('mfu', s)}% "
                     f"· 显存 {g('peak', s, '{:.0f}')}G · 梯度 {g('gnorm', s)}"])
    last = fresh[-1][0]
    # One keyed news line per topic: a key holds one fact and the latest shows, so these
    # refresh in place instead of stacking. 人话 -- the card is read by people.
    for k, text in (
        (f"{name}_run", f"新模型训练到 {last_step}/{total} 步（{100 * last_step / (total or 1):.0f}%），"
                        f"验证误差 {fresh[-1][1]:g}，还要约 {g('eta', last_step, '{:.0f}')} 小时"),
        (f"{name}_speed", f"速度 {g('tps', last_step)}K tok/s/卡，MFU {g('mfu', last_step)}%，"
                          f"显存 {g('peak', last_step, '{:.1f}')}G/卡"),
        # gnorm on this stack is BIMODAL -- consecutive logged steps read 0.17 and 20666 --
        # so one sample is not a reading of it. Report the recent middle and the worst, or a
        # board quoting whichever point the poll happened to land on alarms people by luck.
        (f"{name}_health", f"梯度 中位 {_med([v for _, v in (series.get('gnorm') or [])[-30:]])}"
                           f"／最大 {_mx([v for _, v in (series.get('gnorm') or [])[-30:]])}"
                           f"（忽高忽低是这个结构的常态），"
                           f"告警 {g('alarms', last, '{:.0f}')} 次，"
                           f"空载专家 {g('zero_load', last, '{:.0f}')} 个，"
                           f"参数更新率 {g('opt_moved', last, '{:.0f}')}%"),
    ):
        cmds.append([AWB, "news", "--key", k, text])
    if dry:
        return "dry-run: " + " ; ".join(" ".join(c[1:]) for c in cmds)
    for c in cmds:
        r = subprocess.run(c, env=env, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise SystemExit(f"awb refused {' '.join(c[1:])}: "
                             f"{(r.stderr or r.stdout).strip()[:300]}")
    sent[key] = fresh[-1][0]
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(sent, fh)
    note = f"{name}: {len(fresh)} point(s) to step {fresh[-1][0]}"

    # Mirror the board to its Feishu topic. awb-lark EDITS one card in place, so this cannot
    # spam a group the way a webhook message per poll would -- which is why no webhook is in
    # this path at all. It only runs for a board that is already linked (lark.json), never
    # creating a topic on its own.
    #
    # The daemon cannot be relied on: the running `awb-lark watch` has cwd ~/.aupai-team, so
    # with no AWB_DIR it reads ./.awb -- the board abandoned on 2026-09-25, not ./awb which
    # is the live one. Pointing the sync explicitly is the fix that does not touch someone
    # else's process.
    if os.path.exists(os.path.join(awb_dir, "lark.json")):
        lark = os.path.join(os.path.dirname(AWB), "awb-lark")
        r = subprocess.run([lark, "sync"], env=env, capture_output=True, text=True, timeout=120)
        note += "; feishu card updated" if r.returncode == 0 else \
            f"; FEISHU SYNC FAILED rc={r.returncode}: {(r.stderr or r.stdout).strip()[:200]}"
    else:
        note += f"; no feishu topic linked for {awb_dir} (awb-lark setup CHAT_ID)"
    return note


#: the card's charts: (title, [(series key, legend)]). Everything else goes in the table.
CARD_CHARTS = [
    ("误差（越低越好）", [("val", "验证"), ("loss", "训练")]),
    ("速度", [("tps", "K tok/s/卡"), ("mfu", "MFU %")]),
    ("显存 GiB/卡", [("peak", "峰值")]),
    ("梯度大小", [("gnorm", "gnorm")]),
]
#: latest-value rows under the charts, for series that do not get a curve
CARD_TABLE = ["eta", "qk_median", "grad_muon", "grad_sinkhorn", "grad_adamw_decay",
              "grad_adamw_nodecay", "opt_moved", "zero_load", "alarms"]
FEISHU_MAX = 30 * 1024


def _thin(points, n):
    """At most n points, evenly spaced, always keeping the first and the last.

    The card is capped at 30 KB by Feishu, so the curves there are thinned. The PAGE keeps
    every point -- the thinning is the card's limit, not the measurement's, and the card says
    so rather than implying it drew everything.
    """
    if len(points) <= n:
        return points
    step = (len(points) - 1) / (n - 1)
    idx = sorted({int(round(i * step)) for i in range(n)} | {len(points) - 1})
    return [points[i] for i in idx]


def build_feishu_card(series, meta, title, page_url, per_chart=50):
    """One card carrying every metric: four charts plus a row per remaining series."""
    last_step = max((p[-1][0] for p in series.values() if p), default=0)
    total = meta.get("total_steps") or 0
    elements = []
    drawn = set()
    for label, keys in CARD_CHARTS:
        values = []
        for key, legend in keys:
            pts = series.get(key) or []
            if not pts:
                continue
            drawn.add(key)
            for s, v in _thin(pts, per_chart):
                values.append({"x": str(s), "y": v, "s": legend})
        if not values:
            continue
        elements.append({"tag": "chart", "aspect_ratio": "16:9", "chart_spec": {
            "type": "line", "title": {"text": label},
            "data": {"values": values}, "xField": "x", "yField": "y", "seriesField": "s",
            "legends": {"visible": True, "position": "bottom"}}})
    rows = []
    for key in CARD_TABLE:
        pts = series.get(key) or []
        if not pts:
            continue
        drawn.add(key)
        label = METRICS.get(key, (key, None))[0]
        rows.append(f"**{label}** {pts[-1][1]:.4g}")
    # Anything parsed but neither drawn nor tabled is NAMED, so a new log field cannot go
    # missing from the card in silence.
    missed = [k for k, v in series.items() if v and k not in drawn]
    if rows:
        elements.append({"tag": "markdown", "content": "　·　".join(rows)})
    if missed:
        elements.append({"tag": "markdown",
                         "content": f"<font color='grey'>未上卡片：{'、'.join(sorted(missed))}"
                                    f"（见网页）</font>"})
    n = sum(len(v) for v in series.values())
    foot = (f"<font color='grey'>{len(series)} 项指标、{n} 个点；卡片上的曲线做了抽稀，"
            f"全部点在网页上</font>")
    if page_url:
        foot += f"\n[看全部曲线]({page_url})"
    elements.append({"tag": "markdown", "content": foot})
    return {"schema": "2.0",
            "config": {"update_multi": True, "width_mode": "fill",
                       "summary": {"content": f"{title} 第 {last_step} 步"}},
            "header": {"title": {"tag": "plain_text",
                                 "content": f"{title}　{last_step}/{total}"
                                            f"（{100 * last_step / (total or 1):.0f}%）"},
                       "template": "blue"},
            "body": {"elements": elements}}


def feishu_card_sync(series, meta, title, page_url, chat, state_path, dry=False):
    """Send the metrics card once, then patch that same message forever.

    One card edited in place, so the group never gets the same numbers twice. The message id
    is remembered per (chat, title); losing it would post a second card rather than silently
    stop, which is why it is written only after a send that returned an id.
    """
    card = build_feishu_card(series, meta, title, page_url)
    body = json.dumps(card, ensure_ascii=False)
    per = 50
    while len(body.encode()) > FEISHU_MAX and per > 6:
        per = max(6, per // 2)
        card = build_feishu_card(series, meta, title, page_url, per_chart=per)
        body = json.dumps(card, ensure_ascii=False)
    size = len(body.encode())
    if size > FEISHU_MAX:
        raise SystemExit(f"card is {size} B, over Feishu's {FEISHU_MAX} B even at {per} "
                         f"points per curve -- drop a chart rather than send a truncated one")
    if dry:
        return f"dry-run: {size} B, {per} pts/curve, {len(card['body']['elements'])} elements"
    try:
        with open(state_path, encoding="utf-8") as fh:
            st = json.load(fh)
    except (OSError, ValueError):
        st = {}
    key = f"{chat}::{title}::card"
    mid = st.get(key)
    if mid:
        r = subprocess.run(["lark-cli", "im", "messages", "patch", "--message-id", mid,
                            "--data", json.dumps({"content": body}), "--as", "bot"],
                           capture_output=True, text=True, timeout=120)
        ok = r.returncode == 0 and '"ok": true' in r.stdout
        if ok:
            return f"card patched ({size} B, {per} pts/curve)"
        # A patch fails for good reasons (card older than 14 days, rate limit). Say so and
        # send a fresh one rather than leaving the group with a frozen card.
        mid = None
    r = subprocess.run(["lark-cli", "im", "+messages-send", "--chat-id", chat,
                        "--msg-type", "interactive", "--content", body, "--as", "bot"],
                       capture_output=True, text=True, timeout=120)
    try:
        mid = json.loads(r.stdout)["data"]["message_id"]
    except (ValueError, KeyError, TypeError):
        raise SystemExit(f"sending the metrics card failed: "
                         f"{(r.stderr or r.stdout).strip()[:300]}") from None
    st[key] = mid
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(st, fh)
    return f"card sent {mid} ({size} B, {per} pts/curve)"


def push(rows, meta, page_url, title, dry=False):
    """Post the latest reading of every metric to Feishu, with the page's link.

    Refuses loudly without a webhook. A feed that silently stops feeding reads exactly like
    a healthy quiet one from the outside, which is the failure this whole session kept finding.
    """
    last_step = max((r["step"] for r in rows), default=0)
    total = meta.get("total_steps") or 0
    lines = [f"**{title}**  step {last_step}/{total}"
             + (f"  ({100 * last_step / total:.1f}%)" if total else "")]
    for r in rows:
        if r["key"] in METRICS:
            lines.append(f"{r['label']}: **{r['value']:g}** {r['arrow']}".rstrip())
    if page_url:
        lines.append(f"[curves]({page_url})")
    text = "\n".join(lines)
    if dry:
        print(text)
        return "dry-run"
    url = webhook()
    if not url:
        raise SystemExit(
            "--push needs a Feishu group webhook and found none. Put it in "
            "$AUPAI_FEISHU_WEBHOOK or ~/.aupai_feishu_webhook (chmod 600). Get it from the "
            "Feishu group: 设置 -> 群机器人 -> 添加机器人 -> 自定义机器人, which gives a "
            "https://open.feishu.cn/open-apis/bot/v2/hook/... URL. Refusing rather than "
            "skipping, so a feed that stopped feeding is visible.")
    body = json.dumps({"msg_type": "interactive", "card": {
        "config": {"wide_screen_mode": True},
        "elements": [{"tag": "markdown", "content": text}]}}).encode()
    req = urllib.request.Request(url, data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        got = json.loads(resp.read().decode())
    if got.get("code") not in (0, None):
        raise SystemExit(f"Feishu refused the message: {got}")
    return "sent"


def selftest():
    """Known answer on a fixture built from the real log's own line shapes, both directions."""
    log = (
        "step 10/38146 0% [main] | loss 10.914 | lr 2.00e-05 (muon 2.00e-05) | gnorm 0.92 "
        "| 0.01B tok | 6K tok/s/gpu | MFU 14% | peak 71.52GiB | ETA 186.7h | s/step 17.6\n"
        "step 10/38146 0% [main] | loss 10.914 | lr 2.00e-05 (muon 2.00e-05) | gnorm 0.92 "
        "| 0.01B tok | 6K tok/s/gpu | MFU 14% | peak 71.52GiB | ETA 186.7h | s/step 17.6\n"
        "step 20/38146 0% [main] | loss 5.0 | lr 4.00e-05 (muon 4.00e-05) | gnorm 453.73 "
        "| 0.02B tok | 10K tok/s/gpu | MFU 24% | peak 71.52GiB | ETA 148.0h | s/step 10.3\n"
        "step 500 val 2.803\n"
        "step 1600 health qk_scale top5 layers.11=0.112(mean 0.07) | median=0.0844 layers=24\n"
        "step 1600 health gradnorm top5 head=0.03 | groups muon=0.0449 sinkhorn=0.0331 "
        "adamw_decay=0.00635 adamw_nodecay=0.0127\n"
        "step 1600 health | moe top1 max nan zero-load 0 | opt moved min 51.3% | alarms 0\n"
    )
    s, meta = parse(log)
    assert meta["total_steps"] == 38146, meta
    # the runlog echoes every progress line; a duplicated step must not become two points
    assert s["loss"] == [[10, 10.914], [20, 5.0]], s["loss"]
    assert s["gnorm"] == [[10, 0.92], [20, 453.73]], s["gnorm"]
    assert s["mfu"] == [[10, 14.0], [20, 24.0]], s["mfu"]
    assert s["val"] == [[500, 2.803]], s["val"]
    assert s["qk_median"] == [[1600, 0.0844]], s["qk_median"]
    assert s["grad_muon"] == [[1600, 0.0449]], s["grad_muon"]
    assert s["opt_moved"] == [[1600, 51.3]], s["opt_moved"]
    assert s["alarms"] == [[1600, 0.0]], s["alarms"]
    # "nan" is a diagnostic that did not compute, NOT a data point: it must not enter a series
    assert "moe_top1" not in s or s["moe_top1"] == [], s.get("moe_top1")
    # a log with no progress line yields no points rather than a plausible-looking empty curve
    s2, _ = parse("nothing here\n")
    assert s2 == {}, s2
    rows = summary(s, meta)
    got = {r["key"]: r for r in rows}
    assert got["loss"]["value"] == 5.0 and "ok" in got["loss"]["arrow"], got["loss"]
    assert "worse" in got["mfu"]["arrow"] or "ok" in got["mfu"]["arrow"], got["mfu"]
    # the AWB watermark: a repeated poll must send nothing, and a new point must send only it
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        st = os.path.join(tmp, "state.json")
        first = awb_push(s, meta, "t_val", 1.9, "/nonexistent-board", st, dry=True)
        assert "step 500" in first, first
        assert "news" in first, first
        # the card must carry more than the one number: a train-loss reference curve, a note
        # under the bar, and one keyed news line per topic
        assert "--ref --step 500 t_val_train" in first, first
        assert "训练误差" in first and "MFU" in first and "显存" in first, first
        for k in ("t_val_run", "t_val_speed", "t_val_health"):
            assert f"--key {k}" in first, (k, first)
        # a series the log did not carry degrades to ? -- never a fabricated 0, and never a crash
        thin = {"val": [[500, 2.8]]}
        lean = awb_push(thin, meta, "t_val", 1.9, "/nonexistent-board", st, dry=True)
        assert "MFU ?%" in lean, lean
        assert "t_val_train" not in lean, "no train loss means no reference curve"
        # dry-run must not write the watermark, or a real first push would send nothing
        assert not os.path.exists(st), "dry-run wrote the watermark"
        with open(st, "w", encoding="utf-8") as fh:
            json.dump({"/nonexistent-board::t_val": 500}, fh)
        again = awb_push(s, meta, "t_val", 1.9, "/nonexistent-board", st, dry=True)
        assert "no new point past step 500" in again, again
        s2 = dict(s, val=s["val"] + [[1000, 2.04]])
        nxt = awb_push(s2, meta, "t_val", 1.9, "/nonexistent-board", st, dry=True)
        assert "step 1000" in nxt and "step 500" not in nxt, nxt
        empty = awb_push({}, meta, "t_val", 1.9, "/nonexistent-board", st, dry=True)
        assert "no val reading yet" in empty, empty
    html = render_html(s, meta, rows, "selftest")
    assert "<title>selftest</title>" in html and "Chart.js" in html
    assert '"loss": [[10, 10.914], [20, 5.0]]' in html.replace("'", '"'), "series must reach the page"
    # --push without a webhook must refuse, not skip
    saved = os.environ.pop("AUPAI_FEISHU_WEBHOOK", None)
    home = os.path.expanduser("~/.aupai_feishu_webhook")
    if not os.path.exists(home):
        try:
            push(rows, meta, None, "selftest")
            raise AssertionError("push must refuse without a webhook")
        except SystemExit as e:
            assert "webhook" in str(e), e
    if saved:
        os.environ["AUPAI_FEISHU_WEBHOOK"] = saved
    print(f"metrics_feed selftest OK: {len(s)} series, duplicate steps collapsed, nan rejected, "
          f"empty log empty, push refuses without a webhook")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v42_gate_1001r")
    ap.add_argument("--log", default=None, help="pod path; default /work/aupai/runs/<run>.log")
    ap.add_argument("--local-log", default=None, help="a log already on this machine")
    ap.add_argument("--out", default=os.path.expanduser("~/aupai-metrics.html"))
    ap.add_argument("--json-out", default=None, help="also write the parsed series")
    ap.add_argument("--page-url", default=os.environ.get("AUPAI_METRICS_PAGE_URL", ""),
                    help="where the rendered page is served; goes in the Feishu message")
    ap.add_argument("--push", action="store_true", help="post to the Feishu group webhook")
    ap.add_argument("--awb", action="store_true", help="also put val on the AWB board")
    ap.add_argument("--card", action="store_true",
                    help="send/patch ONE Feishu card carrying every metric as charts")
    ap.add_argument("--chat", default=os.environ.get("AUPAI_FEISHU_CHAT", ""),
                    help="Feishu chat id (oc_...); defaults to the AWB board's linked group")
    ap.add_argument("--awb-dir", default=os.environ.get(
        "AWB_DIR", os.path.expanduser("~/.aupai-team/awb")))
    ap.add_argument("--awb-metric", default="v42r_val",
                    help="series name; NOT v42_val, which holds the retired stage-2 line")
    ap.add_argument("--awb-target", type=float, default=1.954,
                    help="the stage-2 line's own recorded target, so the two read on one scale")
    ap.add_argument("--awb-state", default=os.path.expanduser("~/.aupai-metrics-awb.json"),
                    help="watermark of the last step pushed, per board and series")
    ap.add_argument("--dry-run", action="store_true", help="print the message, send nothing")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    text = read_log(args.run, args.log, args.local_log)
    series, meta = parse(text)
    if not series:
        raise SystemExit(f"no metric lines found in the log for {args.run} -- refusing to "
                         f"publish an empty page over a good one")
    meta["run"] = args.run
    rows = summary(series, meta)
    html = render_html(series, meta, rows, args.run)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"run": args.run, "meta": meta, "series": series}, fh)
    n = sum(len(v) for v in series.values())
    print(f"{args.run}: {len(series)} series, {n} points -> {args.out}")
    for r in rows:
        if r["key"] in METRICS:
            print(f"  {r['label']:<26} {r['value']:<12g} step {r['step']:<6} n={r['n']}")
    if args.awb:
        print("awb:", awb_push(series, meta, args.awb_metric, args.awb_target,
                               args.awb_dir, args.awb_state, dry=args.dry_run))
    if args.card:
        chat = args.chat
        if not chat:  # the board already names the group it is linked to; do not ask twice
            try:
                with open(os.path.join(args.awb_dir, "lark.json"), encoding="utf-8") as fh:
                    chat = json.load(fh).get("chat", "")
            except (OSError, ValueError):
                chat = ""
        if not chat:
            raise SystemExit("--card needs a Feishu chat id: pass --chat oc_..., set "
                             "$AUPAI_FEISHU_CHAT, or link the board with awb-lark setup")
        print("card:", feishu_card_sync(series, meta, args.run, args.page_url, chat,
                                        args.awb_state, dry=args.dry_run))
    if args.push or args.dry_run:
        print(push(rows, meta, args.page_url, args.run, dry=args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
