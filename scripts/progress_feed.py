#!/usr/bin/env python3
"""Append one line to the controller's progress feed and regenerate the HTML.

The page is a plain file the user opens once; a meta refresh plus a reload
timer rewrite the view from disk, so regenerating the file is the whole
update mechanism -- no server, no fetch (file:// would block it anyway).
A ~/.aupai-status.json card file, if present, renders as a status board
above the timeline.

    progress_feed.py <kind> "<text>"

kind is one of: rule (a decision), find (a measurement or finding),
run (a job's state), warn (something needs attention), note (everything else).
"""

# restartable: appends one JSONL line; the page is rewritten from the store.

import calendar
import html
import json
import os
import subprocess
import sys
import time

STORE = os.path.expanduser("~/.aupai-progress.jsonl")
STATUS = os.path.expanduser("~/.aupai-status.json")
CONTROL = os.path.expanduser("~/.aupai-control.json")
PAGE = os.path.expanduser("~/aupai-progress.html")
PATROL = os.path.expanduser("~/.aupai-patrol")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = 60
SHOWN = 10

SRC_TAG = {"machine": ("测", "#0d9488"), "person": ("说", "#7c5cff")}

KINDS = {
    "rule": ("裁定", "#7c5cff"),
    "find": ("发现", "#0d9488"),
    "run": ("运行", "#2563eb"),
    "warn": ("注意", "#d97706"),
    "note": ("", "#6b7280"),
}

BJ = 8 * 3600


def bj_epoch(epoch):
    return time.strftime("%H:%M", time.gmtime(epoch + BJ))


def bj_str(utc):
    t = _parse_ts(utc)
    if t is None:
        return "?"
    return time.strftime("%m-%d %H:%M", time.gmtime(t + BJ))

CSS = """
:root{--bg:#fff;--fg:#1a1a19;--dim:#6b7280;--line:#e7e5e4;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#131312;--fg:#e8e6e3;--dim:#8b8b86;--line:#2a2a28;--card:#1b1b1a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.55 ui-sans-serif,-apple-system,"PingFang SC","Helvetica Neue",sans-serif;
 padding:20px 18px 40px;max-width:720px}
header{display:flex;align-items:baseline;gap:10px;margin-bottom:14px}
h1{font-size:13px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
 color:var(--dim);margin:0}
.live{font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums;
 display:flex;align-items:center;gap:5px}
.live i{width:7px;height:7px;border-radius:50%;background:#10b981;display:inline-block;
 animation:beat 2s infinite}
@keyframes beat{0%,100%{opacity:1}50%{opacity:.25}}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-top:3px solid;
 border-radius:8px;padding:11px 13px}
.card .ti{font-size:11px;color:var(--dim);letter-spacing:.05em}
.card .bi{font-size:22px;font-weight:700;margin:3px 0 2px;font-variant-numeric:tabular-nums}
.card .su{font-size:12.5px;color:var(--dim);line-height:1.5}
.bar{height:6px;background:var(--line);border-radius:3px;margin-top:8px;overflow:hidden}
.bar i{display:block;height:100%;border-radius:3px}
h2{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
 color:var(--dim);margin:0 0 4px}
ol{list-style:none;margin:0;padding:0}
li{padding:7px 0;border-bottom:1px solid var(--line);display:flex;gap:8px;
 font-size:13px;line-height:1.5}
li:last-child{border-bottom:0}
.t{font-size:11px;color:var(--dim);font-variant-numeric:tabular-nums;
 padding-top:2px;min-width:34px}
.k{font-weight:600;font-size:11px;padding-top:2px;min-width:26px}
.s{font-size:10px;padding-top:3px;min-width:14px;font-weight:700}
.m{flex:1}
details{margin-top:10px}
summary{font-size:12px;color:var(--dim);cursor:pointer}
details li{font-size:12.5px;color:var(--dim)}
table{border-collapse:collapse;font-size:12.5px;margin:2px 0 18px}
th{font-size:10px;color:var(--dim);letter-spacing:.06em;text-align:left;
 font-weight:600;padding:2px 14px 4px 0;border-bottom:1px solid var(--line)}
td{padding:3px 14px 3px 0;border-bottom:1px solid var(--line);
 font-variant-numeric:tabular-nums;white-space:nowrap}
td.n{font-weight:700}
td.zero{color:#d97706;font-weight:700}
td.ex{color:var(--dim)}
tr.stale td{opacity:.5}
.head{background:var(--card);border:1px solid var(--line);border-top:3px solid #2563eb;
 border-radius:8px;padding:11px 13px;margin-bottom:18px}
.head .bi{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}
.head .su{font-size:12.5px;color:var(--dim);margin-top:2px}
.card.stale{opacity:.5}
.card .as{font-size:10px;color:var(--dim);margin-top:6px}
"""


def _row(r):
    label, colour = KINDS.get(r["kind"], KINDS["note"])
    chip = f'<span class=k style="color:{colour}">{label}</span>' if label else ""
    src = r.get("src", "")
    stag = ""
    if src in SRC_TAG:
        t, c = SRC_TAG[src]
        stag = f'<span class=s style="color:{c}" title="{src}">{t}</span>'
    return (
        "<li>"
        f'<span class=t>{html.escape(bj_epoch(calendar.timegm(time.strptime(r["at"], "%H:%M"))))}</span>'
        f"{stag}{chip}"
        f'<span class=m>{html.escape(r["text"])}</span>'
        "</li>"
    )


def _parse_ts(s):
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return calendar.timegm(time.strptime(s, fmt))
        except ValueError:
            continue
    return None


def _age(opened, now):
    t = _parse_ts(opened)
    if t is None:
        return "?"
    mins = int((now - t) / 60)
    if mins < 60:
        return f"{mins}m"
    if mins < 60 * 24:
        return f"{mins // 60}h"
    return f"{mins // (60 * 24)}d"


def control_section():
    if not os.path.exists(CONTROL):
        return ""
    d = json.load(open(CONTROL, encoding="utf-8"))
    out = ["<h2>对照实验</h2>",
           '<div class=head style="border-top-color:#0d9488">',
           f'<div class=bi>{html.escape(d["title"])}</div>']
    for r in d["rows"]:
        pending = "" if r.get("final") else ' <b style="color:#d97706">（待定）</b>'
        note = f" — {html.escape(r['note'])}" if r.get("note") else ""
        out.append(f'<div class=su>{html.escape(r["text"])}{pending}{note} '
                   f'<span style="opacity:.7">{html.escape(r["measured"])} 测</span></div>')
    out.append("</div>")
    return "".join(out)


# 98-3: the seven nodes of docs/standards/roadmap_0903.md (fb ruling 2026-09-03).
# Node text mirrors the doc; state is read live from runs/tasks.jsonl by row id.
ROADMAP_NODES = [
    ("fb-5",  "N1 参数腿启动闸", "fb", "exp 行记 PEAK/STARTUP/code_fp，都在 95.22 GiB 以下才放行", "09-03"),
    ("fb-6",  "N2 参数 vs 数据判决", "fb", "两条腿 domain_loss 差值（nat）带配对 SE，定 30B 形状", "09-03"),
    ("e1-29", "N3 基准 v2", "e1", "只留三个指标：humaneval_bpb、math_bpb、lambada_en，差值带配对 SE", "09-05"),
    ("3b-11", "N4 30B 语料盖章", "3b", "8 个域全进 mix_30b.json，每域过审计、13-gram、去重", "09-06"),
    ("e1-30", "N5 预训练后就绪", "e1", "post_pretrain_plan.md §5 零未决", "09-06"),
    ("de-43", "N6 删除轮", "de", "每个 concern 只留唯一产物，reachability 和未关任务数进收尾行", "09-05"),
    ("e1-31", "N7 中层循环", "e1", "只做推理 A/B（中间 4 层走两次），赢了才进 SFT 对照", "A 09-03 / B 09-04"),
]


def roadmap_section():
    tasks_p = os.path.join(REPO, "runs", "tasks.jsonl")
    if not os.path.exists(tasks_p):
        return ""
    state = {}
    for ln in open(tasks_p, encoding="utf-8"):
        if ln.strip():
            t = json.loads(ln)
            state[t["id"]] = t.get("state", "?")
    out = ["<h2>路线图（09-03 裁定）</h2>",
           '<div class=head style="border-top-color:#2563eb">']
    for tid, node, owner, exit_no, date in ROADMAP_NODES:
        st = state.get(tid, "无任务行")
        mark = {"done": "完成", "open": "未完成", "blocked": "卡住"}.get(st, st)
        colour = {"done": "#16a34a", "open": "#d97706", "blocked": "#dc2626"}.get(st, "#6b7280")
        out.append(f'<div class=su>{html.escape(node)}（{html.escape(owner)}，{html.escape(date)}）'
                   f' — <b style="color:{colour}">{mark}</b>：{html.escape(exit_no)}</div>')
    out.append("</div>")
    return "".join(out)


# 98-4: per-member liveness from board.liveness (de's cec7a077). One line per member,
# stalled ones first; the stalled tag must agree with harness check_peer_stalled.
PEER_STALL_MIN = 120


def _age_cn(mins):
    if mins is None:
        return "—"
    if mins < 60:
        return f"{mins} 分钟前"
    return f"{mins // 60} 小时前"


def liveness_section():
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    try:
        from board import liveness
    except Exception:
        return ""
    live = liveness(REPO)
    if not live:
        return ""

    def ages_of(d):
        return [x for x in (d["commit_min"], d["ledger_min"]) if x is not None]

    def stalled(d):
        ages = ages_of(d)
        return bool(d["open_tasks"]) and bool(ages) and min(ages) >= PEER_STALL_MIN

    rows = sorted(live.items(),
                  key=lambda kv: (not stalled(kv[1]), -min(ages_of(kv[1])) if ages_of(kv[1]) else 1))
    out = ["<h2>每人动静</h2>", '<div class=head style="border-top-color:#d97706">']
    for name, d in rows:
        tag = ' <b style="color:#dc2626">两小时没动静</b>' if stalled(d) else ""
        out.append(f'<div class=su><b>{html.escape(name)}</b>：交代码 {_age_cn(d["commit_min"])}、'
                   f'记账 {_age_cn(d["ledger_min"])}，手上 {d["open_tasks"]} 件事{tag}</div>')
    out.append("</div>")
    return "".join(out)


# 98-5: one friction line under the per-member section (6e request, main fc4e7efe).
# Reads runs/friction.jsonl directly: rows, distinct causes, top (kind, cause) pair.
def friction_section():
    fp = os.path.join(REPO, "runs", "friction.jsonl")
    if not os.path.exists(fp):
        return ""
    rows = [json.loads(ln) for ln in open(fp, encoding="utf-8") if ln.strip()]
    if not rows:
        return ""
    groups = {}
    for r in rows:
        key = (r.get("kind", "?"), r.get("cause", ""))
        groups[key] = groups.get(key, 0) + 1
    (kind, cause), n = max(groups.items(), key=lambda kv: kv[1])
    kind_cn = {"merge": "合并冲突", "hook": "钩子", "check": "检查", "launch": "启动"}.get(kind, kind)
    short = cause.split(",")[0][:60]
    ncauses = len({r.get("cause", "") for r in rows})
    return ("<h2>摩擦记录</h2>"
            '<div class=head style="border-top-color:#d97706">'
            f'<div class=su>{len(rows)} 条、{ncauses} 种原因；最多的是{kind_cn}（{n} 条）：{html.escape(short)}'
            ' <a href="file:///Users/bytedance/code/aupai/docs/standards/friction_review.md">复盘全文</a></div>'
            "</div>")


def queue_section():
    tasks_p = os.path.join(REPO, "runs", "tasks.jsonl")
    roster_p = os.path.join(REPO, "runs", "roster.json")
    if not (os.path.exists(tasks_p) and os.path.exists(roster_p)):
        return ""
    members = [m["name"] for m in json.load(open(roster_p, encoding="utf-8"))["members"]]
    exempt = {"fb", "98"}
    tasks = [json.loads(ln) for ln in open(tasks_p, encoding="utf-8") if ln.strip()]
    latest = {}
    for t in tasks:
        latest[t["id"]] = t
    now = time.time()
    stat = {m: {"open": 0, "oldest": None, "closed": None} for m in members}
    for t in latest.values():
        s = stat.get(t.get("owner"))
        if s is None:
            continue
        if t.get("state") == "open":
            if not (t.get("blocked_on") or "").strip():
                s["open"] += 1
            if s["oldest"] is None or t.get("opened", "") < s["oldest"]:
                s["oldest"] = t.get("opened")
        elif t.get("state") == "done" and t.get("closed"):
            if s["closed"] is None or t["closed"] > s["closed"]:
                s["closed"] = t["closed"]
    out = ["<h2>每人队列</h2><table>",
           "<tr><th></th><th>open 未阻塞</th><th>最老 open</th><th>最近关闭</th></tr>"]
    for m in members:
        s = stat[m]
        cls = "ex" if m in exempt else ("zero" if s["open"] == 0 else "n")
        open_cell = "—" if m in exempt else str(s["open"])
        oldest = _age(s["oldest"], now) if s["oldest"] else "—"
        closed = bj_str(s["closed"]) if s["closed"] else "—"
        tag = " 豁免" if m in exempt else ""
        out.append(f'<tr><td>{html.escape(m)}{tag}</td><td class={cls}>{open_cell}</td>'
                   f'<td>{oldest}</td><td>{closed}</td></tr>')
    out.append("</table>")
    names = {m: [] for m in members}
    for t in latest.values():
        if t.get("state") == "open" and not (t.get("blocked_on") or "").strip() and t.get("owner") in names:
            first = t.get("task", "")
            if first.upper().startswith("LONG LINE:"):
                first = first.split(":", 1)[1].strip()
            cut = next((i for i, ch in enumerate(first[:40]) if ch in ":;,("), 0)
            if cut:
                first = first[:cut]
            elif len(first) > 40:
                first = first[:40].rsplit(" ", 1)[0]
            names[t["owner"]].append(first.strip())
    listed = [f"<tr><td>{html.escape(m)}</td><td>{html.escape('；'.join(names[m]))}</td></tr>"
              for m in members if names[m]]
    if listed:
        out.append("<table><tr><th></th><th>open 任务</th></tr>")
        out.extend(listed)
        out.append("</table>")
    return "".join(out)


def render(rows):
    now = time.time()
    now_bj = time.strftime("%H:%M:%S", time.gmtime(now + BJ))
    now_utc = time.strftime("%H:%M:%S", time.gmtime(now))
    patrol = ""
    if os.path.exists(PATROL):
        with open(PATROL, encoding="utf-8") as fh:
            stamp = fh.read().strip()
            patrol = f' · 最后巡检 {html.escape(bj_epoch(calendar.timegm(time.strptime(stamp, "%H:%M"))))} (+0800)'
    parts = [
        "<!doctype html><html lang=zh><head><meta charset=utf-8>",
        '<meta http-equiv="refresh" content="15">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>aupai 进展</title><style>", CSS, "</style></head><body>",
        "<header><h1>aupai 进展</h1>",
        f'<span class=live><i></i>北京时间 {now_bj}（+0800）· UTC {now_utc}{patrol} · 每 15 秒自动刷新</span></header>',
    ]
    if os.path.exists(STATUS):
        with open(STATUS, encoding="utf-8") as fh:
            status = json.load(fh)
        h = status.get("headline")
        if h:
            if h.get("paused"):
                big = (f'{html.escape(h["run"])}：停窗口中，最新 ckpt '
                       f'{html.escape(h["ckpt"])}（step {h["step"]}/{h["total"]}）')
            else:
                pct = round(100 * h["step"] / h["total"])
                big = (f'{html.escape(h["run"])}：step {h["step"]}/{h["total"]}（{pct}%）'
                       f' · loss {h["loss"]} · {html.escape(h["tps"])} tok/s/gpu'
                       f' · ETA {html.escape(h["eta"])} · 最新 ckpt {html.escape(h["ckpt"])}')
            parts.append('<div class=head>')
            parts.append(f'<div class=bi>{big}</div>')
            if h.get("stop"):
                parts.append(f'<div class=su style="color:#7c5cff">{html.escape(h["stop"])}</div>')
            parts.append(f'<div class=su>截至 {bj_str(h["asof"])} (+0800)，来自 pod 训练日志</div></div>')
        cards = status.get("cards", [])
        if cards:
            parts.append('<div class=cards>')
            for c in cards:
                _, colour = KINDS.get(c.get("tone", ""), KINDS["note"])
                stale = ""
                asof_html = ""
                if c.get("asof"):
                    age_h = (now - calendar.timegm(time.strptime(c["asof"], "%Y-%m-%d %H:%M"))) / 3600
                    if age_h > 2:
                        stale = " stale"
                    asof_html = f'<div class=as>截至 {bj_str(c["asof"])} (+0800)</div>'
                parts.append(f'<div class="card{stale}" style="border-top-color:{colour}">')
                parts.append(f'<div class=ti>{html.escape(c["title"])}</div>')
                parts.append(f'<div class=bi>{html.escape(c["big"])}</div>')
                if c.get("bar") is not None:
                    pct = round(100 * float(c["bar"]))
                    parts.append(f'<div class=bar><i style="width:{pct}%;background:{colour}"></i></div>')
                parts.append(f'<div class=su>{html.escape(c["sub"])}</div>{asof_html}</div>')
            parts.append('</div>')
    parts.append(control_section())
    parts.append(roadmap_section())
    parts.append(liveness_section())
    parts.append(friction_section())
    parts.append(queue_section())
    rest = rows
    parts.append("<h2>时间线</h2><ol>")
    parts.extend(_row(r) for r in rest[:SHOWN])
    parts.append("</ol>")
    older = rest[SHOWN:]
    if older:
        parts.append(f'<details><summary>更早的 {len(older)} 条</summary><ol>')
        parts.extend(_row(r) for r in older)
        parts.append("</ol></details>")
    parts.append("<script>setTimeout(function(){location.reload()},15000)</script>")
    parts.append("</body></html>")
    return "".join(parts)


def main():
    if len(sys.argv) >= 3:
        src = sys.argv[-1] if len(sys.argv) >= 4 and sys.argv[-1] in SRC_TAG else None
        text_args = sys.argv[2:-1] if src else sys.argv[2:]
        row = {
            "at": time.strftime("%H:%M", time.gmtime()),
            "kind": sys.argv[1],
            "text": " ".join(text_args),
        }
        if src:
            row["src"] = src
        with open(STORE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = []
    if os.path.exists(STORE):
        with open(STORE, encoding="utf-8") as fh:
            rows = [json.loads(ln) for ln in fh if ln.strip()]
    with open(PAGE, "w", encoding="utf-8") as fh:
        fh.write(render(rows[::-1][:KEEP]))
    audit_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "audit_0904")
    if os.path.isdir(audit_dir):
        subprocess.run(
            [sys.executable, os.path.join(audit_dir, "..", "..", "scripts", "audit_render.py"), "--compose"],
            cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
            capture_output=True,
        )
    print(PAGE)


if __name__ == "__main__":
    main()
