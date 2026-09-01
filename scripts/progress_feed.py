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

import html
import json
import os
import sys
import time

STORE = os.path.expanduser("~/.aupai-progress.jsonl")
STATUS = os.path.expanduser("~/.aupai-status.json")
PAGE = os.path.expanduser("~/aupai-progress.html")
KEEP = 60
SHOWN = 10

KINDS = {
    "rule": ("裁定", "#7c5cff"),
    "find": ("发现", "#0d9488"),
    "run": ("运行", "#2563eb"),
    "warn": ("注意", "#d97706"),
    "note": ("", "#6b7280"),
}

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
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-top:3px solid;
 border-radius:8px;padding:11px 13px}
.card .ti{font-size:11px;color:var(--dim);letter-spacing:.05em}
.card .bi{font-size:20px;font-weight:700;margin:3px 0 2px;font-variant-numeric:tabular-nums}
.card .su{font-size:12px;color:var(--dim);line-height:1.45}
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
.m{flex:1}
details{margin-top:10px}
summary{font-size:12px;color:var(--dim);cursor:pointer}
details li{font-size:12.5px;color:var(--dim)}
"""


def _row(r):
    label, colour = KINDS.get(r["kind"], KINDS["note"])
    chip = f'<span class=k style="color:{colour}">{label}</span>' if label else ""
    return (
        "<li>"
        f'<span class=t>{html.escape(r["at"])}</span>'
        f"{chip}"
        f'<span class=m>{html.escape(r["text"])}</span>'
        "</li>"
    )


def render(rows):
    now = time.strftime("%H:%M:%S")
    parts = [
        "<!doctype html><html lang=zh><head><meta charset=utf-8>",
        '<meta http-equiv="refresh" content="15">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>aupai 进展</title><style>", CSS, "</style></head><body>",
        "<header><h1>aupai 进展</h1>",
        f'<span class=live><i></i>每 15 秒自动刷新 · 生成于 {now}</span></header>',
    ]
    if os.path.exists(STATUS):
        with open(STATUS, encoding="utf-8") as fh:
            cards = json.load(fh).get("cards", [])
        parts.append('<div class=cards>')
        for c in cards:
            _, colour = KINDS.get(c.get("tone", ""), KINDS["note"])
            parts.append(f'<div class=card style="border-top-color:{colour}">')
            parts.append(f'<div class=ti>{html.escape(c["title"])}</div>')
            parts.append(f'<div class=bi>{html.escape(c["big"])}</div>')
            if c.get("bar") is not None:
                pct = round(100 * float(c["bar"]))
                parts.append(f'<div class=bar><i style="width:{pct}%;background:{colour}"></i></div>')
            parts.append(f'<div class=su>{html.escape(c["sub"])}</div></div>')
        parts.append('</div>')
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
        with open(STORE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "at": time.strftime("%H:%M"),
                "kind": sys.argv[1],
                "text": " ".join(sys.argv[2:]),
            }, ensure_ascii=False) + "\n")

    rows = []
    if os.path.exists(STORE):
        with open(STORE, encoding="utf-8") as fh:
            rows = [json.loads(ln) for ln in fh if ln.strip()]
    with open(PAGE, "w", encoding="utf-8") as fh:
        fh.write(render(rows[::-1][:KEEP]))
    print(PAGE)


if __name__ == "__main__":
    main()
