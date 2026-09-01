#!/usr/bin/env python3
"""Append one line to the controller's progress feed and regenerate the HTML.

The page is a plain file the user opens once; it carries a meta refresh, so
rewriting the file on disk is the whole update mechanism -- no server, no JS,
no fetch (file:// would block it anyway).

    progress_feed.py <kind> "<text>"

kind is one of: rule (a decision), find (a measurement or finding),
run (a job's state), warn (something needs attention), note (everything else).
"""

# restartable: the append is one line to a JSONL and the render is a full
# rewrite from that store, so an interrupt loses at most the entry being
# written and the next call reproduces the page exactly.

import html
import json
import os
import sys
import time

STORE = os.path.expanduser("~/.aupai-progress.jsonl")
PAGE = os.path.expanduser("~/aupai-progress.html")
KEEP = 60  # entries rendered; the store keeps everything

KINDS = {
    "rule": ("裁定", "#7c5cff"),
    "find": ("发现", "#0d9488"),
    "run": ("运行", "#2563eb"),
    "warn": ("注意", "#d97706"),
    "note": ("", "#6b7280"),
}

CSS = """
:root{--bg:#fbfbfa;--fg:#1a1a19;--dim:#6b7280;--line:#e7e5e4;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#131312;--fg:#e8e6e3;--dim:#8b8b86;--line:#2a2a28;--card:#1b1b1a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.55 ui-sans-serif,-apple-system,"PingFang SC","Helvetica Neue",sans-serif;
 padding:20px 18px 40px;max-width:720px}
h1{font-size:13px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
 color:var(--dim);margin:0 0 2px}
.sub{font-size:12px;color:var(--dim);margin:0 0 18px;font-variant-numeric:tabular-nums}
ol{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px}
li{background:var(--card);border:1px solid var(--line);border-left-width:3px;
 border-radius:5px;padding:9px 12px}
.t{font-size:11px;color:var(--dim);font-variant-numeric:tabular-nums;
 letter-spacing:.03em;display:flex;gap:8px;margin-bottom:3px}
.k{font-weight:600}
.m{white-space:pre-wrap;word-break:break-word}
"""


def render(rows):
    now = time.strftime("%H:%M:%S")
    parts = [
        "<!doctype html><html lang=zh><head><meta charset=utf-8>",
        '<meta http-equiv="refresh" content="15">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>aupai 进展</title><style>", CSS, "</style></head><body>",
        "<h1>aupai 进展</h1>",
        f'<p class=sub>最新在上 · 每 15 秒自动刷新 · 页面生成于 {now} · {len(rows)} 条</p>',
        "<ol>",
    ]
    for r in rows:
        label, colour = KINDS.get(r["kind"], KINDS["note"])
        parts.append(f'<li style="border-left-color:{colour}">')
        parts.append(f'<div class=t><span>{html.escape(r["at"])}</span>')
        if label:
            parts.append(f'<span class=k style="color:{colour}">{label}</span>')
        parts.append("</div>")
        parts.append(f'<div class=m>{html.escape(r["text"])}</div></li>')
    parts.append("</ol></body></html>")
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
