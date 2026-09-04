#!/usr/bin/env python3
"""Audit 0904 renderer.

Reads every runs/audit_0904/<area>.md report, regenerates
runs/audit_0904/findings.jsonl (never hand-edited), and composes the audit
section on top of ~/aupai-progress.html (old page content moves below).

A findings table must keep the charter column order: id, severity, claim,
evidence, contradiction. A shuffled order is a parse error, not a guess.
A row the parser cannot read lands on the page as "unparsed", never dropped.
"""
import argparse
import calendar
import json
import os
import re
import subprocess
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_DIR = os.path.join(ROOT, "runs", "audit_0904")
FINDINGS_JSONL = os.path.join(AUDIT_DIR, "findings.jsonl")
RULINGS_JSONL = os.path.join(AUDIT_DIR, "rulings.jsonl")
PAGE = os.path.expanduser("~/aupai-progress.html")

AREAS = {
    "model_code": ("model and training code", "b0", "tilerl"),
    "eval_heldout": ("evaluation and held-out", "e1", "3b"),
    "instruments_ledgers": ("instruments and ledgers", "de", "44"),
    "corpus_data": ("corpus and data", "3b", "b0"),
    "pod_repo": ("pod and repository state", "tilerl", "b0"),
    "facts_docs": ("facts and documents", "44", "de"),
    "user_facing": ("user-facing statements", "98", "e1"),
}
AREA_ORDER = list(AREAS)

# A report's filename may differ from the charter's area stem. b0's model_code
# report landed as model_training.md.
STEM_ALIAS = {"model_training": "model_code", "pod_repo_state": "pod_repo"}

SEV_ORDER = {"S1": 0, "S2": 1, "S3": 2, "ND": 3}
ID_RE = re.compile(r"^[A-Z]{1,3}-?\d+$")

HEADER_ALIASES = [
    re.compile(r"^id$", re.I),
    re.compile(r"^(sev|severity)$", re.I),
    re.compile(r"^claim", re.I),
    re.compile(r"^evidence", re.I),
    re.compile(r"^(contradicts?|what contradicts.*)$", re.I),
]

# Plain-word sentences for the "what this means" block, keyed by finding id.
# Every sentence carries its numbers' basis. Fallback below for unknown ids.
MEANING = {
    "E1": "[E1] 评测台账里 60 行域损失没有一行标注走的是哪条前向路径，而修这个标注的提交只改了命令行工具、没改写台账的脚本——已发表的对比可能混着两条路径。",
    "E3": "[E3] 「同算力下大模型更好」这个结论（-0.0108 nat）全部测在有缺陷的路径上，缺陷本身是差值的 7.6 倍；结论方向对不对，要等两条腿都在 doc_cu 路径上重测，这是审计后第一件排队的活。",
    "UF-1": "[UF-1] 进展页 2026-09-01 那条「94.4% vs 0.3%」的对比在事实库查无出处，项目规范 09-03 已点名移除，页面一直没改。",
    "UF-2": "[UF-2] 进展页 math_owm 扫描的两个命中密度（15614/GB、18106/GB）来自已作废的旧扫描输出，3b 已裁定以日志为准，页面没改。",
    "MT-1": "[MT-1] 事实库里那条「服务并发 KV 池只有申报预算一半、修复未做」的记录描述的是已经不存在的代码：b0 在另一个仓库打开现状，现在的配置是预算的 8 倍，修复三天前就做了。",
}


# restartable: regenerates findings.jsonl and the page from the seven reports; an
# interrupt costs one re-run, both outputs are overwrite-only and order-independent.
def _split_row(line):
    """Split a markdown table row on |, ignoring pipes inside inline code.

    Evidence cells carry shell snippets (`grep -c 'a\\|b'`, pipelines); a plain
    str.split("|") reads their pipes as column boundaries (E8 went unparsed that way).
    """
    cells, buf, in_code = [], [], False
    for ch in line.strip():
        if ch == "`":
            in_code = not in_code
            buf.append(ch)
        elif ch == "|" and not in_code:
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    cells.append("".join(buf).strip())
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def read_report(path):
    """Return (meta, findings, unparsed, pair_checks, blind_spots, open_qs).

    Raises ValueError on a findings table whose columns are out of order.
    """
    text = open(path, encoding="utf-8").read()
    stem = os.path.basename(path)[:-3]
    area, owner, pair = AREAS.get(stem, (stem, "?", "?"))

    # frontmatter overrides
    fm = re.match(r"^---\n(.*?)\n---", text, re.S)
    if fm:
        for line in fm.group(1).splitlines():
            k, _, v = line.partition(":")
            v = v.strip()
            if k.strip() == "area" and v:
                area = v
            elif k.strip() == "owner" and v:
                owner = v
            elif k.strip() == "pair" and v:
                pair = v

    findings, unparsed = [], []
    in_findings = False
    header_seen = False
    table_done = False
    for line in text.splitlines():
        if line.startswith("## "):
            in_findings = "finding" in line.lower()
            header_seen = False
            table_done = False
            continue
        if not in_findings or table_done:
            continue
        if not line.startswith("|"):
            # The findings table ends at its first non-table line; a later table
            # in the same section (b0's §4a pod-source list) is not findings and
            # is listed nowhere, per controller ruling 2026-09-04.
            if header_seen and findings:
                table_done = True
            continue
        cells = _split_row(line)
        if not header_seen:
            if len(cells) != 5 or not cells[0].lower() == "id" or not all(
                pat.match(c) for pat, c in zip(HEADER_ALIASES, cells)
            ):
                raise ValueError(
                    f"{path}: findings table header missing or out of order: {cells}"
                )
            header_seen = True
            continue
        if set(cells[0]) <= set("-: "):  # separator row
            continue
        if len(cells) != 5:
            unparsed.append({"raw": line, "reason": "not 5 columns"})
            continue
        fid, sev, claim, evidence, contra = cells
        if sev in ("—", "–", "-"):
            sev = "ND"
        if not ID_RE.match(fid) or sev not in SEV_ORDER:
            unparsed.append({"raw": line, "reason": "bad id or severity"})
            continue
        findings.append(
            {
                "id": fid,
                "area": area,
                "owner": owner,
                "severity": sev,
                "claim": claim,
                "evidence": evidence,
                "contradiction": contra,
            }
        )

    pair_checks = {}
    m = re.search(r"##\s*Pair check(.*?)(?=\n## |\Z)", text, re.S | re.I)
    if m:
        for pm in re.finditer(
            r"\b([A-Z]{1,3}-?\d+)\b.*?\b(held|holds?|fails?|failed)\b",
            m.group(1),
            re.I,
        ):
            verdict = "held" if pm.group(2).lower().startswith("h") else "failed"
            pair_checks[pm.group(1)] = verdict

    blind_spots, open_qs = [], []
    section = None
    for line in text.splitlines():
        if line.startswith("## "):
            low = line.lower()
            section = "blind" if "blind" in low else "open" if "open" in low else None
            continue
        if section == "blind" and line.lstrip().startswith(("- ", "* ")):
            blind_spots.append(line.lstrip()[2:].strip())
        if section == "open" and re.match(r"^\d+\.\s", line):
            open_qs.append(re.sub(r"^\d+\.\s*", "", line).strip())

    return {
        "area": area,
        "owner": owner,
        "pair": pair,
        "stem": stem,
        "findings": findings,
        "unparsed": unparsed,
        "pair_checks": pair_checks,
        "blind_spots": blind_spots,
        "open_qs": open_qs,
    }


def load_rulings():
    rulings = {}
    if os.path.exists(RULINGS_JSONL):
        for line in open(RULINGS_JSONL, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                rulings[r["id"]] = r
    return rulings


def collect():
    reports, all_findings, all_unparsed = [], [], []
    for stem in AREA_ORDER:
        path = os.path.join(AUDIT_DIR, stem + ".md")
        if not os.path.exists(path):
            alt = next((s for s, t in STEM_ALIAS.items() if t == stem), None)
            if alt:
                path = os.path.join(AUDIT_DIR, alt + ".md")
        if not os.path.exists(path):
            reports.append({"stem": stem, "landed": False, **_area_meta(stem)})
            continue
        rep = read_report(path)
        rep["landed"] = True
        rep["mtime"] = time.strftime("%Y-%m-%d %H:%M", time.gmtime(os.path.getmtime(path)))
        reports.append(rep)
        all_findings.extend(rep["findings"])
        for u in rep["unparsed"]:
            u["area"] = rep["area"]
        all_unparsed.extend(rep["unparsed"])

    rulings = load_rulings()
    for f in all_findings:
        r = rulings.get(f["id"], {})
        f["pair_check"] = f.get("pair_check") or next(
            (
                rep["pair_checks"].get(f["id"])
                for rep in reports
                if rep.get("pair_checks") and f["id"] in rep["pair_checks"]
            ),
            "pending",
        )
        f["ruling"] = r.get("ruling", "pending")
        f["ruling_by"] = r.get("by", "")
        f["ruling_note"] = r.get("note", "")
        f["fix_state"] = r.get("fix_state", "frozen-until-audit-close")
    all_findings.sort(key=lambda f: (SEV_ORDER[f["severity"]], f["id"]))
    return reports, all_findings, all_unparsed


def _area_meta(stem):
    area, owner, pair = AREAS[stem]
    return {"area": area, "owner": owner, "pair": pair}


def write_findings_jsonl(findings, unparsed):
    fields = [
        "id",
        "area",
        "owner",
        "severity",
        "claim",
        "evidence",
        "contradiction",
        "pair_check",
        "ruling",
        "fix_state",
    ]
    with open(FINDINGS_JSONL, "w", encoding="utf-8") as fh:
        for f in findings:
            fh.write(json.dumps({k: f[k] for k in fields}, ensure_ascii=False) + "\n")
        for u in unparsed:
            fh.write(
                json.dumps(
                    {
                        "id": "UNPARSED",
                        "area": u["area"],
                        "severity": "?",
                        "claim": u["raw"],
                        "reason": u["reason"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def chip(text, cls):
    return f'<span class="chip {cls}">{text}</span>'


def evidence_html(ev):
    """Repo paths become file:// links when they exist; rest stays code."""
    m = re.search(r"`?([a-z_/]+\.(py|md|json|jsonl|sh|txt)(?::\d+(-\d+)?)?)`?", ev)
    if not m:
        return f"<code>{_esc(ev)}</code>"
    target = m.group(1).split(":")[0]
    if os.path.exists(os.path.join(ROOT, target)):
        link = f"file://{os.path.join(ROOT, m.group(1))}"
        return f'<code><a href="{link}">{_esc(m.group(1))}</a></code>'
    return f"<code>{_esc(ev)}</code>"


def _esc(s):
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def running_jobs():
    """Ledger rows still marked running and started within 24h.

    The ledger carries stale running rows from before the no_ghost_running
    check; an untimed name list would print 45 dead rows. The 24h window is
    the instrument's resolution, stated on the page.
    """
    exp = os.path.join(ROOT, "runs", "experiments.jsonl")
    names, cutoff = [], time.time() - 24 * 3600
    if os.path.exists(exp):
        for line in open(exp, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") != "running":
                continue
            try:
                t = calendar.timegm(
                    time.strptime(r.get("started", ""), "%Y-%m-%d %H:%M")
                )
            except ValueError:
                continue
            if t >= cutoff and r.get("name") not in names:
                names.append(r["name"])
    return names


def _md_inline(s):
    s = _esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", s)


def md_to_html(md):
    # minimal subset: headings, dash lists, pipe tables, paragraphs
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("### "):
            out.append(f"<h4>{_md_inline(ln[4:])}</h4>")
            i += 1
            continue
        if ln.startswith("## "):
            out.append(f"<h3>{_md_inline(ln[3:])}</h3>")
            i += 1
            continue
        if ln.startswith("# "):
            out.append(f"<h2>{_md_inline(ln[2:])}</h2>")
            i += 1
            continue
        if ln.lstrip().startswith("|"):
            tbl = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in _split_row(lines[i].strip().strip("|"))]
                tbl.append(cells)
                i += 1
            tbl = [r for r in tbl if not all(set(c) <= set("-: ") for c in r)]
            if tbl:
                head = "".join(f"<th>{_md_inline(c)}</th>" for c in tbl[0])
                body = "".join(
                    "<tr>" + "".join(f"<td>{_md_inline(c)}</td>" for c in r) + "</tr>"
                    for r in tbl[1:]
                )
                out.append(f"<table class='findings'><tr>{head}</tr>{body}</table>")
            continue
        if ln.lstrip().startswith(("- ", "* ")):
            items = []
            while i < len(lines) and lines[i].lstrip().startswith(("- ", "* ")):
                items.append(f"<li>{_md_inline(lines[i].lstrip()[2:])}</li>")
                i += 1
            out.append("<ul class='meaning'>" + "".join(items) + "</ul>")
            continue
        para = [ln]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "- ", "* ", "|")):
            para.append(lines[i])
            i += 1
        out.append(f"<p>{_md_inline(' '.join(para))}</p>")
    return "".join(out)


def state_block(root):
    p = os.path.join(root, "docs", "standards", "state_0904.md")
    if not os.path.exists(p):
        return ""
    return (
        '<section class="audit"><h1>现状（state_0904）</h1>'
        + md_to_html(open(p, encoding="utf-8").read())
        + "</section>"
    )


def cleanup_block(root):
    p = os.path.join(root, "runs", "audit_0904", "cleanup.jsonl")
    if not os.path.exists(p):
        return ""
    trs = []
    for ln in open(p, encoding="utf-8"):
        if not ln.strip():
            continue
        r = json.loads(ln)
        st = r.get("state") or "open"
        cls = "ok" if st == "done" else "pend"
        trs.append(
            f"<tr><td>{_esc(r.get('id', ''))}</td><td>{_esc(r.get('owner', ''))}</td>"
            f"<td class='claim'>{_esc(r.get('item') or r.get('action', ''))}</td>"
            f"<td>{chip(st, cls)}</td><td>{_esc(r.get('evidence', ''))}</td></tr>"
        )
    return (
        '<section class="audit"><h1>清理清单（cleanup_0904）</h1>'
        "<table class='findings'><tr><th>id</th><th>owner</th><th>项</th>"
        f"<th>状态</th><th>证据</th></tr>{''.join(trs)}</table></section>"
    )


def render_page(reports, findings, unparsed, old_html):
    sha = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )
    now = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    sev_counts = {s: sum(1 for f in findings if f["severity"] == s) for s in SEV_ORDER}
    landed = sum(1 for r in reports if r.get("landed"))
    jobs = running_jobs()

    # headline strip
    strip = (
        f'<div class="strip">{chip("S1 " + str(sev_counts["S1"]), "s1")}'
        f'{chip("S2 " + str(sev_counts["S2"]), "s2")}'
        f'{chip("S3 " + str(sev_counts["S3"]), "s3")}'
        f'{chip("ND " + str(sev_counts["ND"]), "nd")}'
        f'<span class="strip-item">报告 {landed}/7</span>'
        f'<span class="strip-item">ledger 记为在跑（24h 内启动，可能含未关闭的死行）：{_esc("、".join(jobs)) if jobs else "无"}</span>'
        f'<span class="strip-item">删档 12:03Z（charter aed940e8）</span>'
        f'<span class="strip-item">生成 {now} · main {sha}</span></div>'
    )

    # area cards
    cards = []
    for rep in reports:
        if not rep.get("landed"):
            body = (
                f'<div class="kv">owner {_esc(rep["owner"])} · pair {_esc(rep["pair"])}</div>'
                f'<div class="kv warn">未落地</div>'
            )
        else:
            counts = {
                s: sum(1 for f in rep["findings"] if f["severity"] == s)
                for s in SEV_ORDER
            }
            ups = "".join(f"<li>{_esc(b)}</li>" for b in rep["blind_spots"]) or "<li>（未列）</li>"
            body = (
                f'<div class="kv">owner {_esc(rep["owner"])} · pair {_esc(rep["pair"])} · '
                f'落地 {_esc(rep["mtime"])} UTC</div>'
                f'<div class="kv">{chip("S1 " + str(counts["S1"]), "s1")} '
                f'{chip("S2 " + str(counts["S2"]), "s2")} '
                f'{chip("S3 " + str(counts["S3"]), "s3")}</div>'
                f'<details><summary>未覆盖（{len(rep["blind_spots"])}）</summary><ul class="blind">{ups}</ul></details>'
            )
        cards.append(
            f'<div class="card"><h3>{_esc(rep["area"])}</h3>{body}</div>'
        )

    # findings table
    rows_html = []
    for f in findings:
        pc = f["pair_check"]
        pc_cls = "ok" if pc == "held" else "bad" if pc == "failed" else "pend"
        sev_label = "无缺陷/撤回" if f["severity"] == "ND" else f["severity"]
        sev_cls = "nd" if f["severity"] == "ND" else f["severity"].lower()
        ruling = f["ruling"]
        r_cls = "ok" if ruling == "accepted" else "bad" if ruling == "returned" else "pend"
        r_note = f.get("ruling_note", "")
        note_html = f"<div class='rnote'>{_esc(r_note)}</div>" if r_note else ""
        rows_html.append(
            f"<tr><td>{chip(sev_label, sev_cls)}</td>"
            f"<td>{_esc(f['area'])}</td>"
            f"<td class='claim'>{_esc(f['claim'])}</td>"
            f"<td>{evidence_html(f['evidence'])}</td>"
            f"<td>{chip(pc, pc_cls)}</td>"
            f"<td class='ruling'>{chip(ruling, r_cls)}{note_html}</td></tr>"
        )
    for u in unparsed:
        rows_html.append(
            f"<tr><td>{chip('?', 'pend')}</td><td>{_esc(u['area'])}</td>"
            f"<td class='claim'>unparsed: {_esc(u['raw'][:200])}</td>"
            f"<td>{_esc(u['reason'])}</td><td></td><td></td></tr>"
        )
    table = (
        '<table class="findings"><tr><th>级别</th><th>领域</th><th>结论（原文）</th>'
        "<th>证据</th><th>复核</th><th>裁定</th></tr>" + "".join(rows_html) + "</table>"
    )

    # what this means: the curated five S1s, in MEANING's order -- with 8 S1s on the
    # page an alphabetical cut would drop the user-facing ones for CD's.
    by_id = {f["id"]: f for f in findings}
    s1s = [by_id[i] for i in MEANING if i in by_id and by_id[i]["severity"] == "S1"][:5]
    meaning = "".join(
        f"<li>{MEANING.get(f['id'], '[' + f['id'] + '] ' + _esc(f['claim'][:120]))}</li>"
        for f in s1s
    )

    # open questions
    qs = []
    for rep in reports:
        if rep.get("landed"):
            for q in rep["open_qs"]:
                qs.append(f"<li>[{_esc(rep['area'])}] {_esc(q)}</li>")
    questions = "".join(qs) or "<li>（无）</li>"

    audit = f"""
<section class="audit">
<h1>项目审计 2026-09-04</h1>
{strip}
<div class="cards">{''.join(cards)}</div>
<h2>发现（S1 在前）</h2>
{table}
<h2>这对用户意味着什么</h2>
<ul class="meaning">{meaning}</ul>
<h2>待用户拍板</h2>
<ul class="questions">{questions}</ul>
</section>
"""

    # old page body below
    m = re.search(r"<body[^>]*>(.*)</body>", old_html, re.S)
    old_body = m.group(1) if m else f"<pre>{_esc(old_html)}</pre>"
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>aupai 进展 + 审计</title>
<style>
:root {{ --bg:#f7f7f5; --fg:#1a1a1a; --card:#fff; --line:#ddd; --mut:#666; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
  --bg:#141414; --fg:#eaeaea; --card:#1e1e1e; --line:#333; --mut:#999; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:14px/1.6 -apple-system,'PingFang SC',system-ui,sans-serif; }}
.audit {{ max-width:1100px; margin:0 auto; padding:16px; }}
h1 {{ font-size:20px; }} h2 {{ font-size:16px; margin-top:24px; }}
.strip {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center;
  padding:10px 12px; background:var(--card); border:1px solid var(--line); border-radius:8px; }}
.strip-item {{ color:var(--mut); font-size:13px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
  gap:10px; margin-top:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
.card h3 {{ margin:0 0 6px; font-size:14px; }}
.kv {{ font-size:13px; color:var(--mut); margin:2px 0; }}
.kv.warn {{ color:#c0392b; }}
.blind {{ margin:6px 0 0; padding-left:18px; font-size:12.5px; color:var(--mut); }}
.blind li {{ margin:3px 0; }}
details summary {{ cursor:pointer; font-size:13px; color:var(--mut); }}
.chip {{ display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px;
  border:1px solid var(--line); white-space:nowrap; }}
.chip.s1 {{ background:#c0392b; color:#fff; border-color:#c0392b; }}
.chip.s2 {{ background:#d68910; color:#fff; border-color:#d68910; }}
.chip.s3 {{ background:var(--mut); color:#fff; border-color:var(--mut); }}
.chip.nd {{ background:transparent; color:var(--mut); border:1px solid var(--line); }}
.rnote {{ font-size:11px; color:var(--mut); margin-top:2px; }}
.chip.ok {{ background:#1e8449; color:#fff; border-color:#1e8449; }}
.chip.bad {{ background:#c0392b; color:#fff; border-color:#c0392b; }}
.chip.pend {{ background:transparent; color:var(--mut); }}
table.findings {{ width:100%; border-collapse:collapse; margin-top:8px;
  background:var(--card); border:1px solid var(--line); border-radius:8px; }}
.findings th, .findings td {{ padding:6px 8px; border-bottom:1px solid var(--line);
  text-align:left; vertical-align:top; font-size:13px; }}
.findings th {{ color:var(--mut); font-weight:600; }}
.findings td.claim {{ max-width:420px; }}
.findings td.ruling {{ max-width:220px; color:var(--mut); }}
.findings code {{ background:var(--bg); padding:1px 4px; border-radius:4px; font-size:12px; }}
.findings a {{ color:inherit; }}
.meaning li, .questions li {{ margin:4px 0; }}
.old {{ border-top:2px solid var(--line); margin-top:28px; }}
</style></head>
<body>
<section class="old">{old_body}</section>
{state_block(ROOT)}{cleanup_block(ROOT)}{audit}
</body></html>
"""


def selftest():
    good = """# Audit: x
---
area: test area
owner: t
pair: u
---
## 4. Findings

| id | sev | claim as published | evidence | what contradicts it |
|---|---|---|---|---|
| T-1 | S1 | a claim | `eval/x.py:1` | the contradiction |
| T-2 | S2 | another | facts/t.json#t | none |
| T-3 | S3 | pipe in code | `grep -c 'a\\|b'` returns 0 | none |
| T-4 | — | withdrawn, no defect | facts/t.json#t | withdrawn by controller |

A non-finding table later in the same section must be ignored, not unparsed:

| path | bytes |
|---|---|
| x.json | 100 |

## 5. Blind spots

- nothing

## 6. Open questions

1. decide something

## Pair check

T-1 holds on the numbers; T-2 fails the sign check.
"""
    shuffled = good.replace(
        "| id | sev | claim as published | evidence | what contradicts it |",
        "| sev | id | claim as published | evidence | what contradicts it |",
    )
    with tempfile.TemporaryDirectory() as td:
        ap = os.path.join(td, "test_area.md")
        open(ap, "w").write(good)
        rep = read_report(ap)
        assert len(rep["findings"]) == 4, rep["findings"]
        assert rep["findings"][0]["id"] == "T-1"
        assert rep["findings"][2]["evidence"] == "`grep -c 'a\\|b'` returns 0"
        assert rep["findings"][3]["severity"] == "ND"
        assert not rep["unparsed"], rep["unparsed"]
        assert rep["pair_checks"] == {"T-1": "held", "T-2": "failed"}, rep["pair_checks"]
        assert rep["blind_spots"] == ["nothing"]
        assert rep["open_qs"] == ["decide something"]
        open(ap, "w").write(shuffled)
        try:
            read_report(ap)
        except ValueError:
            pass
        else:
            raise AssertionError("shuffled columns did not fail")
        # unparsable row
        open(ap, "w").write(good.replace("| T-2 | S2 |", "| T-X | S9 |"))
        rep = read_report(ap)
        assert len(rep["findings"]) == 3 and len(rep["unparsed"]) == 1
        # state/cleanup blocks: absent -> empty, present -> rendered
        with tempfile.TemporaryDirectory() as td:
            assert state_block(td) == "" and cleanup_block(td) == ""
            os.makedirs(os.path.join(td, "docs", "standards"))
            os.makedirs(os.path.join(td, "runs", "audit_0904"))
            open(os.path.join(td, "docs", "standards", "state_0904.md"), "w").write(
                "# 现状\n\n## user_facing\n\n- **stands**: x\n- retracted: y\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
            )
            open(os.path.join(td, "runs", "audit_0904", "cleanup.jsonl"), "w").write(
                '{"id": "C9", "owner": "98", "item": "amend", "state": "done", "evidence": "abc1234"}\n'
                '{"id": "C1", "owner": "tilerl", "item": "kill loops", "state": "open", "evidence": ""}\n'
            )
            sb = state_block(td)
            assert "<strong>stands</strong>" in sb and "<th>a</th>" in sb and "<td>2</td>" in sb, sb
            cb = cleanup_block(td)
            assert 'class="chip ok">done' in cb and "abc1234" in cb and 'class="chip pend">open' in cb, cb
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--compose", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    reports, findings, unparsed = collect()
    write_findings_jsonl(findings, unparsed)
    if args.compose or os.path.exists(PAGE):
        old_html = open(PAGE, encoding="utf-8").read() if os.path.exists(PAGE) else ""
        # Idempotent recompose: a previous compose left the page as
        # AUDIT <section class="old">OLD</section>, possibly nested. The true old
        # page is the innermost wrapper's content; progress_feed's page has no
        # <section>, so the first </section> after the innermost open is its close.
        while '<section class="old">' in old_html:
            start = old_html.rindex('<section class="old">') + len('<section class="old">')
            end = old_html.index('</section>', start)
            old_html = old_html[start:end]
        with open(PAGE, "w", encoding="utf-8") as fh:
            fh.write(render_page(reports, findings, unparsed, old_html))
    print(
        f"{len(findings)} findings, {len(unparsed)} unparsed, "
        f"{sum(1 for r in reports if r.get('landed'))}/7 reports -> {PAGE}"
    )


if __name__ == "__main__":
    main()
