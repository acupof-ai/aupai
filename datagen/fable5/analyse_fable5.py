#!/usr/bin/env python3
"""
Fable-5 SFT Traces  —  Dataset Analyser  v2
Author: kelexine (https://github.com/kelexine)

Run after clean_fable5.py v3. Produces a full statistical breakdown of the
v3 schema (no uid/session/step; includes messages, normalised response,
anonymised paths).

Usage:
    python analyse_fable5.py [--input cleaned_fable5.jsonl]
"""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentiles(values: list[float], qs=(5, 25, 50, 75, 90, 95, 99)) -> dict:
    if not values:
        return {q: 0 for q in qs}
    s = sorted(values)
    n = len(s)
    result = {}
    for q in qs:
        idx = (q / 100) * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        result[q] = s[lo] + (idx - lo) * (s[hi] - s[lo])
    return result


def histogram(values: list[float], bins: list[tuple[int, int]], label: str) -> list[str]:
    lines = [f"  {label}:"]
    for lo, hi in bins:
        count = sum(1 for v in values if lo <= v < hi)
        bar   = "█" * min(40, count // max(1, len(values) // 40))
        lines.append(f"    [{lo:>6} – {hi:>6})  {count:>5}  {bar}")
    tail = sum(1 for v in values if v >= bins[-1][1])
    if tail:
        lines.append(f"    [{bins[-1][1]:>6}+       )  {tail:>5}")
    return lines


def section(title: str) -> str:
    return f"\n{'─' * 60}\n  {title}\n{'─' * 60}"

# ─────────────────────────────────────────────────────────────────────────────
# Analysis blocks
# ─────────────────────────────────────────────────────────────────────────────

def overview(rows: list[dict]) -> list[str]:
    out = [section("OVERVIEW")]
    out.append(f"  total rows      : {len(rows):,}")
    out.append("  schema version  : v3  (no uid/session/step; messages field present)")
    fields = sorted(rows[0].keys()) if rows else []
    out.append(f"  fields          : {', '.join(fields)}")
    return out


def origin_breakdown(rows: list[dict]) -> list[str]:
    out = [section("ORIGIN BREAKDOWN")]
    by_origin: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_origin[r["origin"]].append(r)

    for origin, group in sorted(by_origin.items()):
        n     = len(group)
        trunc = sum(1 for r in group if r["context_truncated"])
        tasks = Counter(r["task_type"] for r in group)
        types = Counter(r["output_type"] for r in group)
        out.append(f"\n  origin={origin!r}  ({n:,} rows)")
        out.append(f"    truncated ctx   : {trunc:,}  ({trunc/n*100:.1f}%)")
        out.append(f"    task_type dist  : {dict(tasks.most_common())}")
        out.append(f"    output_type dist: {dict(types.most_common())}")
    return out


def cot_length_analysis(rows: list[dict]) -> list[str]:
    out = [section("COT LENGTH ANALYSIS")]
    all_cot = [r["cot_length"] for r in rows]
    p = percentiles(all_cot)
    out.append(f"  global  min={min(all_cot)}  max={max(all_cot):,}  "
               f"mean={statistics.mean(all_cot):.0f}  median={statistics.median(all_cot):.0f}")
    out.append(f"  p5={p[5]:.0f}  p25={p[25]:.0f}  p50={p[50]:.0f}  "
               f"p75={p[75]:.0f}  p90={p[90]:.0f}  p95={p[95]:.0f}  p99={p[99]:.0f}")
    bins = [(0,200),(200,500),(500,1000),(1000,2000),(2000,4000),(4000,6000),(6000,9000)]
    out += histogram(all_cot, bins, "cot_length distribution (all rows)")
    for task in ("agentic", "reasoning", "chat"):
        vals = [r["cot_length"] for r in rows if r["task_type"] == task]
        if not vals:
            continue
        p = percentiles(vals)
        out.append(f"\n  [{task}]  n={len(vals):,}  "
                   f"min={min(vals)}  max={max(vals):,}  mean={statistics.mean(vals):.0f}")
        out.append(f"    p25={p[25]:.0f}  p50={p[50]:.0f}  p75={p[75]:.0f}  p95={p[95]:.0f}")
    return out


def context_length_analysis(rows: list[dict]) -> list[str]:
    out = [section("CONTEXT LENGTH ANALYSIS")]
    all_ctx  = [r["context_length"] for r in rows]
    trunc    = [r["context_length"] for r in rows if r["context_truncated"]]
    complete = [r["context_length"] for r in rows if not r["context_truncated"]]
    out.append(f"  all      n={len(all_ctx):,}  mean={statistics.mean(all_ctx):.0f}  "
               f"median={statistics.median(all_ctx):.0f}  max={max(all_ctx):,}")
    out.append(f"  complete n={len(complete):,}  "
               f"mean={statistics.mean(complete) if complete else 0:.0f}  "
               f"max={max(complete) if complete else 0:,}")
    out.append(f"  truncd   n={len(trunc):,}  "
               f"mean={statistics.mean(trunc) if trunc else 0:.0f}  "
               f"max={max(trunc) if trunc else 0:,}")
    bins = [(0,2000),(2000,5000),(5000,10000),(10000,20000)]
    out += histogram(all_ctx, bins, "context_length distribution")
    return out


def response_length_analysis(rows: list[dict]) -> list[str]:
    out = [section("RESPONSE LENGTH ANALYSIS")]
    all_resp = [r["response_length"] for r in rows]
    p = percentiles(all_resp)
    out.append(f"  global  min={min(all_resp)}  max={max(all_resp):,}  "
               f"mean={statistics.mean(all_resp):.0f}  median={statistics.median(all_resp):.0f}")
    out.append(f"  p25={p[25]:.0f}  p50={p[50]:.0f}  p75={p[75]:.0f}  p95={p[95]:.0f}")
    for task in ("agentic", "reasoning", "chat"):
        vals = [r["response_length"] for r in rows if r["task_type"] == task]
        if not vals:
            continue
        out.append(f"  [{task}]  n={len(vals):,}  mean={statistics.mean(vals):.0f}  "
                   f"max={max(vals):,}")
    return out


def messages_analysis(rows: list[dict]) -> list[str]:
    out = [section("MESSAGES FIELD ANALYSIS")]
    msg_counts  = [len(r["messages"]) for r in rows]
    roles_all: Counter = Counter()
    has_tool_calls = 0
    has_tool_result = 0
    single_turn = 0

    for r in rows:
        msgs = r["messages"]
        roles_all.update(m.get("role") for m in msgs)
        if any(m.get("tool_calls") for m in msgs):
            has_tool_calls += 1
        if any(m.get("role") == "tool" for m in msgs):
            has_tool_result += 1
        # Single-turn: only [user, assistant]
        if len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
            single_turn += 1

    out.append(f"  messages/row  min={min(msg_counts)}  max={max(msg_counts):,}  "
               f"mean={statistics.mean(msg_counts):.1f}  median={statistics.median(msg_counts):.0f}")
    out.append(f"  single-turn rows (user+assistant only) : {single_turn:,}")
    out.append(f"  rows with tool_calls in prior turns    : {has_tool_calls:,}")
    out.append(f"  rows with tool results in prior turns  : {has_tool_result:,}")
    out.append("  role distribution across all messages  :")
    for role, count in roles_all.most_common():
        out.append(f"    {role:<16} {count:>7,}")
    bins = [(1,3),(3,5),(5,10),(10,20),(20,50),(50,100)]
    out += histogram(msg_counts, bins, "messages-per-row distribution")
    return out


def tool_distribution(rows: list[dict]) -> list[str]:
    out = [section("TOOL CALL DISTRIBUTION  (agentic rows only)")]
    agentic = [r for r in rows if r["output_type"] == "tool_use"]
    tool_counts: Counter = Counter()
    for r in agentic:
        name = r["output"].get("tool", "<unknown>")
        tool_counts[name] += 1
    out.append(f"  total tool calls : {len(agentic):,}  unique tools : {len(tool_counts)}")
    out.append("")
    for tool, count in tool_counts.most_common(20):
        bar = "█" * min(40, count // max(1, len(agentic) // 40))
        pct = count / len(agentic) * 100
        out.append(f"    {tool:<28} {count:>5}  ({pct:5.1f}%)  {bar}")
    return out


def anonymization_check(rows: list[dict]) -> list[str]:
    out = [section("PATH ANONYMISATION INTEGRITY CHECK")]
    # Look for any remaining /home/USERNAME/ patterns (but allow /home/user/)
    leak_re      = re.compile(r"/home/(?!user(?:/|$))[^/\s]+/")
    win_back_re  = re.compile(
        r"(?i)[A-Za-z]:\\Users\\"
        r"(?!user(?:\\|$|[\s\"',;]))"
        r"[^\\\"'\s,;]+"
        r"(?=\\|$|[\s\"',;])"
    )
    win_fwd_re   = re.compile(
        r"(?i)[A-Za-z]:/Users/"
        r"(?!user(?:/|$|[\s\"',;]))"
        r"[^/\"'\s,;]+"
        r"(?=/|$|[\s\"',;])"
    )

    leaks: list[tuple[str, str]] = []

    def scan(text: str, label: str) -> None:
        for pat, name in (
            (leak_re,     "unix-path"),
            (win_back_re, "win-backslash-path"),
            (win_fwd_re,  "win-fwdslash-path"),
        ):
            if pat.search(text):
                leaks.append((label, name))

    for i, r in enumerate(rows):
        scan(r.get("context", ""),    f"row[{i}].context")
        scan(r.get("thinking", ""),   f"row[{i}].thinking")
        scan(r.get("response", ""),   f"row[{i}].response")
        scan(r.get("completion", ""), f"row[{i}].completion")
        out_str = json.dumps(r.get("output", {}))
        scan(out_str, f"row[{i}].output")
        for j, m in enumerate(r.get("messages", [])):
            scan(json.dumps(m), f"row[{i}].messages[{j}]")

    if leaks:
        out.append(f"  ⚠  {len(leaks)} leak(s) found:")
        for label, kind in leaks[:20]:
            out.append(f"    {kind}  in  {label}")
        if len(leaks) > 20:
            out.append(f"    … and {len(leaks) - 20} more")
    else:
        out.append("  ✓  No home-directory path leaks detected.")
    return out


def response_format_check(rows: list[dict]) -> list[str]:
    out = [section("RESPONSE FORMAT CHECK")]
    tool_with_tag  = sum(1 for r in rows
                         if r["output_type"] == "tool_use"
                         and r["response"].startswith("<tool_call>"))
    text_no_prefix = sum(1 for r in rows
                         if r["output_type"] == "text"
                         and not r["response"].startswith("ASSISTANT"))
    tool_total = sum(1 for r in rows if r["output_type"] == "tool_use")
    text_total = sum(1 for r in rows if r["output_type"] == "text")
    out.append(f"  tool_use rows with <tool_call> tag : {tool_with_tag}/{tool_total}")
    out.append(f"  text rows without ASSISTANT prefix : {text_no_prefix}/{text_total}")

    bad_completion = sum(1 for r in rows
                         if not r["completion"].startswith("<think>")
                         or "</think>" not in r["completion"])
    out.append(f"  completions with invalid structure : {bad_completion}")
    out.append(f"  all checks passed                  : "
               f"{tool_with_tag == tool_total and text_no_prefix == text_total and bad_completion == 0}")
    return out


def sample_rows(rows: list[dict]) -> list[str]:
    out = [section("SAMPLE ROWS  (1 per task_type × origin)")]
    seen: set[tuple] = set()
    for task in ("agentic", "reasoning", "chat"):
        for origin in ("local", "hf"):
            key = (task, origin)
            if key in seen:
                continue
            r = next((x for x in rows
                      if x["task_type"] == task and x["origin"] == origin), None)
            if not r:
                continue
            seen.add(key)
            out.append(f"\n  [{task} / {origin}]")
            out.append(f"    cot_len={r['cot_length']}  ctx_len={r['context_length']}  "
                       f"resp_len={r['response_length']}  truncated={r['context_truncated']}")
            out.append(f"    output       : {json.dumps(r['output'])[:160]}")
            out.append(f"    thinking[:160]: {r['thinking'][:160]!r}")
            out.append(f"    response[:160]: {r['response'][:160]!r}")
            out.append(f"    messages count: {len(r['messages'])}")
            out.append(f"    messages[0]   : {json.dumps(r['messages'][0])[:160]}")
            if len(r["messages"]) > 1:
                out.append(f"    messages[-1]  : {json.dumps(r['messages'][-1])[:160]}")
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("cleaned_fable5.jsonl"))
    args = p.parse_args()

    print(f"Loading {args.input} …")
    rows = load_jsonl(args.input)
    print(f"  {len(rows):,} rows\n")

    blocks = [
        overview(rows),
        origin_breakdown(rows),
        cot_length_analysis(rows),
        context_length_analysis(rows),
        response_length_analysis(rows),
        messages_analysis(rows),
        tool_distribution(rows),
        anonymization_check(rows),
        response_format_check(rows),
        sample_rows(rows),
    ]

    lines = []
    for block in blocks:
        lines.extend(block)
        lines.append("")

    report = "\n".join(lines)
    print(report)

    out = Path("analysis_report.txt")
    out.write_text(report, encoding="utf-8")
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
