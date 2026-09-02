#!/usr/bin/env python3
"""
Fable-5-traces Dataset Cleaner  —  v3
Author: kelexine (https://github.com/kelexine)

Cleans and normalises Kelexine/Fable-5-traces into a multi-paradigm fine-tuning
dataset. Three training shapes off one JSONL:

  Full SFT          —  messages  (or context + completion)
  Reasoning split   —  context + thinking + response
  Instruction-only  —  context + response

v3 changes over v2:
  - Removed uid, session, step (provenance metadata not needed for training)
  - Added `messages`: structured OpenAI-format conversation list; final element
    is the assistant turn containing the full <think>…</think> completion
  - Normalised `response`:
      text rows     → raw text string (no ASSISTANT wrapper)
      tool_use rows → <tool_call>{"name":…,"arguments":{…}}</tool_call>
  - Path anonymisation: /home/USERNAME/ → /home/user/,
                        C:\\Users\\USERNAME\\ → C:\\Users\\user\\
    applied to context, thinking, output dict, response, completion, messages
  - `completion` now uses the normalised response (consistent with messages)

Usage:
  python clean_fable5.py                            # pull from HF Hub
  python clean_fable5.py --parquet ./0000.parquet   # local parquet
  python clean_fable5.py --help
"""

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATASET_ID  = "Kelexine/Fable-5-traces"
SPLIT       = "train"

DEFAULT_OUTPUT = Path("cleaned_fable5.jsonl")
DEFAULT_REPORT = Path("cleaning_report.json")

# Text-output rows with CoT at or above this are tagged "reasoning"
REASONING_COT_THRESHOLD = 450

# Rows with CoT shorter than this are dropped as malformed
MIN_COT_LENGTH = 50

# ─────────────────────────────────────────────────────────────────────────────
# Compiled patterns
# ─────────────────────────────────────────────────────────────────────────────

TRUNCATION_MARK = "…[earlier truncated]…"

# Claude Code injected noise blocks
LOCAL_CMD_RE = re.compile(
    r"<local-command-caveat>.*?</local-command-caveat>\s*"
    r"(?:"
        r"<command-name>.*?</command-name>\s*"
        r"<command-message>.*?</command-message>\s*"
        r"<command-args>.*?</command-args>\s*"
    r")?"
    r"(?:<local-command-stdout>.*?</local-command-stdout>\s*)?",
    re.DOTALL,
)
ANSI_RE        = re.compile(r"\x1b\[[0-9;]*[mGKHF]|\[1m|\[22m")
THINK_OPEN_RE  = re.compile(r"^\s*<think>\s*", re.MULTILINE)
THINK_CLOSE_RE = re.compile(r"\s*</think>\s*$", re.MULTILINE)

# Path anonymisation
# Matches /home/USERNAME/ but NOT /home/user/ (already anonymised)
UNIX_HOME_RE = re.compile(r"/home/(?!user(?:/|$))([^/\s\"'\\,;>\]]+)/")
# Windows paths — two variants to handle both separator styles:
#   backslash : C:\Users\USERNAME  (trailing \ via positive lookahead, not consumed)
#   fwd-slash : C:/Users/USERNAME  (same; covers cross-platform scripts)
# Both skip already-anonymised "user" and use lookahead so the trailing
# separator / end-of-string is preserved in the output.
WIN_USER_BACK_RE = re.compile(
    r"(?i)([A-Za-z]:\\Users\\)"
    r"(?!user(?:\\|$|[\s\"',;]))"
    r"([^\\\"'\s,;]+)"
    r"(?=\\|$|[\s\"',;])",
)
WIN_USER_FWD_RE = re.compile(
    r"(?i)([A-Za-z]:/Users/)"
    r"(?!user(?:/|$|[\s\"',;]))"
    r"([^/\"'\s,;]+)"
    r"(?=/|$|[\s\"',;])",
)

# Context message-boundary splitter
# Splits at \n that is immediately followed by a known role marker
CONTEXT_SPLIT_RE = re.compile(r"\n(?=USER:|ASSISTANT \()")

# ASSISTANT (tool call) ToolName input={...}
TOOL_CALL_LINE_RE = re.compile(r"^ASSISTANT \(tool call\) (\S+) input=(.*)", re.DOTALL)

# ─────────────────────────────────────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CleanRow:
    """
    Single cleaned training example.

    Training mode field map
    ───────────────────────
    Full SFT        :  messages[-1]["content"] == completion
                       (or use context + completion)
    Reasoning split :  context  +  thinking  +  response
    Instruction-only:  context  +  response
    """
    # ── classification ───────────────────────────────────────────────────────
    model:             str        # "claude-fable-5"
    origin:            str        # "local" | "hf"
    task_type:         str        # "agentic" | "reasoning" | "chat"
    output_type:       str        # "tool_use" | "text"
    context_truncated: bool       # True when upstream context window was clipped

    # ── structured conversation (primary training field) ──────────────────────
    messages: list[dict]          # OpenAI-format: role/content/tool_calls dicts.
                                  # Prior turns parsed from context; final element
                                  # is {"role":"assistant","content": completion}.

    # ── flat fields for flexible sampling ────────────────────────────────────
    context:    str               # cleaned, anonymised raw context string
    thinking:   str               # isolated CoT (no <think> tags), anonymised
    response:   str               # normalised assistant output, no wrapper:
                                  #   text     → raw text string
                                  #   tool_use → <tool_call>{…}</tool_call>
    output:     dict              # anonymised parsed payload dict
    completion: str               # "<think>\n{thinking}\n</think>\n{response}"

    # ── length metadata ───────────────────────────────────────────────────────
    cot_length:      int
    context_length:  int
    response_length: int

# ─────────────────────────────────────────────────────────────────────────────
# Path anonymisation
# ─────────────────────────────────────────────────────────────────────────────

def anonymize_text(text: str) -> str:
    """Replace identifiable home-directory path components in a string."""
    text = UNIX_HOME_RE.sub("/home/user/", text)
    text = WIN_USER_BACK_RE.sub(r"\1user", text)
    text = WIN_USER_FWD_RE.sub(r"\1user", text)
    return text


def anonymize_value(v: Any) -> Any:
    """Recursively anonymise strings inside dicts / lists."""
    if isinstance(v, str):
        return anonymize_text(v)
    if isinstance(v, dict):
        return {k: anonymize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [anonymize_value(i) for i in v]
    return v


def anonymize_messages(messages: list[dict]) -> list[dict]:
    result = []
    for msg in messages:
        m = dict(msg)
        if "content" in m and isinstance(m["content"], str):
            m["content"] = anonymize_text(m["content"])
        if "tool_calls" in m and m["tool_calls"]:
            anon_tcs = []
            for tc in m["tool_calls"]:
                anon_tc = dict(tc)
                if "function" in anon_tc:
                    fn = dict(anon_tc["function"])
                    fn["arguments"] = anonymize_value(fn.get("arguments", {}))
                    anon_tc["function"] = fn
                anon_tcs.append(anon_tc)
            m["tool_calls"] = anon_tcs
        result.append(m)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Context → messages parser
# ─────────────────────────────────────────────────────────────────────────────

def extract_json_prefix(text: str) -> tuple[dict, str]:
    """
    Extract the first complete JSON object from the start of `text`.
    Returns (parsed_dict, remainder_text).  On failure: ({}, text).
    """
    if not text or not text.startswith("{"):
        return {}, text

    depth = escape = 0
    in_str_bool = False

    for i, ch in enumerate(text):
        if escape:
            escape = 0
            continue
        if ch == "\\" and in_str_bool:
            escape = 1
            continue
        if ch == '"':
            in_str_bool = not in_str_bool
            continue
        if in_str_bool:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[: i + 1]), text[i + 1 :].strip()
                except json.JSONDecodeError:
                    break

    # Fallback: try the whole text
    try:
        return json.loads(text), ""
    except json.JSONDecodeError:
        return {}, text


def _make_tool_call_message(tool_name: str, args: dict, tool_result: str) -> list[dict]:
    """Build assistant tool-call message + optional tool-result message."""
    msgs: list[dict] = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": tool_name, "arguments": args},
                }
            ],
        }
    ]
    if tool_result:
        msgs.append({"role": "tool", "name": tool_name, "content": tool_result})
    return msgs


def parse_context_to_messages(context: str) -> list[dict]:
    """
    Parse the raw cleaned context string into an OpenAI-format messages list.

    Recognises:
      USER: …
      ASSISTANT (message): …
      ASSISTANT (tool call) ToolName input={…}

    Content that does not start with a known marker (e.g. embedded tool
    stdout between the tool call and the next USER/ASSISTANT line) is
    appended to the previous message's content.
    """
    # Strip truncation marker — it's metadata, not conversation content
    ctx = context.replace(TRUNCATION_MARK, "").strip()
    if not ctx:
        return []

    chunks = CONTEXT_SPLIT_RE.split(ctx)
    messages: list[dict] = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if chunk.startswith("USER:"):
            content = chunk[5:].strip()
            if content:
                messages.append({"role": "user", "content": content})

        elif chunk.startswith("ASSISTANT (message):"):
            content = chunk[len("ASSISTANT (message):") :].strip()
            if content:
                messages.append({"role": "assistant", "content": content})

        elif chunk.startswith("ASSISTANT (tool call)"):
            m = TOOL_CALL_LINE_RE.match(chunk)
            if m:
                tool_name  = m.group(1)
                raw        = m.group(2).strip()
                args, rest = extract_json_prefix(raw)
                messages.extend(_make_tool_call_message(tool_name, args, rest))
            else:
                # Malformed tool call — preserve as generic assistant turn
                messages.append({"role": "assistant", "content": chunk})

        else:
            # Unrecognised chunk (inline tool output, etc.) — append to last msg
            if messages:
                last = messages[-1]
                existing = last.get("content") or ""
                last["content"] = (existing + "\n" + chunk).strip() if existing else chunk
            # If no prior message yet, discard — likely pre-conversation noise

    return messages

# ─────────────────────────────────────────────────────────────────────────────
# Field builders
# ─────────────────────────────────────────────────────────────────────────────

def extract_thinking(raw_cot: str) -> str:
    text = THINK_OPEN_RE.sub("", raw_cot)
    text = THINK_CLOSE_RE.sub("", text)
    return text.strip()


def validate_output(raw: Any) -> tuple[dict | None, str]:
    if raw is None:
        return None, "null"
    if isinstance(raw, dict):
        return (raw, "") if raw else (None, "empty_dict")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, f"json_error:{exc.msg}"
        if not isinstance(parsed, dict):
            return None, f"wrong_type:{type(parsed).__name__}"
        return parsed, ""
    return None, f"unexpected_type:{type(raw).__name__}"


def build_normalized_response(output_type: str, output: dict) -> str:
    """
    Normalised response string — no ASSISTANT wrapper.

    text rows     → raw text string
    tool_use rows → <tool_call>{"name":…,"arguments":{…}}</tool_call>
    """
    if output_type == "tool_use":
        payload = {
            "name":      output.get("tool", "Unknown"),
            "arguments": output.get("input", {}),
        }
        return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n</tool_call>"
    return output.get("text", "")


def build_completion(thinking: str, response: str) -> str:
    return f"<think>\n{thinking}\n</think>\n{response}"


def clean_context(raw: str) -> tuple[str, bool]:
    truncated = TRUNCATION_MARK in raw
    ctx = LOCAL_CMD_RE.sub("", raw)
    ctx = ANSI_RE.sub("", ctx)
    # Collapse 3+ blank lines → 2
    lines, out, blanks = ctx.splitlines(), [], 0
    for line in lines:
        s = line.rstrip()
        if s == "":
            blanks += 1
            if blanks <= 2:
                out.append("")
        else:
            blanks = 0
            out.append(s)
    return "\n".join(out).strip(), truncated


def classify_task(output_type: str, cot_len: int) -> str:
    if output_type == "tool_use":
        return "agentic"
    return "reasoning" if cot_len >= REASONING_COT_THRESHOLD else "chat"

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def load_source(args: argparse.Namespace):
    if args.parquet:
        print(f"  source : local parquet → {args.parquet}")
        return load_dataset("parquet", data_files=str(args.parquet), split="train")
    print(f"  source : HuggingFace Hub → {DATASET_ID}")
    kwargs: dict[str, Any] = {"path": DATASET_ID, "split": SPLIT}
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    return load_dataset(**kwargs)


def deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    unique, dups = [], 0
    for row in rows:
        uid = row.get("uid", "")
        if uid in seen:
            dups += 1
        else:
            seen.add(uid)
            unique.append(row)
    return unique, dups


def sort_sessions(rows: list[dict]) -> tuple[list[dict], int]:
    by_session: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_session[row.get("session", "__unknown__")].append(row)
    out: list[dict] = []
    for session_rows in by_session.values():
        session_rows.sort(
            key=lambda r: int(r.get("uid", "#-1").rsplit("#", 1)[-1])
            if r.get("uid", "").rsplit("#", 1)[-1].isdigit()
            else -1
        )
        out.extend(session_rows)
    return out, len(by_session)


def process_rows(rows: list[dict]) -> tuple[list[CleanRow], dict, dict]:
    clean: list[CleanRow] = []
    drop_reasons: dict[str, int] = defaultdict(int)
    stats: dict[str, Any] = {
        "task_type":   defaultdict(int),
        "output_type": defaultdict(int),
        "origin":      defaultdict(int),
        "truncated":   0,
    }

    for row in tqdm(rows, desc="  cleaning", unit="row"):
        origin      = row.get("origin", "")
        model       = row.get("model", "")
        output_type = row.get("output_type", "")
        raw_cot     = row.get("cot") or ""
        raw_context = row.get("context") or ""
        raw_output  = row.get("output")

        # ── Guards ──────────────────────────────────────────────────────────
        if output_type not in ("tool_use", "text"):
            drop_reasons[f"bad_output_type:{output_type!r}"] += 1
            continue

        thinking_raw = extract_thinking(raw_cot)
        if len(thinking_raw) < MIN_COT_LENGTH:
            drop_reasons["cot_too_short"] += 1
            continue

        output_raw, err = validate_output(raw_output)
        if output_raw is None:
            drop_reasons[f"bad_output:{err}"] += 1
            continue

        # ── Anonymise ────────────────────────────────────────────────────────
        thinking = anonymize_text(thinking_raw)
        output   = anonymize_value(output_raw)

        # ── Clean & anonymise context ─────────────────────────────────────────
        context_clean, truncated = clean_context(raw_context)
        context = anonymize_text(context_clean)

        # ── Build derived fields ──────────────────────────────────────────────
        cot_len   = len(thinking)
        task_type = classify_task(output_type, cot_len)
        response  = build_normalized_response(output_type, output)
        completion = build_completion(thinking, response)

        # ── Build messages ────────────────────────────────────────────────────
        prior = parse_context_to_messages(context)
        prior = anonymize_messages(prior)
        messages = prior + [{"role": "assistant", "content": completion}]

        clean.append(CleanRow(
            model             = model,
            origin            = origin,
            task_type         = task_type,
            output_type       = output_type,
            context_truncated = truncated,
            messages          = messages,
            context           = context,
            thinking          = thinking,
            response          = response,
            output            = output,
            completion        = completion,
            cot_length        = cot_len,
            context_length    = len(context),
            response_length   = len(response),
        ))

        stats["task_type"][task_type]     += 1
        stats["output_type"][output_type] += 1
        stats["origin"][origin]           += 1
        if truncated:
            stats["truncated"] += 1

    return clean, drop_reasons, stats


def write_outputs(
    clean: list[CleanRow],
    out_jsonl: Path,
    out_report: Path,
    raw_total: int,
    dups: int,
    drop_reasons: dict,
    stats: dict,
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in clean:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    report = {
        "raw_total":               raw_total,
        "after_dedup":             raw_total - dups,
        "duplicates_removed":      dups,
        "clean_total":             len(clean),
        "dropped_invalid":         (raw_total - dups) - len(clean),
        "reasoning_cot_threshold": REASONING_COT_THRESHOLD,
        "truncated_context_rows":  stats["truncated"],
        "task_type_dist":          dict(stats["task_type"]),
        "output_type_dist":        dict(stats["output_type"]),
        "origin_dist":             dict(stats["origin"]),
        "drop_reasons":            dict(drop_reasons),
    }
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def print_summary(
    raw_total: int,
    dups: int,
    clean: list[CleanRow],
    stats: dict,
    drop_reasons: dict,
    out_jsonl: Path,
    out_report: Path,
) -> None:
    n = len(clean)
    dropped = (raw_total - dups) - n
    print()
    print("─" * 56)
    print(f"  Raw rows             {raw_total:>7,}")
    print(f"  Duplicates removed   {dups:>7,}")
    print(f"  Dropped (invalid)    {dropped:>7,}")
    print(f"  Clean rows           {n:>7,}")
    print(f"  Truncated ctx rows   {stats['truncated']:>7,}")
    print()
    print("  Task type:")
    for k, v in sorted(stats["task_type"].items()):
        pct = v / n * 100 if n else 0
        print(f"    {k:<12} {v:>6,}  ({pct:.1f}%)")
    print()
    print("  Output type:")
    for k, v in sorted(stats["output_type"].items()):
        print(f"    {k:<12} {v:>6,}")
    if drop_reasons:
        print()
        print("  Drop reasons:")
        for k, v in sorted(drop_reasons.items(), key=lambda x: -x[1]):
            print(f"    {k:<46} {v:>5,}")
    print()
    print(f"  → {out_jsonl}")
    print(f"  → {out_report}")
    print("─" * 56)

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clean Kelexine/Fable-5-traces for mixed fine-tuning (v3)."
    )
    p.add_argument("--parquet", type=Path, default=None,
                   help="Local .parquet file (skips HF Hub download).")
    p.add_argument("--hf-token", type=str, default=None,
                   help="HuggingFace API token (or set HF_TOKEN env var).")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output JSONL (default: {DEFAULT_OUTPUT}).")
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                   help=f"Output report JSON (default: {DEFAULT_REPORT}).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("[1/5] Loading dataset …")
    ds = load_source(args)
    raw_total = len(ds)
    print(f"  {raw_total:,} rows loaded.")

    print("[2/5] Deduplicating on uid …")
    unique, dups = deduplicate(list(ds))
    print(f"  {dups:,} duplicates removed → {len(unique):,} unique rows.")

    print("[3/5] Sorting within sessions by step index …")
    sorted_rows, n_sessions = sort_sessions(unique)
    print(f"  {n_sessions:,} sessions sorted.")

    print("[4/5] Cleaning, anonymising, and building messages …")
    clean, drop_reasons, stats = process_rows(sorted_rows)

    print("[5/5] Writing outputs …")
    write_outputs(clean, args.output, args.report,
                  raw_total, dups, drop_reasons, stats)

    print_summary(raw_total, dups, clean, stats,
                  drop_reasons, args.output, args.report)


if __name__ == "__main__":
    main()
