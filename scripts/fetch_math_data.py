#!/usr/bin/env python3
"""Fetch open math datasets and normalize to short-step Chinese format.

Output format (jsonl, one file per source in data/math/):
  {"instruction": "<problem, zh>", "output": "<short steps>\n答案是：\\boxed{X}", "src": "..."}

Sources are downloaded in parallel (one process each). Run:
  python scripts/fetch_math_data.py                # all
  python scripts/fetch_math_data.py ape210k belle  # subset
"""

import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from holdout import is_holdout  # noqa: E402

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "math")


def num(s):
    """Parse int/float/percent/fraction string -> float, else None."""
    s = str(s).strip()
    try:
        return float(s)
    except ValueError:
        pass
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100
        except ValueError:
            return None
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", s)
    if m and float(m.group(2)):
        return float(m.group(1)) / float(m.group(2))
    return None


def fmt(x):
    """Render a float without a trailing .0."""
    return str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:.4g}"


# --- per-source normalizers -------------------------------------------------


def ape210k(row):
    """Ape210K: native zh problem + `equation` (x=...) + verified `result`."""
    q = (row.get("question_chinese") or "").strip()
    eq = (row.get("equation") or "").lstrip("x=").strip()
    r = num(row.get("result_float") or row.get("result"))
    if not q or not eq or r is None:
        return None
    expr = eq.replace("%", "/100")
    if not re.fullmatch(r"[\d.+\-*/() ]+", expr):
        return None  # keep only pure arithmetic we can verify
    try:
        val = eval(expr, {"__builtins__": {}})  # noqa: S307 — charset-restricted above
    except (SyntaxError, ZeroDivisionError, TypeError, NameError):
        return None
    if abs(val - r) > 1e-4 * max(1.0, abs(r)):
        return None  # equation disagrees with the recorded answer
    return {
        "instruction": q,
        "output": f"列式：{eq} = {fmt(r)}\n答案是：\\boxed{{{fmt(r)}}}",
        "src": "ape210k",
    }


# Two tiers, and the tier matters more than the position. An explicit answer marker always beats a
# bare \frac buried in the derivation: math23k solutions typically end "因此，现在电视机卖1664元一台"
# with no marker at all, and a single-tier rule that ranks every pattern by end offset returns the
# \frac{1}{5} from the middle of the working (measured: it corrupts 18.6% of math23k this way).
MARKERS = [
    re.compile(r"\\boxed\{([^{}]+)\}"),
    re.compile(r"####\s*(-?[\d,./%]+)"),
    re.compile(r"(?:答案|答)\s*(?:是|为)?\s*[:：]?\s*(-?[\d,./%]+)"),
]
# Only consulted when no marker is present. \frac{10}{3} must be read as a fraction rather than
# falling through to LAST_NUM, which would return the denominator (REVIEW_2026-08-26.md #2).
FALLBACKS = [re.compile(r"\\[dt]?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}")]
# Thousands separators and scientific notation are part of the number: without them "40,000。"
# reads as "000" and "1.763e+04" as "04".
LAST_NUM = re.compile(r"(-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?)(?!.*\d)", re.S)


def _best(text, patterns):
    """The match nearest the END of the text, across the given patterns."""
    best = None
    for rx in patterns:
        for m in rx.finditer(text):
            val = f"{m.group(1)}/{m.group(2)}" if m.re.groups == 2 else m.group(1)
            val = val.replace(",", "").strip().rstrip("。.,，")
            if num(val) is not None and (best is None or m.end() > best[0]):
                best = (m.end(), val)
    return best[1] if best else None


def tail_answer(text):
    """Gold answer: the last explicit answer marker; failing that a fraction or the last number in
    the closing lines. Selecting by end offset within a tier matters too -- '解答：\n\n1. ...' used
    to match the 答 pattern and yield '1.' as the answer for 3.5% of mxode rows."""
    text = text.strip()
    hit = _best(text, MARKERS)
    if hit is not None:
        return hit
    tail = "\n".join(text.split("\n")[-2:])
    hit = _best(tail, FALLBACKS + [LAST_NUM])
    if hit is not None:
        return hit
    return _best(text, FALLBACKS)


def belle(row):
    """BelleGroup school_math: zh prose steps; keep only ones with a numeric tail answer."""
    q, a = (row.get("instruction") or "").strip(), (row.get("output") or "").strip()
    q = re.sub(r"^题目[:：]\s*", "", q)
    ans = tail_answer(a)
    if not q or not ans or len(a) > 800:
        return None
    body = "\n".join(ln.strip() for ln in a.split("\n") if ln.strip())
    return {"instruction": q, "output": f"{body}\n答案是：\\boxed{{{ans}}}", "src": "belle"}


def gsm8k_zh(row):
    q, a = (
        (row.get("question_zh") or row.get("instruction") or "").strip(),
        (row.get("answer_zh") or row.get("output") or "").strip(),
    )
    ans = tail_answer(a)
    if not q or not ans:
        return None
    body = re.sub(r"\n?####.*$", "", a).strip()
    body = re.sub(r"<<[^>]*>>", "", body)
    return {"instruction": q, "output": f"{body}\n答案是：\\boxed{{{ans}}}", "src": "gsm8k_zh"}


def math23k(row):
    """math23k-reborn: chat messages + metadata['reference'] as the verified answer."""
    msgs = row.get("messages") or []
    if isinstance(msgs, str):
        msgs = json.loads(msgs)  # dataset field, literal list
    q = next((m["content"] for m in msgs if m.get("role") == "user"), "").strip()
    a = next((m["content"] for m in msgs if m.get("role") == "assistant"), "").strip()
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, ValueError):
            meta = {}
    ans = str(meta.get("reference") or "").strip() or tail_answer(a)
    if not q or not ans or num(ans) is None or len(a) > 800:
        return None
    body = "\n".join(ln.strip() for ln in a.split("\n") if ln.strip())
    return {"instruction": q, "output": f"{body}\n答案是：\\boxed{{{ans}}}", "src": "math23k"}


def mxode(row):
    """Mxode 220K: use the short `response` field, drop the long `reasoning`."""
    q = (row.get("instruction") or row.get("prompt") or row.get("question") or "").strip()
    a = (row.get("response") or row.get("output") or "").strip()
    ans = tail_answer(a)
    if not q or not ans or len(a) > 800:
        return None
    body = "\n".join(ln.strip() for ln in a.split("\n") if ln.strip())
    return {"instruction": q, "output": f"{body}\n答案是：\\boxed{{{ans}}}", "src": "mxode"}


def tick(row):
    """TICK666: templated arithmetic drills — capped share, keeps numeric fluency up."""
    q = (row.get("instruction") or row.get("question") or row.get("input") or "").strip()
    a = (row.get("output") or row.get("answer") or "").strip()
    ans = tail_answer(a) or (a.strip() if num(a.strip()) is not None else None)
    if not q or not ans or len(a) > 400:
        return None
    return {"instruction": q, "output": f"{a}\n答案是：\\boxed{{{ans}}}", "src": "tick"}


SOURCES = {
    # name: (hf repo, split, normalizer, row cap)
    "ape210k": ("MU-NLPC/Calc-ape210k", "train", ape210k, None),
    "belle": ("BelleGroup/school_math_0.25M", "train", belle, None),
    "gsm8k_zh": ("meta-math/GSM8K_zh", "train", gsm8k_zh, None),
    "math23k": ("Azure99/math23k-reborn", "train", math23k, None),
    "mxode": ("Mxode/School-Math-R1-Distil-Chinese-220K", "train", mxode, None),
    "tick": ("TICK666/Basic-Math-Chinese-1M-V1.1", "train", tick, 150_000),
}


def fetch(name):
    from datasets import load_dataset

    repo, split, norm, cap = SOURCES[name]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.jsonl")
    kept = seen = 0
    try:
        ds = load_dataset(repo, split=split, streaming=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in ds:
                seen += 1
                if cap and kept >= cap:
                    break
                d = norm(row)
                if d and not is_holdout(d["instruction"]):
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
                    kept += 1
    except Exception as e:  # noqa: BLE001 — report and continue with the other sources
        return f"{name}: FAILED {type(e).__name__}: {str(e)[:120]} (kept {kept}/{seen})"
    return f"{name}: {kept}/{seen} kept -> {path}"


if __name__ == "__main__":
    names = sys.argv[1:] or list(SOURCES)
    with ProcessPoolExecutor(max_workers=len(names)) as ex:
        for line in ex.map(fetch, names):
            print(line, flush=True)
