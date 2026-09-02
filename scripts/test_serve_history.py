#!/usr/bin/env python3
"""selftest for serve.format_history -- it must delegate to pure ChatML.

Failure case this guards (deletion audit 2026-09-02, defect 2): serve's body
called loader.format_prompt (ChatML) and then appended a literal '答：', a
hybrid the SFT data never contained. serve.py loads a checkpoint at import
time, so this test asserts on the source of truth (loader.format_history) and
on serve.py's source, not by importing serve.

    python3 scripts/test_serve_history.py --selftest
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.loader import IM_START, format_history


def test_chatml_format():
    out = format_history([
        {"role": "user", "content": "1+1=?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "why"},
    ])
    assert out.count(IM_START) == 4, out
    assert "答：" not in out, out
    assert out.endswith(f"{IM_START}assistant\n"), out[-40:]


def test_serve_delegates_without_answering():
    src = open(os.path.join(ROOT, "serve.py"), encoding="utf-8").read()
    code = "\n".join(
        l for l in src.splitlines()
        if l.strip() and not l.lstrip().startswith("#") and not l.lstrip().startswith('"')
    )
    assert "答：" not in code, "serve.py code still emits the literal 答："
    assert re.search(r"format_history\s*\(", src), "serve.py no longer calls format_history"


if __name__ == "__main__":
    test_chatml_format()
    test_serve_delegates_without_answering()
    print("selftest OK: loader.format_history is pure ChatML; serve.py delegates, no 答：")
