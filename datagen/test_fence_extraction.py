#!/usr/bin/env python3
"""Known-answer tests for vet_textbooks fence extraction.

The fence opener/closer must be LINE-ANCHORED (CommonMark), identical in
semantics to shared/gate_chapter.py. A triple-backtick sequence inside a code or
prose line must not close a fence early — that bug mis-segmented 53 textbook
chapters (45 live / ~159K gate tokens) on 2026-09-14.

Run: python datagen/test_fence_extraction.py   (stdlib only; exit 1 on failure)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vet_textbooks as vt


def py_blocks(md):
    return vt.code_blocks(md)[0]


def check(name, cond):
    if not cond:
        print("FAIL:", name)
        return False
    print("ok:", name)
    return True


def main():
    ok = True

    # 1) single opener, no close -> unclosed; truncated gate drops it, no python run.
    single = "intro\n```python\nprint(1)\n"
    ok &= check("single opener yields 0 blocks (unclosed)", py_blocks(single) == [])
    ok &= check("single opener is truncated", vt.truncated(single) is True)

    # 2) odd fence count (text close, then a stray python opener) -> unclosed/truncated.
    odd = "```text\nx\n```\nmid\n```python\nprint('a')\n"
    ok &= check("odd fences yield 0 blocks", py_blocks(odd) == [])
    ok &= check("odd fences are truncated", vt.truncated(odd) is True)

    # 3) non-python language does not swallow following python blocks.
    mixed = ("```text\nn\n```\n```python\nprint('PY1')\n```\n"
             "```c\nint x;\n```\n```python\nprint('PY2')\n```\n")
    ok &= check("text->py->c->py extracts both py blocks",
                [b.strip() for b in py_blocks(mixed)] == ["print('PY1')", "print('PY2')"])
    ok &= check("well-formed mixed chapter is not truncated", vt.truncated(mixed) is False)

    # 4) REGRESSION: triple-backtick INSIDE a code line (inline literal) must not
    # close the fence; the whole block survives as one python block.
    inline = "```python\nx = 'a ``` b'\nprint(x)\n```\n"
    ok &= check("inline triple-backtick keeps one whole block",
                py_blocks(inline) == ["x = 'a ``` b'\nprint(x)"])
    ok &= check("inline-backtick chapter is not truncated", vt.truncated(inline) is False)

    # 5) REGRESSION: a ```text demo showing code-looking tokens (JWT segments)
    # must be skipped as non-python and NOT split into a spurious python block.
    jwt = ("```text\nheader.payload.signature\n```\n"
           "```python\ndef f():\n    return 1\n```\n")
    blocks = py_blocks(jwt)
    ok &= check("JWT text demo yields exactly one py block", len(blocks) == 1)
    ok &= check("JWT py block is the real function, not the segment string",
                blocks[0].strip() == "def f():\n    return 1")

    # 6) opener must start in column 0 (same as gate_chapter): an indented
    # ``` does not open a fence, so the inner backticks are ordinary text.
    indented = "   ```python\nprint(1)\n   ```\n"
    ok &= check("indented fence is not treated as python", py_blocks(indented) == [])

    # 7) a BARE (no-lang) fence is not assumed python (its closed bodies are
    # mostly tables/hex/math/pseudocode); same for python3 which the exec gate
    # does not tag. Only ```python / ```py execute.
    bare = "```\nprint('bare')\n```\n"
    ok &= check("bare fence is not executed as python", py_blocks(bare) == [])
    py3 = "```python3\nprint('three')\n```\n"
    ok &= check("python3-tagged fence is not executed (gate uses python/py only)",
                py_blocks(py3) == [])
    pytag = "```py\nprint('pyalias')\n```\n"
    ok &= check("py alias IS executed",
                [b.strip() for b in py_blocks(pytag)] == ["print('pyalias')"])

    print("ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
