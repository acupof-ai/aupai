#!/usr/bin/env python3
"""selftest for the four PR #169 blocking findings fixed on ae-1.

python3 scripts/test_p1_tokenizer_scripts.py
"""

import contextlib
import io
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_p1_tokenizer as B  # noqa: E402

# finding 2: held-out textbooks are disjoint from the fit prefix BY CONSTRUCTION
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
    for i in range(100):
        f.write(json.dumps({"text": f"{i:04d}" + "x" * 996}) + "\n")  # 1000 bytes each
    tb_path = f.name
try:
    fit = B.textbook_texts(tb_path, max_bytes=40_000)
    assert len(fit) == 40, f"fit read {len(fit)} chapters, expected 40"
    ho = B.textbook_texts(tb_path, max_bytes=1_000_000, skip_bytes=40_000)
    assert len(ho) == 60, f"held-out read {len(ho)} chapters, expected 60"
    assert not set(fit) & set(ho), "held-out overlaps the fit prefix"
    assert B.textbook_texts(tb_path, max_bytes=1_000_000, skip_bytes=100_000) == [], (
        "a file exhausted by the fit prefix must yield an empty held-out, not fit data"
    )
finally:
    os.remove(tb_path)

# finding 1: one tax convention -- candidate/frozen - 1, positive = frozen costs more
src = open(os.path.join(ROOT, "scripts", "build_p1_tokenizer.py"), encoding="utf-8").read()
assert 'm_new["chars/token"] / m_frz["chars/token"] - 1' in src, "tax ratio must be candidate/frozen - 1"
assert 'f"{100 * tax:+.1f}%"' in src, "tax must print with its own sign, no hardcoded '+'"
real_src = open(os.path.join(ROOT, "scripts", "tokenizer_p1_real.py"), encoding="utf-8").read()
assert real_src.count("(sum(c) / len(c)) / (sum(f) / len(f)) - 1") == 1, "real script keeps its convention"
tax = 3.265 / 3.164 - 1  # the fact's own numbers
assert abs(tax - 0.032) < 0.001 and tax > 0, (
    "the fact's +3.2~3.4% must come out positive under this convention"
)

# finding 3: gates fail loud -- gate_failures names every veto, and the final path
# appears only via os.replace after the gate loop
m_ok, g_ok = {"ref fertility": 1.4}, {"round-trip lossless": True, "_bytes": 256}
assert B.gate_failures("x", m_ok, g_ok) == []
assert B.gate_failures("x", {"ref fertility": 1.6}, g_ok), "ref fertility above 1.55 must fail"
assert B.gate_failures("x", m_ok, {"round-trip lossless": False, "_bytes": 256}), "round-trip must fail"
assert B.gate_failures("x", m_ok, {"round-trip lossless": True, "_bytes": 255}), "a dropped byte must fail"
assert "tok.save(tmp)" in src and "os.replace(tmp, a.out)" in src, "vocab must land via tmp + replace"
assert "tok.save(a.out)" not in src, "a vocab must never be written to the final path before gating"

# nit: sample_domain_mix warns when the code pool cannot reach the 88:12 ratio
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import tokenizer_p1_real as R  # noqa: E402

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    out = R.sample_domain_mix(["a" * 1000], tb_chars=10_000_000, rng=__import__("random").Random(0))
assert len(out) == 1 and "WARNING" in buf.getvalue(), "an undershot 88:12 mix must warn, not pass silently"

print("selftest OK: held-out disjoint by construction, one tax convention, gates fail loud")
