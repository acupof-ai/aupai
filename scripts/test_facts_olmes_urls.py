#!/usr/bin/env python3
"""selftest for 44-14 defect 4: be.gold_bpb_method cited files in AI2's olmes
repo as bare paths, which read as local files and resolve to nothing. External
sources must carry their URL.

    python3 scripts/test_facts_olmes_urls.py
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
text = open(os.path.join(ROOT, "facts/base_eval.json"), encoding="utf-8").read()

assert "github.com/allenai/olmes" in text, "olmes citations lack the repo URL"
bare = re.sub(r"https?://\S+", "", text)  # strip URLs; the path inside the URL is fine
assert "oe_eval/" not in bare, "bare olmes path outside a URL still reads as a local file"

print("selftest OK: olmes citations carry their URL")
