#!/usr/bin/env python3
import json
import sys

tot = 0
by_src = {}
mc = 0  # src contains 'math'
with open(sys.argv[1], encoding="utf-8") as _f:
    for line in _f:
        if not line.strip():
            continue
        tot += 1
    d = json.loads(line)
    s = d.get("src") or ""
    by_src[s] = by_src.get(s, 0) + 1
    if "math" in s.lower():
        mc += 1
top = sorted(by_src.items(), key=lambda x: -x[1])[:10]
print(f"total_rows={tot}")
print("src_top=", top)
print(f"rows_src_contains_math={mc}  share_of_total={mc/tot:.6f} ({mc/tot:.4%})")
