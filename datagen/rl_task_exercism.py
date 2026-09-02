#!/usr/bin/env python3
# 3b-9: scrape Exercism python track into RL task sets. Host-side, no GPU.
"""Exercism python track -> data/rl_tasks/exercism/ as {id, prompt, src_files, hidden_tests, n_tests}.

Source (github raw, 200/0.4s measured 2026-09-02): exercises/{practice,concept}/<slug> with
  .docs/instructions.md  -> task prompt
  .meta/example.py       -> reference impl
  <slug>_test.py         -> canonical hidden tests, self-contained unittest
Each slug becomes one RL task. Host admission: n_tests>=5 AND reference passes hidden_tests.

Raw fetches are rate-fair; the github contents API is 60/h unauthenticated, so we hit it
ONCE for the slug list and use raw for everything else, assuming example.py for reference
(raw 404 -> fall back to a one-off contents API call for the .meta listing).
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

API = "https://api.github.com/repos/exercism/python/contents"
RAW = "https://raw.githubusercontent.com/exercism/python/main"
OUT = os.path.join("data", "rl_tasks", "exercism")


def gh(path):
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return json.load(r)


def raw(path):
    req = urllib.request.Request(RAW + path)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def raw_404(path):
    try:
        raw(path)
        return False
    except urllib.error.HTTPError as e:
        return e.code == 404


def slug_test_file(slug):
    return slug.replace("-", "_") + "_test.py"


def count_tests(test_src):
    return len(re.findall(r"def test_\w+\s*\(", test_src))


def reference_passes(slug, test_src, ref_src):
    """Run the canonical test against the reference in a temp dir; rc 0 = passes.
    Trusted source (Exercism-reviewed examples), host-side non-sandboxed."""
    import tempfile
    d = tempfile.mkdtemp()
    try:
        with open(os.path.join(d, slug.replace("-", "_") + ".py"), "w") as f:
            f.write(ref_src)
        with open(os.path.join(d, slug_test_file(slug)), "w") as f:
            f.write(test_src)
        p = subprocess.run([sys.executable, os.path.join(d, slug_test_file(slug))],
                           cwd=d, capture_output=True, timeout=60)
        return p.returncode == 0
    except Exception:
        return False
    finally:
        import shutil
        shutil.rmtree(d)


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks, issues = [], []
    for track in ("practice", "concept"):
        slugs = [s["name"] for s in gh(f"/exercises/{track}")]
        for slug in slugs:
            tfile = slug_test_file(slug)
            try:
                inst = raw(f"/exercises/{track}/{slug}/.docs/instructions.md")
            except Exception as e:
                issues.append({"slug": slug, "track": track, "err": f"instructions: {e}"})
                continue
            try:
                test_src = raw(f"/exercises/{track}/{slug}/{tfile}")
            except Exception as e:
                issues.append({"slug": slug, "track": track, "err": f"test: {e}"})
                continue
            ref_file = "example.py"
            try:
                ref_src = raw(f"/exercises/{track}/{slug}/.meta/{ref_file}")
            except Exception:
                # fall back: name the reference from .meta listing (rare API call)
                try:
                    meta = [x["name"] for x in gh(f"/exercises/{track}/{slug}/.meta")]
                    ref_file = next((m for m in meta if m.endswith(".py") and m != "tests.toml" and "template" not in m), None)
                    if not ref_file:
                        raise FileNotFoundError("no reference py in .meta")
                    ref_src = raw(f"/exercises/{track}/{slug}/.meta/{ref_file}")
                except Exception as e:
                    issues.append({"slug": slug, "track": track, "err": f"reference: {e}"})
                    continue
            n = count_tests(test_src)
            ok = n >= 5 and reference_passes(slug, test_src, ref_src)
            tasks.append({
                "id": f"exercism-{slug}",
                "track": track,
                "prompt": inst,
                "src_files": {slug.replace("-", "_") + ".py": ref_src},
                "hidden_tests": {tfile: test_src},
                "n_tests": n,
                "reference_passes": ok,
                "license": "MIT",
                "source": f"https://github.com/exercism/python/tree/main/exercises/{track}/{slug}",
            })
            print(f"{'PASS' if ok else 'fail'} {track}/{slug} n={n} ref={ref_file}", flush=True)
            time.sleep(0.1)

    admitted = [t for t in tasks if t["reference_passes"] and t["n_tests"] >= 5]
    with open(os.path.join(OUT, "tasks.jsonl"), "w", encoding="utf-8") as f:
        for t in tasks:
            json.dump(t, f, ensure_ascii=False)
            f.write("\n")
    with open(os.path.join(OUT, "issues.jsonl"), "w", encoding="utf-8") as f:
        for i in issues:
            json.dump(i, f, ensure_ascii=False)
            f.write("\n")
    print(json.dumps({
        "scraped": len(tasks), "admitted": len(admitted),
        "practice": sum(1 for t in tasks if t["track"] == "practice"),
        "concept": sum(1 for t in tasks if t["track"] == "concept"),
        "issues": len(issues), "out": OUT,
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()