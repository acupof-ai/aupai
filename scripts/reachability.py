#!/usr/bin/env python3
"""Reachability analysis: which .py/.sh files are reachable from the entry points.

Entry points: AGENTS.md command blocks and tables, run_ddp.sh, CI workflow,
harness CHECKS, score_matrix registry. Reachability = entry point + transitive
imports (Python) and command citations (shell). Files with no entry point are
"none" -- the deletion candidates for t26.

Usage: python scripts/reachability.py > runs/reachability.txt
"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Collect all .py/.sh files (excluding noise)
ALL_FILES = set()
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in (".git", ".venv", "__pycache__", "node_modules", ".ruff_cache")]
    for fn in filenames:
        if fn.endswith((".py", ".sh")):
            ALL_FILES.add(os.path.relpath(os.path.join(dirpath, fn), ROOT))


def git_last_commit(path):
    r = subprocess.run(
        ["git", "log", "-1", "--format=%h %ad", "--date=short", "--", path],
        cwd=ROOT, capture_output=True, text=True,
    )
    return r.stdout.strip() or "never"


def file_lines(path):
    try:
        with open(os.path.join(ROOT, path), errors="ignore") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# --- Entry point collection ---

def agents_entry_points():
    """Scripts cited in AGENTS.md command blocks and tables."""
    eps = set()
    agents = os.path.join(ROOT, "AGENTS.md")
    if not os.path.exists(agents):
        return eps
    text = open(agents, encoding="utf-8").read()
    # Command blocks: ```bash ... ``` and inline `python scripts/x.py`
    for m in re.finditer(r"(?:scripts|eval|datagen|filters|mathbank|algorithms)/[\w./-]+\.(?:py|sh)", text):
        path = m.group(0)
        if path in ALL_FILES:
            eps.add(path)
    # Top-level entry points: train.py, sft.py, etc.
    for m in re.finditer(r"`([\w.-]+\.py)`", text):
        path = m.group(1)
        if path in ALL_FILES:
            eps.add(path)
    return eps


def run_ddp_entry_points():
    """Scripts called by run_ddp.sh."""
    eps = set()
    p = os.path.join(ROOT, "run_ddp.sh")
    if not os.path.exists(p):
        return eps
    text = open(p, errors="ignore").read()
    for m in re.finditer(r"[\w./-]+\.(?:py|sh)", text):
        path = m.group(0)
        if path in ALL_FILES:
            eps.add(path)
    return eps


def ci_entry_points():
    """Scripts called by CI workflows."""
    eps = set()
    ci_dir = os.path.join(ROOT, ".github", "workflows")
    if not os.path.isdir(ci_dir):
        return eps
    for fn in os.listdir(ci_dir):
        if fn.endswith((".yml", ".yaml")):
            text = open(os.path.join(ci_dir, fn), errors="ignore").read()
            for m in re.finditer(r"(?:scripts|eval|datagen|filters|mathbank|algorithms)/[\w./-]+\.(?:py|sh)", text):
                path = m.group(0)
                if path in ALL_FILES:
                    eps.add(path)
    return eps


def harness_entry_points():
    """Scripts referenced by harness CHECKS and pipeline steps."""
    eps = set()
    p = os.path.join(ROOT, "scripts", "harness.py")
    if not os.path.exists(p):
        return eps
    text = open(p, errors="ignore").read()
    # Pipeline step scripts: "fetch_corpus.py", "clean_corpus.py", etc.
    for m in re.finditer(r'"([\w.-]+\.py)"', text):
        path = os.path.join("scripts", m.group(1))
        if path in ALL_FILES:
            eps.add(path)
    # Scripts in check functions
    for m in re.finditer(r'"(scripts/[\w./-]+\.py)"', text):
        path = m.group(1)
        if path in ALL_FILES:
            eps.add(path)
    return eps


def score_matrix_entry_points():
    """Scripts in the score_matrix registry."""
    eps = set()
    p = os.path.join(ROOT, "scripts", "score_matrix.py")
    if not os.path.exists(p):
        return eps
    text = open(p, errors="ignore").read()
    for m in re.finditer(r'"(scripts/[\w./-]+\.py)"', text):
        path = m.group(1)
        if path in ALL_FILES:
            eps.add(path)
    for m in re.finditer(r'"(eval/[\w./-]+\.py)"', text):
        path = m.group(1)
        if path in ALL_FILES:
            eps.add(path)
    return eps


# --- Import graph ---

def python_imports(path):
    """Files imported by a Python file."""
    deps = set()
    try:
        text = open(os.path.join(ROOT, path), errors="ignore").read()
    except OSError:
        return deps
    for m in re.finditer(r"^\s*(?:from|import)\s+([\w.]+)", text, re.MULTILINE):
        mod = m.group(1)
        # Try module as file: scripts/harness.py -> harness
        candidates = [
            mod.replace(".", "/") + ".py",
            os.path.join("scripts", mod.split(".")[-1] + ".py"),
            os.path.join("eval", mod.split(".")[-1] + ".py"),
            os.path.join("algorithms", mod.split(".")[-1] + ".py"),
            mod.split(".")[-1] + ".py",
        ]
        for c in candidates:
            if c in ALL_FILES:
                deps.add(c)
                break
    return deps


def shell_calls(path):
    """Scripts called by a shell script."""
    deps = set()
    try:
        text = open(os.path.join(ROOT, path), errors="ignore").read()
    except OSError:
        return deps
    for m in re.finditer(r"(?:scripts|eval|datagen|filters|mathbank|algorithms)/[\w./-]+\.(?:py|sh)", text):
        dep = m.group(0)
        if dep in ALL_FILES and dep != path:
            deps.add(dep)
    for m in re.finditer(r"(?<![\w/])([\w.-]+\.py)", text):
        dep = m.group(1)
        if dep in ALL_FILES and dep != path:
            deps.add(dep)
    return deps


def reachable_from(entry_points):
    """BFS from entry points through imports/calls."""
    seen = set(entry_points)
    queue = list(entry_points)
    while queue:
        f = queue.pop()
        if f.endswith(".py"):
            deps = python_imports(f)
        elif f.endswith(".sh"):
            deps = shell_calls(f)
        else:
            deps = set()
        for d in deps:
            if d not in seen:
                seen.add(d)
                queue.append(d)
    return seen


# --- Main ---

def main():
    eps = set()
    sources = {
        "AGENTS.md": agents_entry_points(),
        "run_ddp.sh": run_ddp_entry_points(),
        "CI": ci_entry_points(),
        "harness": harness_entry_points(),
        "score_matrix": score_matrix_entry_points(),
    }
    for src, files in sources.items():
        for f in files:
            eps.add(f)

    reachable = reachable_from(eps)

    # Map each file to the entry point that reaches it
    def reaching_ep(path):
        if path in eps:
            return "ENTRY"
        # Find which entry point's BFS reaches it
        for ep in sorted(eps):
            if path in reachable_from({ep}):
                return ep
        return "none"

    print(f"{'PATH':<55} {'LINES':>6}  {'LAST COMMIT':<20}  REACHED FROM")
    print("-" * 110)
    unreachable = []
    for f in sorted(ALL_FILES):
        lines = file_lines(f)
        commit = git_last_commit(f)
        ep = reaching_ep(f)
        if ep == "none":
            unreachable.append(f)
        print(f"{f:<55} {lines:>6}  {commit:<20}  {ep}")

    print(f"\n{'='*110}")
    print(f"Total: {len(ALL_FILES)} files, {len(eps)} entry points, "
          f"{len(reachable)} reachable, {len(unreachable)} unreachable")
    if unreachable:
        print("\nUnreachable (deletion candidates):")
        for f in unreachable:
            print(f"  {f}")


if __name__ == "__main__":
    main()
