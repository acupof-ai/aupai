#!/usr/bin/env python3
"""Reachability analysis: which .py/.sh files are reachable from the entry points.

Edge kinds:
  ENTRY    — cited in AGENTS.md tables, run_ddp.sh, CI, harness, score_matrix
  import   — Python import or shell command citation (transitive, BFS from ENTRY)
  registry — dynamic dispatch: run_eval._load_module, algorithms lazy-import table
  docs     — cited in docs/**, AGENTS.md, EXPERIMENTS.md
  facts    — cited in a facts/*.json source field
  exps     — cited in a runs/experiments.jsonl cmd field

Files with no edge are "none" — the deletion candidates for t26.

Usage: python scripts/reachability.py > runs/reachability.txt
"""
import glob
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fate rulings from fb (2026-08-31). DELETE files are cited by no doc, fact, run
# row, or registry. KEEP files must become reachable or return to the list.
FATE = {
    "algorithms/prepare_rlvr.py": "DELETE",
    "bench_mem.py": "DELETE",
    "data/sft/quality_check.py": "DELETE",
    "filters/clean_school_math.py": "DELETE (filters_fp hash change — ratchet baseline)",
    "scripts/attn_every_probe.py": "DELETE",
    "scripts/attn_res_gap.py": "DELETE",
    "scripts/audit_ocsg.py": "DELETE",
    "scripts/chat_remote.py": "DELETE",
    "scripts/ckpt_diff.py": "DELETE",
    "scripts/dashboard.py": "DELETE (superseded by harness board)",
    "scripts/fone_probe.py": "DELETE",
    "scripts/ocsg_determ.py": "DELETE",
    "scripts/probe_arith.py": "DELETE",
    "scripts/probe_procedure.py": "DELETE",
    "scripts/repeat_check.py": "DELETE",
    "scripts/rescore_v2.py": "DELETE",
    "scripts/rl_delta_cos.py": "DELETE",
    "scripts/sandbox_batch.py": "DELETE",
    "scripts/short_conv_bench.py": "DELETE",
    "scripts/train_vocab_variants.py": "DELETE",
    "scripts/reachability.py": "KEEP (add to AGENTS.md entry-point table)",
    "scripts/count_cleaned_code.py": "KEEP (add to AGENTS.md entry-point table)",
    "algorithms/test_rlvr_reward_suite.py": "KEEP (add to CI)",
    "mathbank/vet_programs.py": "KEEP (cite in corpus entry-point row)",
    "scripts/ckpt_info.py": "KEEP (AGENTS.md row — ops tool)",
    "eval/ppl.py": "KEEP (AGENTS.md row — eval tool)",
    "scripts/assemble_lambda_probe.py": "KEEP (3b's t05, deprioritised, live)",
    "scripts/validate_lambda_probe.py": "KEEP (3b's t05, deprioritised, live)",
    "scripts/build_math.py": "DELETE (3b confirmed unclaimed, 2026-08-31)",
}

# Collect all .py/.sh files (excluding noise)
ALL_FILES = set()
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in (".git", ".venv", "__pycache__", "node_modules", ".ruff_cache")]
    for fn in filenames:
        if fn.endswith((".py", ".sh")):
            ALL_FILES.add(os.path.relpath(os.path.join(dirpath, fn), ROOT))

SCRIPT_RE = re.compile(r"(?:scripts|eval|datagen|filters|mathbank|algorithms|probes)/[\w./-]+\.(?:py|sh)")
TOPLEVEL_RE = re.compile(r"(?<![\w/])([\w.-]+\.(?:py|sh))")

# Basename -> full path, for resolving bare names like "rlvr.py" -> "algorithms/rlvr.py"
BASENAME_INDEX = {}
for _f in ALL_FILES:
    BASENAME_INDEX.setdefault(os.path.basename(_f), _f)


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


def _read(path):
    try:
        return open(os.path.join(ROOT, path), encoding="utf-8", errors="ignore").read()
    except OSError:
        return ""


def _resolve_script(text):
    """Find script paths in text that match ALL_FILES. Bare names resolve via basename index."""
    found = set()
    for m in SCRIPT_RE.finditer(text):
        if m.group(0) in ALL_FILES:
            found.add(m.group(0))
    for m in TOPLEVEL_RE.finditer(text):
        name = m.group(1)
        if name in ALL_FILES:
            found.add(name)
        elif name in BASENAME_INDEX:
            found.add(BASENAME_INDEX[name])
    return found


# --- Entry point collection ---

def agents_entry_points():
    return _resolve_script(_read("AGENTS.md"))


def run_ddp_entry_points():
    return _resolve_script(_read("run_ddp.sh"))


def ci_entry_points():
    eps = set()
    ci_dir = os.path.join(ROOT, ".github", "workflows")
    if not os.path.isdir(ci_dir):
        return eps
    for fn in os.listdir(ci_dir):
        if fn.endswith((".yml", ".yaml")):
            eps |= _resolve_script(_read(os.path.join(".github", "workflows", fn)))
    return eps


def harness_entry_points():
    return _resolve_script(_read("scripts/harness.py"))


def score_matrix_entry_points():
    return _resolve_script(_read("scripts/score_matrix.py"))


# --- Import graph (transitive) ---

def python_imports(path):
    deps = set()
    text = _read(path)
    for m in re.finditer(r"^\s*(?:from|import)\s+([\w.]+)", text, re.MULTILINE):
        mod = m.group(1)
        # Try module as file: scripts/harness.py -> harness
        candidates = [
            mod.replace(".", "/") + ".py",
            os.path.join("scripts", mod.split(".")[-1] + ".py"),
            os.path.join("eval", mod.split(".")[-1] + ".py"),
            os.path.join("algorithms", mod.split(".")[-1] + ".py"),
            os.path.join("mathbank", mod.split(".")[-1] + ".py"),
            mod.split(".")[-1] + ".py",
        ]
        for c in candidates:
            if c in ALL_FILES:
                deps.add(c)
                break
        else:
            # Bare module name -> basename index (e.g. mathcommon -> mathbank/mathcommon.py)
            basename = mod.split(".")[-1] + ".py"
            if basename in BASENAME_INDEX:
                deps.add(BASENAME_INDEX[basename])
    return deps


def shell_calls(path):
    text = _read(path)
    deps = _resolve_script(text)
    return {d for d in deps if d != path}


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


# --- Additional citation edges (non-transitive) ---

def registry_edges():
    """Dynamic dispatch: run_eval._load_module, algorithms lazy-import table."""
    edges = {}
    # eval/run_eval.py: _load_module("name") -> eval/name.py
    text = _read("eval/run_eval.py")
    for m in re.finditer(r'_load_module\("([\w.]+)"\)', text):
        path = os.path.join("eval", m.group(1) + ".py")
        if path in ALL_FILES:
            edges[path] = ("registry", "eval/run_eval.py")
    # MC_BENCHMARKS keys: "ceval", "gsm8k", etc.
    for m in re.finditer(r'"(\w+)":\s*\(', text):
        path = os.path.join("eval", m.group(1) + ".py")
        if path in ALL_FILES:
            edges.setdefault(path, ("registry", "eval/run_eval.py"))
    # algorithms/__init__.py: _LAZY dict mapping to module names
    text = _read("algorithms/__init__.py")
    for m in re.finditer(r'"([\w.]+)":\s*"([\w.]+)"', text):
        mod = m.group(2)
        path = os.path.join("algorithms", mod + ".py")
        if path in ALL_FILES:
            edges.setdefault(path, ("registry", "algorithms/__init__.py"))
    # mathbank generator registry: ["math_programs_l1", ...] in vet_programs/run_math_short
    for mb in ("mathbank/vet_programs.py", "mathbank/run_math_short.py"):
        text = _read(mb)
        for m in re.finditer(r'"(math_programs_[\w.]+)"', text):
            path = os.path.join("mathbank", m.group(1) + ".py")
            if path in ALL_FILES:
                edges.setdefault(path, ("registry", mb))
    # mathbank glob dispatch: run_math_short.py globs math_programs_l*_ext*.py
    for f in ALL_FILES:
        if f.startswith("mathbank/math_programs_l") and "_ext" in f:
            edges.setdefault(f, ("registry", "mathbank/run_math_short.py"))
        if f.startswith("mathbank/math_programs_short_"):
            edges.setdefault(f, ("registry", "mathbank/run_short_sol.py"))
    # harness _TRAINING_PROCS: string references to training scripts
    text = _read("scripts/harness.py")
    for m in re.finditer(r'"([\w./-]+\.py)"', text):
        name = m.group(1)
        path = name if name in ALL_FILES else BASENAME_INDEX.get(name)
        if path:
            edges.setdefault(path, ("registry", "scripts/harness.py"))
    # build_corpus source handlers: --source fineweb2 etc. are data sources, not scripts.
    # But build_corpus imports filters and datagen modules — covered by python_imports.
    return edges


def docs_edges():
    """Scripts cited in docs/**, AGENTS.md, EXPERIMENTS.md, and shell scripts."""
    edges = {}
    docs = glob.glob(os.path.join(ROOT, "docs", "**", "*.md"), recursive=True)
    docs += [os.path.join(ROOT, "AGENTS.md"), os.path.join(ROOT, "EXPERIMENTS.md")]
    # Shell scripts cite generators in comments (e.g. build_math_expand.sh -> mathbank/)
    docs += glob.glob(os.path.join(ROOT, "scripts", "*.sh"))
    docs += glob.glob(os.path.join(ROOT, "*.sh"))
    for doc in docs:
        rel = os.path.relpath(doc, ROOT)
        text = _read(rel)
        for f in _resolve_script(text):
            edges.setdefault(f, ("docs", rel))
    return edges


def facts_edges():
    """Scripts cited in JSON files (facts/*.json, scripts/*_baseline.json, etc.)."""
    edges = {}
    for pattern in ("facts/*.json", "scripts/*_baseline.json", "data/*.json"):
        for path in glob.glob(os.path.join(ROOT, pattern)):
            rel = os.path.relpath(path, ROOT)
            try:
                text = open(path, encoding="utf-8").read()
            except OSError:
                continue
            for f in _resolve_script(text):
                edges.setdefault(f, ("facts", rel))
    return edges


def experiments_edges():
    """Scripts cited in runs/experiments.jsonl cmd fields."""
    edges = {}
    p = os.path.join(ROOT, "runs", "experiments.jsonl")
    if not os.path.isfile(p):
        return edges
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        cmd = row.get("cmd", "")
        for f in _resolve_script(cmd):
            edges.setdefault(f, ("exps", "experiments.jsonl"))
    return edges


# --- Main ---

def main():
    eps = set()
    for fn in (agents_entry_points, run_ddp_entry_points, ci_entry_points,
               harness_entry_points, score_matrix_entry_points):
        eps |= fn()

    bfs_reachable = reachable_from(eps)

    # Collect additional citation edges
    all_edges = {}
    for edge_fn in (registry_edges, docs_edges, facts_edges, experiments_edges):
        for f, (kind, source) in edge_fn().items():
            all_edges.setdefault(f, (kind, source))

    # Citation edges are transitive: a file reached via registry/docs/facts/exps
    # also reaches everything it imports.
    citation_reached = set(all_edges.keys()) - bfs_reachable
    if citation_reached:
        bfs_reachable |= reachable_from(citation_reached)

    # Package __init__.py is reached when any module in the package is reached.
    for f in list(bfs_reachable | set(all_edges.keys()) | eps):
        pkg = os.path.dirname(f)
        init = os.path.join(pkg, "__init__.py")
        if init in ALL_FILES and init not in bfs_reachable and init not in all_edges:
            all_edges[init] = ("import", f)

    # Determine reaching edge for each file
    def reaching(path):
        if path in eps:
            return "ENTRY"
        if path in all_edges:
            kind, source = all_edges[path]
            return f"{kind}:{source}"
        if path in bfs_reachable:
            # Find which entry point reaches it via BFS
            for ep in sorted(eps):
                if path in reachable_from({ep}):
                    return f"import:{ep}"
            return "import"
        return "none"

    print(f"{'PATH':<55} {'LINES':>6}  {'LAST COMMIT':<20}  {'REACHED FROM':<45}  FATE")
    print("-" * 130)
    unreachable = []
    for f in sorted(ALL_FILES):
        lines = file_lines(f)
        commit = git_last_commit(f)
        ep = reaching(f)
        fate = FATE.get(f, "")
        if ep == "none":
            unreachable.append(f)
        print(f"{f:<55} {lines:>6}  {commit:<20}  {ep:<45}  {fate}")

    print(f"\n{'='*130}")
    delete_count = sum(1 for f in unreachable if FATE.get(f, "").startswith("DELETE"))
    keep_count = sum(1 for f in unreachable if FATE.get(f, "").startswith("KEEP"))
    print(f"Total: {len(ALL_FILES)} files, {len(eps)} entry points, "
          f"{len(bfs_reachable)} BFS-reachable, {len(all_edges)} citation edges, "
          f"{len(unreachable)} unreachable ({delete_count} DELETE, {keep_count} KEEP)")
    if unreachable:
        print("\nUnreachable (deletion candidates):")
        for f in unreachable:
            print(f"  {f:<55} {FATE.get(f, '')}")


if __name__ == "__main__":
    main()
