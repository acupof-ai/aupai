#!/usr/bin/env python3
"""Reachability analysis: which .py/.sh files are reachable from the entry points.

Edge kinds:
  ENTRY    — cited in AGENTS.md tables, run_ddp.sh, CI, harness, score_matrix, or run by
             the pre-commit hook's SELFTEST_FILES map
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
    "datagen/count_cleaned_code.py": "KEEP (add to AGENTS.md entry-point table)",
    "algorithms/test_rlvr_reward_suite.py": "KEEP (add to CI)",
    "mathbank/vet_programs.py": "KEEP (cite in corpus entry-point row)",
    "scripts/ckpt_info.py": "KEEP (AGENTS.md row — ops tool)",
    "eval/ppl.py": "KEEP (AGENTS.md row — eval tool)",
    "eval/assemble_lambda_probe.py": "KEEP (3b's t05, deprioritised, live)",
    "eval/validate_lambda_probe.py": "KEEP (3b's t05, deprioritised, live)",
    "scripts/build_math.py": "DELETE (3b confirmed unclaimed, 2026-08-31)",
    # N6 pass, de 2026-09-03. Every one of the 23 was RUN before it was judged, and the
    # per-directory glob/importlib grep found no runtime loader reaching any of them --
    # the bare `*.py` globs in harness.py and eval_load_cost.py are tree walks that
    # enumerate for reporting, not loaders, so treating them as edges marks all 23 live
    # and is a criterion that says yes to everything.
    #
    # Citations were counted EXCLUDING data/pod_head_manifest.txt, runs/reachability.txt,
    # the pod inventories and the previous deletion audit: those list files by name by
    # construction, so counting them makes every file in the tree look cited. On that
    # basis exactly four of the 23 have a real citation, and zero are named by any fact.
    #
    # KEEP: a live test of live code. Each one RUNS GREEN here and asserts something no
    # other test covers.
    "scripts/test_e1_28_leak_scan.py": "KEEP (green; e1-28 open; gram width + refuse-on-missing-field)",
    "scripts/test_e1_28_matched.py": "KEEP (green; e1-28 open; universal-form exclusion, doubt=contamination)",
    "scripts/test_e1_29_floor_by_class.py": "KEEP (green; e1-29 open, N3 row; cited by runs/review.jsonl:150)",
    "datagen/test_near_dedup_known.py": "KEEP (green, PASS with known pos/neg; the only known-answer case for near-dedup)",
    "scripts/test_attn_res_fp32_logits.py": "KEEP (green from repo root; the only check that --attn_res_fp32_logits is not inert)",
    "scripts/attnres_logits_reference.py": "KEEP (green, rejects all three controls; cited by algorithms/attnres_fused.py:11)",
    # KEEP: named by an owner's open task or an N row, so deleting it deletes work in flight.
    "datagen/build_code_tests_v1.py": "KEEP (3b's N4 code_tests Phase A, committed today with a wip marker)",
    "scripts/e1_28_matched.py": "KEEP (e1-28 open; test_e1_28_matched.py is its test)",
    "scripts/e1_30_case_table.py": "KEEP (e1-30 open, N5 row; runs/e1_30_case_table.md is its output)",
    "datagen/numma_to_jsonl.py": "KEEP (cited by data/PROVENANCE.md as the numina converter)",
    "datagen/code_dedup_build.py": "KEEP (3b ruled 2026-09-03: the executing half of N4's MinHash 0.8 cross-source dedup; 3b-10's code_dedup08 is its output, near_dedup_scale.py is its report side)",
    "datagen/rl_task_exercism.py": "KEEP (3b ruled 2026-09-03: the exercism RL task-set source, 3b-9; RL is scheduled after N5, which retires the schedule slot and not the task set)",
    # e1/58's ruling, 2026-09-03, and its CRITERION SUPERSEDES THE ONE ABOVE: a file is
    # KEEP because it is the sole producer of a PUBLISHED number, not because a task is
    # still open. Closing a task is not a reason to delete its production path -- the
    # number stays published either way, and deleting the producer downgrades it from
    # recomputable to merely re-readable. Verified against the audit before recording.
    "scripts/e1_27_read.py": "KEEP (sole producer of control_pythia160m_vs_ours.md:76's 0.293989 denominator evidence -- the lr_scale 0.1 bit-identical rerun plus the lr_scale 1.0 negative control; train.py:848 stores lr_scale in neither Cfg nor the checkpoint, so nothing else can recover it)",
    "scripts/e1_27_score.sh": "KEEP (produces the log e1_27_read.py reads; keeping the reader without the producer is recomputable downgraded to re-readable)",
    "scripts/e1_28_clean_score.sh": "KEEP (sole producer of control_pythia160m_vs_ours.md:562's clean-subset verdict -- 10,105 ids, sha 7231156c5698c210, floor 2.0243x, lead 16.6290%; its docstring also carries the 1e scope ruling on WHICH three of the nine points were recomputed, which lives nowhere else)",
    # b0/62's ruling, 2026-09-03, after running all six on the pod. TWO OF MY "same-name
    # trap" readings were BACKWARDS: I read a citation that does not match the path as
    # naming a different file, when it was the CITATION that carried the typo. Checking
    # which of the two is wrong takes one `git log --all` per path, and I skipped it.
    "scripts/attnres_bench.py": "KEEP (it IS the file docs/lessons/fused_attnres_is_slower_in_torch.md cites: both created by 3ed56306, its docstring is the source of that doc's 'nothing is imported' line, and its :144-149 print columns are the doc's table header. The doc's /tmp/ path is where it lived before entering git, not another file)",
    "scripts/logit_dist.py": "KEEP (docs/standards/structure_experiments.md:529 cites scripts/_logit_dist.py, which has never existed on any branch -- `git log --all` is empty for it. The underscore belongs to the OUTPUT: this file's :112 writes runs/_logit_dist.json, whose head_row_norm_med 76.61315 and cos_to_argmax_row_mean 0.22061723 are that doc table's 76.61 and 0.2206, field names identical)",
    "scripts/attnres_triton_gate.py": "KEEP (PASSes on the pod at 3 shapes, bar 1e-05; ModuleNotFoundError here is its hardcoded sys.path.insert(0, '/work/aupai') at :12, so a Mac run cannot work by construction. Its :5 names attnres_triton_bf16_gate.py -- cited by docs/lessons/upper_bound_is_not_an_effect.md:4 -- and explains the split of labour, so deleting the fp32 half leaves the bf16 half's docstring pointing at a control that does not exist)",
    "bench_eff/parse_ddp.py": "KEEP (runs on the pod and produces its result: 'HtoD memcpy: 0.00 ms/step (0.00%)'. My 'missing trace file' reading was of a Mac run only)",
    "bench_eff/parse_kernels.py": "KEEP (ran it on the pod: prints 'total GPU kernel time: 2074.5 ms/step (20 active steps)' then the per-kernel table, e.g. 247.9 ms x3000 nvjet_qqtst_128x160 under 'FP8 linear (cuBLASLt scaled_mm): 657.8 ms/step (31.7%)'. CPU-only, no card. Its input /work/aupai/bench_eff/ddp_trace_rank0.json is 415,798,937 B on the pod, 2026-08-31 21:51, uncommitted by design -- so a Mac run raises FileNotFoundError and reads as broken, same misreading as parse_ddp.py above. It is the per-kernel resolution behind facts/efficiency.json's trace attributions)",
    "bench_eff/parse_trace.py": "KEEP (ran it on the pod with the rank0 trace as argv[1]: 'total GPU kernel time: 41.5 ms over 20 steps = 2074.53 ms/step' and the category table -- other 36.9%, triton_compiled 24.7%, ddp_comm 17.5%, kda_kernel 3.5%, attention 2.4%. Its 2074.53 agrees with parse_kernels.py's 2074.5 on the same trace, which is why both are kept: same input, one aggregates by category and one by kernel, and neither derives the other's number)",
}

# Awaiting a measurement, not a ruling: these two build the model and compile, then OOM
# because another job holds the card (b0/62, 2026-09-03: 'Process 2569878 has 83.88 GiB in
# use'). PYTHONPATH=. clears their ModuleNotFoundError, so they are card-blocked rather than
# broken, and a candidate is judged after it runs. They stay unruled deliberately --
# unreached_files_ruled keeps naming them, which is the correct state for "nobody has run
# this yet" and is what an empty FATE entry cannot say.
NEEDS_A_FREE_CARD = (
    "bench_eff/bench_eff.py",
    "bench_eff/bench_opt.py",
)

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


def hook_entry_points():
    """Files the pre-commit hook runs. A REAL edge the citation graph could not see.

    scripts/hooks/pre-commit carries a SELFTEST_FILES map: every path in it has its
    `--selftest` run on every commit that stages it. That is a stronger guarantee of
    liveness than a doc mention -- it executes -- and this scan reported all 21 of them as
    unreached, including algorithms/code_reward.py (524 lines), isolate.py (473) and
    rollout.py (329), all three live.

    Read as an edge SOURCE rather than filtered out downstream: a classifier that sorts the
    false candidates after the fact has to be re-run and re-read by a person every time, and
    the 21 come back on every scan. Same fix as the vet_programs.py:37 glob -- tell the
    static analysis about the edge instead of annotating its output."""
    return _resolve_script(_read("scripts/hooks/pre-commit"))


def score_matrix_entry_points():
    return _resolve_script(_read("eval/score_matrix.py"))


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
               harness_entry_points, score_matrix_entry_points, hook_entry_points):
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

    # Printed with the table, not only in this file's docstring: the committed
    # listing is what someone reads before deleting, and a warning that lives only
    # in the generator is a warning they never see.
    print("NOT A DELETION ORACLE. This is a citation graph: a doc mention counts as an edge,")
    print("so 'reachable' can mean 'named by a doc nobody runs'. And a file loaded by a runtime")
    print("glob is live while invisible here -- vet_programs.py:37 globs math_programs_l*_ext*.py,")
    print("23 generators. Before deleting anything, grep for glob/importlib on its directory.")
    print()
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
