"""Find batch jobs that lose everything when interrupted.

`datagen/train_quality_head.py` accumulated scores in memory and called np.save once at the
end. Killed at 50% of a two-hour run, it lost 100% of the work. The property was invisible
until the moment it cost something.

The shape is mechanical, so the check is static: a loop that accumulates into a list, a write
that happens only after the loop, and nothing that reads previously-finished output back. Any
one of those alone is fine; all three together means an interrupt costs the whole run.

A script that is genuinely safe -- because it is short, or because it writes per shard in a way
this cannot see -- declares it in the file, next to the code:

    # restartable: writes one file per shard, rerun skips existing

That keeps the exemption where the next reader will see it, instead of in a registry that rots.

The existing tree has many scripts with this shape and most are short, so the gate is a
RATCHET: `scripts/restartability_baseline.json` lists what was already at risk when the check
landed, and only a NEW offender fails. The baseline shrinks as scripts are fixed or marked; it
must never grow.

Run: python scripts/restartability_audit.py [--json out.json] [--all]
     python scripts/restartability_audit.py --update-baseline
Exit code 1 if any script is at risk and not in the baseline.
"""

import argparse
import ast
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "scripts", "restartability_baseline.json")
DIRS = ["datagen", "scripts", "eval", "filters", "mathbank", "algorithms", "."]

WRITERS = {
    "save",
    "savez",
    "savez_compressed",
    "dump",
    "to_parquet",
    "to_csv",
    "to_json",
    "write_text",
    "write_bytes",
    "savetxt",
}
# There is deliberately no "looks resumable" heuristic. The first version suppressed a finding
# when words like glob( or checkpoint appeared anywhere in the file, and that silenced the exact
# script this check exists for: train_quality_head.py globs its INPUTS and loads a model
# CHECKPOINT, neither of which makes its two-hour run restartable. A substring is not evidence.
# The only exemption is an explicit marker, written by someone who checked.


def _loop_depth_map(tree):
    """Map each node to whether it sits inside a loop."""
    inside = {}

    def walk(node, in_loop):
        inside[node] = in_loop
        for child in ast.iter_child_nodes(node):
            deeper = in_loop or isinstance(node, (ast.For, ast.While, ast.AsyncFor))
            walk(child, deeper)

    walk(tree, False)
    return inside


def _writes(tree, inside):
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
        if name in WRITERS:
            out.append((name, node.lineno, inside.get(node, False)))
        elif (
            name == "open"
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and any(m in str(node.args[1].value) for m in ("w", "a"))
        ):
            out.append(("open(w)", node.lineno, inside.get(node, False)))
    return out


def _fns_called_in_loops(tree, inside):
    """Names invoked from inside a loop. A write in one of those runs per iteration even though
    it is not lexically inside the loop -- missing this flagged train.py, whose checkpoint save
    lives in a helper the training loop calls."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and inside.get(node, False):
            f = node.func
            names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return names


def _enclosing_fn(tree):
    """node -> the name of the function lexically containing it."""
    owner = {}

    def walk(node, fn):
        owner[node] = fn
        for child in ast.iter_child_nodes(node):
            walk(child, node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn)

    walk(tree, None)
    return owner


def _accumulates(tree, inside):
    """A list grown inside a loop -- the in-memory buffer that an interrupt throws away."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("append", "extend")
            and inside.get(node, False)
        ):
            return node.lineno
    return None


def audit_file(path):
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {"path": path, "error": f"unparseable: {e}"}
    exempt = "# restartable:" in src
    inside = _loop_depth_map(tree)
    writes = _writes(tree, inside)
    acc = _accumulates(tree, inside)
    called_in_loop = _fns_called_in_loops(tree, inside)
    owner = _enclosing_fn(tree)
    write_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    by_line = {n.lineno: owner.get(n) for n in write_nodes}
    has_incremental = any(w[2] or by_line.get(w[1]) in called_in_loop for w in writes)
    at_risk = bool(writes) and bool(acc) and not has_incremental
    return {
        "path": os.path.relpath(path, ROOT),
        "exempt": exempt,
        "at_risk": at_risk and not exempt,
        "accumulates_at_line": acc,
        "writes": [{"call": w[0], "line": w[1], "in_loop": w[2]} for w in writes],
    }


def collect():
    rows = []
    for d in DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, files in os.walk(base):
            dirnames[:] = [x for x in dirnames if not x.startswith((".", "__")) and x != "node_modules"]
            if d == "." and dirpath != base:
                continue  # root: top level only, the subdirs are listed explicitly
            for f in sorted(files):
                if f.endswith(".py"):
                    rows.append(audit_file(os.path.join(dirpath, f)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--all", action="store_true", help="print every file, not just the risky ones")
    ap.add_argument(
        "--update-baseline", action="store_true", help="record today's risky set (only ever shrink it)"
    )
    args = ap.parse_args()

    rows = collect()
    risky = [r for r in rows if r.get("at_risk")]
    exempt = [r for r in rows if r.get("exempt")]
    known = set()
    if os.path.exists(BASELINE):
        with open(BASELINE) as f:
            known = set(json.load(f)["at_risk"])
    if args.update_baseline:
        with open(BASELINE, "w") as f:
            json.dump({"at_risk": sorted(r["path"] for r in risky)}, f, indent=2)
        print(f"baseline updated: {len(risky)} entries")
        return 0
    new = [r for r in risky if r["path"] not in known]

    for r in rows if args.all else risky:
        mark = "RISK" if r.get("at_risk") else ("exempt" if r.get("exempt") else "ok")
        w = ", ".join(f"{x['call']}@{x['line']}" for x in r.get("writes", [])) or "-"
        print(
            f"[{mark}] {r['path']}\n        accumulates at line {r.get('accumulates_at_line')}, writes: {w}"
        )
    for r in new:
        print(
            f"[NEW] {r['path']} is at risk and not in the baseline -- write per shard, or add "
            f"a '# restartable: <why>' line saying why an interrupt is cheap"
        )
    print(
        f"restartability_audit: {len(risky)} at risk ({len(new)} new, {len(known)} SILENCED by baseline), "
        f"{len(exempt)} exempt, {len(rows)} scanned"
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 1 if new else 0


def selftest():
    """The regression this exists for: accumulate in a loop, save once at the end."""
    import tempfile

    bad = "import numpy as np\nxs = []\nfor i in range(10):\n    xs.append(i)\nnp.save('o.npy', xs)\n"
    good = bad.replace("np.save('o.npy', xs)", "    np.save(f'o{i}.npy', xs)")
    # The words that fooled the first version: globbing inputs and loading a model checkpoint
    # say nothing about whether a rerun is cheap.
    decoy = bad.replace("xs = []", "xs = []\nfiles = glob('in/*')\nckpt = 'checkpoint.pt'")
    with tempfile.TemporaryDirectory() as d:
        for name, src, want in (("bad.py", bad, True), ("good.py", good, False), ("decoy.py", decoy, True)):
            p = os.path.join(d, name)
            with open(p, "w") as fh:
                fh.write(src)
            got = audit_file(p)["at_risk"]
            assert got is want, f"{name}: at_risk={got}, expected {want}"
    return "accumulate-then-save-once is flagged, per-iteration save is not, and resume-sounding words do not excuse it"


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print("selftest OK:", selftest())
        sys.exit(0)
    sys.exit(main())
