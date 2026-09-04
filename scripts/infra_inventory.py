#!/usr/bin/env python3
# infra-layer: moves to aupai-infra at the split (6e ruling 2026-09-04).
# restartable: read-only, ~3 s over the tracked tree, no output file -- an interrupt
# loses nothing and a rerun is cheaper than any checkpointing would be.
"""Which tracked files touch the compute layer, and how deeply.

Generated, never hand-typed: a hand list of "the pod files" is what the split
would be wrong about, and the cost of being wrong is a file that compiles in
neither repo. Every classification below is a count of matched symbols, so the
output is checkable by re-running it rather than by trusting the author.

Five classes, and the four non-`mixed` ones exist to keep `mixed` honest. Counts
move with the tree — run the script, do not quote these:

  infra-only      transport/allocation only; names no project subject          7
  mixed           implements a verb AND knows project concepts — the seam     29
  contract-caller calls a verb, implements none; no function carries it       26
  project-only    one or two mentions inside a project file                   25
  reference       a doc or ledger that NAMES the compute without calling it   48

The last three were all `mixed` in the first version, which put 103 of 152 files
there — a number that is not a plan. Three corrections, each measured:

  A `runs/*.jsonl` row saying "card 5" creates no import and splits at no
  function boundary. Only code can be a seam, so only code is classified as one.

  `export CUDA_VISIBLE_DEVICES=3` uses a verb; it does not own it. 26 files
  looked like seams on symbol count alone and have nothing to cut. They stay in
  the project and depend on the contract, which is what a contract is for.

  A keyword count is not a subject, in EITHER direction. `run_pretrain.sh` read
  infra-only on alloc:3 proj:0, where proj:0 meant "the vocabulary does not speak
  launcher" (6e caught it) — so the vocabulary widened. That immediately pushed
  three transport files into `mixed` on one incidental `torchrun`, so the rule is
  symmetric: mostly-transport plus a passing mention is still transport.

`mixed` is the answer that costs work, so among code the rule stays biased: a
code file lands there unless it is unambiguously one side. An over-large mixed
set is a longer plan; an under-large one is a broken repo.
"""
import ast
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The compute layer's vocabulary: how a caller reaches, allocates, or syncs the box.
INFRA = {
    "exec": r"~/bin/pod\b|bin/pod\b|tn exec|crictl|pod-exec|podx",
    "sync": r"pod_push|pod_drift|pod_pull_ledgers|podput|pod_sync_check|pod_synced_head|bootstrap_pod",
    "alloc": r"card_claim|card_assignment|nvidia-smi|CUDA_VISIBLE_DEVICES|free_card",
    "hygiene": r"sweep\.py|env_hygiene|disk_inventory|_gpu_descendants",
}
# The project's vocabulary: what the compute is being used FOR.
#
# The DATA half was here from the start. The other two halves were missing, and their
# absence put four launcher/tooling files in infra-only on a proj=0 that meant "my regex
# does not speak this dialect", not "this file has no project subject" (6e, 2026-09-04):
#   run_pretrain.sh   alloc:3 proj:0 -- a training launcher, subject is the run
#   n8_conv_cost.py   alloc:1 proj:0 -- a throughput experiment
#   merge_main.sh / merge_drivers.py  sync:1 proj:0 -- git tooling, not transport
# This is the same reasoning I rejected for the 24 harness checks, applied to files:
# a keyword count is not a subject. Widening the vocabulary fixes the cause; an
# override list would have hidden it behind a name.
PROJECT = "|".join((
    # what is trained on
    r"\b(mix_|corpus|tokeniz|checkpoint|ckpt|eval|score_matrix|grpo|sft|ladder|domain"
    r"|vocab|milestone|anneal|lr_schedule)",
    # how a run is launched -- a launcher's subject is the run it starts. Only terms
    # that START one: `run_ddp` and `worktree` are how the SYNC tooling refers to what
    # it guards, so including them put pod_push.sh itself in project territory (caught
    # by the selftest, which is what it is for).
    r"\b(torchrun|train\.py|exp\.py|nproc_per_node|HYP=|throughput|tok/s)",
    # repository tooling: it manipulates git, not the compute
    r"\b(merge_main|merge-driver)",
))

# One-day artifacts, not a layer. runs/audit_0904/*.py were written for a single audit
# and will not exist in either repo (6e, 2026-09-04).
EXCLUDE = ("runs/audit_0904/",)


def tracked():
    out = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True).stdout
    keep = (".py", ".sh", ".md", ".json", ".jsonl", ".yml", ".yaml")
    return [f for f in out.split("\n")
            if f.endswith(keep) and not f.startswith(EXCLUDE)]


def carriers(path, text):
    """For a .py file, the top-level defs whose body carries an infra match.

    The split cuts at function boundaries, so 'this file is mixed' is not
    actionable on its own -- the plan needs the function names."""
    if not path.endswith(".py"):
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    lines = text.split("\n")
    pat = re.compile("|".join(INFRA.values()))
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        if pat.search("\n".join(lines[node.lineno - 1 : end])):
            out.append(node.name)
    return out


def classify(path):
    try:
        text = open(os.path.join(ROOT, path), encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    hits = {k: n for k, p in INFRA.items() if (n := len(re.findall(p, text)))}
    total = sum(hits.values())
    if not total:
        return None
    proj = len(re.findall(PROJECT, text, re.I))
    carrier_names = carriers(path, text)
    code = path.endswith((".py", ".sh"))
    if not code:
        # A doc or ledger NAMES the compute; it never calls it. No import, no
        # function boundary, nothing to cut -- so it cannot be a seam, and calling
        # it one buried the 13 real seams under 90 rows of prose. These move by a
        # topic decision, which is a person's call, not a regex's.
        cls = "reference"
    elif proj <= 2 and total >= 4:
        # Mostly transport, with a passing mention. `pod_sh_offset.py` says "torchrun"
        # once while doing nothing but byte offsets; `test_pod_drift_root.py` tests
        # drift. One incidental token is not a subject -- the same rule that keeps a
        # keyword count from deciding, applied symmetrically. Without this, widening
        # the project vocabulary moved three transport files into `mixed` on a single
        # word, which is the original error pointed the other way.
        cls = "infra-only"
    elif proj == 0:
        cls = "infra-only"
    elif total <= 2 and proj > 20:
        cls = "project-only"
    elif not carrier_names and set(hits) <= {"alloc"}:
        # CALLS the contract, does not implement it: `export CUDA_VISIBLE_DEVICES=3`
        # or one card_claim invocation, with no function carrying transport code.
        # 28 files looked like seams on symbol count alone; none of them has
        # anything to cut, because using a verb is not owning it. These stay in the
        # project and depend on the contract, which is the whole point of having one.
        cls = "contract-caller"
    else:
        cls = "mixed"
    return {
        "path": path,
        "class": cls,
        "kind": "code" if code else "text",
        "infra_hits": hits,
        "infra_total": total,
        "project_hits": proj,
        "carriers": carrier_names,
    }


def main():
    rows = [r for r in (classify(f) for f in tracked()) if r]
    rows.sort(key=lambda r: (r["class"], -r["infra_total"], r["path"]))
    by = {}
    for r in rows:
        by.setdefault(r["class"], []).append(r)
    if "--json" in sys.argv:
        json.dump(rows, sys.stdout, indent=1)
        return 0
    print(f"{len(rows)} tracked file(s) reference the compute layer\n")
    for cls in ("infra-only", "mixed", "contract-caller", "project-only", "reference"):
        rs = by.get(cls, [])
        print(f"## {cls} — {len(rs)}")
        if cls == "contract-caller":
            print("   (calls a contract verb; implements none — stays in the project)")
            for r in rs[:10]:
                print(f"  {r['path']:52s} infra:{r['infra_total']:4d} proj:{r['project_hits']}")
            if len(rs) > 10:
                print(f"  ... and {len(rs) - 10} more (--json for all)")
            print()
            continue
        if cls == "reference":
            print("   (docs and ledgers that name the compute; no import, no seam — "
                  "they move by topic, listed for that decision)")
            for r in rs[:12]:
                print(f"  {r['path']:52s} infra:{r['infra_total']:4d} proj:{r['project_hits']}")
            if len(rs) > 12:
                print(f"  ... and {len(rs) - 12} more (--json for all)")
            print()
            continue
        for r in rs:
            sym = " ".join(f"{k}:{v}" for k, v in r["infra_hits"].items())
            print(f"  {r['path']:52s} {sym:34s} proj:{r['project_hits']}")
            if cls == "mixed" and r["carriers"]:
                print(f"      carriers: {', '.join(r['carriers'][:12])}")
        print()
    return 0


if __name__ == "__main__":
    # One runnable check: the classifier must SEPARATE, not just label. A rule that
    # calls everything mixed passes any test that only counts rows, which is how the
    # first version shipped with 103 of 152 files in `mixed`.
    assert classify("scripts/pod_push.sh")["class"] == "infra-only"
    assert classify("scripts/harness.py")["class"] == "mixed"
    assert classify("AGENTS.md")["class"] == "reference"
    # A file with MANY alloc hits and no carrier: the branch that would otherwise
    # have called 28 contract callers seams. eval_math.sh does not test it -- one
    # hit and 27 project hits makes it project-only two branches earlier.
    assert classify("eval/eval_all.sh")["class"] == "contract-caller"
    # 6e's catch, 2026-09-04: a training launcher is project by SUBJECT however many
    # allocate calls it makes. It read infra-only on alloc:3 proj:0, where proj:0 meant
    # "the vocabulary does not speak launcher", not "no project subject".
    assert classify("scripts/run_pretrain.sh")["class"] == "contract-caller"
    # ...and the symmetric case, which widening the vocabulary then created: mostly
    # transport plus one incidental token is still transport.
    assert classify("scripts/pod_sh_offset.py")["class"] == "infra-only"
    sys.exit(main())
