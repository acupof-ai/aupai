"""Which eval/ tools are ever launched DIRECTLY, and which only ever run inside a wrapper?

The open question left by card_claim_coverage.md: 19 eval/ tools take a card and none claims it.
The right acquire unit is the process that holds the card, so the question is empirical -- a tool
only ever reached through eval_all.sh is claimed once by the wrapper, and a tool people run by hand
has to claim for itself.

EVIDENCE, in decreasing strength:
  1. runs/experiments.jsonl `cmd` -- a real launch that happened. The strongest evidence there is:
     the row exists because someone ran it.
  2. .sh wrappers under eval/ and scripts/ -- a tool invoked there is covered by the wrapper's
     claim IF the wrapper is what people launch.
  3. doc command blocks (AGENTS.md, docs/**) -- an instruction to run it directly. Weaker than a
     ledger row (a doc can rot) but it is what the next session will type.

A tool in NEITHER 1 nor 3, and in no wrapper, is a deletion-broadcast candidate rather than an
acquire site: nothing in the repo records it ever being run.

  python3 runs/audit_0904/eval_acquire_unit.py
  python3 runs/audit_0904/eval_acquire_unit.py --selftest
"""

import json
import os
import re
import subprocess
import sys

ROOT = os.environ.get(
    "AUDIT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)


def _tracked(pattern):
    r = subprocess.run(["git", "ls-files", pattern], cwd=ROOT, capture_output=True, text=True)
    return [x for x in r.stdout.splitlines() if x.strip()]


def population():
    """The card-taking, unclaimed tools under eval/.

    Uses card_claim_population.py's OWN regexes, read out of its source, rather than a second copy
    of the predicate: two predicates for one population is how the counts in this directory drifted
    before (that scan's first version missed score_matrix and the doc had to publish a correction).
    It is a script with no importable function, so its two patterns are lifted by name and compiled
    here -- and the selftest asserts they are still there, so a rewrite of that file fails loudly
    instead of silently reverting this one to its own predicate.
    """
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "card_claim_population.py"), encoding="utf-8").read()
    ns = {"re": re}
    body = []
    keep = False
    for line in src.splitlines():
        if line.startswith(("TAKES = ", "LOADS = ")):
            keep = True
        elif keep and not line.startswith((" ", ")", "r\"", "r'", '"', "'")):
            keep = False
        if keep:
            body.append(line)
    exec("\n".join(body), ns)
    takes, loads = ns["TAKES"], ns["LOADS"]
    out = []
    for rel in _tracked("eval/*.py"):
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            continue
        s = open(p, encoding="utf-8", errors="replace").read()
        if not (loads.search(s) and takes.search(s)):
            continue
        if "claim_my_cards" in s or "card_claim" in s:
            continue
        out.append(rel)
    return sorted(out)


def ledger_cmds():
    """Every `cmd` string in runs/experiments.jsonl -- launches that actually happened."""
    p = os.path.join(ROOT, "runs", "experiments.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        c = d.get("cmd")
        if isinstance(c, str) and c:
            out.append(c)
    return out


def wrapper_bodies():
    """Every tracked .sh, by path -- the wrappers a tool can hide behind."""
    out = {}
    for rel in _tracked("*.sh"):
        p = os.path.join(ROOT, rel)
        if os.path.isfile(p):
            out[rel] = open(p, encoding="utf-8", errors="replace").read()
    return out


_BLOCK = re.compile(r"```(?:bash|sh|console)?\n(.*?)```", re.S)
#: A backtick span is a COMMAND only if it starts like one. AGENTS.md wraps bare paths in backticks
#: constantly -- `docs/lessons/kept_methods.md`, `data/tokenizer.json` -- and counting those as
#: launches made every path-mentioned tool read as directly run. Caught by the negative control
#: below, which asserted a prose path is not a command and failed on the first version.
_CMD = re.compile(r"^\s*(?:[A-Z_]+=\S+\s+)*(?:python3?|bash|sh|\./|torchrun|setsid|pod\s)")


def doc_commands():
    """Text inside fenced command blocks, plus backtick spans that LOOK like commands.

    Both are needed and neither alone is right: AGENTS.md's entry-point table -- the repo's actual
    answer to "how do I run this" -- puts its commands in backticks inside table cells, not in
    fenced blocks, so fenced-only misses them; and backticks also hold bare paths, so unfiltered
    backticks count prose. Hence _CMD.
    """
    paths = ["AGENTS.md", "CLAUDE.md", "README.md"] + _tracked("docs/*.md") \
        + _tracked("docs/**/*.md")
    body = []
    for rel in dict.fromkeys(paths):
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            continue
        src = open(p, encoding="utf-8", errors="replace").read()
        for m in _BLOCK.finditer(src):
            for ln in m.group(1).splitlines():
                if _CMD.match(ln):
                    body.append((rel, ln))
        for m in re.finditer(r"`([^`\n]{6,200})`", src):
            if _CMD.match(m.group(1)):
                body.append((rel, m.group(1)))
    return body


def hook_registered():
    """Tools the pre-commit hook runs a --selftest for, or records as unrunnable-here.

    A FOURTH EVIDENCE SOURCE, added after the first run put six live tools in the never-run bucket.
    A file in SELFTEST_FILES is executed on every commit that stages it, and a file in NEEDS_DATA
    carries a written reason why it cannot run on a dev box -- both are records of a tool being
    maintained, which is what the never-run bucket is supposed to exclude. Without this,
    eval/lambada_en.py read as a deletion candidate while its selftest runs in CI, and
    eval/base_matrix.py read the same way while the hook holds a sentence explaining its exemption.

    Note what this does NOT establish: a selftest is not a card-taking launch. So a hook-registered
    tool is not moved into "acquire in the tool" -- it is moved out of "no record", into its own
    row, because the question this scan answers is where the acquire goes and a selftest takes no
    card.
    """
    p = os.path.join(ROOT, "scripts", "hooks", "pre-commit")
    if not os.path.isfile(p):
        return set()
    src = open(p, encoding="utf-8", errors="replace").read()
    return {m.group(1) for m in re.finditer(r'"(eval/[A-Za-z0-9_]+\.py)"\s*[,:)]', src)}


def imported_by_registry():
    """Tools reached through eval/run_eval.py's `import_module(f"eval.{name}")` registry.

    THE FIFTH SOURCE, and the one AGENTS.md warns about by name: no static analysis sees a runtime
    loader, so arc/mmlu/piqa read as never-run by every other predicate here. They are not dead and
    they are not acquire sites -- run_eval imports them and holds the card, exactly as
    vet_programs.py:37's glob makes 23 live generators look unreferenced.

    Read from MC_BENCHMARKS' own table, so a benchmark added there is covered without editing this.
    """
    p = os.path.join(ROOT, "eval", "run_eval.py")
    if not os.path.isfile(p):
        return set()
    src = open(p, encoding="utf-8", errors="replace").read()
    m = re.search(r"MC_BENCHMARKS\s*=\s*[\{\[\(](.*?)[\}\]\)]\s*\n", src, re.S)
    names = set()
    if m:
        names = {x for x in re.findall(r"['\"]([a-z0-9_]+)['\"]", m.group(1))}
    # The adapter functions are keyed by benchmark name too; union them in so a table written as a
    # dict of callables is still covered.
    names |= set(re.findall(r"def\s+_load_([a-z0-9_]+)\s*\(", src))
    return {f"eval/{n}.py" for n in names
            if os.path.isfile(os.path.join(ROOT, "eval", f"{n}.py"))}


def classify():
    tools = population()
    cmds = ledger_cmds()
    wraps = wrapper_bodies()
    docs = doc_commands()
    hooked = hook_registered()
    registry = imported_by_registry()
    rows = []
    for t in tools:
        base = os.path.basename(t)
        in_ledger = [c for c in cmds if t in c or f"/{base}" in c or base in c.split()]
        in_wrap = sorted({w for w, body in wraps.items() if t in body or base in body})
        in_doc = sorted({d for d, body in docs if t in body})
        rows.append((t, len(in_ledger), in_wrap, in_doc, t in hooked, t in registry))
    return rows


def main():
    rows = classify()
    print(f"{len(rows)} card-taking eval/ tools with no claim\n")
    print(f"{'tool':30s} {'ledg':>4s} {'hook':>4s} {'reg':>4s}  {'wrappers':24s} docs")
    direct, wrapped, registry, hooked_only, unrun = [], [], [], [], []
    for t, n, wraps, docs, hooked, reg in rows:
        print(f"{t:30s} {n:4d} {'yes' if hooked else '-':>4s} {'yes' if reg else '-':>4s}  "
              f"{','.join(os.path.basename(w) for w in wraps)[:24]:24s} {len(docs)}")
        if n or docs:
            direct.append(t)
        elif reg:
            registry.append(t)
        elif wraps:
            wrapped.append(t)
        elif hooked:
            hooked_only.append(t)
        else:
            unrun.append(t)

    print(f"\nACQUIRE IN THE TOOL ({len(direct)}) -- a ledger row or a doc command block launches "
          f"it directly, so the tool is the process that holds the card:")
    for t in direct:
        print(f"  {t}")
    print(f"\nRUN_EVAL HOLDS THE CARD ({len(registry)}) -- reached through run_eval's "
          f"import_module registry, never as a process of their own. One acquire in run_eval.py "
          f"covers all of them; an acquire inside each would refuse run_eval's claim:")
    for t in registry:
        print(f"  {t}")
    print(f"\nWRAPPER IS THE UNIT ({len(wrapped)}) -- reached only through a .sh, which is where "
          f"one claim covers the whole run:")
    for t in wrapped:
        print(f"  {t}")
    print(f"\nMAINTAINED BUT NEVER LAUNCHED ({len(hooked_only)}) -- the hook runs or exempts its "
          f"selftest, so it is not dead; a selftest takes no card, so it is not an acquire site "
          f"either. No action:")
    for t in hooked_only:
        print(f"  {t}")
    print(f"\nNO RECORD OF EVER BEING RUN ({len(unrun)}) -- deletion broadcast, not an acquire "
          f"site (AGENTS.md: run a deletion candidate before judging it):")
    for t in unrun:
        print(f"  {t}")
    return 0


def _selftest():
    fails = []

    def case(ok, label):
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")
        if not ok:
            fails.append(label)

    cmds = ledger_cmds()
    case(len(cmds) > 50, f"the ledger yields launch commands: {len(cmds)}")
    # KNOWN ANSWER: score_matrix is documented in AGENTS.md's entry-point table AND was run
    # directly (b0's card-5 rescore, e1's C11), so it must not read as wrapper-only. It already
    # claims, so it is outside the population -- which is itself the assertion: the population is
    # the UNCLAIMED ones.
    pop = population()
    case("eval/score_matrix.py" not in pop,
         "score_matrix, which already claims, is excluded from the population")
    case(all(p.startswith("eval/") for p in pop), f"the population is eval/-only: {len(pop)} tools")

    docs = doc_commands()
    case(any("eval/math_hard.py" in b for _d, b in docs),
         "the doc reader finds eval/math_hard.py, which AGENTS.md's pass@k row cites in backticks")
    # THE NEGATIVE CONTROL, and it caught the first version. AGENTS.md wraps bare paths in
    # backticks throughout, so an unfiltered backtick reader counts `docs/lessons/kept_methods.md`
    # as a launch and every path-mentioned tool reads as directly run. A command span must start
    # like a command.
    case(not any(b.strip().startswith(("docs/", "data/", "runs/", "facts/")) for _d, b in docs),
         "a bare path in backticks is not read as a command")
    case(any(b.strip().startswith(("python", "bash", "./", "NGPU", "COMPILE", "E2E")) or "=" in
             b.split()[0] for _d, b in docs),
         "what IS collected looks like commands")

    wraps = wrapper_bodies()
    case(any("eval_all.sh" in w for w in wraps), f"wrappers are read: {len(wraps)} .sh files")

    # THE TWO SOURCES THAT MOVED THE ANSWER, each with its known answer. The first run of this
    # scan put 9 tools in the deletion bucket and every one of them was reachable; both of these
    # were added because of that, so both are asserted rather than trusted.
    hooked = hook_registered()
    case("eval/lambada_en.py" in hooked,
         "the hook source names eval/lambada_en.py, whose selftest runs on every commit")
    case("eval/base_matrix.py" in hooked,
         "and eval/base_matrix.py, which the hook exempts with a written reason")

    reg = imported_by_registry()
    for want in ("eval/arc.py", "eval/mmlu.py", "eval/piqa.py"):
        case(want in reg, f"run_eval's import_module registry reaches {want} -- a RUNTIME loader "
                          f"no name scan sees")
    case("eval/l1_fewshot.py" not in reg,
         "a tool that is NOT in the registry table is not claimed to be (the negative control)")

    rows = classify()
    case(len(rows) == len(pop), "every population member gets a row")
    # No tool may land in two buckets: the classification is a chain of elif, so this asserts the
    # ORDER is total rather than that the buckets happen not to overlap today.
    buckets = [sum(1 for r in rows if (r[1] or r[3])),
               sum(1 for r in rows if not (r[1] or r[3]) and r[5]),
               sum(1 for r in rows if not (r[1] or r[3]) and not r[5] and r[2]),
               sum(1 for r in rows if not (r[1] or r[3]) and not r[5] and not r[2] and r[4]),
               sum(1 for r in rows if not (r[1] or r[3]) and not r[5] and not r[2] and not r[4])]
    case(sum(buckets) == len(rows), f"the buckets partition the population: {buckets}")
    print(f"eval_acquire_unit selftest: {'ok' if not fails else 'FAIL'} ({len(fails)} failing)")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    sys.exit(main())
