"""Which broken() worlds are coupled to a live file's CURRENT bytes, and are those bytes still there?

6e's item, from the world-8 stale-pycache race (e5b73d40) and from _broken_reported_path, which
STOPPED REPRODUCING ITS DEFECT AND STAYED GREEN for a day.

Two classes are reported, and one of them is the class this scan was commissioned to add:

  DEAD MARKER -- the world locates its mutation with a string literal that is no longer in the
  file it reads. Decidable: a `replace` whose target is absent is a no-op, and the world then
  mutates nothing while its selftest stays green.

  SIZE-PRESERVING MUTATION -- the replacement is the same byte length as what it replaces. That is
  what blocked b0: world 8's `return 0.0` -> `return 1.0` kept rlvr_reward.py at 3565 bytes, and
  pyc invalidation is (source mtime in WHOLE SECONDS, size), so a mutation landing in the same
  second as a prior green run reused stale bytecode and the defect never executed. Reported whether
  or not the marker is live -- it is a property of the edit, not of drift.

THE CEILING, MEASURED AGAINST THE FOUNDING CASE AND STATED RATHER THAN PAPERED OVER. The
dead-marker predicate does NOT catch _broken_reported_path's failure. That world reverted
`print(f"preds saved: {out_path}")` to `{preds_path}`, and its marker -- the out_path form -- is
still in eval/l1_fewshot.py today. What died was the REPLACEMENT: e1's 29b31367 deleted the
`preds_path` variable, so the reverted print referenced an unbound name and produced a NameError
instead of the stale-name defect. A machine cannot see that without resolving names in the
mutated file, which is a different tool. So this scan decides two properties and is blind to a
third; the third is why every (b) world should carry its own post-mutation assertion, which is the
second fix in e5b73d40.

Class (a), `git show <sha>:<path>`, is immune by construction and reported separately.
scan_broken_worlds.py already separates hand-written content, which is out of scope here.

  python3 runs/audit_0904/dead_worlds.py
  python3 runs/audit_0904/dead_worlds.py --selftest
"""

import ast
import functools
import os
import subprocess
import sys

ROOT = os.environ.get(
    "AUDIT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
HARNESS = os.path.join(ROOT, "scripts", "harness.py")
# THE HOOK IS SCANNED TOO, and it is the reason this scan exists. Its ten worlds live inside one
# `_selftest()` rather than in named `_broken_*` functions, so a scan that only reads harness.py
# would have missed world 8 -- the world that blocked b0 for a size-preserving mutation. Scanned
# as one unit, which is coarse but honest: the alternative is inventing a boundary inside a
# 400-line function.
HOOK = os.path.join(ROOT, "scripts", "hooks", "pre-commit")


def _funcs(path):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    return src, {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _str_consts(fn):
    """Every string literal in fn, including the pieces of an f-string, longest first.

    Longest first because a world's mutation marker is its longest literal far more often than
    not, and a short literal ("\\n", "0") matches every file.
    """
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append(n.value)
    return sorted(set(out), key=len, reverse=True)


def _reads_git_show(fn):
    """Does the world read its bytes from a pinned revision (`git show <sha>:<path>`)?"""
    for n in ast.walk(fn):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            v = n.value
            if ":" in v and ("~" in v or len(v.split(":")[0]) >= 7):
                head = v.split(":")[0].rstrip("~0123456789^")
                if head and all(c in "0123456789abcdef" for c in head) and len(head) >= 7:
                    return True
    return False


def _reads_real_files(fn):
    """Does the world get its bytes from the real tree at all -- by path, or by glob?

    THE LITERAL-PATH TEST ALONE MISSES THE ONE WORLD KNOWN TO HAVE DIED. _broken_reported_path (at
    e1f8c56f~1, the version that stayed green for a day while reproducing nothing) copies
    `glob.glob(os.path.join(ROOT, "eval", "*.py"))` and then opens
    os.path.join(d, "eval", "l1_fewshot.py") -- three separate literals, no path literal at all.
    Scored by _repo_paths it came back c-or-linked, i.e. out of scope, which is the scan reporting
    clean on its own founding case.

    So class (b) is "reads real bytes", and marker liveness is then checked against the WHOLE
    tracked tree rather than against the files the world happens to name as one string.
    """
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = getattr(n.func, "attr", None) or getattr(n.func, "id", None)
            if f in READ_REAL or f == "glob":
                return True
            if f == "open" and any(
                    isinstance(x, ast.Call) and (getattr(x.func, "attr", None) == "join")
                    for x in ast.walk(n)):
                return True
    return False


READ_REAL = {"copy", "copy2", "copytree", "copyfile"}


@functools.lru_cache(maxsize=1)
def _tracked_python():
    """Every tracked text file's contents, as one blob per path. ~1s, read once."""
    r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    out = {}
    for rel in r.stdout.splitlines():
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p) or os.path.getsize(p) > 4_000_000:
            continue
        try:
            out[rel] = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
    return out


def _executes_python(fn):
    """Does the world RUN python on the file it mutated, or only read it?

    THE RACE NEEDS AN IMPORT. A stale __pycache__ can only matter if the mutated module is
    executed: check_gemm_dims parses train.py with `ast.parse` and never imports it, so its
    size-preserving `ffn_hidden = 3072` -> `3400` edit cannot be defeated by a pyc, and reporting
    it beside world 8 would be a false positive of exactly the kind this scan is supposed to avoid.
    World 8 spawns the hook, which subprocesses the test, which imports rlvr_reward -- that is the
    difference.
    """
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = getattr(n.func, "attr", None) or getattr(n.func, "id", None)
            if f in ("run", "check_output", "Popen", "call", "check_call", "system"):
                return True
            if f in ("import_module", "__import__", "exec"):
                return True
    return False


def _repo_paths(fn):
    """Repo-relative paths the world names as string literals, that exist in the tree.

    NO `"/" in s` REQUIREMENT. The first version had one and it produced a false DEAD immediately:
    _broken_doc_commands copies README.md and asserts `"data/mix_scale_0.2b.json" in s` against
    it, but README.md has no slash, so the only file the world reads was not in the list and its
    live marker was scored against the four data/*.json paths it merely creates.
    """
    out = []
    for s in _str_consts(fn):
        if s.startswith("/") or len(s) > 120 or not s.strip():
            continue
        cand = os.path.join(ROOT, s)
        if os.path.isfile(cand):
            out.append(s)
    return out


_META = set("\\[](){}|+*?^$")


def _looks_like_regex(s):
    """A pattern or a process-output marker, not a file marker.

    Two grounds, both measured against this repo's own worlds:
      - two or more regex metacharacters: the hook's world 8 matches
        `REFUSING:.*test_rlvr_reward_suite` on the inner hook's stderr.
      - the string is a REFUSAL MESSAGE. Every world that spawns a check or a hook asserts on its
        stderr, and `REFUSING: this is the integration tree` is printed by the hook at runtime --
        it lives in an f-string that the scan reads as several fragments, so a plain containment
        test against the file finds neither half and reports DEAD on three working worlds.
    Both are out of this scan's scope -- it decides file coupling -- and both are counted rather
    than dropped.
    """
    if s.startswith(("REFUSING", "WARNING", "FAIL", "SELFTEST")):
        return True
    return sum(1 for c in s if c in _META) >= 2


def _replace_pairs(fn):
    """(old, new) for every `X.replace(old, new)`, resolving the one non-literal form that matters.

    `s.replace(M, M[:-3] + "1.0")` IS DECIDABLE and it is world 8's own edit, the one that blocked
    b0. Reading it as opaque would make this scan blind to the single instance it was written for.
    The shape: the first argument is a name, the second is that same name sliced by a negative
    constant plus a literal. The length delta is then (len(literal) - k) for `[:-k]`, and zero
    delta means size-preserving whatever the marker's own length is.

    Anything else non-literal stays opaque and is reported as opaque, never as safe.
    """
    pairs, opaque = [], 0
    for n in ast.walk(fn):
        if not isinstance(n, ast.Call):
            continue
        if getattr(n.func, "attr", None) != "replace" or len(n.args) < 2:
            continue
        a, b = n.args[0], n.args[1]
        if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                and isinstance(b, ast.Constant) and isinstance(b.value, str):
            pairs.append((a.value, b.value))
            continue
        delta = _slice_concat_delta(a, b)
        if delta is None:
            opaque += 1
        elif delta == 0:
            # Reported with the shape rather than the bytes: the marker's value is not literal
            # here, so only the delta is known, and the delta is the whole property.
            pairs.append(("<marker>", "<marker>"))
    return pairs, opaque


def _slice_concat_delta(a, b):
    """Byte-length delta of `replace(M, M[:-k] + "lit")`, or None if not that shape."""
    if not isinstance(a, ast.Name) or not isinstance(b, ast.BinOp) \
            or not isinstance(b.op, ast.Add):
        return None
    left, right = b.left, b.right
    if not isinstance(right, ast.Constant) or not isinstance(right.value, str):
        return None
    if not isinstance(left, ast.Subscript) or not isinstance(left.slice, ast.Slice):
        return None
    if getattr(left.value, "id", None) != a.id or left.slice.lower is not None:
        return None
    up = left.slice.upper
    if isinstance(up, ast.UnaryOp) and isinstance(up.op, ast.USub) \
            and isinstance(up.operand, ast.Constant) and isinstance(up.operand.value, int):
        return len(right.value) - up.operand.value
    return None


def _replacement_values(fn):
    """Every literal a world INSERTS -- the second argument of a replace, and any written value.

    A world that mutates and then guards its own edit asserts the POST-mutation string:
    _broken_gemm_dims does `src.replace("ffn_hidden = 3072", "ffn_hidden = 3400")` and then
    `assert "ffn_hidden = 3400" in src`. That assert matches the MUTATED string, not the file, so
    scoring it against train.py reports DEAD on a world that works. The pre-mutation half of the
    same pair -- "ffn_hidden = 3072" -- is the real coupling and is still checked.
    """
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "replace" \
                and len(n.args) >= 2:
            b = n.args[1]
            if isinstance(b, ast.Constant) and isinstance(b.value, str):
                out.add(b.value)
    return out


def _match_literals(fn):
    """Only the literals a world uses AS A MATCH TARGET against a file it read.

    THE FIRST VERSION TOOK EVERY LITERAL IN THE FUNCTION and reported 5 dead worlds, all five
    false: three were the world's own DOCSTRING (prose about the mutation, never matched against
    anything) and two were values the world WRITES into its fixture --
    `blocked_on="frozen until the run ends"` is inserted by _broken_owner_queue_depth, so "absent
    from the file" is what it is supposed to be. A literal is only evidence of coupling when the
    world's own correctness depends on finding it.

    The four positions that are match targets:
      s.replace(OLD, ...)        -- the mutation itself
      OLD in s / OLD not in s    -- the world's own guard
      s.index(OLD) / s.find(OLD) -- locating the edit
      re.sub/search(PAT, ...)    -- same, as a pattern
    """
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            attr = getattr(n.func, "attr", None)
            fid = getattr(n.func, "id", None)
            if attr in ("replace", "index", "find", "rindex", "count", "split") and n.args:
                a = n.args[0]
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    out.append(a.value)
            if (attr in ("sub", "search", "match", "findall") or fid in ("sub", "search")) \
                    and n.args:
                a = n.args[0]
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    out.append(a.value)
        # `X in s` / `X not in s`: the literal is the LEFT operand of the comparison.
        if isinstance(n, ast.Compare) and len(n.ops) == 1 \
                and isinstance(n.ops[0], (ast.In, ast.NotIn)) \
                and isinstance(n.left, ast.Constant) and isinstance(n.left.value, str):
            out.append(n.left.value)
    return sorted(set(out), key=len, reverse=True)


def _marker_state(fn, paths):
    """For each literal the world MATCHES ON, is it still in one of the files it reads?

    Returns (live, dead) lists of (literal, path-or-None). A literal shorter than 8 chars is
    skipped: it matches too much to be evidence either way, and saying so beats a false verdict.
    """
    bodies = {p: open(os.path.join(ROOT, p), encoding="utf-8").read() for p in paths}
    if not bodies:
        # A GLOB-COPYING WORLD names no path literal, so its markers are scored against the whole
        # tracked tree. Coarser -- a literal present in some other file reads LIVE -- but the
        # alternative is what happened to _broken_reported_path: no path literal, so no scoring at
        # all, so the scan reported clean on the one world known to have died.
        bodies = _tracked_python()
    inserted = _replacement_values(fn)
    live, dead, patterns = [], [], []
    for s in _match_literals(fn):
        if len(s) < 8 or s in inserted:
            continue
        if _looks_like_regex(s):
            patterns.append(s)
            continue
        hit = next((p for p, b in bodies.items() if s in b), None)
        (live if hit else dead).append((s, hit))
    return live, dead, patterns


def scan():
    rows = []
    for path, prefixes in ((HARNESS, ("_broken_", "_positive_")), (HOOK, ("_selftest",))):
        _src, funcs = _funcs(path)
        rel = os.path.relpath(path, ROOT)
        for name, fn in sorted(funcs.items()):
            if not name.startswith(prefixes):
                continue
            label = name if path == HARNESS else f"{rel}:{name}"
            if _reads_git_show(fn):
                rows.append((label, "a-git-revision", [], [], [], [], 0))
                continue
            paths = _repo_paths(fn)
            if not paths and not _reads_real_files(fn):
                rows.append((label, "c-or-linked", [], [], [], [], 0))
                continue
            live, dead, pats = _marker_state(fn, paths)
            pairs, opaque = _replace_pairs(fn)
            # Only a world that EXECUTES what it mutated can be defeated by a stale pyc, so the
            # size-preserving list is gated on that. Without the gate check_gemm_dims's ast-only
            # world reads as carrying the race.
            size_pres = []
            if _executes_python(fn):
                size_pres = [(o, n) for o, n in pairs if len(o) == len(n) and o != n] \
                    + [(o, n) for o, n in pairs if o == "<marker>"]
            rows.append((label, "b-live-file-string", paths, dead, size_pres, pats, opaque))
    return rows


def main():
    rows = scan()
    by = {}
    for r in rows:
        by.setdefault(r[1], []).append(r)
    print(f"{len(rows)} worlds scanned in scripts/harness.py + scripts/hooks/pre-commit\n")
    for cls in ("a-git-revision", "b-live-file-string", "c-or-linked"):
        print(f"  {cls:20s} {len(by.get(cls, []))}")

    dead = [r for r in rows if r[3]]
    print(f"\nDEAD MARKERS -- a literal the world matches on is no longer in the file it reads "
          f"({len(dead)} worlds):")
    if not dead:
        print("  none")
    for name, _cls, paths, deadlits, _sp, _pats, _op in dead:
        print(f"  {name}  reads {paths}")
        for s, _ in deadlits[:3]:
            print(f"      absent: {s[:90]!r}")

    sp = [r for r in rows if r[4]]
    print(f"\nSIZE-PRESERVING MUTATIONS -- same byte length, so a stale __pycache__ can survive "
          f"them ({len(sp)} worlds):")
    if not sp:
        print("  none")
    for name, _cls, _paths, _d, pairs, _pats, _op in sp:
        for o, n in pairs[:2]:
            if o == "<marker>":
                print(f"  {name}  replace(M, M[:-k] + lit) with a zero length delta -- "
                      f"world 8's own shape")
            else:
                print(f"  {name}  {o[-40:]!r} -> {n[-40:]!r}  ({len(o)} bytes both)")

    # NOT SILENTLY DROPPED (env_hygiene's rule and this repo's): what the scan could not decide is
    # printed as a count, because a scan that reports only its findings reads as complete.
    pats = sum(len(r[5]) for r in rows)
    opaq = sum(r[6] for r in rows)
    print(f"\nOUT OF SCOPE, stated rather than omitted: {pats} match literals look like regex "
          f"patterns (matched against process OUTPUT, not against a file); {opaq} replace() calls "
          f"have non-literal arguments this scan cannot resolve, so their size delta is UNKNOWN, "
          f"not zero.")
    return 0


def _selftest():
    """Known answers on this repo's own harness, plus a constructed pair for each predicate."""
    fails = []

    def case(ok, label):
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")
        if not ok:
            fails.append(label)

    # _reads_git_show: the sha:path form is class (a), a bare path is not.
    m = ast.parse("def f():\n    subprocess.run(['git','show','47cb01c2~1:eval/l1_fewshot.py'])\n")
    case(_reads_git_show(m.body[0]), "a sha:path literal reads as class (a) git-revision")
    m2 = ast.parse("def f():\n    open('eval/l1_fewshot.py')\n")
    case(not _reads_git_show(m2.body[0]), "a bare repo path is NOT class (a)")

    # size-preserving: world 8's own pair, which is the case that cost b0 a merge, beside a
    # length-changing pair that must not be flagged.
    m3 = ast.parse("def f():\n    s.replace('return 0.0', 'return 1.0')\n")
    p3, _ = _replace_pairs(m3.body[0])
    case([(o, n) for o, n in p3 if len(o) == len(n)],
         "world 8's `return 0.0` -> `return 1.0` is flagged size-preserving")
    m4 = ast.parse("def f():\n    s.replace('domains', '')\n")
    p4, _ = _replace_pairs(m4.body[0])
    case(not [(o, n) for o, n in p4 if len(o) == len(n)],
         "a length-CHANGING replace is not flagged (the negative control)")
    # THE FORM THIS SCAN EXISTS FOR: world 8's actual edit is not two literals, it is
    # `marker[:-3] + "1.0"`, whose delta is len("1.0") - 3 = 0. Reading it as opaque would leave
    # the scan blind to its own founding case, so it is resolved, and reported as the <marker>
    # shape because only the delta is known.
    m5 = ast.parse("def f():\n    s.replace(marker, marker[:-3] + '1.0')\n")
    p5, o5 = _replace_pairs(m5.body[0])
    case(p5 == [("<marker>", "<marker>")] and o5 == 0,
         "world 8's real form, marker[:-3] + '1.0', resolves to a ZERO delta")
    # And a slice-concat whose delta is NOT zero must not be flagged.
    m6 = ast.parse("def f():\n    s.replace(marker, marker[:-3] + '10.0')\n")
    p6, _o6 = _replace_pairs(m6.body[0])
    case(p6 == [], "a slice-concat with a nonzero delta is not flagged (the negative control)")
    # Anything else non-literal stays opaque: unknown is not safe.
    m7 = ast.parse("def f():\n    s.replace(a, b)\n")
    p7, o7 = _replace_pairs(m7.body[0])
    case(p7 == [] and o7 == 1,
         "a replace this scan cannot resolve is reported opaque, never assumed safe")

    # _marker_state on the real tree: a literal from a real file is live, an invented one is dead.
    real = open(os.path.join(ROOT, "scripts", "sweep.py"), encoding="utf-8").read()
    lit = [ln for ln in real.splitlines() if len(ln.strip()) > 30][0].strip()
    src = ("def f():\n"
           f"    a = s.replace({lit!r}, 'x')\n"
           "    b = s.replace('this literal is in no file in this repository at all', 'y')\n")
    fn = ast.parse(src).body[0]
    live, dead, _pats = _marker_state(fn, ["scripts/sweep.py"])
    case(any(s == lit for s, _ in live), "a literal really in sweep.py reads LIVE")
    case(any("no file in this repository" in s for s, _ in dead),
         "an invented literal reads DEAD")

    # AN END-TO-END DEAD WORLD, because the two cases above test the predicate and not the scan.
    # A real harness copy with one world's marker corrupted must come back DEAD by name, and the
    # unmodified copy must not -- which is the difference between a scan and a predicate.
    import tempfile
    hsrc = open(HARNESS, encoding="utf-8").read()
    old = 'src.replace("ffn_hidden = 3072", "ffn_hidden = 3400", 1)'
    case(old in hsrc, "the end-to-end fixture's anchor is in the real harness")
    if old in hsrc:
        d = tempfile.mkdtemp(prefix="deadworlds_")
        broke = os.path.join(d, "harness_broken.py")
        clean = os.path.join(d, "harness_clean.py")
        open(clean, "w", encoding="utf-8").write(hsrc)
        # EVERY OCCURRENCE, not the first. _broken_mutation_asserted_took embeds a verbatim copy of
        # _broken_gemm_dims's source as the string it mutates, so the anchor appears twice; a
        # count-1 replace corrupted only the first and the second kept the marker live, so the
        # world reported no dead marker and this case failed on a working scan. Caught by the hook
        # when that check landed -- the fixture was coupled to there being exactly one copy.
        open(broke, "w", encoding="utf-8").write(hsrc.replace(
            old, 'src.replace("ffn_hidden = 9999", "ffn_hidden = 3400", 1)'))
        saved = globals()["HARNESS"]
        try:
            globals()["HARNESS"] = broke
            got = {n: d2 for n, _c, _p, d2, _s, _pt, _o in scan()}
            case(bool(got.get("_broken_gemm_dims")),
                 "a world whose marker was corrupted comes back DEAD end to end")
            globals()["HARNESS"] = clean
            got2 = {n: d2 for n, _c, _p, d2, _s, _pt, _o in scan()}
            case(not got2.get("_broken_gemm_dims"),
                 "the SAME world on the unmodified harness is not DEAD (the negative control)")
        finally:
            globals()["HARNESS"] = saved
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    # The whole scan runs on the real harness and classifies every world.
    rows = scan()
    case(len(rows) >= 80, f"the scan reaches every world: {len(rows)}")
    case(any(c == "a-git-revision" for _n, c, _p, _d, _s, _pt, _o in rows),
         "at least one world is class (a) -- _broken_reported_path was converted to git show")

    print(f"dead_worlds selftest: {'ok' if not fails else 'FAIL'} ({len(fails)} failing)")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    sys.exit(main())
