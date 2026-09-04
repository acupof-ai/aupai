#!/usr/bin/env python3
"""Removing a registry entry must go red at all THREE enforcement points.

The bug this guards: datagen/holdout.py's EVAL_FILES held four paths, the corpus builders took
their exclusion population from it, and data/sft/control_sft_text_heldout.jsonl was not one of
them -- so 2,114 of 7,523 measurable held-out items reached the pretraining corpus with the
guard green, fingerprinted and loud (facts/contamination.json#cont.heldout_in_pretrain_corpus,
e1 2026-09-04). The guard was correct on its own population. Its population was wrong.

A registry whose removal changes nothing is not enforcing anything, which is how the
four-entry list passed every check while missing the file that mattered. So each point is
tested for going RED on removal, and each has its positive beside it -- without the positive,
every assertion here passes on an implementation that rejects everything.

    python3 scripts/test_eval_registry.py
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDOUT = os.path.join(ROOT, "datagen", "holdout.py")
VICTIM = "control_sft_text_heldout"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world(drop=None):
    """A tree with the REAL holdout.py (optionally minus one entry) and a real data population.

    Mutated from the real file, never hand-written: a hand-made registry shares this test's
    assumptions about the schema, which is the defect it is checking for.
    """
    d = tempfile.mkdtemp(prefix="evalreg_")
    os.makedirs(os.path.join(d, "datagen"), exist_ok=True)
    # A .git, because these worlds model a CHECKOUT. Without it is_pod() reads the fixture as the
    # pod's hand-pushed tree, where every registry path must exist -- and since `data/` is
    # gitignored the fixture cannot hold them all, so the (c+) world FAILed for a reason that had
    # nothing to do with what it was testing. The world under test has to be the world it claims.
    os.makedirs(os.path.join(d, ".git"), exist_ok=True)
    text = open(HOLDOUT, encoding="utf-8").read()
    entry_rel = None
    if drop:
        key = f'    "{drop}": {{'
        start = text.index(key)
        end = text.index("\n    },\n", start) + len("\n    },\n")
        import re
        entry_rel = re.search(r'"path":\s*"([^"]+)"', text[start:end]).group(1)
        text = text[:start] + text[end:]
    open(os.path.join(d, "datagen", "holdout.py"), "w", encoding="utf-8").write(text)
    # The victim's file EXISTS in the world either way: `data/` is gitignored, so without this
    # the "broken" world is an entry gone AND its file gone, which is consistent rather than
    # broken -- and the check passes green on it. Measured: that is exactly what happened.
    if entry_rel is None:
        mod = _load(os.path.join(d, "datagen", "holdout.py"), f"h_{os.path.basename(d)}")
        entry_rel = mod.REGISTRY[VICTIM]["path"]
    p = os.path.join(d, entry_rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for i in (1, 2, 3):
            fh.write(json.dumps({"id": f"x{i}", "question": f"planted holdout {i}?",
                                 "answer": str(i), "src": "fixture"}) + "\n")
    # And the rest of the real eval population, so a hash-set comparison is against a realistic
    # set rather than one planted file. Without this the drop world hashed NOTHING and wrote an
    # empty set with rc 0 -- which made the "set shrinks" assertion pass on a None, the
    # false-pass shape. It also found a real gap: main() now refuses to write an empty set.
    for sub in ("data/eval", "data/synthetic"):
        realsub = os.path.join(ROOT, sub)
        if not os.path.isdir(realsub):
            continue
        for dirpath, _dn, filenames in os.walk(realsub):
            for fn in filenames:
                if not fn.endswith((".jsonl", ".json")):
                    continue
                s = os.path.join(dirpath, fn)
                t = os.path.join(d, os.path.relpath(s, ROOT))
                os.makedirs(os.path.dirname(t), exist_ok=True)
                if not os.path.exists(t):
                    shutil.copy(s, t)
    return d, entry_rel


def main():
    bad = 0

    def case(cond, label):
        nonlocal bad
        bad += 0 if cond else 1
        print(f"  {'ok  ' if cond else 'BUG '} {label}")

    # (a) THE ACCESSOR. eval_path raises on a name no entry covers, so a loader cannot score a
    #     file the guard has never heard of.
    full = _load(HOLDOUT, "h_full")
    case(full.eval_path(VICTIM).endswith(full.REGISTRY[VICTIM]["path"]),
         f"(a+) eval_path resolves a registered name ({VICTIM})")
    d_drop, rel = _world(drop=VICTIM)
    dropped = _load(os.path.join(d_drop, "datagen", "holdout.py"), "h_drop")
    try:
        dropped.eval_path(VICTIM)
        case(False, "(a) eval_path must RAISE on the de-registered name")
    except Exception as e:
        case(type(e).__name__ == "UnregisteredEval" and VICTIM in str(e),
             f"(a) eval_path raises UnregisteredEval naming it ({type(e).__name__})")

    # (b) THE HASH SET. main() must hash every entry's question_field, so dropping an entry
    #     drops its questions and a planted holdout question stops being recognised.
    #     Run as a subprocess so it writes into the fixture tree, not the repo.
    def _hashes(tree):
        # HOLDOUT_ALLOW_PARTIAL, because this fixture IS a partial tree on purpose: it copies
        # the eval files that exist here (5 of 13) and de-registers one, so main()'s
        # cross-machine refusal fires and writes nothing. Without the override the three
        # assertions below read "no hash file" and the test reports 10/13 -- measured
        # 2026-09-04, the same afternoon the refusal landed. The override is the documented
        # way to say "a partial set is what I want here", and what this test asserts is the
        # SHRINK between two partial sets, which needs both to exist.
        env = dict(os.environ, HOLDOUT_ALLOW_PARTIAL="1")
        r = subprocess.run([sys.executable, os.path.join(tree, "datagen", "holdout.py")],
                           capture_output=True, text=True, cwd=tree, timeout=300, env=env)
        hp = os.path.join(tree, "data", "eval", "holdout_hashes.txt")
        if not os.path.exists(hp):
            return None, r.stdout + r.stderr
        return {x.strip() for x in open(hp, encoding="utf-8") if x.strip()
                and not x.startswith("#")}, r.stdout + r.stderr

    d_full, _rel2 = _world()
    h_full, out_full = _hashes(d_full)
    h_drop, out_drop = _hashes(d_drop)
    planted = full.qhash("planted holdout 1?")
    case(h_full is not None and planted in h_full,
         f"(b+) a registered file's question IS hashed ({'present' if h_full and planted in h_full else 'MISSING'})")
    # `h is not None` on BOTH sides, explicitly: a world whose hash run failed returns None, and
    # `planted not in None`-style leniency would read a crashed run as a passing assertion. That
    # is what happened here before the population was copied in -- `3 -> ?` in the output was the
    # tell, and the ? was h_drop being None.
    case(h_drop is not None and planted not in h_drop,
         f"(b) de-registering drops its questions, so the planted one is no longer recognised "
         f"({'no hash file' if h_drop is None else 'absent as required'})")
    case(h_full is not None and h_drop is not None and len(h_full) > len(h_drop),
         f"(b) and the set SHRINKS ({len(h_full) if h_full is not None else 'NO FILE'} -> "
         f"{len(h_drop) if h_drop is not None else 'NO FILE'})")

    # A registered file that hashes NOTHING must be loud, not a 0 among the counts. This is the
    # silent no-op e1 named: a registry that adds files while the hasher reads one field name.
    d_wrong, _ = _world()
    hp = os.path.join(d_wrong, "datagen", "holdout.py")
    t = open(hp, encoding="utf-8").read().replace(
        '"question_field": ["question"],', '"question_field": ["nonexistent_field"],', 1)
    open(hp, "w", encoding="utf-8").write(t)
    r = subprocess.run([sys.executable, hp], capture_output=True, text=True, cwd=d_wrong, timeout=300)
    case(r.returncode != 0 and "NO hashes" in (r.stdout + r.stderr),
         f"(b) a registered file whose field matches nothing RAISES rather than hashing 0 "
         f"(rc={r.returncode})")

    # (c) THE COMPLETENESS CHECK. Covered by harness --selftest's broken world too; asserted
    #     here against both worlds so this file tests all three points as one story.
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import harness
    st_full, ev_full = harness.check_eval_registry_complete(d_full)
    st_drop, ev_drop = harness.check_eval_registry_complete(d_drop)
    case(st_full == harness.PASS, f"(c+) a complete registry PASSes ({ev_full[:60]})")
    case(st_drop == harness.FAIL and rel in ev_drop,
         f"(c) a de-registered file FAILs, naming it ({ev_drop[:80]})")

    # THE POSITIVE FOR THE WHOLE FILE: the three points must not be a constant. If (c) failed
    # on both worlds the assertions above would still read green one at a time.
    case(st_full != st_drop, "the three points DIFFER between the two worlds (not constants)")

    # THE PATH-EXISTS HALF MUST NOT DEPEND ON HOW MUCH DATA A LAPTOP HAPPENS TO HOLD. The first
    # version gated it on `data_present > len(reg)//2`, which read 5 of 13 in one worktree and
    # SKIPPED, then 7 of 13 in the integration tree and FAILED on the same commit -- and because
    # the pre-commit hook runs `harness check`, that red blocked every commit and merge into main
    # from a laptop for about an hour (6e, 2026-09-04). A threshold over incidental files is not a
    # test of "is the population supposed to be complete here"; is_pod is.
    #
    # The world below is a git checkout (it has .git) holding 7 of 13 registered paths -- the
    # integration tree's exact shape -- and it must not FAIL.
    d_seven, _ = _world()   # _world() already gives it a .git: it models a checkout
    seven_mod = _load(os.path.join(d_seven, "datagen", "holdout.py"), "h_seven")
    present = [n for n, e in seven_mod.REGISTRY.items()
               if os.path.exists(os.path.join(d_seven, e["path"]))]
    st_seven, ev_seven = harness.check_eval_registry_complete(d_seven)
    case(st_seven != harness.FAIL,
         f"a git checkout with {len(present)}/{len(seven_mod.REGISTRY)} paths present does NOT "
         f"FAIL on the absent ones ({st_seven}: {ev_seven[:70]})")
    case("speaks on the pod" in ev_seven,
         "...and it says which half it skipped and where that half speaks")
    # The same tree WITHOUT .git is the pod's shape, where every path must exist -- so the half
    # is not merely disabled, it still fires where it is meant to.
    import shutil as _sh2
    _sh2.rmtree(os.path.join(d_seven, ".git"))
    st_pod, ev_pod = harness.check_eval_registry_complete(d_seven)
    case(st_pod == harness.FAIL and "does not exist" in ev_pod,
         f"...while the same tree with no .git (the pod's shape) DOES FAIL on them ({st_pod})")

    for t in (d_full, d_drop, d_wrong, d_seven):
        shutil.rmtree(t, ignore_errors=True)
    print(f"test_eval_registry: {13 - bad}/13 pass")
    return 1 if bad else 0


if __name__ == "__main__":
    # --selftest accepted and ignored: the hook invokes registered files with that flag, and
    # this file's whole body IS the selftest.
    sys.exit(main())
