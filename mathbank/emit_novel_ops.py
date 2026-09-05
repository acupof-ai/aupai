#!/usr/bin/env python3
"""Emit the constructed-operator test and train sets for the conversion-rate curve.

    python3 mathbank/emit_novel_ops.py            # write the sets
    python3 mathbank/emit_novel_ops.py --selftest # verify the properties they must have

WHAT THE CURVE NEEDS AND WHY EACH PROPERTY IS ASSERTED RATHER THAN INTENDED.

The readout is accuracy on family S minus accuracy on family P, as a function of n
training instances in {1, 8, 64, 512, 4096}. Three things can silently destroy it:

  TEST/POOL OVERLAP -- an instance in both sets turns the curve into recall. Checked
      two ways, because one is not enough: by exact prompt text, and by the semantic
      key (operands + structure). Two draws can produce identical operands under
      different phrasing, and identical phrasing is not the only way to leak an item.
  NON-NESTED n-SETS -- if the n=8 set is a fresh draw rather than the n=1 item plus
      seven more, then two adjacent points on the curve differ by WHICH items were
      seen as well as by how many, and the difference between them is not attributable
      to n. Nested by construction here: pool[:n].
  DRIFTING SEEDS -- a test set regenerated at a different seed is a different test.
      Both seeds are constants in this file and the header records them, so a rebuilt
      file that disagrees with a scored one is detectable rather than silent.

The test seed is 20260905 and is FIXED FOREVER (4c, 2026-09-05). The pool seed is
20260906, deliberately different: drawing both from one stream would make the pool's
first instances a continuation of the test set's, which is not overlap but is not
independence either.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random  # noqa: E402

from math_programs_l5_ext_novel import FAMILY, PROGRAMS, RULE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "probes", "novel_ops")
TEST_SEED = 20260905
POOL_SEED = 20260906
N_TEST = 1000
N_POOL = 4096
CURVE_N = (1, 8, 64, 512, 4096)
COLLISION_FACT = "facts/contamination.json#cont.novel_operator_collision"
ABSENCE_FACT = "facts/contamination.json#cont.generator_families_in_owm"

_BY_FAMILY = {}
for _lvl, _name, _fn in PROGRAMS:
    _BY_FAMILY.setdefault(FAMILY[_name], []).append((_name, _fn))


def _key(rec):
    """Semantic identity: the operands and the program that made it.

    NOT the prompt text. Two instances with the same operands under two phrasings are
    the same problem for a model that has learned the rule, and an overlap check that
    only compares strings would call them distinct.
    """
    return (rec["program"], tuple(rec["operands"]))


def _operands(instruction):
    """The instance's operands: prose integers minus the rule's own constants.

    The rule sentence states 3, 2 and 1, so they appear in every instruction and are
    not part of any instance's identity. Returned in order of appearance.
    """
    import re
    RULE_CONSTS = {1, 2, 3}
    out = []
    for tok in re.findall(r"\d+", instruction.split("求")[-1] if "求" in instruction else instruction):
        v = int(tok)
        if v not in RULE_CONSTS:
            out.append(v)
    return out


def generate(family, n, seed, exclude=()):
    """n DISTINCT instances of one family, cycling its programs so the mix is even.

    Drawn without replacement on the semantic key, and against `exclude` as well as
    against itself. Random draws collide long before a space is exhausted: family P
    at its original range held 381 distinct instances and the curve wants 5,096, but
    even S's diamond_chain (4,913) cannot supply 5,096 unique items by chance. A
    bigger space lowers the collision rate and does not remove it; dedup does.

    RAISES rather than returning short. A silently truncated set makes the last curve
    point smaller than it claims to be, which reads as a plateau in the readout.
    """
    rng = random.Random(seed)
    progs = _BY_FAMILY[family]
    seen = set(exclude)
    out = []
    tries = 0
    limit = 200 * max(n, 1)
    while len(out) < n:
        tries += 1
        if tries > limit:
            raise SystemExit(
                f"REFUSING: {family} yielded {len(out)} distinct instances in {tries} draws, "
                f"wanted {n}. The generator's operand space is too small for this set size -- "
                f"widen the range rather than accepting a short or duplicated set.")
        name, fn = progs[len(out) % len(progs)]
        ins, lines, ans = fn(rng)
        rec = {"family": family, "program": name, "instruction": ins,
               "solution": lines, "answer": ans, "operands": _operands(ins)}
        k = _key(rec)
        if k in seen:
            continue
        seen.add(k)
        out.append(rec)
    return out


def _header(family, kind, n, items):
    h = {"_header": True, "family": family, "kind": kind, "n": n,
         "generator": "mathbank/math_programs_l5_ext_novel.py",
         "programs": sorted({r["program"] for r in items}),
         "rule": RULE,
         "seed": TEST_SEED if kind == "test" else POOL_SEED,
         "absence_basis": ("CONSTRUCTED, not selected: the operator, its rule and its phrasing "
                           "were invented 2026-09-05, after every corpus in the mix was built. "
                           f"A scan cannot certify absence -- see {ABSENCE_FACT}, where every "
                           "probed pre-existing family was found present in OpenWebMath."),
         "collision_bound": ("rule and rule-phrasing 0 hits in 105,000 docs over 7 domains "
                             "(bounded < 0.0029% by the rule of three, NOT zero); bare '@' glyph "
                             f"0.039%, all 41 read and all '@'-as-at. {COLLISION_FACT}")}
    if kind == "pool":
        h["curve_n"] = list(CURVE_N)
        h["nesting"] = "the n-set for each n is items[:n]; adjacent points differ only by n"
    return h


def write_sets(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for family in sorted(_BY_FAMILY):
        test = generate(family, N_TEST, TEST_SEED)
        # exclude=test: the pool is disjoint BY CONSTRUCTION, and _assert_disjoint then
        # verifies the construction rather than hoping two seeds missed each other.
        pool = generate(family, N_POOL, POOL_SEED, exclude={_key(r) for r in test})
        _assert_disjoint(family, test, pool)
        for kind, items, n in (("test", test, N_TEST), ("pool", pool, N_POOL)):
            p = os.path.join(out_dir, f"{family}_{kind}.jsonl")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(_header(family, kind, n, items), ensure_ascii=False) + "\n")
                for r in items:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            written.append((os.path.relpath(p, ROOT), len(items)))
    return written


def _assert_disjoint(family, test, pool):
    """RAISES on overlap. Both units, because either alone misses a real leak."""
    t_text = {r["instruction"] for r in test}
    p_text = {r["instruction"] for r in pool}
    both_text = t_text & p_text
    t_key = {_key(r) for r in test}
    p_key = {_key(r) for r in pool}
    both_key = t_key & p_key
    if both_text or both_key:
        raise SystemExit(
            f"REFUSING to write {family}: test and pool overlap -- "
            f"{len(both_text)} by exact prompt, {len(both_key)} by (program, operands). "
            f"A shared instance turns the curve into recall. Example: "
            f"{sorted(both_key)[:2] if both_key else sorted(both_text)[:1]}")


def _selftest():
    fails = []
    import tempfile
    d = tempfile.mkdtemp(prefix="novel_ops_st_")
    try:
        written = write_sets(d)
        if len(written) != 4:
            fails.append(f"wrote {len(written)} files, want 4 (S/P x test/pool)")
        for family in ("S", "P"):
            test = generate(family, N_TEST, TEST_SEED)
            pool = generate(family, N_POOL, POOL_SEED, exclude={_key(r) for r in test})
            # 1. determinism: the same seed twice is the same set, or the test set is
            #    not fixed and a rescore silently measures something else.
            if [r["instruction"] for r in test] != [r["instruction"] for r in generate(family, N_TEST, TEST_SEED)]:
                fails.append(f"{family}: test set is not reproducible at its own seed")
            # 2. disjointness, both units, asserted here as well as at write time
            if {r["instruction"] for r in test} & {r["instruction"] for r in pool}:
                fails.append(f"{family}: test/pool share a prompt")
            if {_key(r) for r in test} & {_key(r) for r in pool}:
                fails.append(f"{family}: test/pool share a (program, operands) key")
            # 3. NESTING: every curve point must be a prefix of the next.
            for a, b in zip(CURVE_N, CURVE_N[1:]):
                if [r["instruction"] for r in pool[:a]] != [r["instruction"] for r in pool[:b]][:a]:
                    fails.append(f"{family}: n={a} is not a prefix of n={b}")
            # 4. the answers are the generator's, not re-derived here -- verify a sample
            #    against run_math_short.verify, the bank's own checker.
            from run_math_short import verify
            for r in test[:50] + pool[:50]:
                _out, ok = verify(r["instruction"], r["solution"], r["answer"])
                if not ok:
                    fails.append(f"{family}: emitted item fails the bank verifier: {r['program']}")
                    break
            # 5. operand extraction must actually find operands, or _key collapses to
            #    the program name and the semantic overlap check silently passes.
            empty = [r for r in test if not r["operands"]]
            if empty:
                fails.append(f"{family}: {len(empty)} item(s) with no extracted operands")
        # 6. THE OVERLAP CHECK MUST HAVE POWER: feed it a known collision.
        t = generate("S", 10, TEST_SEED)
        try:
            _assert_disjoint("S", t, list(t))
            fails.append("_assert_disjoint accepted a set overlapping itself")
        except SystemExit:
            pass
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    for f in fails:
        print(f"BUG {f}", file=sys.stderr)
    print(f"emit_novel_ops selftest: {'PASS (6 worlds)' if not fails else f'{len(fails)} BUG(S)'}")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    for rel, n in write_sets(a.out):
        print(f"wrote {rel}: {n} instances")
    return 0


if __name__ == "__main__":
    sys.exit(main())
