#!/usr/bin/env python3
"""Assert experiment 1's arm mixes schedule the exposures the axis claims.

    python3 scripts/e1_arm_plan_check.py            # all five arms
    python3 scripts/e1_arm_plan_check.py --arm n64  # one

Run BEFORE each arm launches (4c's ruling (4), 2026-09-05). The readout of
runs/prereg.jsonl#conversion_rate_0905 is a curve against n, the exposure count, so a mix that
schedules 0.95n or 1.02n produces a curve whose x-axis is wrong by a factor nothing downstream
can recover. Two defects already found this way before any arm ran:

  the 5%% val split came off the FRONT of every domain, injection shards included, so the epoch
  cap truncated the tail of the interleaved permutation -- n64 wanted 1,625 rows and could draw
  1,542, and ~40 of n1's 1,000 documents would have had ZERO exposures

  build_mix's budget is total_tokens/seq (4096) while a pool packs at seq+1 (4097), so a weight
  derived from a token count asks for one row more than the pool holds at every size

WHAT THIS CHECKS AND WHY IT IS NOT THE CONFIG ALONE. The row arithmetic is recomputed from the
SHARD -- token count, pool rows, exposures per document -- and compared against what the mix's
weight will draw. That is the gap both defects above live in: the weight and the rows the weight
actually gets. The exposure count comes from the shard's own `_exposure` field and document
identity from `_s_index`, neither of which build_mix reads.

TWO LEVELS, AND ONLY ONE OF THEM RUNS OFF THE POD. Everything above is arithmetic over files and
runs anywhere. The strongest assertion -- build the real plan through train.build_mix and count
what the plan CONTAINS -- needs the domains' token caches, which live beside the run on the pod.
That world is attempted and SKIPS with a named reason when the caches are absent, so a green run
here is not the same claim as a green run there. It prints which level ran.

THE WORLD-2 STRIPE IS WHY BOTH RANKS ARE BUILT when the plan level does run. The plan is striped
`plan[:, rank::world]` (train.py:2241), so rank 0 alone holds about half the exposures and a
per-rank count would read n/2 and look like a defect. The claim "each document appears exactly n
times" is a claim about the UNION of the ranks, which is what the model sees.
"""

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

ARMS = (("control_arm", 0), ("n1", 1), ("n8", 8), ("n64", 64), ("n256", 256))
SEQ = 4096


def _shard_docs(domain):
    """(_s_index, _exposure) per document line of a shard, in file order.

    File order is what matters: _domain_seqs concatenates documents in the order they appear and
    packs them into seq+1 rows, so the k-th document lands in row k*len/(seq+1) -- but this
    function is only used for the COUNTS, which are order-independent, and for the header.
    """
    p = os.path.join(ROOT, "data", "corpus", domain, f"{domain}_000.jsonl")
    header, docs = None, []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("_header"):
                header = r
                continue
            docs.append((r["_s_index"], r["_exposure"]))
    return header, docs


def check_arm(name, n, verbose=True):
    """Returns (problems, numbers) for one arm. Nothing is asserted about the natural domains
    beyond their weights summing correctly -- their row counts are the control's by construction
    and any drift there shows up as a weight-sum failure."""
    problems, nums = [], {}
    mix_path = os.path.join(ROOT, "data", f"mix_e1_{name}.json")
    if not os.path.exists(mix_path):
        return [f"{name}: {os.path.relpath(mix_path, ROOT)} does not exist"], nums
    with open(mix_path, encoding="utf-8") as fh:
        mix = json.load(fh)

    # 1. THE WEIGHTS STILL SUM TO ONE. Cheap, and it is the only thing that would catch a natural
    #    domain having been reweighted by hand after the build.
    w_sum = sum(v["weight"] for v in mix["domains"].values())
    nums["weight_sum"] = w_sum
    if abs(w_sum - 1.0) > 1e-12:
        problems.append(f"{name}: weights sum to {w_sum:.15f}, not 1 within 1e-12")

    # 2. THE INJECTION DOMAINS CARRY val_frac 0. Without it the split comes off the front of the
    #    shard and the cap truncates the tail -- checked here rather than trusted because the mix
    #    is a file anyone can edit, and the failure it causes is invisible in the loss.
    inj = [d for d in mix["domains"] if d.startswith(("s_inject_", "p_format"))]
    for d in inj:
        if mix["domains"][d].get("val_frac") != 0:
            problems.append(f"{name}: domain {d} has val_frac "
                            f"{mix['domains'][d].get('val_frac')!r}, want 0 -- a 5% split off the "
                            f"front makes the realised exposure count 0.95n")
        if mix["domains"][d].get("epochs") != 1:
            problems.append(f"{name}: domain {d} has epochs "
                            f"{mix['domains'][d].get('epochs')!r}, want 1 -- the repeats are "
                            f"materialised in the shard, so epochs>1 multiplies the exposures")

    # 3. EXPOSURE COUNTS COME FROM THE SHARD, not from the arm's name. If the wrong shard is wired
    #    into the mix, everything else here still passes.
    s_dom = [d for d in mix["domains"] if d.startswith("s_inject_")]
    if n and len(s_dom) != 1:
        problems.append(f"{name}: {len(s_dom)} s_inject domains in the mix, want exactly 1")
    elif n:
        header, docs = _shard_docs(s_dom[0])
        per_doc = collections.Counter(i for i, _ in docs)
        nums["s_docs"] = len(per_doc)
        nums["s_exposures_per_doc"] = sorted(set(per_doc.values()))
        if sorted(set(per_doc.values())) != [n]:
            problems.append(f"{name}: shard {s_dom[0]} gives each document "
                            f"{sorted(set(per_doc.values()))} exposures, want exactly [{n}]")
        if header.get("exposures") != n:
            problems.append(f"{name}: shard header says exposures={header.get('exposures')}, "
                            f"the arm is n={n}")
        # The header is a claim; per_doc is the count. Both are checked because they can disagree.
        if len(per_doc) != 1000:
            problems.append(f"{name}: shard holds {len(per_doc)} distinct documents, want 1000")
    if not n and s_dom:
        problems.append(f"{name}: control_arm must inject NO S, but the mix names {s_dom}")

    # 4. P IS ONE EXPOSURE IN EVERY ARM, control included. This is what makes a P movement
    #    readable at all: if P's count varied with n, a P change could not be attributed.
    if "p_format" not in mix["domains"]:
        problems.append(f"{name}: no p_format domain -- P is the composition control and runs in "
                        f"every arm, including control_arm")
    else:
        _h, pdocs = _shard_docs("p_format")
        pc = collections.Counter(i for i, _ in pdocs)
        nums["p_docs"] = len(pc)
        nums["p_exposures_per_doc"] = sorted(set(pc.values()))
        if sorted(set(pc.values())) != [1]:
            problems.append(f"{name}: p_format gives each document {sorted(set(pc.values()))} "
                            f"exposures, want exactly [1] in every arm")

    # 5. THE ROWS THE WEIGHT DRAWS EQUAL THE ROWS THE POOL HOLDS. This is the one that needed the
    #    row-derived weights: `want` is int(budget_rows * weight) and the pool is
    #    tokens//(seq+1). Computed here from the mix and the shard, so it does not depend on
    #    build_mix -- and then the plan is built below and compared against it.
    budget_rows = mix["total_tokens"] / mix["seq"]
    for d in inj:
        _h, dd = _shard_docs(d)
        # Token count is measured once by the caller and cached in the role string; recount it
        # rather than parse prose.
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
        p = os.path.join(ROOT, "data", "corpus", d, f"{d}_000.jsonl")
        toks = 0
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if not r.get("_header"):
                    toks += len(tk.encode(r["text"]).ids)
        pool_rows = toks // (mix["seq"] + 1)
        want = int(budget_rows * mix["domains"][d]["weight"])
        nums[f"{d}_pool_rows"] = pool_rows
        nums[f"{d}_want_rows"] = want
        if want != pool_rows:
            problems.append(
                f"{name}: domain {d} wants {want} rows and its pool holds {pool_rows}. The epoch "
                f"cap will {'truncate' if want > pool_rows else 'under-draw'} the difference, so "
                f"the realised exposure count is not the arm's n. Derive the weight from the row "
                f"count ({pool_rows}/{budget_rows:.0f}), not from the token count: build_mix's "
                f"budget divides by seq ({mix['seq']}) while a pool packs at seq+1.")
    if verbose:
        print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in nums.items()))
    return problems, nums


def check_plan(name, n):
    """Build the REAL plan through train.build_mix and count the exposures it schedules.

    This is the level the file-arithmetic above cannot reach: everything else infers what the
    weight will draw, and this reads what build_mix actually put in the plan. It needs each
    domain's token cache, so it SKIPS with a named reason rather than passing when they are
    absent -- a skip that read as a pass would be the worst outcome here, because this is the
    only assertion that would survive a change to build_mix's own arithmetic.

    Both ranks, unioned: the plan is striped plan[:, rank::world] (train.py:2241), so rank 0
    holds about half the exposures and a single-rank count reads n/2.
    """
    import train

    mix_path = os.path.join(ROOT, "data", f"mix_e1_{name}.json")
    with open(mix_path, encoding="utf-8") as fh:
        mix = json.load(fh)
    missing = [d for d in mix["domains"]
               if not os.path.exists(train._domain_cache_path(d))]
    if missing:
        return None, (f"{len(missing)} of {len(mix['domains'])} token cache(s) absent "
                      f"({', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}) -- "
                      f"run on the pod, where the caches live beside the run")
    tok = train.build_tokenizer() if hasattr(train, "build_tokenizer") else None
    problems = []
    inj = [d for d in mix["domains"] if d.startswith(("s_inject_", "p_format"))]
    names = list(mix["domains"])
    rows_by_dom = collections.Counter()
    for rank in (0, 1):
        mine, _val = train.build_mix(mix_path, tok, rank == 0, False, rank=rank, world=2)
        doms = train.Cfg._plan_domains
        for di, dname in enumerate(names):
            rows_by_dom[dname] += int((doms == di).sum()) if doms is not None else 0
    for d in inj:
        want = int((mix["total_tokens"] / mix["seq"]) * mix["domains"][d]["weight"])
        got = rows_by_dom[d]
        if got != want:
            problems.append(f"{name}: the built plan holds {got} rows of {d}, the weight asks "
                            f"{want}. The cap bound, or the stripe dropped rows.")
    return problems, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, choices=[a for a, _ in ARMS])
    ap.add_argument("--plan", action="store_true",
                    help="also build the real plan through train.build_mix (needs the token "
                         "caches; SKIPs with a reason when they are absent)")
    a = ap.parse_args()
    todo = [(n, k) for n, k in ARMS if a.arm is None or n == a.arm]
    allp, skips = [], []
    for name, n in todo:
        problems, _ = check_arm(name, n)
        allp += problems
        if a.plan:
            pp, skip = check_plan(name, n)
            if skip:
                skips.append(f"{name}: {skip}")
            else:
                allp += pp
    for p in allp:
        print(f"BUG {p}", file=sys.stderr)
    for s in skips:
        print(f"SKIP plan level -- {s}")
    level = "file arithmetic" if (not a.plan or skips) else "file arithmetic + built plan"
    print(f"e1 arm plan check: {'PASS' if not allp else f'{len(allp)} BUG(S)'} "
          f"({len(todo)} arm(s), {level})")
    return 1 if allp else 0


if __name__ == "__main__":
    sys.exit(main())
