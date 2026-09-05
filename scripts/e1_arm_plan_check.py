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
#: Rows ckpt_b0_headmix_armA.pt's row_cursor already consumed, summed from the checkpoint on the
#: pod (as_of_step 3815, cursor_seed 42): math_owm_stage2 64450, en_c4_stage2 40082, cot 19776,
#: textbook_30b 24700, chatml 1796, chat_qa 1714, zh_web 7496, code_py_starcoder 80380,
#: code_py_rp1t 3766. Hard-coded rather than read, because this check must run where the
#: checkpoint is not; --plan reads the real cursor and cross-checks it against this number.
CURSOR_ROWS = 244160


def _shard_docs(domain):
    """(header, [(_s_index, _exposure)]) for a shard, the header from its sidecar.

    File order is what matters: _domain_seqs concatenates documents in the order they appear and
    packs them into seq+1 rows, so the k-th document lands in row k*len/(seq+1) -- but this
    function is only used for the COUNTS, which are order-independent, and for the header.

    THE HEADER IS A SIDECAR, not line 1. It was in-band until 2026-09-05, which is what made
    every injection domain's cache build raise KeyError: 'content' -- train._jsonl_content reads
    that key on every line with no header skip. datagen/build_s_inject.py now writes
    <shard>.meta.json and asserts the shard itself is readable; this reads the same pair.
    """
    p = os.path.join(ROOT, "data", "corpus", domain, f"{domain}_000.jsonl")
    meta = p + ".meta.json"
    header, docs = None, []
    if os.path.exists(meta):
        with open(meta, encoding="utf-8") as fh:
            header = json.load(fh)
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("_header"):
                # An in-band header means the shard predates the sidecar and its cache CANNOT be
                # built. Refuse rather than skip: skipping is what let the old format reach the pod.
                raise SystemExit(
                    f"REFUSING: {os.path.relpath(p, ROOT)} carries an in-band _header line. "
                    f"train._jsonl_content reads [\"content\"] on every line with no header skip, "
                    f"so this domain's token cache cannot be built. Rebuild with "
                    f"datagen/build_s_inject.py, which writes the header to "
                    f"{os.path.basename(meta)}.")
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
    # THE REMAINDER, NOT total_tokens. build_mix subtracts the resume cursor before computing
    # want: `rows = total_tokens/seq`, then `rows = max(0, rows - spent)`, then
    # `want = int(rows * frac * weight)`. An arm resumes ckpt_b0_headmix_armA.pt, whose row_cursor
    # sums to CURSOR_ROWS, so its total_tokens carries the RESUMED run's budget and the rows the
    # weights are shares of is the difference. Reading total_tokens here instead would compute
    # want against 276,160 rows and report every arm as over-drawing by 8.6x -- the check would
    # be wrong in exactly the direction that hides the defect it exists for.
    budget_rows = mix["total_tokens"] / mix["seq"] - CURSOR_ROWS
    for d in inj:
        _h, dd = _shard_docs(d)
        # Token count is measured once by the caller and cached in the role string; recount it
        # rather than parse prose.
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
        p = os.path.join(ROOT, "data", "corpus", d, f"{d}_000.jsonl")
        toks, ndocs = 0, 0
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                # THE SAME KEY train._jsonl_content READS, and every line is a document now: the
                # header moved to a sidecar on 2026-09-05 because an in-band one made the cache
                # build raise KeyError: 'content'. Reading "text" here would have counted zero
                # tokens against the rebuilt shards and reported every pool as empty.
                if "_header" in r:
                    raise SystemExit(
                        f"REFUSING: {os.path.relpath(p, ROOT)} carries an in-band _header line, so "
                        f"train._jsonl_content cannot build this domain's cache. Rebuild with "
                        f"datagen/build_s_inject.py.")
                toks += len(tk.encode(r["content"]).ids)
                ndocs += 1
        # PLUS ONE <eos> PER DOCUMENT. train.encode emits a single <eos>-separated stream --
        # np.append(p, eos) per document (train.py:1707) -- so the stream is sum(len(ids)) + ndocs,
        # not sum(len(ids)). Measured against the caches the pod actually built: without the eos
        # this read 202/1623/6495 rows where the caches hold 204/1639/6557, and the error was
        # INVISIBLE because `want` and `pool_rows` were both derived from the same undercount, so
        # want == pool passed on a shared error. p_format and n1 agree either way (25.4 rows, the
        # floor absorbs 1000 tokens), which is why the two smallest arms could not reveal it.
        toks += ndocs
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
        # AND AGAINST THE CACHE THE TRAINER WILL ACTUALLY READ, which is the only reading neither
        # side of the comparison above produced. `want` and `pool_rows` are both derived from this
        # script's token count, so a miscount cancels: dropping the per-document <eos> made both
        # 202 where the cache holds 204, and want == pool passed on the shared error through five
        # arms and two reviews. The cache is train.py's own output -- reshaped at
        # `n = len(data) // (seq+1)` -- so it cannot share this script's arithmetic.
        #
        # Imported inside the branch: check_arm is the level that runs anywhere, and importing
        # train costs seconds and pulls Triton. Absent cache = no assertion, reported as such
        # below rather than passing quietly.
        import train
        cp = train._domain_cache_path(d)
        if os.path.exists(cp):
            # THE CO-RESIDENCY REFUSAL BEFORE THE READ, not because this read is large -- an
            # injection cache is 204-6557 rows, a few MB -- but because the rule keys on the READ
            # PATH and not on the reader's estimate of its own size. torch.load by path skips
            # train._domain_seqs, which is where the other half of the check looks, so a
            # by-path read is exactly the hole coresident_cache_refusal exists to close. It
            # returns 0 silently when no claim holds cards, which is the case for this script's
            # normal off-pod run.
            sys.path.insert(0, os.path.join(ROOT, "eval"))
            import cache_guard
            cache_guard.assert_not_co_resident([d])
            import torch
            _len = int(torch.load(cp, map_location="cpu", weights_only=False).shape[0])
            _n = _len // (mix["seq"] + 1)
            nums[f"{d}_cache_rows"] = _n
            if _n != pool_rows:
                problems.append(
                    f"{name}: domain {d} packs to {pool_rows} rows by this script's count, but the "
                    f"token cache at {cp} holds {_n} ({_len} tokens). The cache is what the run "
                    f"reads, so the weight is a share of the wrong number. A per-document <eos> "
                    f"(train.py:1707) is the difference this has been before: the stream is "
                    f"sum(len(ids)) + n_docs, not sum(len(ids)).")
        else:
            nums[f"{d}_cache_rows"] = "absent"
    if verbose:
        print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in nums.items()))
    return problems, nums


def _ckpt_cursor(path):
    """(cursor dict, seed, srcfp, reason-it-is-absent) from the checkpoint the arms resume.

    THE CURSOR HAS TO COME FROM THE CHECKPOINT, not from CURSOR_ROWS. check_plan's two
    cursor assertions -- a natural domain drew rows at all, and its FIRST drawn row is its
    cursor -- both begin `if d not in cur`, so an empty cur skips them silently and
    build_mix is handed row_cursor=None, which builds a FRESH plan. The check then prints
    PASS having asserted nothing about the thing it exists to assert: that no arm re-reads
    rows the control already trained on. Found 2026-09-05 before the pod run, by reading
    main() rather than the function (main called check_plan(name, n) and never passed a
    cursor -- the parameter had no caller).
    """
    if not os.path.exists(path):
        return None, None, None, f"checkpoint absent at {path}"
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cur = ck.get("row_cursor")
    if not cur:
        why = ck.get("row_cursor_refused") or "the checkpoint carries no row_cursor key"
        return None, None, None, str(why)
    return (dict(cur), ck.get("row_cursor_seed"), dict(ck.get("row_cursor_srcfp") or {}),
            None)


def check_plan(name, n, cursor=None, cursor_seed=None, cursor_srcfp=None):
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
    # THE CURSOR IS CHECKED BEFORE THE CACHES, so this refusal is reachable without a pod. Below
    # the cache check it would be dead code off-pod -- exactly the shape being fixed here, where a
    # guard sits behind an earlier return and can never fire on the path that needs it.
    if not cursor:
        return [f"{name}: check_plan called with NO cursor, so build_mix would build a FRESH plan "
                f"and both cursor assertions (a natural domain drew rows; its first drawn row is "
                f"its cursor) would be skipped on `d not in cur`. A PASS would then say nothing "
                f"about whether the arm re-reads rows the control already trained on."], None
    missing = [d for d in mix["domains"]
               if not os.path.exists(train._domain_cache_path(d))]
    if missing:
        return None, (f"{len(missing)} of {len(mix['domains'])} token cache(s) absent "
                      f"({', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}) -- "
                      f"run on the pod, where the caches live beside the run")
    # build_tokenizer(texts) takes a `texts` argument its body never reads -- it only loads
    # TOK_PATH -- so [] is passed. It is called rather than Tokenizer.from_file because it is what
    # sets train.VOCAB_ID, and _domain_seqs REFUSES to stamp a cache without it. `hasattr` was
    # here instead and hid this: the attribute exists, so the guard passed and the CALL raised
    # TypeError on the pod. A capability check that does not check the signature is not one.
    tok = train.build_tokenizer([])
    problems = []
    inj = [d for d in mix["domains"] if d.startswith(("s_inject_", "p_format"))]
    nat = [d for d in mix["domains"] if d not in inj]
    names = list(mix["domains"])
    # THE REAL CURSOR, and cross-checked against the constant the file-level check uses. If the
    # checkpoint's cursor ever differs from CURSOR_ROWS, every want above was computed against the
    # wrong remainder and its PASS meant nothing -- so this compares them rather than trusting one.
    cur = dict(cursor or {})
    if cur:
        spent = sum(cur.values())
        if spent != CURSOR_ROWS:
            problems.append(f"{name}: the checkpoint's row_cursor sums to {spent} rows but this "
                            f"check's CURSOR_ROWS is {CURSOR_ROWS}. Every want the file-level "
                            f"check computed used the wrong remainder.")
    else:
        # NOT A PASS. Everything below that mentions the cursor is keyed on `d in cur`, so an
        # empty cursor turns the two assertions that matter into no-ops while the count checks
        # still print PASS. The caller is responsible for supplying it; refuse rather than
        # measure a fresh plan and report it as the resumed one.
        problems.append(f"{name}: check_plan ran with NO cursor, so build_mix built a FRESH plan "
                        f"and both cursor assertions (a natural domain drew rows; its first drawn "
                        f"row is its cursor) were skipped. A PASS here would mean nothing about "
                        f"whether the arm re-reads rows the control already trained on.")
    rows_by_dom = collections.Counter()
    # LOWEST ROW INDEX PER DOMAIN, unioned over both ranks. mine[1] is the row index into that
    # domain's pool and mine[0] the domain id (train.py:2316-2317), so this is the pool position
    # the arm actually starts reading from -- the quantity 4c's assertion is about.
    lowest = {}
    for rank in (0, 1):
        mine, _val = train.build_mix(mix_path, tok, rank == 0, False, rank=rank, world=2,
                                     row_cursor=cur or None, cursor_seed=cursor_seed,
                                     cursor_srcfp=cursor_srcfp)
        for di, dname in enumerate(names):
            sel = mine[0] == di
            k = int(sel.sum())
            rows_by_dom[dname] += k
            if k:
                lo = int(mine[1][sel].min())
                lowest[dname] = min(lowest.get(dname, lo), lo)
    for d in inj:
        # The REMAINDER, as above: build_mix subtracts the cursor before computing want.
        want = int((mix["total_tokens"] / mix["seq"] - sum(cur.values() or [0]))
                   * mix["domains"][d]["weight"])
        got = rows_by_dom[d]
        if got != want:
            problems.append(f"{name}: the built plan holds {got} rows of {d}, the weight asks "
                            f"{want}. The cap bound, or the stripe dropped rows.")
    # 4c's assertion: A NATURAL DOMAIN CONTINUES FROM ITS CURSOR RATHER THAN RESTARTING. This is
    # the one that catches a floored budget and a discarded cursor, both of which are silent: a
    # floored budget draws nothing at all, and a discarded cursor draws the SAME row count from
    # row 0 -- re-reading rows the control already saw while every count still matches.
    for d in nat:
        if d not in cur:
            continue
        if rows_by_dom[d] == 0:
            problems.append(f"{name}: natural domain {d} drew 0 rows. Its cursor is at "
                            f"{cur[d]}; a budget that floors to zero after the cursor "
                            f"subtraction looks exactly like this.")
            continue
        # THE FIRST DRAWN ROW IS THE CURSOR, modulo the pool length -- build_mix reads
        # `torch.arange(used[name], used[name] + want) % len(pool)`, so a cursor past one epoch
        # wraps. A DISCARDED cursor restarts the domain at row 0 and draws the SAME COUNT, so
        # every count assertion above still passes while the arm re-reads rows the control has
        # already seen. This is the only assertion that separates those two.
        want_lo = cur[d]
        if lowest.get(d) is not None and lowest[d] != want_lo:
            problems.append(
                f"{name}: natural domain {d} starts at pool row {lowest[d]}, its checkpoint "
                f"cursor is {want_lo}. 0 means the cursor was discarded and the domain restarted, "
                f"which re-reads rows the control arm already trained on while the row COUNT still "
                f"matches -- the failure no count check can see. (If the pool is shorter than the "
                f"cursor the read wraps, and this assertion needs the modulo.)")
    return problems, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, choices=[a for a, _ in ARMS])
    ap.add_argument("--plan", action="store_true",
                    help="also build the real plan through train.build_mix (needs the token "
                         "caches; SKIPs with a reason when they are absent)")
    ap.add_argument("--ckpt", default="ckpt_b0_headmix_armA.pt",
                    help="the checkpoint the arms resume; --plan reads its row_cursor and "
                         "asserts each natural domain continues from it")
    a = ap.parse_args()
    todo = [(n, k) for n, k in ARMS if a.arm is None or n == a.arm]
    allp, skips = [], []
    cur = seed = srcfp = None
    if a.plan:
        # THE CURSOR IS READ ONCE, HERE, and a missing one SKIPS the plan level rather than
        # building a fresh plan and calling it a pass. Before 2026-09-05 main called
        # check_plan(name, n) with no cursor at all: build_mix got row_cursor=None, both cursor
        # assertions were skipped on `d not in cur`, and the pod run would have printed
        # "file arithmetic + built plan / PASS" while asserting nothing about the resume.
        ck = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(ROOT, a.ckpt)
        cur, seed, srcfp, why = _ckpt_cursor(ck)
        if cur is None:
            skips.append(f"all arms: {why} -- the plan level needs the resumed checkpoint's "
                         f"row_cursor, and without it the two cursor assertions do not run")
    for name, n in todo:
        problems, _ = check_arm(name, n)
        allp += problems
        if a.plan and cur is not None:
            pp, skip = check_plan(name, n, cursor=cur, cursor_seed=seed, cursor_srcfp=srcfp)
            if skip:
                skips.append(f"{name}: {skip}")
            else:
                allp += pp
    for p in allp:
        print(f"BUG {p}", file=sys.stderr)
    for s in skips:
        print(f"SKIP plan level -- {s}")
    if a.plan and not skips:
        print(f"cursor: {len(cur)} domains, sums to {sum(cur.values())} rows "
              f"(CURSOR_ROWS {CURSOR_ROWS}), seed {seed}")
    level = "file arithmetic" if (not a.plan or skips) else "file arithmetic + built plan"
    print(f"e1 arm plan check: {'PASS' if not allp else f'{len(allp)} BUG(S)'} "
          f"({len(todo)} arm(s), {level})")
    return 1 if allp else 0


if __name__ == "__main__":
    sys.exit(main())
