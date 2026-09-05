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
`plan[:, rank::world]` (train.py:2316), so rank 0 alone holds about half the exposures and a
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
#: The anneal_frac the LAUNCH LINES pass (--anneal_frac 0, 4c's ruling 2026-09-05, prereg
#: amendment 8), and therefore the value this check builds the plan at. build_mix reads
#: Cfg.anneal_frac (0.10 by default, train.py:377) and never the mix file's own "anneal_frac"
#: key, so all five arm files declare 0.0 and would silently get two phases. A check built at
#: the default would certify a two-phase plan while the run builds a one-phase one -- and the
#: two differ: `want = int(rows * frac * weight)` runs once per phase and int(0.9x) + int(0.1x)
#: <= int(x), which cost exactly one row in every injection arm (n1 25->24, n8 204->203,
#: n64 1639->1638, n256 6557->6556). One row is ~39 document exposures in an interleaved shard;
#: at n1 that is ~40 of 1,000 documents at ZERO exposures while the axis reads 1.
ANNEAL_FRAC = 0.0


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

    THE CURSOR HAS TO COME FROM THE CHECKPOINT, not from CURSOR_ROWS. check_plan's natural-domain
    assertions -- the domain drew rows at all, and it drew the count its cursor and weight name --
    both begin `if d not in cur`, so an empty cur skips them silently and
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


def pool_rows_of(domain, mix):
    """len(pools[domain]) as build_mix computes it: cache rows minus the val split.

    THE VAL SPLIT IS PART OF IT, and leaving it out would make the set assertion below wrong by
    5% of the pool for every natural domain -- an off-by-thousands modulus, so every expected
    index after the first wrap would be wrong and the assertion would fire on a healthy plan.
    Mirrors train.py's per-domain override (`val_frac` on the domain, else Cfg.val_frac, floored
    at 1 by max(1, ...) and capped at Cfg.val_rows_max) rather than restating a number.

    Read from the TOKEN CACHE, not recomputed from the corpus: the cache is what build_mix
    reshapes, and this whole check exists because a count recomputed alongside the thing it
    checks cannot disagree with it.

    mmap=True, SO THIS READS THE HEADER AND NOT THE TENSOR. The previous form was
    `torch.load(cp, map_location="cpu").shape[0]`, which pulls the whole file in to learn a
    shape: 85.2 GB for zh_web, which eval/cache_guard rightly refused while cards 0 were held
    by tilerl-mathrun2, killing three of five solo arms on 2026-09-05. Measured on the pod
    (/proc/self/io around the call): mmap reports the shape at 0.0 MB rchar, and on the three
    domains small enough to load both ways -- chatml, chat_qa, code_py_rp1t -- the mmap shape
    equals the full-load shape exactly, where the full load charges 156.0 / 152.8 / 1683.4 MB.
    So the co-residency call is gone from this function: there is no longer a large read for it
    to refuse, and leaving a refusal in front of a header read would refuse a cost that is not
    paid. The caches are flat 1-D token tensors (train.py:1958 `data[: n*(seq+1)].view(-1, seq+1)`),
    so shape[0] // (seq+1) is the row count under either load.
    """
    import train
    cp = train._domain_cache_path(domain)
    if not os.path.exists(cp):
        return 0
    import torch
    n_rows = int(torch.load(cp, map_location="cpu", weights_only=True, mmap=True).shape[0]
                 ) // (mix["seq"] + 1)
    dcfg = mix["domains"][domain]
    if "val_frac" in dcfg:
        vf = dcfg["val_frac"]
        n_val = 0 if vf == 0 else min(max(1, int(n_rows * vf)), train.Cfg.val_rows_max)
    else:
        n_val = min(max(1, int(n_rows * train.Cfg.val_frac)), train.Cfg.val_rows_max)
    return n_rows - n_val


def check_plan(name, n, cursor=None, cursor_seed=None, cursor_srcfp=None):
    """Build the REAL plan through train.build_mix and count the exposures it schedules.

    This is the level the file-arithmetic above cannot reach: everything else infers what the
    weight will draw, and this reads what build_mix actually put in the plan. It needs each
    domain's token cache, so it SKIPS with a named reason rather than passing when they are
    absent -- a skip that read as a pass would be the worst outcome here, because this is the
    only assertion that would survive a change to build_mix's own arithmetic.

    Both ranks, unioned: the plan is striped plan[:, rank::world] (train.py:2316), so rank 0
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
                f"and the natural-domain assertions (the domain drew rows; it drew the count its "
                f"cursor and weight name) would be skipped on `d not in cur`. A PASS would then "
                f"say nothing about whether the arm re-reads rows the control already trained "
                f"on."], None
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
    # THE PLAN IS BUILT AT THE LAUNCH'S anneal_frac, NOT AT Cfg's DEFAULT. build_mix reads
    # Cfg.anneal_frac (0.10 by default) and never the mix file's own "anneal_frac" key, so all five
    # arm files declare 0.0 and all five would silently get two phases. The launch lines pass
    # --anneal_frac 0 (4c's ruling 2026-09-05, amendment 8), and a check that built at 0.10 would
    # certify a two-phase plan while the run builds a one-phase plan -- disagreeing with the run by
    # construction, which is the class of failure this gate exists to catch.
    #
    # It also removes the lost row: `want = int(rows * frac * weight)` runs once per phase, and
    # int(0.9x) + int(0.1x) <= int(x), so double flooring cost exactly one row in every injection
    # arm (n1 25->24, n8 204->203, n64 1639->1638, n256 6557->6556). One row is ~39 document
    # exposures in an interleaved shard; at n1 that is ~40 of 1,000 documents at ZERO exposures
    # while the axis reads 1.
    _old_af = train.Cfg.anneal_frac
    train.Cfg.anneal_frac = ANNEAL_FRAC
    try:
        return _check_plan_inner(name, n, mix, mix_path, tok, dict(cursor), cursor_seed,
                                 cursor_srcfp)
    finally:
        train.Cfg.anneal_frac = _old_af


def _check_plan_inner(name, n, mix, mix_path, tok, cur, cursor_seed, cursor_srcfp):
    import train
    problems = []
    inj = [d for d in mix["domains"] if d.startswith(("s_inject_", "p_format"))]
    nat = [d for d in mix["domains"] if d not in inj]
    names = list(mix["domains"])
    # THE REAL CURSOR, cross-checked against the constant the file-level check uses. If the
    # checkpoint's cursor ever differs from CURSOR_ROWS, every want above was computed against the
    # wrong remainder and its PASS meant nothing -- so this compares them rather than trusting one.
    # `cur` is the caller's dict; check_plan refuses an empty one before reaching here, so the
    # no-cursor branch that used to live at this point is gone rather than duplicated.
    spent = sum(cur.values())
    if spent != CURSOR_ROWS:
        problems.append(f"{name}: the checkpoint's row_cursor sums to {spent} rows but this "
                        f"check's CURSOR_ROWS is {CURSOR_ROWS}. Every want the file-level "
                        f"check computed used the wrong remainder.")
    rows_by_dom = collections.Counter()
    # THE ROW COUNT PER DOMAIN, FROM Cfg._plan_domains -- NOT from build_mix's return value.
    #
    # WHAT WAS WRONG HERE (found 2026-09-05 on the pod, 10 BUG lines, all of them artifacts):
    # this read `mine, _val = train.build_mix(...)` and then `mine[0] == di` / `mine[1][sel]`.
    # build_mix does not return the plan. It returns the TOKEN tensor, shape (rows, seq+1):
    # train.py:2319 `out = torch.empty((mine.shape[1], Cfg.seq + 1), dtype=torch.int32)` and
    # train.py:2333 `return out, vcat`. So `mine[0]` was ROW ZERO'S TOKEN IDS compared against a
    # domain index, and `mine[1][sel]` was row one's token ids read as pool indices. The counts
    # that came out summed to 9 rows across a 31,994-row plan, and eight domains reported "drew 0
    # rows" while train.py's own mix lines in the same output showed them drawing thousands.
    #
    # The (domain, pool_index) plan is LOCAL to build_mix: train.py:2316 `mine = plan[:, :n][:,
    # rank::world]`, consumed at :2324 `out[m] = pools[name][mine[1][m]]`, never returned. What
    # escapes is Cfg._plan_domains (:2317, int8, one domain index per row of THIS rank's plan) --
    # enough for the count, and it carries no pool index, which is why the set assertion below
    # cannot be built from it and reads bytes instead.
    plan_dom = {}
    for rank in (0, 1):
        out, _val = train.build_mix(mix_path, tok, rank == 0, False, rank=rank, world=2,
                                    row_cursor=cur or None, cursor_seed=cursor_seed,
                                    cursor_srcfp=cursor_srcfp)
        pd = getattr(train.Cfg, "_plan_domains", None)
        if pd is None:
            problems.append(f"{name}: build_mix published no Cfg._plan_domains, so the per-domain "
                            f"row count could not be read -- that is a missing assertion, not a "
                            f"pass (train.py:2317 sets it)")
            return problems, None
        if int(pd.shape[0]) != int(out.shape[0]):
            # The two must describe the same rows or the count is about a different plan than
            # the tokens. Cheap, and it is the assertion that would have caught the defect above.
            problems.append(f"{name}: Cfg._plan_domains holds {int(pd.shape[0])} rows but the "
                            f"token tensor holds {int(out.shape[0])} -- they do not describe the "
                            f"same plan, so neither the count nor the row-count assertions are "
                            f"meaningful")
            return problems, None
        plan_dom[rank] = int(pd.shape[0])
        for di, dname in enumerate(names):
            rows_by_dom[dname] += int((pd == di).sum())
        del out, pd
    # EVERY DOMAIN INDEX IN THE PLAN IS A DOMAIN THIS CHECK KNOWS. Without it a plan carrying an
    # index past len(names) would contribute to no domain's count and every count would still
    # agree -- the shape of the defect this rewrite fixes, where a wrong axis produced counts that
    # summed to 9 out of 31,994 and nothing compared the parts to the whole.
    if sum(rows_by_dom[d] for d in names) != sum(plan_dom.values()):
        problems.append(f"{name}: the per-domain counts sum to "
                        f"{sum(rows_by_dom[d] for d in names)} but the plan holds "
                        f"{sum(plan_dom.values())} rows across both ranks -- some rows carry a "
                        f"domain index this check did not count.")
    # THE REMAINDER AND THE PHASES, hoisted: both loops below assert against them, and computing
    # them once is what keeps the injection and natural assertions talking about the same plan.
    # build_mix subtracts the cursor from the budget ONCE, globally (train.py:2251
    # `rows = max(0.0, rows - spent)`), and then applies each domain's weight to that remainder.
    # SUMMED OVER THE PHASES build_mix will run: `want = int(rows * frac * weight)` once per phase,
    # and int() truncates in EACH, so sum(int(rows*frac*w)) can be a row less than int(rows*w).
    # Phase by phase rather than a single int() is what makes this correct at any ANNEAL_FRAC
    # instead of only at 0 -- the single-phase form was right only because the launch passes
    # --anneal_frac 0, and a check that is right by coincidence goes wrong when the coincidence
    # ends (train.py:2174 builds `phases` from Cfg.anneal_frac).
    rows_rem = mix["total_tokens"] / mix["seq"] - sum(cur.values() or [0])
    _phases = ([(1.0, "weight")] if not ANNEAL_FRAC
               else [(1 - ANNEAL_FRAC, "weight"), (ANNEAL_FRAC, "anneal")])

    def _want_of(d, used_d):
        """(want, pool_len, cap_bound) -- what build_mix will allocate to domain `d`.

        The cap is not decoration here: p_format's pool is 20 rows with epochs 1, so an
        uncapped expectation and the plan disagree by construction and the check reports a
        defect in the run that is really a defect in the check. train.py:2255-2262 -- want is
        capped at int(len(pool) * epochs) - used[name], floored at 0, and `used` advances by
        the capped amount within the phase loop.

        cap_bound SAYS WHETHER pool_len ACTUALLY ENTERED THE ANSWER. Measured 2026-09-05 by
        mutation: overstating code_py_rp1t's pool by 4,000 rows left the check GREEN, because
        want (507) is nowhere near the cap (pool 410,992 minus a 3,766 cursor) so the min() never
        selects it. A quantity the verdict cannot depend on is not being checked by reading it --
        it is being read and discarded. The caller uses this flag to assert pool_len some other
        way when the cap did not bind.
        """
        pool_len = pool_rows_of(d, mix)
        if not pool_len:
            return None, None, False
        dcfg = mix["domains"][d]
        total, bound = 0, False
        for frac, key in _phases:
            raw = int(rows_rem * frac * dcfg.get(key, dcfg["weight"]))
            cap = int(pool_len * dcfg.get("epochs", 1)) - used_d
            w = max(0, min(raw, cap))
            if w != raw:
                bound = True
            total += w
            used_d += w
        return total, pool_len, bound

    def _check_pool_len(d, pool_len, cap_bound):
        """When the cap did not bind, assert pool_len against a source that is not itself.

        The plan level reads pool_len from the cache header. If the cap does not bind, nothing
        downstream depends on it, so a wrong pool_len is invisible here -- and pool_len is exactly
        the quantity a rebuilt or truncated cache changes. So compare it against the SHARD for the
        domains that have one (the injection domains, whose row count IS the measurement and whose
        pool the file level already counts from the shard bytes), and against the epoch-cap
        arithmetic for the rest.
        """
        if d not in inj:
            # Natural domains have no shard to count; their pool is the cache's whole row count
            # minus the val split. The one thing assertable without a second source is that the
            # cursor lies inside the pool -- a cursor past the pool means the cache shrank under
            # a cursor written against the old one, and `% len(pool)` would silently re-read
            # from the start rather than continue.
            if d in cur and cur[d] >= pool_len:
                return (f"{name}: {d}'s cursor is at {cur[d]} but its pool holds only {pool_len} "
                        f"rows. build_mix reads `arange(used, used+want) % len(pool)`, so a cursor "
                        f"past the pool wraps to the start and silently re-reads rows the control "
                        f"already trained on instead of continuing.")
            return None
        # Injection domains: the shard is the independent source, and it is the one the axis
        # depends on. _shard_docs reads the file the cache was built from.
        try:
            _hdr, docs = _shard_docs(d)
        except (OSError, KeyError, SystemExit):
            return (f"{name}: {d}'s shard could not be read, so its pool length ({pool_len}) rests "
                    f"on the cache header alone -- that is a missing assertion, not a pass")
        if not docs:
            return f"{name}: {d}'s shard holds no documents"
        return None

    for d in inj:
        # Injection domains carry NO cursor (measured 2026-09-05: ckpt_b0_headmix_armA.pt's
        # row_cursor holds exactly the 9 natural domains), so they start at used=0.
        want, pool_len, cap_bound = _want_of(d, 0)
        if want is None:
            problems.append(f"{name}: cannot read {d}'s pool length, so its row-count assertion "
                            f"did not run -- that is a missing assertion, not a pass")
            continue
        bad = _check_pool_len(d, pool_len, cap_bound)
        if bad:
            problems.append(bad)
        got = rows_by_dom[d]
        if got != want:
            problems.append(f"{name}: the built plan holds {got} rows of {d}, the weight asks "
                            f"{want} (summed over {len(_phases)} phase(s) at anneal_frac "
                            f"{ANNEAL_FRAC}, pool {pool_len}). The stripe dropped rows.")
    # 4c's assertion, AS MUCH OF IT AS THE PLAN'S PUBLISHED STATE SUPPORTS: a natural domain
    # continues from its cursor rather than restarting. The floored-budget half is fully covered
    # here (it draws nothing at all, so a count sees it). The discarded-cursor half is NOT covered
    # by this loop and is stated as uncovered below rather than assumed away.
    for d in nat:
        if d not in cur:
            continue
        if rows_by_dom[d] == 0:
            problems.append(f"{name}: natural domain {d} drew 0 rows. Its cursor is at "
                            f"{cur[d]}; a budget that floors to zero after the cursor "
                            f"subtraction looks exactly like this.")
            continue
        # THE COUNT MUST BE THE REMAINDER TIMES THE WEIGHT, capped by the epoch bound -- the same
        # helper the injection domains use, seeded with THIS domain's cursor as `used`. It catches a
        # remainder computed against the wrong cursor, which moves every want.
        want, pool_len, cap_bound = _want_of(d, cur[d])
        if want is None:
            problems.append(f"{name}: cannot read {d}'s pool length, so its count assertion did "
                            f"not run -- that is a missing assertion, not a pass")
            continue
        bad = _check_pool_len(d, pool_len, cap_bound)
        if bad:
            problems.append(bad)
        if rows_by_dom[d] != want:
            problems.append(f"{name}: natural domain {d} drew {rows_by_dom[d]} rows, the weight "
                            f"asks {want} from a remainder of {rows_rem:.0f} at cursor {cur[d]} "
                            f"(pool {pool_len}). The remainder was computed against a different "
                            f"cursor, or the epoch cap bound.")
    # WHAT THIS CHECK DOES NOT COVER, stated because a silent gap reads as a pass. A cursor that is
    # DISCARDED (train.py:2201 corpus-fingerprint drift, :2190 sample_seed drift) makes a domain
    # restart at row 0, drawing the SAME COUNT from a DIFFERENT SET -- invisible to every assertion
    # above. Asserting the set needs the pool indices, and those never leave build_mix (:2324).
    # Two things make the gap narrow rather than open:
    #   - train.py:2236 REFUSES to start when any cursor would be discarded, unless
    #     --allow_partial_cursor is passed. The launch lines do not pass it, so a discarded cursor
    #     is a crash and not a silent restart.
    #   - Cfg._cursor_discarded is published (:2231), and the assertion below reads it.
    # So this reads the flag build_mix sets rather than re-deriving the fact.
    disc = list(getattr(train.Cfg, "_cursor_discarded", []) or [])
    if disc:
        problems.append(f"{name}: build_mix discarded {len(disc)} cursor(s) -- "
                        f"{'; '.join(disc[:4])}. Those domains restart at row 0 and re-read rows "
                        f"the control already trained on, at an unchanged row count.")
    if not getattr(train.Cfg, "_cursor_seeded", False):
        problems.append(f"{name}: Cfg._cursor_seeded is False, so NO domain's cursor seeded the "
                        f"plan -- every natural domain restarted at row 0 (train.py:2230).")
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
