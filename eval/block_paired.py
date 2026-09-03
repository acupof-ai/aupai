#!/usr/bin/env python3
# restartable: pure arithmetic over two JSONL score records; writes only its --json output.
"""b0-23: paired delta and paired SE over BLOCKS, not over domains.

    python3 eval/block_paired.py --from runs/x.jsonl --arms A B [C] [--weight token|row]
    python3 eval/block_paired.py --selftest

WHY THIS IS NOT paired_stats. eval/domain_loss.py:267 pairs by DOMAIN: n=9, and its own
docstring says the sd measures cross-domain consistency and must not be read as a reseeding
claim. N2's exit number needs the SAMPLING ERROR of a delta, and the sample available is the
val-cache blocks -- 64 per domain, 576 over the nine, not nine. MEASURED 2026-09-03, because
this docstring first said "hundreds to thousands" from an estimate I never checked: 64 blocks
per domain of 4096 tokens each. n=576 is still two orders of magnitude more pairing units than
9, and the paired sd is what makes it enough -- on the first real pair (e1's n7 Stage B arms)
the within-domain sd is 0.34-0.44 unpaired and 0.0437 paired, 9.2x tighter, giving SE 0.0018
over 576 blocks and 0.001-0.003 per domain at n=64.

The unit is the BLOCK because that is what both runs actually share. val_seqs is
deterministic given a vocabulary and a seed, so run A's block i and run B's block i are the
same tokens in the same order; the delta on block i is a real paired observation. Aggregating
to a domain first throws that away.

TWO WEIGHTINGS, BOTH REPORTED, because they are different quantities and neither is a
rounding of the other:
  row     the mean of the per-block mean CEs -- every block counts once
  token   total CE difference over total tokens -- every TOKEN counts once
They coincide only when every block has the same token count. Real val blocks are packed to a
fixed width, and MEASURED 2026-09-03 they are ALWAYS the same width -- distinct n_tokens over
every block of every arm is [4096], one value -- so on production data the two are an identity
and their agreement is not evidence of anything. The selftest fixture deliberately is not
(10/90/400 tokens), because a fixture where they coincide cannot tell a correct implementation
from one that silently uses the wrong one, and that fixture is the only place the distinction
is ever exercised.

THE BLOCK SETS MUST MATCH EXACTLY AND THE REFUSAL IS THE POINT. Taking the intersection would
silently change which blocks the mean is over, and at block granularity that has NO visible
symptom -- a domain-set mismatch shows up as nine names against eight, a block-set mismatch is
two integers nobody compares. So a symmetric difference refuses and reports both counts
(controller's ruling, 2026-09-03).
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _domains(rec):
    """The per-domain dict, from EITHER record shape.

    THE TWO PRODUCERS NEST IT DIFFERENTLY AND ONLY ONE WAS HANDLED. domain_loss.py's CLI
    writes {"ckpt", "domains", "unweighted_mean"}; score_matrix.py writes it under
    {"metrics": {"domain_loss": {...}}}. b0-23 pairs the data leg (scored by the first, on a
    lane card) against the params leg (scored by the second, at end of run), so a reader that
    knows one shape returns {} for the other -- and an empty block set reaches the "no
    per-block data" refusal, which names the record rather than the shape mismatch and sends
    the reader looking for a truncated file. Verified against the real files 2026-09-03:
    score_matrix records have NO top-level "domains" key.

    metrics.domain_loss also carries non-domain entries -- "_split" (a string), "_wall_s" and
    "unweighted_mean" (floats) -- so the caller must skip anything that is not a dict rather
    than assume every key is a domain.
    """
    if rec.get("domains"):
        return rec["domains"]
    return ((rec.get("metrics") or {}).get("domain_loss") or {})


def _blocks(rec):
    """{block_id: (ce_sum, n_tokens)} from one score record.

    A record carries per-domain blocks, so the id is (domain, index): block 3 of cot and block
    3 of zh_web are different observations and a bare index would collide them into one key,
    silently dropping every domain but the last.
    """
    out = {}
    for dom, d in sorted(_domains(rec).items()):
        if not isinstance(d, dict):
            continue  # _split, _wall_s, unweighted_mean -- see _domains
        rows = d.get("blocks")
        if not rows:
            continue
        for i, r in enumerate(rows):
            ce, ntok = (r["ce_sum"], r["n_tokens"]) if isinstance(r, dict) else (r[0], r[1])
            out[(dom, i)] = (float(ce), int(ntok))
    return out


def block_paired(rec_by_name, a_name, b_name, c_name=None):
    """Paired per-block delta, its SE, and t. (B-A)-(C-A) when a control arm is given.

    The control arm is subtracted for the reason b0-16 established: B-A alone does not isolate
    the tensor under test when the control is also uniformly damaged. Same logic as
    paired_stats, one granularity down.
    """
    names = [n for n in (a_name, b_name, c_name) if n]
    missing = [n for n in names if n not in rec_by_name]
    if missing:
        raise KeyError(f"block_paired needs records for {missing}; got {sorted(rec_by_name)}")
    per = {n: _blocks(rec_by_name[n]) for n in names}
    empty = [n for n, b in per.items() if not b]
    if empty:
        raise ValueError(
            f"no per-block data in {empty}. A record scored before --per-block existed carries "
            f"only the domain mean, and pairing on it would give n=9 while reporting an SE that "
            f"reads as n=hundreds. Re-score those checkpoints with per-block output.")

    ref = per[names[0]]
    for n in names[1:]:
        only_ref = set(ref) - set(per[n])
        only_n = set(per[n]) - set(ref)
        if only_ref or only_n:
            sample = sorted(only_ref)[:3] + sorted(only_n)[:3]
            raise ValueError(
                f"block sets differ: {names[0]} has {len(ref)} block(s), {n} has {len(per[n])}; "
                f"{len(only_ref)} only in {names[0]}, {len(only_n)} only in {n}. e.g. {sample}. "
                f"REFUSING rather than intersecting: the intersection would change which blocks "
                f"the mean is over, and at block granularity that has no visible symptom.")
        # Same block id must also be the same LENGTH, or it is not the same block. A rebuilt
        # cache can keep the ordering and change the packing.
        bad = [k for k in ref if ref[k][1] != per[n][k][1]]
        if bad:
            k = bad[0]
            raise ValueError(
                f"{len(bad)} block(s) have the same id but different token counts between "
                f"{names[0]} and {n}, e.g. {k}: {ref[k][1]} vs {per[n][k][1]}. The rows are not "
                f"the same rows, so these are not paired measurements.")

    ids = sorted(ref)

    def mean_ce(n, k):
        ce, ntok = per[n][k]
        return ce / ntok

    if c_name:
        row_d = {k: (mean_ce(b_name, k) - mean_ce(a_name, k))
                    - (mean_ce(c_name, k) - mean_ce(a_name, k)) for k in ids}
        sum_d = {k: (per[b_name][k][0] - per[a_name][k][0])
                    - (per[c_name][k][0] - per[a_name][k][0]) for k in ids}
        stat = f"(B-A)-(C-A) per block with B={b_name}, C={c_name}, A={a_name}"
    else:
        row_d = {k: mean_ce(b_name, k) - mean_ce(a_name, k) for k in ids}
        sum_d = {k: per[b_name][k][0] - per[a_name][k][0] for k in ids}
        stat = f"B-A per block with B={b_name}, A={a_name}"

    vals = [row_d[k] for k in ids]
    n = len(vals)
    row_mean = sum(vals) / n
    if n > 1:
        sd = math.sqrt(sum((v - row_mean) ** 2 for v in vals) / (n - 1))
        se = sd / math.sqrt(n)
    else:
        sd = se = 0.0
    tok_total = sum(per[a_name][k][1] for k in ids)
    tok_mean = sum(sum_d[k] for k in ids) / tok_total
    pos = sum(1 for v in vals if v > 0)
    neg = sum(1 for v in vals if v < 0)
    k_maj = max(pos, neg)
    # Sign test over the blocks that MOVED. Ties carry no direction, and counting them in the
    # denominator makes a real effect look weaker the more exact zeros the fixture has.
    moved = pos + neg
    p = (sum(math.comb(moved, i) for i in range(k_maj, moved + 1)) / (2 ** moved)
         if moved else 1.0)
    return {"statistic": stat, "n_blocks": n, "block_ids": ids,
            "row_weighted_mean": row_mean, "sd": sd, "se": se,
            "t": row_mean / se if se else None,
            "token_weighted_mean": tok_mean, "n_tokens": tok_total,
            "positive": pos, "negative": neg, "ties": n - moved,
            "sign_test_p_one_sided": p,
            "domains": sorted({d for d, _ in ids})}


def print_block_paired(bp):
    print(f"\n=== paired per-block: {bp['statistic']} ===")
    print(f"  n = {bp['n_blocks']} blocks over {len(bp['domains'])} domain(s), "
          f"{bp['n_tokens']:,} tokens")
    print(f"  row-weighted   mean {bp['row_weighted_mean']:+.6f}  sd {bp['sd']:.6f}  "
          f"SE {bp['se']:.6f}" + (f"  t {bp['t']:+.2f}" if bp["t"] is not None else "  t n/a"))
    print(f"  token-weighted mean {bp['token_weighted_mean']:+.6f}")
    if abs(bp["row_weighted_mean"] - bp["token_weighted_mean"]) > 1e-9:
        print("    the two differ, so the blocks are NOT equal-length; say which one a "
              "reported number is.")
    else:
        # SILENCE HERE READS AS AGREEMENT, WHICH IS THE ONE THING IT IS NOT. On equal-length
        # blocks the two weightings are the same arithmetic, so a match is an identity and
        # carries no information about whether this code picked the right one. Measured
        # 2026-09-03 on the real val cache: every block is 4096 tokens, distinct n_tokens =
        # [4096], so this branch is what production always takes. The only evidence that the
        # weightings are implemented correctly is --selftest, whose fixture is 10/90/400.
        print("    IDENTITY, not a check: every block has the same token count, so the two "
              "weightings are the same arithmetic here and agreeing tells you nothing about "
              "either. Correctness of the weighting lives in --selftest (10/90/400 fixture).")
    print(f"  sign: {bp['positive']} up, {bp['negative']} down, {bp['ties']} tie; "
          f"one-sided p {bp['sign_test_p_one_sided']:.2e}")
    print("  THE SE IS THE SAMPLING ERROR OF THIS DELTA over blocks. It is not seed spread: "
          "reseeding the run is a different experiment and 0.24 nat "
          "(ds.seed_variance_0p2b) measures that one.")


def _rec_by_name(paths):
    """Load score records from one or more JSONL files, keyed by ckpt.

    Multiple files because b0-23's two legs are scored by different producers (see --from).

    LAST ROW WINS, EXCEPT THAT A BLOCKLESS ROW NEVER DISPLACES ONE WITH BLOCKS. The ledger
    convention is fold-by-key-last-row-wins, and applying it literally here destroyed the very
    record this tool needs: MEASURED 2026-09-03, runs/b0_23_blocks.jsonl holds
    ckpt_data_leg_206m_8b.pt with 576 blocks and runs/score_matrix.jsonl holds THE SAME ckpt
    name with 0 blocks (it was scored before per_row existed). Loading both in that order left
    the blockless row, and block_paired then refused with "no per-block data" -- naming the
    checkpoint as unscored when a good record for it was in hand. The two rows are not two
    versions of one measurement: one carries a strictly larger payload, so the fold has to be
    by payload and not by position.

    Ties -- both rows blockless, or both with blocks -- still go to the later row, which is the
    convention where it actually applies.
    """
    if isinstance(paths, str):
        paths = [paths]
    out = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                k = r["ckpt"]
                if k in out and _blocks(out[k]) and not _blocks(r):
                    continue
                out[k] = r
    return out


def _selftest():
    fails = []

    def mk(name, per_dom):
        return {"ckpt": name, "domains": {d: {"blocks": [{"ce_sum": c, "n_tokens": t}
                                                         for c, t in rows]}
                                          for d, rows in per_dom.items()}}

    def mk_sm(name, per_dom):
        """The OTHER producer's shape: score_matrix.py nests it under metrics.domain_loss and
        writes no top-level "domains". b0-23 pairs one of each, so a reader that knows only
        mk()'s shape returns {} here and the failure surfaces as "no per-block data" -- which
        names the record and not the shape, sending the reader after a truncated file."""
        inner = {d: {"blocks": [{"ce_sum": c, "n_tokens": t} for c, t in rows]}
                 for d, rows in per_dom.items()}
        # the real file carries these three non-domain entries beside the domains
        inner.update({"_split": "val", "_wall_s": 124.8, "unweighted_mean": 1.9443})
        return {"ckpt": name, "metrics": {"domain_loss": inner}, "type": "base"}

    # BOTH SHAPES MUST READ, AND THE NON-DOMAIN KEYS MUST NOT BECOME BLOCKS. _split is a
    # string and _wall_s a float, so a reader that treats every key as a domain raises
    # AttributeError inside _blocks -- and a crash is not a refusal. Both halves are therefore
    # probed through _domains FIRST, with an early return, so each mutation fails through a
    # named assertion rather than through a traceback from somewhere downstream: dropping the
    # metrics.domain_loss branch otherwise surfaces as block_paired's unrelated "no per-block
    # data" ValueError, which sends the reader after a truncated file.
    _sm = mk_sm("SM", {"cot": [(3.0, 10), (5.0, 10)], "zh_web": [(7.0, 10)]})
    _sm_doms = _domains(_sm)
    if not _sm_doms:
        print("block_paired selftest FAILED: _domains returns {} for a score_matrix-shaped "
              "record (no top-level 'domains', nested under metrics.domain_loss). b0-23 pairs "
              "one record of each shape, so this reader sees no blocks in the params leg and "
              "block_paired raises its 'no per-block data' error -- which names the record, "
              "not the shape, and sends the reader looking for a truncated file.")
        return 1
    _nondict = sorted(k for k, v in _sm_doms.items() if not isinstance(v, dict))
    if _nondict != ["_split", "_wall_s", "unweighted_mean"]:
        print(f"block_paired selftest FAILED: fixture's non-domain keys are {_nondict}; the "
              f"real metrics.domain_loss carries _split (str), _wall_s and unweighted_mean "
              f"(float) beside the domains, and the fixture must carry them too or the "
              f"isinstance guard in _blocks is never exercised.")
        return 1
    _sm_blocks = None
    try:
        _sm_blocks = _blocks(_sm)
    except AttributeError as e:
        # _blocks iterated a non-domain entry: metrics.domain_loss holds _split (str) and
        # _wall_s/unweighted_mean (float) beside the domains. Caught and NAMED here because a
        # traceback from `d.get("blocks")` tells the reader nothing about which key did it.
        print(f"block_paired selftest FAILED: _blocks treats every metrics.domain_loss key as "
              f"a domain and crashed on a non-dict entry ({e}). The real record carries "
              f"_split, _wall_s and unweighted_mean beside the nine domains; skip anything "
              f"that is not a dict.")
        return 1
    if set(_sm_blocks) != {("cot", 0), ("cot", 1), ("zh_web", 0)}:
        fails.append(f"score_matrix-shaped record read as {sorted(_sm_blocks)}; the "
                     f"metrics.domain_loss nesting is not handled, or _split/_wall_s/"
                     f"unweighted_mean leaked in as domains")
    # A BLOCKLESS ROW MUST NOT DISPLACE ONE WITH BLOCKS, even though it comes later. Both real
    # files key ckpt_data_leg_206m_8b.pt: b0_23_blocks.jsonl with 576 blocks, score_matrix.jsonl
    # with 0 (scored before per_row existed). Literal last-row-wins left the blockless one and
    # block_paired refused with "no per-block data" for a checkpoint whose blocks were in hand.
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        _good = os.path.join(_td, "good.jsonl")
        _bare = os.path.join(_td, "bare.jsonl")
        with open(_good, "w", encoding="utf-8") as _f:
            _f.write(json.dumps(mk("K", {"cot": [(3.0, 10)]})) + "\n")
        with open(_bare, "w", encoding="utf-8") as _f:
            _f.write(json.dumps({"ckpt": "K", "metrics": {"domain_loss": {"cot": {"loss": 1.0}}}}) + "\n")
        _folded = _rec_by_name([_good, _bare])          # blockless comes LAST
        if not _blocks(_folded["K"]):
            print("block_paired selftest FAILED: a blockless row loaded after a row with "
                  "blocks displaced it. Both real files key ckpt_data_leg_206m_8b.pt -- "
                  "b0_23_blocks.jsonl with 576 blocks, score_matrix.jsonl with 0 -- so literal "
                  "last-row-wins reports the checkpoint as unscored while its blocks are in "
                  "hand. Fold by payload, not by position.")
            return 1
        _folded2 = _rec_by_name([_bare, _good])          # and the other order still works
        if not _blocks(_folded2["K"]):
            print("block_paired selftest FAILED: a row with blocks did not win over an "
                  "earlier blockless row.")
            return 1

    # and pairing ACROSS the two shapes must work, since that is exactly b0-23's case
    _mixed = block_paired({"A": mk("A", {"cot": [(3.0, 10), (5.0, 10)], "zh_web": [(7.0, 10)]}),
                           "SM": _sm}, "A", "SM")
    if _mixed["n_blocks"] != 3 or abs(_mixed["row_weighted_mean"]) > 1e-12:
        fails.append(f"pairing a domain_loss-shaped record against a score_matrix-shaped one "
                     f"gave n={_mixed['n_blocks']} mean={_mixed['row_weighted_mean']}, want "
                     f"n=3 mean=0 (identical values in both shapes)")

    # BLOCK IDS ARE (domain, index), CHECKED FIRST. A bare index collapses two domains' block 0
    # onto one key and every domain but the last vanishes -- with a clean-looking n. This sits
    # at the top because with a bare index the calls below raise TypeError while unpacking
    # `for d, _ in ids`, and a crash is not a refusal: the mutation would go red through a
    # traceback instead of through this named assertion, which tells a reader far less.
    two = mk("T", {"d1": [(10.0, 10)], "d2": [(20.0, 10)]})
    if len(_blocks(two)) != 2:
        fails.append(f"_blocks collapsed two domains' block 0 into {len(_blocks(two))} key(s); "
                     "the id must carry the domain, or every domain but the last vanishes")
        for f in fails:
            print(f"  FAIL {f}")
        print("\n1 failure(s) -- stopping here: every check below unpacks a (domain, index) id")
        return 1

    # NON-UNIFORM ON PURPOSE: 10 / 90 / 400 tokens. Row-weighting and token-weighting must
    # disagree, or the fixture cannot tell them apart -- which is the whole reason the
    # controller specified a non-uniform one.
    A = mk("A", {"d1": [(10.0, 10), (90.0, 90), (400.0, 400)]})
    # per-block mean CE goes 1.0 -> 1.1 on the SHORT block only.
    B = mk("B", {"d1": [(11.0, 10), (90.0, 90), (400.0, 400)]})
    bp = block_paired({"A": A, "B": B}, "A", "B")
    if bp["n_blocks"] != 3:
        fails.append(f"n_blocks {bp['n_blocks']}, want 3 -- blocks, not domains")
    # row-weighted: (0.1 + 0 + 0)/3
    if abs(bp["row_weighted_mean"] - 0.1 / 3) > 1e-12:
        fails.append(f"row-weighted mean {bp['row_weighted_mean']} want {0.1/3}")
    # token-weighted: 1.0 nat of CE over 500 tokens
    if abs(bp["token_weighted_mean"] - 1.0 / 500) > 1e-12:
        fails.append(f"token-weighted mean {bp['token_weighted_mean']} want {1/500}")
    if abs(bp["row_weighted_mean"] - bp["token_weighted_mean"]) < 1e-6:
        fails.append("the two weightings agree on the non-uniform fixture, so this fixture "
                     "cannot distinguish them and neither can any test built on it")
    if bp["se"] <= 0 or bp["t"] is None:
        fails.append(f"SE must be positive with a real spread, got {bp['se']}")
    # SE is over n BLOCKS: sd/sqrt(3), not sd/sqrt(1 domain).
    want_se = bp["sd"] / math.sqrt(3)
    if abs(bp["se"] - want_se) > 1e-12:
        fails.append(f"SE {bp['se']} is not sd/sqrt(n_blocks) {want_se}")
    if bp["ties"] != 2 or bp["positive"] != 1:
        fails.append(f"sign counts wrong: {bp['positive']} up / {bp['ties']} tie, want 1 / 2")
    # TIES ARE EXCLUDED FROM THE SIGN TEST, and this is the check that pins it. A tied block
    # carries no direction, so counting it in the denominator makes a real effect look weaker
    # the more exact zeros there are. Here 1 up and 2 ties: over the ONE block that moved,
    # one-sided p = 1/2. Counting all three blocks gives the tail sum from k=1, which is
    # 7/8 = 0.875 -- not a weaker version of the same number, the p-value of a different
    # hypothesis. Without this assertion the fixture leaves both formulas passing.
    if abs(bp["sign_test_p_one_sided"] - 0.5) > 1e-12:
        fails.append(f"sign-test p {bp['sign_test_p_one_sided']} over 1 moved block should be "
                     f"0.5; 0.875 means the 2 ties are in the denominator")

    # A CONTROL ARM IS SUBTRACTED. Same reason as b0-16's correction one granularity up: if C
    # moves the same way, (B-A)-(C-A) must cancel it. Without this the check would pass an
    # implementation that ignores C entirely.
    C = mk("C", {"d1": [(11.0, 10), (90.0, 90), (400.0, 400)]})
    bpc = block_paired({"A": A, "B": B, "C": C}, "A", "B", "C")
    if abs(bpc["row_weighted_mean"]) > 1e-12 or bpc["sd"] > 1e-12:
        fails.append(f"(B-A)-(C-A) with B==C must be exactly 0, got "
                     f"{bpc['row_weighted_mean']} sd {bpc['sd']} -- the control is being ignored")
    if bpc["statistic"] == bp["statistic"]:
        fails.append("the controlled and uncontrolled statistics carry the same label")

    # THE REFUSALS, by exception TYPE. A crash is not a refusal: deleting the block-set guard
    # leaves a KeyError from the length loop, and `except Exception` reads that as the guard
    # working. Same shape the domain-level version was caught by (c5b4d002 check 5).
    short = mk("S", {"d1": [(10.0, 10), (90.0, 90)]})
    try:
        block_paired({"A": A, "S": short}, "A", "S")
        fails.append("a 3-block record paired against a 2-block record was ACCEPTED; the "
                     "intersection would silently change what the mean is over")
    except ValueError as e:
        if "3" not in str(e) or "2" not in str(e):
            fails.append(f"the block-set refusal does not report both counts: {e}")
    except KeyError:
        fails.append("block-set mismatch raised KeyError, not ValueError -- that is a crash "
                     "downstream of a missing guard, not a refusal")

    relen = mk("R", {"d1": [(10.0, 10), (90.0, 91), (400.0, 400)]})
    try:
        block_paired({"A": A, "R": relen}, "A", "R")
        fails.append("same block id with a different token count was accepted; a repacked "
                     "cache keeps the ordering and changes the rows")
    except ValueError as e:
        if "90" not in str(e) or "91" not in str(e):
            fails.append(f"the length refusal does not name the two counts: {e}")

    # A record with no per-block data must refuse WITH ITS OWN REASON, not fall back to the
    # domain mean and report an SE that reads as n=hundreds. The `per-block` substring is the
    # assertion: without the empty check the block-set guard fires instead and says "N has 0
    # blocks", which is true, refuses correctly, and tells the reader to go fix a set mismatch
    # that does not exist. Two guards catching one fault is fine; the wrong DIAGNOSIS is not.
    try:
        block_paired({"A": A, "N": {"ckpt": "N", "domains": {"d1": {"loss": 1.0}}}}, "A", "N")
        fails.append("a record with no per-block data was accepted")
    except ValueError as e:
        if "per-block" not in str(e):
            fails.append(f"the no-block refusal does not say why: {e}")

    # A-vs-A is an IDENTITY, and is labelled as one rather than as proof of determinism: there
    # is only one measurement in it.
    same = block_paired({"A": A}, "A", "A")
    if same["row_weighted_mean"] != 0.0 or same["sd"] != 0.0:
        fails.append("A-vs-A is not exactly zero; the arithmetic or the plumbing is wrong")

    for f in fails:
        print(f"  FAIL {f}")
    if fails:
        print(f"\n{len(fails)} failure(s)")
        return 1
    print("block_paired selftest OK: pairs on BLOCKS not domains (n=3 from one domain), SE is "
          "sd/sqrt(n_blocks), and the two weightings DISAGREE on the deliberately non-uniform "
          "10/90/400 fixture (row 0.033333 vs token 0.002000) so a test built on it can tell "
          "them apart. A control arm identical to the test arm cancels to exactly 0, which is "
          "what fails an implementation that ignores C. Four refusals checked by exception "
          "TYPE, not by 'something raised': a differing block set (reporting BOTH counts, never "
          "intersecting), a same-id block whose token count moved, a record carrying no "
          "per-block data, and block ids that must be (domain, index) or two domains collapse "
          "onto one key with a clean-looking n. A-vs-A is exactly 0 and is an identity, not "
          "evidence of determinism.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", nargs="+", metavar="JSONL",
                    help="one or more JSONL files of score records (one record per "
                         "checkpoint). MORE THAN ONE IS THE NORMAL CASE for b0-23: the two "
                         "legs are scored by different producers into different files -- the "
                         "data leg's blocks are in runs/b0_23_blocks.jsonl (domain_loss.py, "
                         "lane card) and the params leg's in runs/score_matrix.jsonl "
                         "(score_matrix.py, end of run). Later files win on a repeated ckpt, "
                         "matching the ledger convention.")
    ap.add_argument("--arms", nargs="+", metavar="CKPT",
                    help="A B [C]: baseline, test, optional control. With C the statistic is "
                         "(B-A)-(C-A).")
    ap.add_argument("--json", help="write the result object here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not (a.src and a.arms and 2 <= len(a.arms) <= 3):
        ap.error("need --from FILE and --arms A B [C], or --selftest")
    recs = _rec_by_name(a.src)
    bp = block_paired(recs, *a.arms)
    print_block_paired(bp)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in bp.items() if k != "block_ids"}, f,
                      ensure_ascii=False, indent=1)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
