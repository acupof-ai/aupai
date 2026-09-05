#!/usr/bin/env python3
"""What does each eval in eval/ cost a co-resident training rank? (de-27)

Written because "likelihood metrics may share a card" was measured on ONE eval
(score_matrix, 2.3 GiB) and applied to eval/ppl.py, which torch.loads a whole token cache
per domain -- 85 GB for zh_web, ~166 GB across the nine domains of mix_500m. Same metric
class, 36x the host IO. The rule named the wrong axis.

Two columns, and the second is the one no rule had:

  ckpt_load   does it load a checkpoint (about 2.1 GB for p500m)? Every scoring eval does.
  host_io     does it read a TOKEN CACHE off /data00? That is the >10 GB axis, and it is
              what separates ppl from score_matrix inside one metric class.

MEASURED cost, not predicted: the third column comes from p500m_20b_0902.log, by taking
the 10-step interval whose wall-clock window contains the eval and differencing it against
the run's sustained 12K tok/s/gpu. The run supplies its own control -- --save_every 500,
and steps 500/1000/1500/2000 all read 7K with NO eval running, which is a 2.1 GB
torch.save plus a val pass. An eval that costs less than 78 s costs less than the run
already spends on itself every 500 steps.

Read the numbers as seconds. The ETA field extrapolates one interval over 19,151 steps, so
a 54-second interval overrun prints as 29 lost hours (gate_failure_shapes.md §50).

    python3 scripts/eval_load_cost.py            # the table
    python3 scripts/eval_load_cost.py --selftest # the derivation, on known answers

# restartable: reads source files and a table of measurements. Writes nothing.
"""

import ast
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# p500m_20b_0902: batch 32, seq 4096, 8 ranks, sustained 12K tok/s/gpu.
PER_STEP_PER_RANK = 32 * 4096
SUSTAINED_K = 12


def interval_secs(k_per_gpu):
    """Wall-clock seconds for one 10-step logging interval at k K tok/s/gpu."""
    return PER_STEP_PER_RANK * 10 / (k_per_gpu * 1000)


def cost_secs(rates):
    """Seconds lost versus the sustained rate, over the intervals an event spans."""
    base = interval_secs(SUSTAINED_K)
    return sum(interval_secs(k) - base for k in rates)


# MEASURED on p500m_20b_0902, 2026-09-02. Attribution is anchored, not guessed: the
# checkpoint's own mtime (06:07:47 for step1500) plus each interval's reported rate gives
# a wall-clock time per dip, matched against each eval log's mtime -- score_matrix within
# 2 min, l1_fewshot within 40 s, ppl within 6 s.
MEASURED = {
    "_control_save_val": ([7], "ckpt save + val, NO eval running (steps 500/1000/1500/2000)"),
    "score_matrix.py": ([9, 11], "4 likelihood metrics: minimal_pairs, mc_ceval, lambada_zh, math_v2_like"),
    "l1_fewshot.py": ([7, 6, 10], "generative, 497 problems x 512 new tokens"),
    "ppl.py": ([8, 8], "killed during checkpoint load; had NOT reached a token cache"),
}

# WALL-CLOCK ON A DEDICATED CARD -- A DIFFERENT QUANTITY FROM MEASURED ABOVE, kept in its own
# table for exactly that reason. MEASURED holds training-throughput dips: seconds of TRAINING lost
# while an eval shares the machine, which is what the co-residency rule turns on. The numbers here
# are how long the eval itself takes with a card to itself, which is what you need to decide
# whether to wait for it or run it beside something. Putting 970s into MEASURED would have claimed
# lambada_en costs 970 seconds of training time; it costs 970 seconds of YOUR time.
#
# Both are needed and neither substitutes: an eval can be quick and disruptive (score_matrix's
# likelihood metrics) or slow and gentle (a generative eval that leaves the GPU 74% idle).
WALL_SECS = {
    "lambada_en.py": (970.7, 5153, "generative, ONE ITEM AT A TIME -- eval/lambada_en.py:286 "
                      "loops greedy_word + target_nll_per_byte per item with no batching, so the "
                      "card sits at ~26% util for 16 minutes. Measured 2026-09-03 in b0-17's "
                      "readout (runs/b0_17_readout.log, ckpt_ab_untiehead_untiehead.pt.ep1) on a "
                      "dedicated card 3. The log goes SILENT for the whole 16 min, which reads as "
                      "a hang: py-spy showed the parent blocked in subprocess.communicate at "
                      "score_matrix.py:269, i.e. working. 5153 items, not the 1000 I assumed "
                      "before reading n_items."),
    "l1_fewshot.py": (2134.3, 497, "generative, 497 problems x 512 new tokens. THE ONLY EVAL "
                      "WITH BOTH COLUMNS FILLED, and they disagree 10x: the dip in MEASURED is "
                      "209.1s of training lost, the wall here is 2134.3s of yours. Neither is "
                      "the other's estimate. Measured twice on a dedicated card 3 in b0-17's "
                      "readout, one run per arm (2134.3s untiehead, 2133.6s untieheadlr -- 0.7s "
                      "apart over 36 minutes, so the number is the eval's shape, not a "
                      "condition of one run). 4.29 s/item against lambada_en's 0.188 s/item: "
                      "23x slower per item at a tenth the item count, because 512 new tokens "
                      "per problem is 512 serial forwards. Both arms read acc 0.0 with "
                      "answer_present_rate 0.002 (1 of 497), i.e. 36 minutes to learn that a "
                      "500-step 200M base emits no parseable answer -- the cost is worth "
                      "recording precisely because the signal was known to be absent."),
}

# HOST BYTES PER TOKEN CACHE, measured on the pod 2026-09-03:
#   ~/bin/pod "ls -la /data00/tokens_*.pt | awk '{print \$5, \$9}'"
# 22 natural/chat caches, 247.8 GB total, plus the five experiment-1 injection caches measured
# 2026-09-05 (below), which add 0.14 GB. This is the quantity the co-residency rule turns on, and
# the one AGENTS.md's coverage table records as "nothing in the repo records it per eval run" --
# recorded here now, per DOMAIN, which is the level at which it is a property of the corpus rather
# than of a run. An eval's host read is the sum over the domains its mix names, because
# _domain_seqs torch.loads one file per domain.
#
# Why bytes and not "reads a cache: yes/no": zh_web alone is 85 GB and tokens_sample is 5 MB, so
# the boolean column groups a 16,000x range into one bucket. `ppl` over mix_500m reads ~166 GB and
# `ppl` over a chat-only mix reads 0.3 GB -- the rule "an eval that reads a token cache waits for
# the run" is right for the first and absurd for the second.
#
# THE THRESHOLD IS DECLARED, NOT FITTED, and saying which matters because the table looks like
# it could supply one. There is no measured bytes->seconds point above ~2 GB: the three 166 GB
# rows have an empty `measured` column, and ppl.py's 109 s was recorded at the moment it was
# killed, BEFORE it reached a cache (§50: the kill was 事前止损). So the largest cache read ever
# allowed to finish beside a live run is unknown, and no regression through the measured points
# can name a byte bound.
#
# 10 GB is this file's own existing line: "that is the >10 GB axis, and it is what separates ppl
# from score_matrix inside one metric class" (docstring, host_io). Reused rather than invented so
# the number has one home. It also lands between the two populations with room on both sides --
# the gap between the two populations runs from web_hq at 5.7 GB to en_c4 at 19.2 GB.
CO_RESIDENCY_BYTES = 10e9

CACHE_BYTES = {
    "zh_web": 85173617415, "code_py_starcoder": 35147667156, "code_rp1t": 30276327900,
    "math_owm_stage2": 26114186246, "en_c4": 19236278784, "math_owm": 16139300949,
    "en_c4_stage2": 9604578481, "textbook_30b": 6440843057, "textbook": 6440843029,
    "web_hq": 5737618119, "cot": 1696226494, "code_py_rp1t": 1683425021,
    "wiki_chat": 1135614653, "wiki": 982864005, "en": 643070455, "math_seed": 326728936,
    "math": 326728901, "code": 230279173, "chatml": 155984979, "chat_qa": 152752218,
    "chat": 152752197, "sample": 5271699,
    # Experiment 1's injection domains, measured on the pod 2026-09-05 with os.path.getsize.
    # RECORDED BECAUSE UNRECORDED MEANS ZERO, not because these are large. domains_cache_bytes
    # contributes 0 for a domain absent from this table, so a mix of only injection domains summed
    # to 0 GB and the co-residency refusal compared 0 against the 10 GB threshold -- fail-open.
    # It does WARN by name, which is the designed behaviour for unknown, so nothing was hidden
    # today: all five really are far under the threshold and the 0 and the truth agreed. That
    # agreement is a coincidence of these five sizes, and the next injection domain need not be
    # small. A guard that is right because the unmeasured quantity happens to be small is right
    # by coincidence.
    "s_inject_n256": 107470468, "s_inject_n64": 26868861, "s_inject_n8": 3360054,
    "s_inject_n1": 421430, "p_format": 343329,
}


def mix_cache_bytes(mix_path):
    """(bytes, [domains with no cache on the pod]) for one mix file."""
    try:
        with open(mix_path, encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, ValueError):
        return None, []
    doms = list((obj.get("domains") or {}).keys())
    return (sum(CACHE_BYTES.get(d, 0) for d in doms),
            [d for d in doms if d not in CACHE_BYTES])


def domains_cache_bytes(domains):
    """(bytes, [domains with no recorded cache]) for an explicit domain list.

    The refusal in eval/cache_guard.py calls THIS, not mix_cache_bytes: it is handed the
    domains an eval is about to read, and those are not always a mix file's whole set --
    `--mix` narrowed, one domain re-scored, a probe over three. Summing the mix would
    refuse on bytes the caller was never going to read.

    An unrecorded domain contributes 0 and is RETURNED BY NAME rather than silently
    skipped, because the caller's decision differs on it: a sum with a hole in it is a
    lower bound, not a measurement, and cache_guard warns on unknown rather than
    refusing.
    """
    doms = list(domains)
    return (sum(CACHE_BYTES.get(d, 0) for d in doms),
            [d for d in doms if d not in CACHE_BYTES])

# Static facts read off each file, so a new eval cannot be silently absent from the table.
# host_io means "reads a token cache from /data00", i.e. goes through train._domain_seqs.
CACHE_READERS = ("_domain_seqs", "val_seqs")


def _executable_src(src):
    """src with string literals and comments blanked out, so a substring test cannot match
    prose. Two defects came from testing the raw text (de, 2026-09-03):

      cache_guard.py  read as host_io=YES on five occurrences of _domain_seqs/val_seqs, ALL
                      of them in its docstring. It is the guard that checks a cache without
                      loading it -- it reads only the tiny .vocab stamp -- so the column was
                      exactly backwards on the one file whose job is to not read the cache.
      nan_probe.py    read as ckpt_load='-' because it hard-codes its paths instead of taking
                      --ckpt, while it torch.loads a 2 GB checkpoint AND a 6 GB SFT pack.

    Blanking prose fixes the first. The second is not a prose problem, so classify also
    matches a literal .pt path in torch.load.
    """
    out = list(src)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    starts = [0]
    for ln in src.splitlines(keepends=True):
        starts.append(starts[-1] + len(ln))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.lineno:
            a = starts[node.lineno - 1] + node.col_offset
            b = starts[node.end_lineno - 1] + node.end_col_offset
            for i in range(a, min(b, len(out))):
                out[i] = " "
    return re.sub(r"#[^\n]*", "", "".join(out))


def classify(path):
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    src = _executable_src(raw)
    return {
        # A hard-coded "<name>.pt" counts: nan_probe.py loads two large files without ever
        # naming --ckpt, and a table that calls that "no load" understates it by 8 GB.
        "ckpt_load": bool(re.search(r"load_checkpoint\(|--ckpt", src))
        or bool(re.search(r"torch\.load\([^)]*\.pt[\"']", raw)),
        "host_io": any(r in src for r in CACHE_READERS),
        "generative": bool(re.search(r"generate_batch|max_new|open_artifact\(", src)),
    }


def rows():
    out = []
    d = os.path.join(ROOT, "eval")
    for b in sorted(os.listdir(d)):
        if not b.endswith(".py") or b == "__init__.py":
            continue
        c = classify(os.path.join(d, b))
        rates, note = MEASURED.get(b, (None, ""))
        out.append((b, c, cost_secs(rates) if rates else None, note))
    return out


def main():
    ctrl = cost_secs(MEASURED["_control_save_val"][0])
    print(f"Control: the run's own 2.1GB checkpoint save + val costs {ctrl:.0f}s, every 500 "
          f"steps, with no eval running.")
    print(f"An eval below {ctrl:.0f}s costs less than the run already spends on itself.\n")
    print(f"{'eval':28s} {'ckpt':5s} {'host GB':>8s} {'gen':4s} {'measured':>9s} {'wall':>9s}  note")
    print("-" * 118)
    mixes = sorted(glob.glob(os.path.join(ROOT, "data", "mix_*.json")))
    live = [m for m in mixes if os.path.basename(m).startswith(("mix_500m", "mix_200m"))]
    ref = live[0] if live else (mixes[0] if mixes else None)
    ref_bytes, _ref_missing = mix_cache_bytes(ref) if ref else (None, [])
    for name, c, secs, note in rows():
        got = f"{secs:.0f}s" if secs is not None else "-"
        # host GB is what THIS eval reads: the reference mix's whole cache set if it goes through
        # _domain_seqs, else just the checkpoint. Not a boolean, because the caches span 5 MB to
        # 85 GB and a yes/no column puts both in one bucket.
        if c["host_io"] and ref_bytes:
            gb = f"{ref_bytes / 1e9:.0f}"
        elif c["ckpt_load"]:
            gb = "~2"
        else:
            gb = "-"
        wall = WALL_SECS.get(name)
        wcol = f"{wall[0]:.0f}s" if wall else "-"
        print(f"{name:28s} {'yes' if c['ckpt_load'] else '-':5s} "
              f"{gb:>8s} {'yes' if c['generative'] else '-':4s} "
              f"{got:>9s} {wcol:>9s}  {note or (wall[2] if wall else '')}")
    unmeasured = [n for n, _, s, _ in rows() if s is None]
    print(f"\n{len(unmeasured)} of {len(rows())} evals have NO THROUGHPUT-DIP measurement -- "
          f"listed as '-' in the measured column, never as zero:")
    print("  " + ", ".join(unmeasured))
    # An eval with a wall time but no dip is still unmeasured IN THE COLUMN THAT MATTERS FOR
    # CO-RESIDENCY, and saying "measured" of it would answer the wrong question -- so it stays in
    # the list above and the gap is named here instead of quietly closed.
    wall_only = sorted(n for n in WALL_SECS if n in unmeasured)
    if wall_only:
        print(f"\n{len(wall_only)} of those DO have a dedicated-card wall time (the 'wall' column) "
              f"but no throughput dip: {', '.join(wall_only)}.\nThat is not the same measurement: "
              f"wall time says how long you wait, the dip says what it costs a run beside it. An "
              f"eval can be slow and gentle (26% util for 16 min) or quick and disruptive.")
    # The case that makes the two-table split concrete rather than argued: an eval with BOTH
    # numbers. Until l1_fewshot had a wall time, every WALL_SECS entry lacked a dip and the
    # distinction rested on reasoning; one row with both turns it into a measured ratio.
    both = sorted(n for n, _, s, _ in rows() if s is not None and n in WALL_SECS)
    for n in both:
        dip = next(s for m, _, s, _ in rows() if m == n)
        wall = WALL_SECS[n][0]
        print(f"\n{n} HAS BOTH: dip {dip:.1f}s of training lost, wall {wall:.1f}s of yours "
              f"-- {wall / dip:.1f}x apart. Reading either as the other is the error the two "
              f"tables exist to prevent; the dip does not bound the wait, and the wait does "
              f"not bound the damage.")
    print("\nThe three measured points do not rank by metric class. score_matrix runs four "
          "likelihood\nmetrics for 46s -- cheaper than the control. l1_fewshot is "
          "generative and costs 209s. ppl\nwas killed at 109s before touching a cache; the "
          "166GB it was about to read is why it was\nstopped, and that cost is unmeasured "
          "because it never happened.")
    print("\nHOST BYTES BY MIX -- what a cache-reading eval loads, per domain set (pod, "
          "2026-09-03):")
    for m in mixes:
        b, missing = mix_cache_bytes(m)
        if not b:
            continue
        note = f"  ({len(missing)} domain(s) with no cache)" if missing else ""
        mark = "  <-- reference for the table above" if m == ref else ""
        print(f"  {os.path.basename(m):32s} {b / 1e9:7.1f} GB{note}{mark}")
    # COUNTED, NOT TYPED: this line read "ALL 22 caches" as a literal while the dict grew to 27
    # when the injection domains were recorded, so the label disagreed with the sum beside it.
    print(f"  {f'ALL {len(CACHE_BYTES)} caches on /data00':32s} "
          f"{sum(CACHE_BYTES.values()) / 1e9:7.1f} GB")
    print("\nzh_web alone is 85 GB and tokens_sample is 5 MB, a 16,000x range. That is why this "
          "column\nis bytes: 'reads a token cache' put both in one bucket, and the rule derived "
          "from it\n(wait for the run) is right for the first and absurd for the second.")
    return 0


def _selftest():
    """Known answers for the arithmetic, and one property of the table.

    The arithmetic is the whole claim -- if interval_secs is wrong, every number in the
    table is wrong in the same direction and the table still looks plausible.
    """
    # 12K/gpu x 8 ranks over 10 steps of 32x4096 tokens per rank.
    assert abs(interval_secs(12) - 109.2) < 0.1, interval_secs(12)
    assert abs(interval_secs(6) - 218.5) < 0.1, interval_secs(6)
    # Halving the rate doubles the interval.
    assert abs(interval_secs(6) - 2 * interval_secs(12)) < 1e-6
    # The sustained rate costs nothing by definition.
    assert cost_secs([12, 12, 12]) == 0
    # And the published figures reproduce.
    assert abs(cost_secs([7]) - 78) < 1, cost_secs([7])
    assert abs(cost_secs([9, 11]) - 46) < 1, cost_secs([9, 11])
    assert abs(cost_secs([7, 6, 10]) - 209) < 1, cost_secs([7, 6, 10])
    assert abs(cost_secs([8, 8]) - 109) < 1, cost_secs([8, 8])

    # The ETA amplification in §50: one interval at 8K instead of 12K.
    eta = lambda k: 19151 * PER_STEP_PER_RANK / (k * 1000) / 3600  # noqa: E731
    assert abs((eta(8) - eta(12)) - 29.1) < 0.2, eta(8) - eta(12)
    assert abs(cost_secs([8]) - 54.6) < 0.5, cost_secs([8])

    # ppl.py must classify as a cache reader and score_matrix too (via val_seqs), because
    # that is the column the old rule lacked. If this ever flips, the table is lying about
    # the axis it exists to show.
    r = {n: c for n, c, _, _ in rows()}
    assert r["ppl.py"]["host_io"], "ppl.py no longer reads a token cache -- re-derive"
    assert r["domain_loss.py"]["host_io"], "domain_loss.py should read caches via val_seqs"
    assert not r["arc.py"]["host_io"], "arc.py should not touch a token cache"
    # THE NEGATIVE CASES, which the three above do not cover: every assertion here was a
    # POSITIVE (a file that does reach a cache, or one that plainly does not mention it), so
    # the column could be wrong in the one direction nobody tested -- a file that MENTIONS
    # the names without reading. Both defects below were live and green (de, 2026-09-03).
    assert not r["cache_guard.py"]["host_io"], (
        "cache_guard.py names _domain_seqs five times IN ITS DOCSTRING and reads only the "
        "tiny .vocab stamp; counting it as a >10GB reader inverts the column on the one "
        "file whose job is to check a cache without loading it"
    )
    assert r["nan_probe.py"]["ckpt_load"], (
        "nan_probe.py hard-codes its paths instead of taking --ckpt, and torch.loads a 2GB "
        "checkpoint plus a 6GB SFT pack; a '-' here understates it by 8GB"
    )
    assert not r["ceval.py"]["ckpt_load"], (
        "ceval.py's only --ckpt is a usage line in its docstring: it is a benchmark module "
        "run_eval.py invokes, and run_eval.py is what loads the checkpoint"
    )
    # A prose-only mention must never set a column. Checked as a property on real source
    # rather than per file, so a new eval that only documents the names is covered too.
    import glob as _glob

    for p in _glob.glob(os.path.join(ROOT, "eval", "*.py")):
        with open(p, encoding="utf-8") as fh:
            raw = fh.read()
        exe = _executable_src(raw)
        for name in CACHE_READERS:
            if name in raw and name not in exe:
                assert not classify(p)["host_io"] or any(
                    n in exe for n in CACHE_READERS
                ), f"{os.path.basename(p)}: host_io set by a prose-only mention of {name}"
    # And the table must not silently omit an eval.
    assert len(rows()) >= 25, f"only {len(rows())} evals found"

    # THE HOST-BYTES COLUMN REPRODUCES THE NUMBER AGENTS.md CITES, from a different source. The
    # rule says ppl reads "~166 GB across the nine domains of mix_500m"; that figure came from
    # per-domain estimates, and summing the pod's actual file sizes gives 166.2 GB. Two
    # derivations agreeing is the only reason to believe either. If this ever drifts, one of them
    # changed and the rule is quoting a number nothing measures.
    b, missing = mix_cache_bytes(os.path.join(ROOT, "data", "mix_500m.json"))
    assert b is not None, "data/mix_500m.json unreadable -- the host-bytes column has no reference"
    assert not missing, f"mix_500m names domains with no recorded cache: {missing}"
    assert 160e9 < b < 172e9, f"mix_500m cache total {b / 1e9:.1f} GB, AGENTS.md says ~166"

    # The spread is the whole reason this column is bytes rather than a boolean. If it ever
    # collapses, the boolean was adequate after all and this complexity is not earned.
    lo, hi = min(CACHE_BYTES.values()), max(CACHE_BYTES.values())
    assert hi / lo > 1000, f"cache sizes span only {hi / lo:.0f}x -- a boolean would do"

    # THE TWO TABLES MUST NOT BE READ AS ONE. l1_fewshot is the only eval with a dip AND a
    # wall time, so it is the only place the claim "these are different quantities" can be
    # checked rather than argued. If they ever came within 2x, keeping two tables would be
    # unearned complexity -- and if a future edit copies one number into the other table,
    # this is what catches it (measured 209.1s dip vs 2134.3s wall, 10.2x).
    both = [n for n, _, s, _ in rows() if s is not None and n in WALL_SECS]
    assert both, (
        "no eval has both a dip and a wall time -- the two-table split is then a claim with "
        "no measurement behind it; l1_fewshot.py was that measurement"
    )
    for _n in both:
        _dip = next(s for m, _, s, _ in rows() if m == _n)
        _wall = WALL_SECS[_n][0]
        assert _wall != _dip, (
            f"{_n}: wall and dip are the same number -- one table was copied into the other")
        assert _wall / _dip > 2, (
            f"{_n}: wall {_wall:.1f}s is only {_wall / _dip:.1f}x the dip {_dip:.1f}s. Two "
            f"tables are earned by that gap; without it, one column would do"
        )
    # A WALL_SECS entry must name a real eval, or the column silently describes nothing.
    _names = {n for n, _, _, _ in rows()}
    for _n in WALL_SECS:
        assert _n in _names, f"WALL_SECS names {_n}, which is not an eval in the table"

    print(
        f"selftest OK: interval arithmetic on 4 known answers, 4 published costs "
        f"reproduce, ETA amplification 29.1h vs 54.6s real, cache-reader column correct "
        f"on 6 files (3 positive, 3 negative) plus the prose-only property over all "
        f"{len(rows())} evals listed, and mix_500m's cache total {b / 1e9:.1f} GB "
        f"independently reproduces AGENTS.md's ~166 GB from pod file sizes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else main())
