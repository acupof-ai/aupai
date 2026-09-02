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

# Static facts read off each file, so a new eval cannot be silently absent from the table.
# host_io means "reads a token cache from /data00", i.e. goes through train._domain_seqs.
CACHE_READERS = ("_domain_seqs", "val_seqs")


def classify(path):
    src = open(path, encoding="utf-8").read()
    return {
        "ckpt_load": bool(re.search(r"load_checkpoint\(|--ckpt", src)),
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
    print(f"{'eval':28s} {'ckpt':5s} {'>10GB':6s} {'gen':4s} {'measured':>9s}  note")
    print("-" * 108)
    for name, c, secs, note in rows():
        got = f"{secs:.0f}s" if secs is not None else "-"
        print(f"{name:28s} {'yes' if c['ckpt_load'] else '-':5s} "
              f"{'YES' if c['host_io'] else '-':6s} {'yes' if c['generative'] else '-':4s} "
              f"{got:>9s}  {note}")
    unmeasured = [n for n, _, s, _ in rows() if s is None]
    print(f"\n{len(unmeasured)} of {len(rows())} evals are NOT MEASURED -- listed as '-', "
          f"never as zero:")
    print("  " + ", ".join(unmeasured))
    print("\nThe three measured points do not rank by metric class. score_matrix runs four "
          "likelihood\nmetrics for 46s -- cheaper than the control. l1_fewshot is "
          "generative and costs 209s. ppl\nwas killed at 109s before touching a cache; the "
          "166GB it was about to read is why it was\nstopped, and that cost is unmeasured "
          "because it never happened.")
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
    # And the table must not silently omit an eval.
    assert len(rows()) >= 25, f"only {len(rows())} evals found"
    print(f"selftest OK: interval arithmetic on 4 known answers, 4 published costs "
          f"reproduce, ETA amplification 29.1h vs 54.6s real, cache-reader column correct "
          f"on 3 files, {len(rows())} evals listed")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest() if "--selftest" in sys.argv else main())
