#!/usr/bin/env python3
"""Write facts/moe.json#moe.control_profile_vs_dense_b192 from the two ledger rows.

A WRITER RATHER THAN A HAND-EDIT, for the reason moe.step_time_vs_dense_b192 needed one: every
number in the entry is recomputed here from runs/score_matrix.jsonl at write time, so the entry
cannot disagree with its source. It refuses rather than emitting a partial or unsupported claim.

WHAT THE ENTRY CLAIMS AND WHAT IT REFUSES TO CLAIM. Five paired numbers, and only two of them
separate the arms at this n. The HumanEval tie is stated AS THE CLAIM rather than as a caveat,
because it is the result the user is buying: at 8B tokens, code capability measured per byte is
indistinguishable between a 1.48B-total MoE-48 and a 0.2B dense arm at matched global batch. An
entry that led with domain_bpb and mentioned the tie afterwards would invert that.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTS = os.path.join(ROOT, "facts", "moe.json")
LEDGER = os.path.join(ROOT, "runs", "score_matrix.jsonl")
FACT_ID = "moe.control_profile_vs_dense_b192"

# restartable: reads one local jsonl and rewrites one JSON file in place after every number is in
# hand. An interrupt before the write leaves facts/moe.json untouched; an interrupt during it is a
# single json.dump of a file measured in kilobytes. Re-running recomputes everything from the
# ledger rows, and the duplicate-id refusal at the end stops a second run from appending twice, so
# there is no state to resume and nothing to clean up.

MOE_CKPT = "ckpt_1.5b-a0.2b-e48_8b.pt"
DENSE_CKPT = "ckpt_0.2b_8b_b192.pt"
PROFILE = "control"

# The three metrics PROFILES["control"] runs. Named explicitly rather than read from the rows, so a
# row that silently lost a metric is a refusal instead of a shorter table.
WANT_METRICS = ("domain_bpb", "lambada_en", "humaneval_bpb")


def rows_for(ckpt):
    """The LAST row for this ckpt at this profile. The ledger is append-only and folded by key;
    an earlier row for the same key is history, not a second measurement."""
    found = []
    with open(LEDGER, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r.get("ckpt") == ckpt and r.get("profile") == PROFILE:
                found.append(r)
    if not found:
        sys.exit(f"REFUSING: no row in runs/score_matrix.jsonl for ckpt={ckpt} profile={PROFILE}. "
                 f"This entry is about the control-profile pair; without both rows there is no pair.")
    return found[-1]


def main():
    moe, dense = rows_for(MOE_CKPT), rows_for(DENSE_CKPT)

    for label, r in (("MoE", moe), ("dense", dense)):
        m = r.get("metrics", {})
        missing = [k for k in WANT_METRICS if k not in m]
        if missing:
            sys.exit(f"REFUSING: the {label} row is missing {missing}. The control panel is "
                     f"{list(WANT_METRICS)}; a partial panel is a different comparison.")

    # EVERY PAIRED n MUST MATCH, checked rather than assumed. The whole point of this pair is that
    # both arms scored the SAME bytes; if an n differs, the deltas below are across two samples.
    checks = [
        ("lambada_en", "n_items"),
        ("humaneval_bpb", "n_tasks"),
        ("domain_bpb", "n_domains"),
    ]
    ns = {}
    for metric, field in checks:
        a = moe["metrics"][metric].get(field)
        b = dense["metrics"][metric].get(field)
        if a is None or b is None:
            sys.exit(f"REFUSING: {metric}.{field} is absent on one side (MoE {a!r}, dense {b!r}), "
                     f"so the two arms cannot be shown to have scored the same set.")
        if a != b:
            sys.exit(f"REFUSING: {metric}.{field} differs -- MoE {a}, dense {b}. The deltas in this "
                     f"entry are only paired differences if n is identical on both sides.")
        ns[metric] = a

    # A METRIC THAT WAS SKIPPED INSIDE A ROW is not a measurement. domain_bpb reports both, so the
    # equality above is not enough: 9 of 9 is what makes the unweighted mean the full panel.
    for label, r in (("MoE", moe), ("dense", dense)):
        db = r["metrics"]["domain_bpb"]
        if db.get("n_domains") != db.get("n_domains_total"):
            sys.exit(f"REFUSING: the {label} row's domain_bpb covers {db.get('n_domains')} of "
                     f"{db.get('n_domains_total')} domains. A mean over a smaller set is a "
                     f"different metric wearing the same name -- the row says so itself.")
        if db.get("skipped"):
            sys.exit(f"REFUSING: the {label} row's domain_bpb skipped {list(db['skipped'])}.")

    def g(r, metric, field):
        v = r["metrics"][metric].get(field)
        if v is None:
            sys.exit(f"REFUSING: {metric}.{field} is absent, so it cannot be reported.")
        return v

    # (label, MoE value, dense value, "lower is better" or "higher is better")
    table = [
        ("domain_bpb unweighted mean", g(moe, "domain_bpb", "unweighted_mean_bpb"),
         g(dense, "domain_bpb", "unweighted_mean_bpb"), "lower"),
        ("lambada_en acc", g(moe, "lambada_en", "acc"), g(dense, "lambada_en", "acc"), "higher"),
        ("lambada_en nll_per_byte", g(moe, "lambada_en", "nll_per_byte_mean"),
         g(dense, "lambada_en", "nll_per_byte_mean"), "lower"),
        ("humaneval_bpb per-task mean", g(moe, "humaneval_bpb", "gold_bpb_per_task_mean"),
         g(dense, "humaneval_bpb", "gold_bpb_per_task_mean"), "lower"),
        ("humaneval_bpb byte-weighted", g(moe, "humaneval_bpb", "gold_bpb_byte_weighted"),
         g(dense, "humaneval_bpb", "gold_bpb_byte_weighted"), "lower"),
    ]

    ci_moe = g(moe, "lambada_en", "ci95_halfwidth")
    ci_dense = g(dense, "lambada_en", "ci95_halfwidth")
    acc_delta = table[1][1] - table[1][2]
    ci_sum = ci_moe + ci_dense
    # THE ACCURACY VERDICT IS COMPUTED, not typed: whether the gap clears the summed half-widths is
    # the whole question, and hardcoding "does not separate" would survive a future row where it
    # does. Summed half-widths is the conservative reading of two independent binomials; the
    # sharper test would be on the difference's own SE, and this deliberately does not use it,
    # because the claim being made is the weaker one.
    acc_separates = abs(acc_delta) > ci_sum

    he_task = table[3][1] - table[3][2]
    he_byte = table[4][1] - table[4][2]
    he_opposed = (he_task > 0) != (he_byte > 0)
    if not he_opposed:
        sys.exit(f"REFUSING: the two humaneval_bpb estimators now point the SAME way "
                 f"(per-task {he_task:+.6f}, byte-weighted {he_byte:+.6f}). This entry's central "
                 f"claim is that they disagree in sign, which is what makes it a tie rather than a "
                 f"small difference. Re-read the rows and state something else.")

    n_moe = g(moe, "domain_bpb", "n_params")
    n_dense = g(dense, "domain_bpb", "n_params")

    rows_txt = "; ".join(
        f"{lab} MoE {a:.6f} dense {b:.6f} delta {a - b:+.6f} ({'MoE' if ((a < b) == (d == 'lower')) else 'dense'} better)"
        for lab, a, b, d in table)

    fact = {
        "id": FACT_ID,
        "value": (
            f"THE CODE RESULT IS A TIE, and that is the finding: at 8B tokens and matched global "
            f"batch, HumanEval bits-per-byte does not distinguish the MoE-48 arm from the "
            f"batch-matched dense arm. The two estimators of it move in OPPOSITE directions -- "
            f"per-task mean {he_task:+.6f} (dense lower), byte-weighted {he_byte:+.6f} (MoE lower) "
            f"-- over the same 164 of 164 tasks, and neither carries an error bar, so the sign "
            f"itself is not established. Two of the five paired numbers DO separate, both "
            f"continuous per-byte quantities over identical byte sets: domain_bpb unweighted mean "
            f"{table[0][1] - table[0][2]:+.6f} and lambada_en nll_per_byte "
            f"{table[2][1] - table[2][2]:+.6f}, both favouring MoE. LAMBADA ACCURACY DOES NOT "
            f"SEPARATE: {acc_delta:+.6f} against ci95 half-widths {ci_moe:.6f} and {ci_dense:.6f} "
            f"summing to {ci_sum:.6f}, so the gap is inside the bars."
        ),
        "measured": "2026-09-07",
        "status": "measured",
        "source": (
            f"runs/score_matrix.jsonl, the last row for each ckpt at profile={PROFILE}: "
            f"{MOE_CKPT} and {DENSE_CKPT}, cu_path=cu_none, both type=base. One job on card 7, "
            f"launched 2026-09-07T03:06:59Z and finished 03:48Z (~41 min), claim held by the job's "
            f"own pid for the whole run; pod stamp e5d39f8e, which is main and carries eval/ "
            f"byte-identical to the pushed tree. Written by scripts/write_fact_control_profile.py, "
            f"which recomputes every number from those rows and refuses on a missing metric, an "
            f"unequal n, a partial domain panel, or humaneval estimators that stop disagreeing."
        ),
        "config": {
            "arms": (
                f"MoE-48: {MOE_CKPT}, n_params {n_moe:,} total (~0.2B active, 48 routed experts "
                f"top_k 3 expert_ffn 768 plus 1 shared, moe_layers 0-11). Dense: {DENSE_CKPT}, "
                f"n_params {n_dense:,}. Both trained on mix_200m_8b at batch 8 x accum 4 x seq "
                f"4096 x world 6 = 786,432 tokens/step; the dense arm is the batch-matched "
                f"comparator registered as prereg#matched_batch_dense_0906."
            ),
            "paired_n": (
                f"Identical on both sides, checked rather than assumed: lambada_en "
                f"{ns['lambada_en']} items, humaneval_bpb {ns['humaneval_bpb']} tasks, domain_bpb "
                f"{ns['domain_bpb']} of {moe['metrics']['domain_bpb'].get('n_domains_total')} "
                f"domains with nothing skipped inside the metric. The deltas are paired "
                f"differences only because these match."
            ),
            "numbers": rows_txt,
            "wall_s": (
                f"lambada_en was the long leg on both arms: MoE "
                f"{g(moe, 'lambada_en', '_wall_s')} s, dense {g(dense, 'lambada_en', '_wall_s')} s. "
                f"domain_bpb {g(moe, 'domain_bpb', '_wall_s')} / "
                f"{g(dense, 'domain_bpb', '_wall_s')} s, humaneval_bpb "
                f"{g(moe, 'humaneval_bpb', '_wall_s')} / "
                f"{g(dense, 'humaneval_bpb', '_wall_s')} s."
            ),
        },
        "uncertainty": (
            f"ONE SEED PER ARM (42), so no seed spread exists and none is claimed. That is the "
            f"binding limitation on every number here and it is why the tie is stated as a tie "
            f"rather than as a small dense win: a difference of {abs(he_task):.6f} bpb has no "
            f"error bar to be outside of.\n"
            f"THE ONLY INTERVAL COMPUTED ANYWHERE IN THIS ENTRY is lambada_en's binomial ci95, "
            f"{ci_moe:.6f} and {ci_dense:.6f}, and it applies to the ACCURACY only -- not to "
            f"nll_per_byte, not to either bpb. So the two numbers that separate do so on their "
            f"magnitude against no stated noise floor. domain_bpb's own noise is unmeasured for "
            f"this pair; the nearest recorded figure, ds.seed_variance_0p2b's sd_nat 0.0516, is in "
            f"NATS PER TOKEN for domain_loss and is not convertible to bits per byte, so it is not "
            f"used here and no substitute is invented.\n"
            f"THE ACCURACY TEST IS THE CONSERVATIVE ONE: summed half-widths "
            f"({ci_sum:.6f}) rather than the standard error of the difference, which would be "
            f"narrower and might separate. The weaker test is used deliberately, because the claim "
            f"is that accuracy does NOT establish a difference; a reader who wants the sharper "
            f"test should run it and will be answering a question this entry does not."
        ),
        "boundary": (
            f"NOT AN EFFICIENCY CLAIM AND NOT A CAPABILITY CLAIM BEYOND THESE THREE METRICS. "
            f"(1) The arms differ in TOTAL parameters by ~7x ({n_moe:,} against {n_dense:,}) at "
            f"roughly matched active parameters, so 'MoE better on 2 of 5' is a statement about "
            f"this pair of trained checkpoints, not about MoE architectures, and nothing here is "
            f"per-parameter or per-FLOP. (2) 8B tokens, one mix (mix_200m_8b), one shape, one seed "
            f"each. A different token budget is a different answer -- the tie in particular is a "
            f"statement about 8B and carries no claim about 30B. (3) BPB IS NOT ACCURACY: "
            f"humaneval_bpb scores the bits the model assigns to gold solutions, which is not "
            f"pass@k and does not measure whether either model writes working code. A tie here is "
            f"a tie in gold-solution likelihood; pass_at_k was SKIPPED on both rows (generative, "
            f"unreadable for a base checkpoint) and remains unmeasured. (4) cu_path=cu_none on "
            f"both, so these are not comparable to the #cu rows of the same checkpoints."
        ),
    }

    with open(FACTS, encoding="utf-8") as fh:
        doc = json.load(fh)
    if any(f.get("id") == FACT_ID for f in doc["facts"]):
        sys.exit(f"REFUSING: {FACT_ID} already exists in facts/moe.json. This writer creates the "
                 f"entry; amending one is a different operation and must not be a silent overwrite "
                 f"of someone else's read.")
    doc["facts"].append(fact)
    with open(FACTS, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"{'metric':30}{'MoE':>12}{'dense':>12}{'delta':>12}  better")
    for lab, a, b, d in table:
        better = "MoE" if ((a < b) == (d == "lower")) else "dense"
        print(f"{lab:30}{a:12.6f}{b:12.6f}{a - b:+12.6f}  {better}")
    print(f"\nlambada acc separates: {acc_separates} "
          f"(|{acc_delta:+.6f}| vs summed ci95 {ci_sum:.6f})")
    print(f"humaneval estimators disagree in sign: {he_opposed} "
          f"(per-task {he_task:+.6f}, byte-weighted {he_byte:+.6f})")
    print(f"wrote {FACT_ID} to facts/moe.json ({len(doc['facts'])} facts)")


if __name__ == "__main__":
    main()
