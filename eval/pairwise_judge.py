#!/usr/bin/env python3
"""Pairwise judge scorer for RL stage 4(b) -- stdlib only.

Consumes judge outputs (two position-swapped judgements per pair) and optional
test winners, emits per-pair verdicts and the acceptance report. Protocol:
docs/lessons/pairwise_judge_protocol.md.

Precondition (protocol section 1): pairs reaching the judge in production are
within the all-pass or all-fail subgroup -- tests decide first, the judge only
breaks ties. The judge must be a different model family from the policy
(section 1b).

Validation is decoupled from production (tilerl review, 2026-09-02): acceptance
runs on "mixed" pairs (one rollout passes tests, one fails), where test_winner
is a real gold standard -- a within-subgroup pair has no distinguishable gold,
so validating on it measures nothing (a tie-everything judge scores 1.0).
Agreement is measured only on mixed rows where the judge took a side; ties and
abstentions are reported beside it, never counted in the denominator.
Acceptance also requires the judge to have sided on at least MIN_VALIDATED of
the 20 pairs -- a 1.0 agreement on 2/20 sided is not a judge
(tilerl review, 2026-09-02).

Input JSONL, one line per pair:
  {"pair_id": "task01", "order_ab": "A", "order_ba": "B",
   "test_winner": "A", "subgroup": "mixed"}
subgroup is "all_pass" / "all_fail" (production) or "mixed" (validation).
test_winner is null for tasks without tests (reward issued, not validated).

Usage:
  python3 eval/pairwise_judge.py judgements.jsonl
  python3 eval/pairwise_judge.py --selftest
"""
import argparse
import json
import sys

VALID = {"A", "B", "tie"}

# Pre-registered 2026-09-02 (protocol section 4): of the 20 mixed validation
# pairs, the judge must take a side on at least 15. This floors n_validated
# (mixed+gold pairs where the judge sided), not the agreement denominator --
# if non-mixed rows enter the validation file, n_validated drops while the
# denominator stays the sided count, so the floor fails closed.
MIN_VALIDATED = 15


def verdict(order_ab, order_ba):
    """The pair's verdict from two position-swapped judgements.

    Same call both orders -> that call. Anything inconsistent (including
    tie-vs-side) -> abstain. Ties only count when BOTH orders say tie.
    """
    if order_ab not in VALID or order_ba not in VALID:
        return "abstain"
    if order_ab == order_ba:
        return order_ab
    return "abstain"


def score(rows, min_validated=MIN_VALIDATED):
    """Returns (verdicts, report). verdicts: pair_id -> verdict. report: dict.

    Agreement is measured only on mixed pairs (test differences exist) where
    the judge took a side A/B. Ties and abstentions never enter the
    denominator -- a tie-everything judge must score nothing, not 1.0.
    min_validated is injectable so the selftest can prove the floor is wired
    into accepted (bump it past the passing world and it must go red).
    """
    verdicts = {}
    agree = disagree = abstained = ties = unknown = 0
    n_validated = n_production = 0
    for r in rows:
        v = verdict(r["order_ab"], r["order_ba"])
        verdicts[r["pair_id"]] = v
        sub = r.get("subgroup")
        if sub not in ("mixed", "all_pass", "all_fail"):
            unknown += 1
        if v == "abstain":
            abstained += 1
            continue
        if v == "tie":
            ties += 1
            continue
        if sub == "mixed" and r.get("test_winner") is not None:
            n_validated += 1
            if v == r["test_winner"]:
                agree += 1
            else:
                disagree += 1
        elif sub in ("all_pass", "all_fail"):
            n_production += 1
    judged = agree + disagree
    return verdicts, {
        "agreement": agree / judged if judged else None,
        "abstain_rate": abstained / len(rows) if rows else None,
        "tie_rate": ties / len(rows) if rows else None,
        "n": len(rows),
        "n_validated": n_validated,
        "n_production": n_production,
        "n_unknown_subgroup": unknown,
        "accepted": (agree / judged >= 0.8 and n_validated >= min_validated) if judged else False,
    }


def _selftest():
    # Known-answer worlds. One must abstain, one must agree, one must disagree.
    rows = [
        {"pair_id": "consistent", "order_ab": "A", "order_ba": "A", "test_winner": "A", "subgroup": "mixed"},
        {"pair_id": "swapped", "order_ab": "A", "order_ba": "B", "test_winner": "A", "subgroup": "mixed"},
        {"pair_id": "tie_one_side", "order_ab": "tie", "order_ba": "A", "test_winner": "A", "subgroup": "mixed"},
        {"pair_id": "both_tie", "order_ab": "tie", "order_ba": "tie", "test_winner": "A", "subgroup": "mixed"},
        {"pair_id": "wrong", "order_ab": "B", "order_ba": "B", "test_winner": "A", "subgroup": "mixed"},
        {"pair_id": "no_test", "order_ab": "A", "order_ba": "A", "test_winner": None, "subgroup": "mixed"},
        {"pair_id": "prod", "order_ab": "A", "order_ba": "A", "test_winner": None, "subgroup": "all_pass"},
    ]
    verdicts, rep = score(rows)
    assert verdicts["consistent"] == "A", verdicts
    assert verdicts["swapped"] == "abstain", verdicts
    assert verdicts["tie_one_side"] == "abstain", verdicts
    assert verdicts["both_tie"] == "tie", verdicts
    assert verdicts["wrong"] == "B", verdicts
    # agreement: consistent(agree) + wrong(disagree) = 1/2; both_tie excluded,
    # no_test has no gold, prod is a production row
    assert rep["agreement"] == 0.5, rep
    assert rep["abstain_rate"] == 2 / 7, rep
    assert rep["tie_rate"] == 1 / 7, rep
    assert rep["n_validated"] == 2 and rep["n_production"] == 1, rep
    assert rep["accepted"] is False, rep
    # The blocker world (tilerl 2026-09-02): a tie-everything judge on mixed
    # pairs must NOT pass -- ties never enter the denominator.
    degenerate = [{"pair_id": str(i), "order_ab": "tie", "order_ba": "tie",
                   "test_winner": "A", "subgroup": "mixed"} for i in range(5)]
    _, rep_d = score(degenerate)
    assert rep_d["agreement"] is None and rep_d["accepted"] is False, rep_d
    assert rep_d["tie_rate"] == 1.0, rep_d
    # A world that DOES pass the bar: 16 agree, 4 disagree -> 0.8 exactly,
    # n_validated 20 >= MIN_VALIDATED.
    ok = [{"pair_id": str(i), "order_ab": "A", "order_ba": "A",
           "test_winner": "A", "subgroup": "mixed"} for i in range(16)]
    ok += [{"pair_id": f"d{i}", "order_ab": "B", "order_ba": "B",
            "test_winner": "A", "subgroup": "mixed"} for i in range(4)]
    _, rep2 = score(ok)
    assert rep2["agreement"] == 0.8 and rep2["accepted"] is True, rep2
    # Negative control (tilerl 2026-09-02): bump the floor past the passing
    # world and it MUST go red -- proves the constant is wired into accepted,
    # not just present in the file.
    _, rep2b = score(ok, min_validated=21)
    assert rep2b["accepted"] is False, rep2b
    # The participation hole (tilerl 2026-09-02): a judge that only sides on
    # 2/20 pairs, both right, must NOT pass -- agreement 1.0 means nothing
    # when the judge judged almost nothing.
    shy = [{"pair_id": "s0", "order_ab": "A", "order_ba": "A",
            "test_winner": "A", "subgroup": "mixed"},
           {"pair_id": "s1", "order_ab": "B", "order_ba": "B",
            "test_winner": "B", "subgroup": "mixed"}]
    shy += [{"pair_id": f"a{i}", "order_ab": "A", "order_ba": "B",
             "test_winner": "A", "subgroup": "mixed"} for i in range(18)]
    _, rep3 = score(shy)
    assert rep3["agreement"] == 1.0 and rep3["accepted"] is False, rep3
    assert rep3["n_validated"] == 2 and rep3["abstain_rate"] == 0.9, rep3
    print("selftest OK: verdict, abstention, tie-exclusion, agreement, the 0.8 bar "
          "and the participation floor all behave")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not a.jsonl:
        ap.error("jsonl required (or --selftest)")
    rows = [json.loads(l) for l in open(a.jsonl, encoding="utf-8") if l.strip()]
    verdicts, rep = score(rows)
    for pid, v in sorted(verdicts.items()):
        print(f"{pid}\t{v}")
    print(json.dumps(rep, ensure_ascii=False))
    if not rep["accepted"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
