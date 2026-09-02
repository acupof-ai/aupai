#!/usr/bin/env python3
"""Pairwise judge scorer for RL stage 4(b) -- stdlib only.

Consumes judge outputs (two position-swapped judgements per pair) and optional
test winners, emits per-pair verdicts and the acceptance report. Protocol:
docs/lessons/pairwise_judge_protocol.md.

Precondition (protocol section 1): pairs reaching the judge are within the
all-pass or all-fail subgroup -- tests decide first, the judge only breaks
ties. The judge must be a different model family from the policy (section 1b).

Input JSONL, one line per pair:
  {"pair_id": "task01", "order_ab": "A", "order_ba": "B", "test_winner": "A"}
test_winner is null for tasks without tests (reward issued, not validated).

Usage:
  python3 eval/pairwise_judge.py judgements.jsonl
  python3 eval/pairwise_judge.py --selftest
"""
import argparse
import json
import sys

VALID = {"A", "B", "tie"}


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


def score(rows):
    """Returns (verdicts, report). verdicts: pair_id -> verdict. report: dict."""
    verdicts = {}
    agree = disagree = abstained = ties = 0
    for r in rows:
        v = verdict(r["order_ab"], r["order_ba"])
        verdicts[r["pair_id"]] = v
        tw = r.get("test_winner")
        if v == "abstain":
            abstained += 1
            continue
        if v == "tie":
            ties += 1
        if tw is None:
            continue  # no ground truth: reward issued, not validated
        if v == tw:
            agree += 1
        else:
            disagree += 1
    judged = agree + disagree
    return verdicts, {
        "agreement": agree / judged if judged else None,
        "abstain_rate": abstained / len(rows) if rows else None,
        "tie_rate": ties / len(rows) if rows else None,
        "n": len(rows),
        "n_with_tests": judged,
        "accepted": (agree / judged >= 0.8) if judged else False,
    }


def _selftest():
    # Known-answer worlds. One must abstain, one must agree, one must disagree.
    rows = [
        {"pair_id": "consistent", "order_ab": "A", "order_ba": "A", "test_winner": "A"},
        {"pair_id": "swapped", "order_ab": "A", "order_ba": "B", "test_winner": "A"},
        {"pair_id": "tie_one_side", "order_ab": "tie", "order_ba": "A", "test_winner": "A"},
        {"pair_id": "both_tie", "order_ab": "tie", "order_ba": "tie", "test_winner": "tie"},
        {"pair_id": "wrong", "order_ab": "B", "order_ba": "B", "test_winner": "A"},
        {"pair_id": "no_test", "order_ab": "A", "order_ba": "A", "test_winner": None},
    ]
    verdicts, rep = score(rows)
    assert verdicts["consistent"] == "A", verdicts
    assert verdicts["swapped"] == "abstain", verdicts
    assert verdicts["tie_one_side"] == "abstain", verdicts
    assert verdicts["both_tie"] == "tie", verdicts
    assert verdicts["wrong"] == "B", verdicts
    # agreement: consistent(agree) + both_tie(agree, tie==tie) + wrong(disagree) = 2/3
    assert rep["agreement"] == 2 / 3, rep
    assert rep["abstain_rate"] == 2 / 6, rep
    assert rep["accepted"] is False, rep  # 0.667 < 0.8
    # A world that DOES pass the bar: 4 agree, 1 disagree -> 0.8 exactly.
    ok = [{"pair_id": str(i), "order_ab": "A", "order_ba": "A", "test_winner": "A"}
          for i in range(4)]
    ok.append({"pair_id": "x", "order_ab": "B", "order_ba": "B", "test_winner": "A"})
    _, rep2 = score(ok)
    assert rep2["agreement"] == 0.8 and rep2["accepted"] is True, rep2
    print("selftest OK: verdict, abstention, tie, agreement and the 0.8 bar all behave")


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
