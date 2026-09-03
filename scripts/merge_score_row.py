"""de-41 follow-up: merge the one real score_matrix contradiction, by 6e's written rule.

ckpt_p200m_4b_0902.pt was scored twice and each side holds real numbers the other lacks.
6e's rule, 2026-09-03: on the same (ckpt, metric), a value on one side against None or an
error on the other takes the value; two differing VALUES are a real contradiction for a human.

MEASURED before merging, because the rule's second branch decides whether this is mergeable
at all: of the nine metrics, four are one-sided (humaneval_bpb and lambada_en only on the pod;
l1_fewshot only locally, with the pod carrying ArtifactExists; domain_bpb is None locally and
an error on the pod, so it stays absent) and the five present on BOTH differ only on `_wall_s`.
Not one measured value disagrees. So this is complementary, not contradictory, and no human
choice is required -- which is what the rule was for.

ArtifactExists is NOT "the metric measured zero" (6e). It means an older result is still on
disk and the writer refused to overwrite it; b0 verified the 977,922-byte preds file is that
run's only copy. Merging it away as a missing value is right; recording it as a measured zero
would not be.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
PATH = os.path.join(ROOT, "runs/score_matrix.jsonl")
KEY = ("ckpt_p200m_4b_0902.pt", "full")


def _absent(v):
    """A metric with no usable number: never run, or run and refused."""
    return v is None or (isinstance(v, dict) and "error" in v)


def merge_metrics(local, pod):
    """(merged, conflicts). conflicts are (metric, local, pod) where both hold values that
    differ on something other than wall time."""
    out, conflicts = dict(local), []
    for m in sorted(set(local) | set(pod)):
        lv, pv = local.get(m), pod.get(m)
        if lv == pv:
            continue
        if _absent(lv) and not _absent(pv):
            out[m] = pv
        elif _absent(pv) and not _absent(lv):
            out[m] = lv
        elif _absent(lv) and _absent(pv):
            # NEITHER side has a usable number, and the two absences differ in kind: local
            # None against the pod's `domain_bpb.py exited 1` is the live shape. Without this
            # branch it fell to the else and filed as a conflict -- "a human decides which
            # measurement is right" about two non-measurements. Keep the LOCAL absence: an
            # error string is provenance for why nothing was measured, and promoting it into
            # the metric would make a failed eval look like a recorded one.
            pass
        elif isinstance(lv, dict) and isinstance(pv, dict):
            # _wall_s is how long the eval took, not what it measured. Two runs of the same
            # eval differ on it by construction, and treating it as a disagreement would make
            # every re-scoring a contradiction.
            real = [f for f in set(lv) | set(pv) if f != "_wall_s" and lv.get(f) != pv.get(f)]
            if real:
                conflicts.append((m, lv, pv))
            else:
                out[m] = lv
        else:
            conflicts.append((m, lv, pv))
    return out, conflicts


def selftest():
    v = {"acc": 0.05, "_wall_s": 1.0}
    v_slow = {"acc": 0.05, "_wall_s": 9.0}
    v_other = {"acc": 0.99, "_wall_s": 1.0}
    err = {"error": "l1_fewshot.py exited 1: ArtifactExists"}

    m, c = merge_metrics({"a": None}, {"a": v})
    assert m["a"] == v and not c, (m, c)
    m, c = merge_metrics({"a": v}, {"a": err})
    assert m["a"] == v and not c, "an error on one side must not displace a real value"
    m, c = merge_metrics({"a": None}, {"a": err})
    assert m["a"] is None and not c, "no usable number on either side is not a conflict"
    # wall time is not a measurement
    m, c = merge_metrics({"a": v}, {"a": v_slow})
    assert m["a"] == v and not c, "_wall_s differing must not read as a contradiction"
    # and the negative that makes the rule a rule
    m, c = merge_metrics({"a": v}, {"a": v_other})
    assert len(c) == 1 and c[0][0] == "a", "two differing measured values MUST stay a conflict"
    assert "a" in m
    print(
        "selftest OK: value-over-absent both directions, error is absent not zero, "
        "_wall_s excluded, and two differing measured values stay a conflict"
    )


def main():
    if "--selftest" in sys.argv:
        return selftest()
    rows = [json.loads(ln) for ln in open(PATH, encoding="utf-8") if ln.strip()]
    mine = [r for r in rows if (r.get("ckpt"), r.get("profile")) == KEY]
    if len(mine) != 1:
        raise SystemExit(f"expected exactly 1 local row for {KEY}, found {len(mine)}")
    local = mine[0]

    from pod_pull_ledgers import parse, read_pod

    text, err = read_pod("runs/score_matrix.jsonl")
    if err:
        raise SystemExit(f"cannot read the pod: {err}")
    pod_rows, _ = parse(text)
    pod = [r for r in pod_rows if (r.get("ckpt"), r.get("profile")) == KEY]
    if len(pod) != 1:
        raise SystemExit(f"expected exactly 1 pod row for {KEY}, found {len(pod)}")
    pod = pod[-1]

    merged, conflicts = merge_metrics(local.get("metrics", {}), pod.get("metrics", {}))
    for m in sorted(set(merged)):
        lv, pv = local.get("metrics", {}).get(m), pod.get("metrics", {}).get(m)
        src = "both" if merged[m] == lv and merged[m] == pv else ("local" if merged[m] == lv else "pod")
        got = "absent" if _absent(merged[m]) else "value"
        print(f"  {m:16s} <- {src:5s} ({got})")
    if conflicts:
        print(f"\n{len(conflicts)} REAL conflict(s), not merged -- a human picks:")
        for m, lv, pv in conflicts:
            print(f"  {m}: local={str(lv)[:70]}")
            print(f"  {m}:   pod={str(pv)[:70]}")
        return 1
    if "--apply" not in sys.argv:
        print("\nno real conflicts. Re-run with --apply to write the merged row.")
        return 0

    out = dict(local)
    out["metrics"] = merged
    nt = dict(local.get("noise_thresholds") or {})
    for k, v in (pod.get("noise_thresholds") or {}).items():
        if k not in nt:
            nt[k] = v
        elif isinstance(v, dict) and isinstance(nt[k], dict):
            nt[k] = {**v, **nt[k]}
    out["noise_thresholds"] = nt
    out["merged_from"] = (
        "de-41: the pod's row for the same (ckpt, profile) held humaneval_bpb and lambada_en "
        "that this row lacked; this row held l1_fewshot where the pod's carried ArtifactExists. "
        "No measured value disagreed -- the five metrics present on both sides differ only on "
        "_wall_s. Merged by 6e's rule 2026-09-03: value over None-or-error, two differing "
        "values would have been a human's choice."
    )
    sys.path.insert(0, ROOT)
    from eval.score_matrix import write_records

    write_records(PATH, [out if (r.get("ckpt"), r.get("profile")) == KEY else r for r in rows])
    print("\nwrote the merged row through eval.score_matrix.write_records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
