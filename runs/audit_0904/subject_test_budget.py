#!/usr/bin/env python3
"""Pick the hook's subject->test pairs under a per-subject wall-time budget.

    python3 runs/audit_0904/derive_subject_tests.py --strict --depth > /tmp/depth_strict.tsv
    python3 runs/audit_0904/subject_test_budget.py

6e's item was to derive the hook's TESTS_FOR_SUBJECT from what each test imports or opens rather
than hand-list it. The derivation alone is not the map, and the numbers are why:

  - the loose "carries --selftest" set calls 167 files tests, including tools like
    scripts/profile_step_cost.py and eval/cache_guard.py. test_*.py is the honest set: 56 runnable
    pairs, not 146.
  - an import edge is not evidence a test exercises its subject. 23 hook-runnable tests import
    train.py and nearly all want `Cfg`, where every shape constant lives. Ranking by how many
    names a test takes from the subject and how often it uses them cuts train.py from 23 tests to
    6 at uses>=3.
  - cost is the binding constraint, not correctness. MEASURED on this laptop 2026-09-04, all 27
    candidate tests: 186.8s total, and at uses>=3 a train.py commit would pay 126.1s. The three
    most expensive are test_cursor_save 51.3s, test_arch_compat 18.0s, test_num_id_resolve 16.8s.

So the map is chosen deepest-first under a per-subject budget: 45s buys 27 pairs over 15 subjects
with a worst case of 42.6s (train.py). The five pairs that do not fit are PRINTED, never silently
dropped -- a hook that quietly skips coverage reads as "covered everything".

The budget and the depth threshold are judgements; the timings and the depths are measurements.
Re-run both scripts rather than trusting the numbers above.
"""
import collections

TIME = {
    "algorithms/test_rlvr_reward_suite.py": 0.12, "datagen/test_near_dedup_known.py": 0.15,
    "scripts/test_arch_compat.py": 18.02, "scripts/test_cursor_save.py": 51.33,
    "scripts/test_cvd_pattern.py": 0.13, "scripts/test_drop_zombies.py": 0.06,
    "scripts/test_e1_28_leak_scan.py": 0.10, "scripts/test_e1_28_matched.py": 0.07,
    "scripts/test_e1_29_floor_by_class.py": 0.16,
    "scripts/test_eval_base_prompt_format.py": 2.18, "scripts/test_eval_registry.py": 0.61,
    "scripts/test_fewshot_demos.py": 10.53, "scripts/test_fewshot_stop.py": 10.08,
    "scripts/test_launch_claims.py": 4.04, "scripts/test_ledger_predicates.py": 0.64,
    "scripts/test_muon_shape_lr.py": 15.30, "scripts/test_num_id_resolve.py": 16.84,
    "scripts/test_plan_length.py": 10.40, "scripts/test_pod_ps_judge.py": 0.09,
    "scripts/test_serve_history.py": 0.05, "scripts/test_sft_pack.py": 1.76,
    "scripts/test_sft_prefix.py": 1.89, "scripts/test_spawned_fast.py": 0.18,
    "scripts/test_split_bitwise.py": 8.25, "scripts/test_untie_head.py": 14.19,
    "scripts/test_value_embed.py": 10.03, "scripts/test_zero_init_out.py": 9.56,
}
rows = []
for line in open("/tmp/depth_strict.tsv", encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) == 4:
        rows.append((p[0], p[1], int(p[2]), int(p[3])))

BUDGET = 45.0  # seconds a commit touching one subject may add
per = collections.defaultdict(list)
for s, t, _k, u in sorted(rows, key=lambda r: -r[3]):
    if u >= 3:
        per[s].append((t, u, TIME.get(t, 0.0)))
kept, dropped = {}, []
for s, cands in per.items():
    tot, keep = 0.0, []
    for t, u, sec in cands:  # deepest first
        if tot + sec <= BUDGET:
            keep.append(t)
            tot += sec
        else:
            dropped.append((s, t, u, sec))
    kept[s] = (keep, tot)
print(f"budget {BUDGET:.0f}s per subject: {sum(len(v[0]) for v in kept.values())} pairs over "
      f"{len(kept)} subjects; {len(dropped)} pair(s) dropped for cost")
for s in sorted(kept, key=lambda x: -kept[x][1]):
    keep, tot = kept[s]
    print(f"  {tot:6.1f}s  {s} -> {', '.join(sorted(keep))}")
print("\ndropped for cost (named, never silent):")
for s, t, u, sec in dropped:
    print(f"  {sec:6.1f}s  {s} -> {t}  ({u} uses)")
