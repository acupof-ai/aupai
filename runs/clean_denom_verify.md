# E0 CLEAN denominator zero-contamination verification (2026-09-15)

Read-only, zero-GPU, no data generation. Confirms `eval/e0_merge_score.py` excludes exactly the r3 contamination unions so E0's CLEAN numbers use the verified denominators.

## Merger wiring (verified in source)

`eval/e0_merge_score.py` (on pod, main):
- `_exclude_he()` reads `runs/contam_r3_he_union.json#r3_humaneval_union` → 8 ids → CLEAN denom 164−8=**156**.
- `_exclude_mbpp()` reads `runs/contam_r3_mbpp_union.json#r3_mbpp_union` → 89 `mbpp427:N` ids → CLEAN denom 427−89=**338**.
- Exclusion is by task_id VALUE; shards pass `--no_clean` and the merger recomputes CLEAN — no clean-only data file that could drift.
- Both union files on pod byte-match main (sha256 MATCH for `runs/contam_r3_he_union.json`, `runs/contam_r3_mbpp_union.json`). No modification made.

## HumanEval — final id list (8 excluded, CLEAN 156)

Excluded: **19, 66, 71, 78, 105, 123, 129, 156**.

E0 n=10 temp0.2, metric of record = **per-sample pass@1 over all n=10 samples** (paired denominator 156 tasks x 10 = **1560**), NOT per-problem any-pass. Official merger HE: FULL 293/1640 = 17.87%, **CLEAN 292/1560 = 18.72%**, empty 0; the 8 excluded problems contribute **0 passing samples** (FULL−CLEAN = 293−292 = 1 passing sample sits on a problem that is neither; excluded-8 pass = 0). Any-pass (34/156 problems with >=1 pass) is an auxiliary column only, not the reported number.

## MBPP sanitized — final id list (89 excluded, CLEAN 338)

Excluded (89): 16, 67, 71, 83, 97, 101, 103, 108, 111, 117, 129, 141, 161, 164, 166, 167, 223, 224, 228, 229, 237, 249, 255, 259, 262, 265, 267, 290, 291, 296, 297, 388, 393, 399, 400, 401, 407, 408, 411, 414, 422, 428, 429, 431, 434, 438, 452, 454, 463, 468, 471, 479, 555, 578, 579, 580, 584, 597, 603, 608, 610, 612, 615, 616, 617, 625, 626, 627, 630, 635, 640, 721, 730, 733, 735, 745, 747, 749, 752, 755, 756, 758, 766, 778, 779, 780, 793, 804, 805.

E0 MBPP shards were still landing during this check (26/427 tasks present). Per-sample paired denominator 338 tasks x 10 = **3380** (FULL 4270). During shard landing excluded task 16 produced >=1 passing sample on an any-pass basis.

**Final merger result (E0 n=10 t=0.2): FULL 1120/4270 = 26.23%, CLEAN 909/3380 = 26.89%, empty 45.** The 89 excluded problems account for 1120−909 = **211 passing samples** (FULL over its 890 excluded-sample denominator = 23.7%), which CLEAN removes — concrete inflation the denominator protects against. (A shard0 shell briefly blocked on a stale humaneval_gen zombie claim; 0e released it and the merger completed over all 8 shards.)

## Auxiliary eval exclusion sets — unchanged (contam_evalscreen_r3_final.json)

| set | excluded | CLEAN |
|---|---:|---:|
| arc-easy | 20 (94,279,338,442,473,614,697,736,776,866,948,1182,1493,1572,1600,1802,2005,2145,2173,2213) | 2221 |
| MMLU | 478 (machine-readable `excluded_ids`; 52 answer-bearing) | 13564 |
| lambada_en | 0 (3235 FP, 360 prompt-only) | 5153 |
| code500/codev2/math500/cmmlu | 0 (idiom/LaTeX FP) | full n |

These are auxiliary metrics, not in the 30% primary criterion (primary HE CLEAN 156; secondary MBPP CLEAN 338).

## Exceptions

None in denominator wiring. HE: 0/8 excluded pass at n10. MBPP: at least 1 excluded task (16) produces a passing n10 sample while shards land — expected and exactly why CLEAN excludes it; full count deferred to the merged result. No GPU used, no union file changed.
