# C3 deletion candidates — FINAL LIST, awaiting the user's named instruction

Owner 3b, task 3b-16. Produced 2026-09-06. **Nothing has been deleted.** The user's
standing order (2026-09-05) is that no deletion happens without an explicit named
instruction; this file is the list that instruction would name.

## Verdict

| group | count | size | verdict |
|---|---|---|---|
| `data/corpus/web_cci3_p*` | 24 dirs | **79.5 GiB** | DELETABLE, with three exceptions below |
| loose `batch_*.jsonl` | 143 files | 2.4 MiB | **NOT deletable — the task text is wrong** |

## The task text names a target that must not be deleted

3b-16 says "delete the unclaimed `web_cci3_p*` dirs and loose `batch_*.jsonl`". Every one
of the 143 `batch_*.jsonl` on the pod is in `data/corpus/sample/`, and **148 files there
are tracked in git** — it is the 2,000-document sample a checkout ships, named in
AGENTS.md under Mix. Deleting them would remove tracked content from the pod and leave a
checkout's sample unreproducible there. They are struck from this list. No other directory
on the pod holds a `batch_*.jsonl`.

## web_cci3: 24 dirs, 79.5 GiB

Reference scan (`data/mix_*.json`, `facts/`, `docs/`, `AGENTS.md`, `runs/*.jsonl`):

- **No mix names `web_cci3`.** `grep -l web_cci3 data/mix_*.json` is empty, so no ladder
  point and no A/B reads these directories, and the frozen-corpus rule does not apply.
- **Three facts cite them**, all `status: measured`, and this is what makes the three
  exceptions:
  - `facts/corpus_supply.json#cs.provenance_tiers_0903` names **`web_cci3_p0`** by path.
  - `facts/data_quality.json#dq.corpus_exact_dup_is_zero` names the `web_cci3_p*` family.
  - `facts/contamination.json#cont.cci3_cross_group_dup` names the family.
- **Zero KEEP claims** in `runs/board.jsonl` — `grep -i web_cci3` there returns nothing,
  and the 24h broadcast window (from 2026-09-05 ~05:00Z) has elapsed.

### Three dirs carry NO build stamp

`build_corpus_stats.json` is present in 21 of 24 and **absent in `p0`, `p22`, `p23`**.
The task's own instruction is "read each first, keep stamps" — for these three there is no
stamp to keep, so a delete is unrecoverable in a way the other 21 are not: nothing records
what filters produced them. `p0` is additionally the one directory cited by path in a
measured fact.

**Recommendation: delete 21, hold `p0`, `p22`, `p23`.** That releases 66.9 GiB and leaves
12.6 GiB. If the user's instruction names all 24, the three facts above should be marked
with a boundary saying their source directory no longer exists, in the same commit.

### Per-directory

| dir | size | parts | stamp |
|---|---|---|---|
| web_cci3_p0 | 4.1G | 43 | **absent** |
| web_cci3_p22 | 4.4G | 47 | **absent** |
| web_cci3_p23 | 4.1G | 43 | **absent** |
| web_cci3_p1 | 3.9G | 41 | present |
| web_cci3_p6 | 3.9G | 41 | present |
| web_cci3_p3 | 3.8G | 40 | present |
| web_cci3_p2 | 3.7G | 40 | present |
| web_cci3_p4 | 3.6G | 38 | present |
| web_cci3_p5 | 3.6G | 38 | present |
| web_cci3_p7 | 3.5G | 38 | present |
| web_cci3_p11 | 3.5G | 37 | present |
| web_cci3_p8 | 3.4G | 37 | present |
| web_cci3_p9 | 3.4G | 36 | present |
| web_cci3_p10 | 3.3G | 35 | present |
| web_cci3_p12 | 3.1G | 33 | present |
| web_cci3_p13 | 3.1G | 32 | present |
| web_cci3_p15 | 3.1G | 33 | present |
| web_cci3_p14 | 3.0G | 33 | present |
| web_cci3_p20 | 3.0G | 32 | present |
| web_cci3_p16 | 2.9G | 31 | present |
| web_cci3_p18 | 2.8G | 30 | present |
| web_cci3_p17 | 2.7G | 29 | present |
| web_cci3_p19 | 1.8G | 20 | present |
| web_cci3_p21 | 1.8G | 20 | present |

## What is NOT in this list

`runs/pod_ckpt_candidates_0905.txt` is a separate 85.7 GiB / 85-file checkpoint list owned
by fb, not by 3b-16. 4c's dispatch described this task as "C3 checkpoint deletion"; 3b-16's
text is corpus directories. They are different lists and this file covers only the latter.

## Method

Sizes and stamp presence read from the pod in the container view on 2026-09-06
(`du -sh`, `ls`). Reference scan run against the working tree at the same time. The
sizes are `du` GiB as the pod reports them, not summed part bytes.
