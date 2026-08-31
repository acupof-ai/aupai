---
question: Does the math SFT corpus (data/corpus/math/) contaminate the eval holdouts (math-500, math-hard-1k)?
status: measured
source: datagen/scan_math_contamination.py, 2026-08-30
---

# Audit — math SFT corpus vs eval holdouts

Run 2026-08-30. Scanner: `datagen/scan_math_contamination.py` (broken-world tests:
`datagen/test_scan_math_contamination.py`). No mix was edited.

## Which corpus — read this first

There are two math corpora in play and the numbers below are the LOCAL one. The
training pod (`iv-yeozpb5g5cbw80bls64e`, `/work/aupai/data/corpus/math/`) holds a
**different, smaller corpus** — 5 shards (en_math_text ×3, math_short_v10, v8), not
the local 9 — and it is the one 0830v1 trained on. Pod numbers
(`facts/contamination.json#cont.union_pod`): **math-500 0/500 (0.0%), math-hard
36/1,032 (3.5%)**, 0 exact, FPR 0/1,498 on 269 pod web docs. The local 30.0%/7.1%
(`facts/contamination.json#cont.union`) describes a copy that never trained anything.
**When explaining 0830v1 scores, cite the pod numbers.** The local audit stands on
its own as the gate record for the local shards, and the pod/local divergence is
itself the finding: both are called `data/corpus/math` and nothing but this audit
distinguishes them.

## Verdict (local corpus)

**Contaminated. 7 of 9 shards REJECT at containment 0.8 and do not enter the mix.**
223 of 1,498 long holdouts (14.9%) have a corpus row containing ≥80% of their text;
0 exact-normalized matches. The false-positive rate at this threshold is 0/1,498 on
2,000 assumed-clean web docs (`facts/contamination.json#cont.fpr`).

| shard | rows | holdouts hit ≥0.8 | verdict |
|---|---|---|---|
| ape210k_000 | 53,772 | 21 | REJECT |
| gsm8k_zh_000 | 7,398 | 0 | clean |
| math23k_000 | 19,916 | 0 | clean |
| math_short_sol_v1_000 | 2,262 | 11 | REJECT |
| math_short_v6_000 | 95,311 | 43 | REJECT |
| math_short_v7_000 | 73,682 | 37 | REJECT |
| math_short_v8_000 | 66,803 | 30 | REJECT |
| mxode_000 | 174,860 | 141 | REJECT |
| mxode_001 | 36,230 | 107 | REJECT |

Per-shard counts: `facts/contamination.json#cont.shards`. Union (a holdout hit in
any shard counts once): 507 / 223 / 84 holdouts at 0.7 / 0.8 / 0.9
(`facts/contamination.json#cont.union`). By holdout file at 0.8: math-500
150/500 (30.0%), math-hard 73/1,032 (7.1%) (`facts/contamination.json#cont.split`).

## Method — containment, not Jaccard

Containment = |holdout bigrams ∩ row bigrams| / |holdout bigrams|. It answers "how
much of the holdout is present in this row" and is monotone in row length. Jaccard
answers the wrong question: a verbatim holdout embedded in an 841-char document
scores Jaccard 0.339, so the pre-containment Jaccard screen (math-500 12.2%,
math-hard 0.3%, 949 mxode rows) undercounted — math-hard alone moved 0.3% → 7.1%
under containment. The Jaccard-era numbers are retained as
`facts/contamination.json#cont.jaccard_era` (status: recorded, method superseded).

Threshold validation: 0.8 was chosen before the scan; the FPR baseline (0 hits on
2,000 web docs) is the evidence it means anything. The 34 holdouts with <20 bigrams
are scored in a separate bucket and excluded from every count above — containment
saturates on short strings.

## What this measurement cannot answer

Containment ≥0.8 counts **number-swapped template variants** as hits: a corpus row
with the same problem template but different numbers shares most zh char-bigrams.
The same-numbers subset — true contamination vs template leakage — is unmeasured
(`facts/contamination.json#cont.same_numbers`); the Jaccard-era gap (12.2% vs 30.0%
on math-500) suggests template variants may be the majority. This changes severity,
not the verdict: template leakage still inflates scores on the holdout problems'
structure, and the scanner's policy is REJECT THE WHOLE SOURCE, not row filtering.

`gsm8k_zh_000` and `math23k_000` are clean at 0.8 (0 hits each) and are the only
math shards cleared for the mix.

## Throughput and incremental scanning (hardened 2026-08-30)

The scan is the gate every new source passes (`facts/contamination.json#cont.throughput`).
The old Python postings loop did 0.74M tok/s; containment is now one sparse matmul
(`R @ H`, exact, in C) — 6.8M tok/s single-process, 22M tok/s with 9 fork workers.
A 100B-token full rescan is **1.26h**, under the 2h budget, so LSH was rejected:
it is approximate and would need a recall proof against the 223 known hits where the
exact scan already fits. Two cheaper speedups were measured and rejected first:
df-pruning drops 0 bigrams (no zh bigram sits in >50% of holdouts), length-bucketing
has a 1.4× upper bound (rows and holdouts share one size distribution). The matmul
rewrite reproduces every old number exactly — per-shard hits at 0.7/0.8/0.9, the
507/223/84 union, FPR 0 — and the self-check plus six broken-world tests pass.

Incremental: `data/scan_ledger.jsonl` keys on (path, bytes, mtime, holdout_hash,
threshold). A re-run serves unchanged shards from the cache (verified: 9/9 shards
cached, verdicts and exit codes preserved); a holdout-set change invalidates every
entry and forces a full rescan (`facts/contamination.json#cont.ledger`). New deps:
numpy, scipy — the pod needs both.

## The math_short contamination is self-produced

The pod's REJECT shards are our own synthetic batches (`math_short_v8`, `math_short_v10`),
and every batch measured fails: v3 28, v5 39, v6 43, v7 37, v8 30, v10 12, v11 33
holdouts at 0.8 (`facts/contamination.json#cont.math_short_leak`). The leak path is
not seeds or a shared pool — it is **template DNA**: `mathbank/run_math_short.py` and
math_hard_eval_1k's older generator implement the same canon of Chinese hard-math
types, so hand-written templates converge in surface wording. `split_bank.py`
recorded 0.3% *identical-template* sharing and understated it 10×: containment ≥0.8
does not require template identity. `make_v11_band.py` aggravates it by design —
v11 batches are LP-matched to math_hard_eval_1k's surface statistics. Consequence:
math-hard v1 was the recorded metric, and 3.5% contamination puts every ~3%-level
score within contamination noise. **"Fix the bank" is no longer a todo** — it was
closed 2026-08-30 not by a fix but by v2's construction: math_short batches never
contaminated math-500 (0 hits; 141/150 of math-500's contaminated holdouts are
mxode's), they only ever contaminated math-hard v1, which is retired. The v1
REJECT records above are history, not open work. The executable gate is
`eval/gate_math_short.sh`, wired into `datagen/build_math_expand.sh`: it scans
against math-500 + v2, and all six local batches pass with 0 hits
(`facts/contamination.json#cont.gate`, `#cont.math_short_leak`).

The English shards are **not clean** — they are unmeasured: zh surface bigrams cannot
match them, and the cross-language number-multiset screen flags 6,754 rows (6.3%)
for human review (`facts/contamination.json#cont.cross_language_pod`).

## Corpus identity

`datagen/corpus_fingerprint.py` hashes each domain's (shard name, size, mtime);
build_corpus.py stamps it into `build_corpus_stats.json`, `save_checkpoint` stamps
it next to `vocab_id`, and harness check `corpus_fp_matches` fails on drift
(`facts/contamination.json#cont.corpus_fp`). The pod/local divergence that motivated
it is now impossible to repeat silently.

## math-hard v2 — the replacement eval (2026-08-30)

math_hard_eval_1k is retired: every math_short batch contaminated it because
both descend from the same elementary-olympiad canon (`facts/contamination.json#cont.math_short_leak`).
Its scores are void (`facts/contamination.json#cont.math_hard_v1_void`). The
replacement is `data/synthetic/math_hard_eval_v2_1k.jsonl` (1,080 problems,
generator `mathbank/eval_hard_v2_gen.py`, every problem constructed from its
answer, self-checked), and the scanner's default holdout set is now math-500 + v2.

**Type inventory — why the intersection is empty at the type level.** The bank's
943 L3/L4 programs cover the entire *elementary* olympiad canon: 行程 (相遇/追及/
火车过桥/流水行船/环形/扶梯/发车/接送/多次相遇), 工程, 浓度, 经济, 年龄, 鸡兔同笼,
盈亏, 还原, 和差倍, 平均数, 小学数论 (gcd/lcm/余数/位值/整除), 小学几何 (面积周长
体积/阴影/切割/排水/展开), 组合 (抽屉/容斥/排列组合基础/概率基础), 智巧 (蜗牛/空瓶/
过桥/称球/取石子), 周期, 方阵, 植树, 牛吃草, 钟表, 页码, 幻方, 定义新运算, 分数裂项.
A grep across all `math_programs_l*.py` finds **zero** occurrences of 方程/设未知数/
函数 — the bank is 100% arithmetic. v2's families live in the space that bank
structurally cannot generate:

| cluster | families |
|---|---|
| symbolic algebra | quadratic, abs-value equation, fractional equation, 2-var system, linear/inverse/quadratic function, factorize, variance, floor [x], perfect-square pattern |
| geometry | Pythagoras, 30-60-90 right triangle, similar triangles |
| elementary gaps | number-line moving points, cryptarithm, number-array, hound-and-hare |

**Acceptance (the gate, not a courtesy check):** with v2 as holdout, the scanner
exits 0 on all seven math_short batches (local v3/v5/v6/v7/v8/v11, pod v10 — v10
measured on the training pod itself) — 0 exact, 0 containment ≥0.8, short-holdout
bucket p100 ≤ 0.714. Reverse direction: v2 as corpus vs math-500 + math-hard-1k is
also clean (0/0/0 at 0.7/0.8/0.9), so the new eval carries no old-holdout problem
either (`facts/contamination.json#cont.math_hard_v2`).

## The holdout set itself must be rebuilt

math-500 is 30.0% contaminated **by the local corpus** and 0.0% by the pod corpus
that trained 0830v1 (`facts/contamination.json#cont.union_pod`) — host-specific
numbers, never merge them — and stays only as the active eval until a replacement
exists. The v2 rule (`facts/contamination.json#cont.holdout_v2`, tool:
`datagen/holdout_split.py`): each **new** source is split *before* ingest —
`qhash[:8] % 100 < 2` goes to `data/eval/holdout_v2/`, the rest is the corpus
candidate — and the remainder is gated against the full holdout; any hit refuses the
whole source and discards the slice. Eligibility is enforced: only never-ingested
sources (no ledger entry, not in PROVENANCE) can seed the holdout, so the
local-corpus contamination that hit math-500 is impossible by construction. The v2 set is
empty until the first eligible source arrives; switching the eval pipeline off
math-500 is a separate decision, recorded as the fact's uncertainty.
