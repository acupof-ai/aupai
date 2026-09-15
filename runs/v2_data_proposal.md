# V4.1 → v2 data proposal: HumanEval ≥30% go/no-go (2026-09-15, 3b; gap clustering genB; corpus cross-read 0e)

Cardless, no generation, no GPU, no deletion. Every number below is recomputed from pod
artifacts. This is a decision document for the user: pick a token tier and a mix ruling.

**Bottom line.** The gate gap is coverage, measured two ways. 108/156 clean HumanEval tasks
are never solved in 10 draws (E0); the 63.7M-token narrow SFT proved narrow data does not fix
this and taxes language. The fix is more broad, primitive-balanced pretraining tokens, mixed
with base corpus (not narrow SFT). **~31B of already-decontaminated, already-tokenized code
sits on `/data00` unused by the 30B gate run**, so a first +30–40B tier costs zero generation
and 1.7–2.3 days. Hitting 30% is plausible but not guaranteed: it requires converting roughly
25–35 of the 108 never-correct tasks to partial competence, on top of the observed
~0.6–0.95 HE-point-per-B-token main-phase rate which was already decaying before anneal.

---

## 1. The gap, measured (not estimated)

Source: `data/eval/e0_{he,mbpp}_merged.n10temp0.2.jsonl`,
`psft_{he,mbpp}_merged.n10temp0.2.jsonl`, unions
`runs/contam_r3_{he,mbpp}_union.json`. Per-task c_i/10 over the clean sets (HE 156, MBPP 338).

- HE E0: 108 tasks c=0, 15 c=10, 33 unstable; per-sample mean 18.72, coverage (≥1/10) 30.77.
- MBPP E0: 202 tasks c=0, 48 c=10; mean 26.89, coverage 40.24.
- After phi SFT (63.7M × 6 epoch): HE mean 20.45 but 97 of the 108 c=0 tasks stay c=0
  (11 gained, 9 previously-nonzero collapsed to 0); MBPP net-negative tasks (24 gained,
  27 lost). Narrow SFT redistributes competence, it does not add coverage.

### 1.1 Primitive clustering of the c=0 / near-zero tasks

Single-primary-label, keyword-first-hit over the real problem text (HE docstring, MBPP
`prompt` joined by task id from `data/eval/sanitized-mbpp.json`). genB ran
`scripts/cluster_gap.py` on the pod; 3b independently re-derived both tables from the same
raw files; counts agree within label-order noise.

HE c=0 (108), avg c per class E0→psft:

| primitive class | n | avg c E0 | avg c psft | example ids |
|---|--:|--:|--:|---|
| string transform / parse (case, split, encode, vowels, substring) | ~43 | 0 | 0.3 | 1,7,16,64,89,93,101,140 |
| number theory / digits / prime / modular | ~42 | 0 | 0.3 | 36,39,65,75,79,84,94,131,150 |
| early-exit / condition / boolean validation | 20 | 0 | 0.4 | 54,72,76,80,92,124,141 |
| sort / search / order / kth | 15 | 0 | 0.4 | 20,33,37,47,69,70,126,145 |
| dict / group / count / frequency | 9 | 0 | **0.0** | 16,91,95,111,128 |
| boundary / threshold / closest | 7 | 0 | 0.71 | 5,20,102,107,118 |
| stack / parenthesis / bracket match | 6 | 0 | **1.33** | 1,6,72,119,132 |
| string palindrome / reverse | 5 | 0 | 0.2 | 10,73,112,161 |
| fractional / decimal / float / binary text | 4 | 0 | **0.0** | 2,79,103,133 |
| geometry / polygon / interval | ~3 | 0 | ~0 | 32,127,145 (HE has only ~3) |

MBPP c≤2 (228 of 338), 3b independent recount:

| primitive class | n (3b) | n (genB) | avg c E0 | avg c psft |
|---|--:|--:|--:|--:|
| boundary / threshold / divisibility | 34 | 57 | 0.1 | 0.65 |
| sort / search / order | 45 | 46 | 0.3 | 0.72 |
| early-exit / validation | 42 | 33 | 0.1 | 0.42 |
| dict / count / group | 29 | 16 | 0.2 | 0.38 |
| string / words / palindrome | 21 | 14 | 0.2 | ~0 |
| list transform | 16 | 6 | 0.0 | 1.0 |
| geometry / grid / matrix | 15 | 12 | 0.4 | 0.6 |
| numeric sequence / digits | 11 | 4 | — | 2.5 (small n) |
| fractional / decimal | 3 | 6 | 0.0 | 0.2 |
| other / uncategorized | 12 | 23 | 0.1 | 0.8 |

Two distinct sub-populations:
- **SFT-recoverable primitives** — stack/paren (HE c1 1.33), boundary/threshold (0.71),
  MBPP sort-search (0.72): the model can learn these from a few thousand narrow examples.
  These are the cheapest gains.
- **Hard gaps — c≈0 even after SFT, and the textbook pool predicts little material**
  (genB ab_coverage): fractional/decimal, string palindrome, geometry/polygon,
  dict-aggregation. These need broad pretraining exposure, not another narrow pack.

### 1.2 What 30% requires

30% per-sample = 468/1560; E0 has 292, so +176 correct draws out of the 108 dead tasks:

| new avg c per converted task | dead tasks that must convert | resulting coverage |
|--:|--:|--:|
| 3/10 | 59 | 68% |
| 5/10 | 35 | 53% |
| 7/10 | 25 | 47% |

Realistic band: **convert ~25–35 tasks to partial (c 4–7)**. Coverage is the ceiling RL or
SFT can stabilise; it cannot exceed the set of tasks that ever succeed. v2 must move coverage,
which is what broad tokens do and narrow SFT did not.

---

## 2. What corpus already exists (zero-generation headroom)

Packed token pools on pod `/data00` (byte size / 4 for int32; measured
`facts/corpus_supply.json#cs.v41_gate_token_caches_rebuilt_f1f86097_0911` for the five
non-ultra domains; ultra from 0e's build_corpus_stats). "drawn" = gate weight × 30B
(cot repeats 3×). All are 13-gram-decontaminated at the gate vocab f1f86097.

| domain | pool | drawn in 30B run | unconsumed |
|---|--:|--:|--:|
| code_ultra_l3_noexec_dc | 26.70B | 9.00B | **17.70B** |
| code_py_starcoder_dc | 7.95B | 2.10B | **5.85B** |
| math_owm_stage2_dc | 5.86B | 2.40B | 3.46B |
| code_ultra_l2_dc | 15.34B | 13.50B | 1.84B |
| code_keep_p1_dc | 2.63B | 0.90B | 1.73B |
| en_c4_stage2_dc | 1.99B | 1.35B | 0.64B |
| code_py_rp1t_dc | 0.38B | 0.30B | 0.08B |
| zh_c4_dc / zh_wiki_dc (off-mix, ready) | 0.50 / 0.27B | 0 | 0.77B |
| **total ready, unused** | | | **≈ 32.1B** |

Plus `tokens_code_ultra_l3_stub_dc.excl56d…pt` ≈ 4.9B, holdout-conditional (the phi SFT
provenance reserve; usable only outside the holdout exclusion and with the same D5 scan).

Boundary: pools are a packed stream, not recomputed per-primitive; the unconsumed share of L3
noexec is static-filtered (no execute() solution filter, user order 2026-09-11), so its
exercise quality distribution is measured but its primitive balance is not. Per-domain build
funnels (0e, pod-measured 2026-09-15, kept/scanned docs): L2 16.36M/29.38M (44% dropped),
L3-noexec 15.31M/20.75M, starcoder 6.18M (188 decon hits, drop 0.025%), math_owm 4.13M,
keep_p1 3.06M; the five code pools total 53.0B packed (math 5.86B is separate).

Primitive-targeted coverage over these pools is the open analysis. `scripts/ab_coverage.py`
(pod, 58 lines; product `runs/coverage_anchor.json`) counts 14 primitive regex sets over
title/body of the 1,938-row textbook pool. It extends to the code domains as a body-keyword
screening pass only, with four measured limits (0e): (a) its vocabulary is algorithmic-kata
primitives with unknown precision on natural/library code (starcoder/rp1t); (b) body
co-occurrence has no AST and overstates semantic coverage; (c) the title and per-row token
fields exist only in the textbook pool — other domains give title=0 and need live encoding;
(d) it loads fully in memory, so the 15M-row L3 pool needs a streaming rewrite. It is a
screening signal, not a coverage metric. The conservative mix move is to reweight toward L2
(natural code) + keep_p1 + math, which carry the control-flow/number-theory primitives, and
treat L3 noexec as volume.

---

## 3. v2 data recipe (recommendation)

### 3.1 Tier A — zero generation, +30B (recommended first decision gate)

Resume from `ckpt_v41_r3_0914.pt` (step 38070) into a **new mix** (`data/mix_v42_tierA.json`,
new domain-name refs where needed — never edit `mix_v41_gate.json`, frozen) composed only of
the unused decontaminated pools above, roughly:

- code: ~60% (L3 noexec 17.7B as volume + starcoder 5.9B + L2 1.8B + keep_p1 1.7B, cap each so
  no domain dominates; L3 noexec ≤ ~45% of the tier)
- math: ~15% (math_owm 3.5B — digit/modular/sequence primitives)
- English retention: ~15% (remaining en_c4 + cot)
- Chinese: ~10% (zh_c4 + zh_wiki 0.77B; code comments + reasoning — Chinese open-acc is at
  floor and MMLU 23% says general coverage is thin)

Mix, do not SFT: the phi run showed a narrow pack costs −2.7 LAMBADA-EN / −3.2 ARC-E. Code
share 60% is up from gate's 86% code composition but every token is co-present with language;
the general-ability retention signal is LAMBADA-EN ≥ 21 (r3 baseline 21.93) at the half point.

- **Epochs: 1.** All these rows are unseen by r3; a second epoch is a separate decision.
- Decontamination is two complementary gates (0e), both required when a source could contain
  eval text verbatim: (1) `filters/decontam_ngram.py` whitespace-13-gram (normaliser fp
  0aefe6a2) against HE-164 and **`data/eval/mbpp_holdouts.jsonl`** — note the decon gate uses
  the holdouts file, not the sanitized-427 we score against; (2) sha1(normalised content)[:16]
  against the 305,007-entry holdout registry/manifest for exact documents (the 273,879-excluded
  channel). Engine is vocab-independent and carries decontam_fp so a changed gate file forces a
  rebuild. Whole-domain path: `scripts/filter_gate_domains.py --domains a,b` (measured non-ultra
  drop ~0.025%, CPU-cheap); ultra domains decontaminate inside `datagen/ultradata_shards.py
  --aggregate`. A new eval set must first enter `datagen/holdout.py` REGISTRY
  (`check_eval_registry_complete` gates it). The phi pack D5 scan is the pack-slice template.

### 3.2 Tier B — generation required, +100B

If Tier A plateaus like the anneal did, the hard-gap primitives (fraction/decimal,
palindrome, geometry, dict-aggregate) need targeted material: the ~8.4M eligible existing
textbook/exercise pool (user order 2026-09-14: no new generators; mutate/select from the
existing pool via the mutation gate) plus, only if that pool's primitive coverage is
insufficient, a user-approved generation. Primitive-targeted data must itself be
13-gram-scanned against HE/MBPP/holdout before packing (the phi pack scan is the template:
21 rows / 3 union tasks, 0 new).

### 3.3 Training mechanics (no script changes without a PR)

- Resume: `train.py` loads weights AND continues the per-domain data cursor; the schedule
  recomputes total_steps from the new mix and prints `SCHEDULE MOVED` if the prior total
  differs (train.py:3821) — that is the expected path, extend total by the new steps.
  Confirm the new mix's cursor joins at 38070 per-domain (the #361 failure: stale step24000
  cursor gave a false rc0). Gate: `--dry` must show join38070 and six-domain cursor match,
  verified by json.load of derived files, not exit code.
- LR: do not restart warmup on already-trained weights; low constant LR
  (`--lr_scale` ≈ 0.2, matching the stage-2 A/B setting) with a short cosine warmdown is the
  prior used in this repo. Exact value is a controller ruling before launch.
- Validation half-point: after +15B, score HE/MBPP n=10 CLEAN, MMLU, LAMBADA-EN. Continue
  only if coverage moved (≥5 new c≥1 HE tasks) and LAMBADA held.

---

## 4. Cost on the current 8×H20 (204k tok/s measured, 147,194 s / 29.94B)

| added tokens | days at 204k tok/s | calendar |
|--|--:|---|
| +30B (Tier A) | 1.7 | ~2 days |
| +40B | 2.3 | |
| +60B | 3.4 | |
| +100B (Tier B) | 5.7 | ~6 days |
| +150B | 8.5 | ~9 days |
| 1T | 56.7 | 8 weeks single-node; multi-node B200 with an fp8/SM100 kernel port is a separate project |

Exclusive-card time; pretokenisation of the unused pools is already done (they are packed), so
Tier A has no CPU build queue. Tier B adds pool-build time on top.

## 5. Expected lift and uncertainty — honest

- Observed main-phase greedy CLEAN rate: 18k→24k +4.49 pt over 4.7B (+0.95 pt/B),
  24k→36k +5.76 pt over 9.4B (+0.61 pt/B), then 36k→37k −3.84 pt over 0.8B and anneal
  likelihood −0.005. The per-token rate was already decaying; extrapolating +0.6 pt/B to +30B
  would be wrong (that would imply +18 pt). A defensible prior for broad balanced tokens after
  a plateau is sharply sublinear: a literature-scaled prior (Chinchilla-shaped returns) puts
  +30B at roughly **+3 to +8 HE points**, i.e. ~22–27%, **below 30 at the low end**.
- 30% specifically needs ~25–35 of the 108 dead tasks converted (§1.2). Tier A is a coverage
  probe that buys the measurement: if +15–30B of balanced unseen code does not move the
  c=0 count by ≥~15 tasks, no amount of repeat tokens or narrow SFT reaches 30 and Tier B
  targeted generation is the only path (its own ~6 days).
- What is NOT evidence: the n=1 greedy trajectory swings (±4 pt at 36k→37k is one-sample
  noise); the SFT +1.73 HE (paired CI crosses 0, P .82). The decision metric is n=10 CLEAN
  per-sample mean and the dead-task count, both paired and bootstrapped as in
  `runs/psft_vs_e0_he_clean156.json`.

## 6. Decision requested

1. **Tier A go/no-go** (+30B, zero generation, ~2 days, exclusive 8-card block, resume r3,
   new mix from §3.1, half-point gate). Recommend go: the tokens are already paid for and
   decontaminated; the only cost is card time, and it settles whether scale alone moves the
   coverage count.
2. Pre-authorise the Tier B pool-build (CPU, no GPU) now so it is ready if Tier A plateaus.
3. Rulings needed before any launch: LR scale/warmup for resume (§3.3), card grant in
   `runs/card_assignment.json`, and explicit user go. Nothing here launches anything.
