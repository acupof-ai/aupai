# V4.1 → v2 data proposal: HumanEval ≥30% go/no-go (2026-09-15, 3b; gap clustering genB; corpus cross-read 0e)

Cardless, no generation, no GPU, no deletion. Every number below is recomputed from pod
artifacts. This is a decision document for the user: pick a token tier and a mix ruling.

**Bottom line.** The gate gap is coverage, measured two ways. 108/156 clean HumanEval tasks
are never solved in 10 draws (E0); the 63.7M-token narrow SFT proved narrow data does not fix
this and taxes language. The fix is more broad, primitive-balanced pretraining tokens, mixed
with base corpus (not narrow SFT). **~28B of fully-novel, already-tokenized tokens sit on
`/data00` never drawn by the r3 run** (plus ~24B whose documents r3 saw only as short stubs),
so a first +30B tier costs zero generation and ~1.7 days of the 8-card block. Hitting 30% is plausible but not guaranteed: it requires converting roughly
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

**The run that happened is `data/mix_v41_r3.json` (+ anneal `data/mix_v41_r3_anneal.json`),
not `mix_v41_gate.json`.** Domains/epochs differ; consumption below is read from the r3 final
checkpoint's own `row_cursor` / `row_cursor_srcfp` (ckpt step 38070, mix field
`data/mix_v41_r3.json`), not reconstructed from weights. Pools are pod `/data00/tokens_<d>.pt`,
int32 stream (bytes/4), row width 4097; all at gate vocab `f1f860970d15d623` (`.vocab`
sidecars read back), seed 42.

r3 main mix and its measured cursor:

| domain | epochs | pool | cursor rows | consumed | status |
|---|--:|--:|--:|--:|---|
| code_ultra_l3_stub_dc (srcfp 12ec3cd2) | 3 | 4.99B | 3,427,726 / 1,218,935 | 14.04B | cycled ~2.8×, exhausted |
| code_keep_p1_dc (d7b4f3a0) | 3 | 2.63B | 1,908,579 / 641,196 | 7.82B | cycled ~3×, exhausted |
| cot_dc | 3 | 0.40B | 234,373 / 97,631 | 0.96B | exhausted |
| code_ultra_l2_dc (adf2ff20) | 1 | 15.34B | 922,846 / 3,744,417 | 3.78B | **11.56B unseen tail** |
| math_owm_stage2_dc (4b1469bf) | 1 | 5.86B | 603,515 / 1,429,819 | 2.47B | **3.39B unseen tail** |
| en_c4_stage2_dc (c59c2e42) | 1 | 1.99B | 212,401 / 484,553 | 0.87B | **1.12B unseen tail** |

Plus anneal 3B: textbooks_claude_v41 0.5 + stub 0.3×3B + keep_p1 0.2×3B.

Domains **never named in the r3 mix** (full pool unseen as a scheduled domain), with srcfp:

| domain | pool | srcfp | caveat |
|---|--:|---|---|
| code_ultra_l3_noexec_dc | 26.70B | 63a3b0e6 | parent of stub — see bucket split below |
| code_py_starcoder_dc | 7.95B | 390a13a5 | fully novel; 427-rescan not yet run |
| code_py_rp1t_dc | 0.38B | af4988f9 | fully novel |
| zh_c4_dc / zh_wiki_dc | 0.50 / 0.27B | — | fully novel, off-mix |

The 26.70B L3-noexec pool needs a content-freshness split. Its build_stats
(`survivor_source_dir: data/corpus/code_ultra_l3_noexec_dc`) show the stub pool r3 cycled is
a **last-top-function stub transform of the same L3 docs**: stub kept 13,682,629 of noexec's
15,308,313 docs (89.4%). The uuid arithmetic gives the document partition, but the **token
content tells a different story** (0e independent audit `runs/v2_pool_audit_0e.md`, #372;
66 re-ran `runs/v2_overlap_probe.py` independently at K=50, n=350,140 pairs, plus a per-uuid
micro-sample n=6,396):

- uuid partition: 1,625,684 docs (10.62%) were never carried into the stub; sampled at 1,859
  tok/doc = **~3.0B new-uuid problems**. The other 13,682,629 (89.4%) are same-uuid full
  solutions, 1,723 tok/doc = **~23.6B** (stub rendered only the 365-tok last-def, 4.7× shorter).
- same-uuid token overlap is tiny: the stub text is a literal substring of the full solution
  only **5.95%** of pairs (whitespace-normalised 11.5%), so r3 did **not** read these tokens.
- of the 23.6B, **51.3% of full solutions fail `ast.parse` as a single module** (stub fails
  0.0%). This is not garbage: an unbiased hash-sample of all parse-fail rows (n=5,102) is
  0.82% markdown-fenced, 3.51% REPL, and 25.9% contain ≥2 def/class (median 5,239 chars,
  p10 3,199 / p90 9,397; 19.8% carry imports, 0.02% `__main__`) — English spec + full
  multi-function/script solutions. On the both-parse-ok 48.7%, the stub AST node-set is a
  subset of the full solution **99.99%** of the time (exact conditional counts in
  `runs/indep_overlap_66b.log`), so the parse-ok half (~11.5B) mostly re-contains the seen
  short def (lowest marginal novelty), while the parse-fail half (~12.1B) carries new
  function-external orchestration / multi-function context.

Honest buckets (all 26.7B are r3-UNREAD tokens; the split is by marginal value):

1. **High-novelty (never read, parse-agnostic):** L2 unseen tail 11.56 + starcoder 7.95 +
   rp1t 0.38 + zh 0.77 + en_c4 tail 1.12 (code/language pools) + noexec new-uuid 3.0, plus the
   separate math_owm tail 3.39 counted under math below.
2. **noexec same-uuid full solutions 23.6B** = ~12.1B parse-FAIL multi-function long context
   (high value for complex c=0 tasks, mild prose+code format risk) + ~11.5B parse-ok (AST
   already contains the seen def; lowest marginal novelty).
3. **Exhausted (skip):** stub 4.99, keep_p1 2.63, cot 0.40.

Tier A (+30B) is built from bucket 1 plus a **targeted slice of bucket 2's parse-FAIL
multi-function half** (see §3.1), not from "novel vs already-seen" token bookkeeping.

Per-domain build funnels (0e, pod-measured 2026-09-15, kept/scanned docs): L2 16.36M/29.38M
(44% dropped), L3-noexec 15.31M/20.75M, starcoder 6.18M (188 build-decon hits, drop 0.025%),
math_owm 4.13M, keep_p1 3.06M. Pool rows are 4097-token packed rows, a different unit from
corpus kept-doc counts.

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
new domain-name refs where needed — never edit `mix_v41_gate.json` or `mix_v41_r3.json`,
both frozen). Composition from §2 bucket 1 (fully novel) + a bucket-2 slice, ≈30B:

- code ~24.2B (~81%): L2 unseen tail 11.56 + starcoder 7.95 + rp1t 0.38 + **noexec 4.3**.
  The 4.3B noexec is **not** "novel + some stub-touched": it is (a) all ~3.0B new-uuid
  problems, plus (b) ~1.3B sampled from the **parse-FAIL multi-function / long-context half**
  of the same-uuid pool (post-exclusion, fence/REPL-stripped). Hold the ~11.5B parse-ok
  same-uuid body for Tier B — it mostly re-contains the seen 365-tok def and has the lowest
  marginal novelty. noexec stays capped at ~14% of the tier, now on **distribution/format
  risk** (prose+code render, same-uuid problem concentration; L2/starcoder cover primitive
  diversity better), not on an "already seen" basis. Same-uuid tokens keep the ~0.5 lift
  discount in §5. L2 and starcoder (~19.5B) are the anchors.
- math 3.4B (~11%): math_owm unseen tail (digit/modular/sequence primitives).
- English retention 1.5B (~5%): en_c4 tail 1.1 + a 2nd cot pass 0.4 (exhausted but small,
  high-signal).
- Chinese 0.77B (~3%): zh_c4 0.5 + zh_wiki 0.27 (Chinese open-acc is at floor).

Code ~81% matches r3 main mix's own ~85% code share (stub .47 + keep_p1 .264 + L2 .12), so this
is not a narrower distribution than produced r3 — the difference is all rows are novel and L2/
starcoder natural code replace 3×-repeated stub/keep_p1. Mix, do not SFT: the phi run showed a
narrow pack costs −2.7 LAMBADA-EN / −3.2 ARC-E. Retention gate at the half point is
LAMBADA-EN ≥ 21 (r3 21.93); if the 8% language share is judged too thin to hold that, raise
English by reusing more en_c4/cot (already tokenized) at the expense of the noexec slice.

- **Epochs: 1** over unseen rows (cot reuse aside). A second epoch is a separate decision.
- Decontamination — two stages, and they are NOT interchangeable (0e + 3b evidence):
  1. **Build-time gate (`filters/decontam_ngram.py`, normaliser fp 0aefe6a2) scans HE-164 and
     `data/eval/mbpp_holdouts.jsonl` — that file is 974 MBPP-*train* problems
     (ids `mbpp-train-0..973`), and has ZERO task_id intersection and only 190/427
     prompt-prefix overlaps with the sanitized-427 eval set** (3b, measured 2026-09-15).
     So a pool that passed this gate is decontaminated against HE and MBPP-train, **not
     certified clean against MBPP-427.**
  2. **What actually protects the scored 427 is the post-hoc r3 union rescan**
     (`runs/contam_mbpp427_r3.json` → 89 tasks excluded, CLEAN 338; whitespace-13-gram key =
     prompt text and code+newline-joined test_list). Starcoder, rp1t and the zh domains never
     entered r3, so they were never in that rescan. **Hard prerequisite for Tier A: run the
     same 427/HE-164 13-gram rescan over every newly introduced pool (starcoder, rp1t, zh, and
     the selected noexec slice) and score the eval as CLEAN against whatever union comes
     out** — do not assume 156/338 hold. This is CPU-only (the r3 six-domain rescan was a
     minutes-class offline job), no GPU, and must land before the mix is accepted by
     `_assert_mix_domains`.
  3. **noexec needs the same holdout exclusion r3's stub had — it has NO `.excl` cache.** r3
     read `tokens_code_ultra_l3_stub_dc.excl56d12083dc30bdf2.pt` (the 273,879-doc
     `phi_l3_stub_holdout_manifest` removed); the noexec pool ships only as the full
     `tokens_code_ultra_l3_noexec_dc.pt`, so feeding it raw reintroduces the exact holdout
     full solutions. Build an `excl56d` noexec variant (drop the same uuids) **before**
     selecting the 4.3B. Separately, strip the ~5% markdown-fence / REPL wrappers found in the
     parse-fail slice (keep multi-def modules — do NOT filter on `ast.parse` failing, which
     deletes the highest-novelty long multi-function context). The sha1 holdout channel is a third, independent guard for verbatim
     eval/holdout docs: `data/eval/holdout_hashes.txt` carries length header `# n:305007`
     (305,007 registered hashes, `# fp:0dbff3db…`); building the r3 stub pool against it
     excluded 273,879 documents (asserted PASS in `runs/verify_stub_exclusion.py`). The two
     numbers are different quantities (registry size vs docs dropped on one build). A new eval
     set enters `datagen/holdout.py` REGISTRY first (`check_eval_registry_complete`).
  - Whole-domain 13-gram path: `scripts/filter_gate_domains.py --domains a,b` (measured non-
    ultra drop ~0.025%, CPU-cheap); ultra decontaminates inside
    `datagen/ultradata_shards.py --aggregate`. The phi pack D5 scan is the pack-slice template.

### 3.2 Tier B — generation required, +100B

If Tier A plateaus like the anneal did, the hard-gap primitives (fraction/decimal,
palindrome, geometry, dict-aggregate) need targeted material: the ~8.4M eligible existing
textbook/exercise pool (user order 2026-09-14: no new generators; mutate/select from the
existing pool via the mutation gate) plus, only if that pool's primitive coverage is
insufficient, a user-approved generation. Primitive-targeted data must itself be
13-gram-scanned against HE/MBPP/holdout before packing (the phi pack scan is the template:
21 rows / 3 union tasks, 0 new).

### 3.3 Training mechanics (no script changes without a PR)

- Resume: `train.py` resume loads weights and the per-domain cursor; `build_mix`
  (`train.py:2672`) calls `_assert_mix_derived_against` (`train.py:2597-2669`), which compares
  every domain the NEW mix names against the checkpoint's `row_cursor`/`row_cursor_srcfp` and
  refuses on row-count or srcfp mismatch (`train.py:2635-2656`). Two facts make a changed mix
  safe: a domain the mix does not name is ignored (rename precedent, `train.py:2629-2636`), and
  a brand-new domain absent from the checkpoint starts at row 0 (`train.py:2777`, "cursor …
  discarded", and `2713` comment: a discarded domain contributes 0 base by design). So
  starcoder/rp1t/zh start fresh, and reused L2/math/en_c4 MUST be regenerated with
  `--resume-cursor` pointing at ckpt_v41_r3_0914.pt so their `epochs` totals are derived from
  the real cursor (L2 922,846; math 603,515; en_c4 212,401) — not hand-set. Schedule
  recomputes total_steps and prints `SCHEDULE MOVED` (`train.py:3821`); gate before go is the
  #361 lesson — `--dry` join38070 + per-domain cursor match verified by json.load of derived
  files, not rc0.
- LR basis (the only value in this proposal that is a prior, not a measurement): the repo's
  one precedent for continuing already-trained r3 weights at low LR is the stage-2 A/B setting
  `--lr_scale 0.20` with a 137-step cosine warmdown (0.36% of total). The phi SFT used 0.1 but
  with a *fresh* optimizer from step 0, which is a different regime. For Tier A I recommend
  **lr_scale 0.2, no 500-step warmup restart (weights are warm), short cosine warmdown over the
  last ~5–10%**; this is a controller/user ruling before launch because no LR ablation exists
  on a continued-pretrain from r3. Flag it as an assumption in the prereg row.
- Half-point eval cost (uses cards): the existing `runs/e0_n10.sh` is 8 shards, HE+MBPP
  serialised per card, n=10. Measured wall for the E0 run: launch log 03:29 → HE merged 04:23 →
  MBPP merged 06:00, **~2.5 h of the full 8-card block** (training must pause or get an
  exclusive window). A cheaper interim gate — greedy n=1 HE rstrip CLEAN per
  `runs/he_r3_step*_rstrip.log`, single card, minutes — can gate early, but the continue/stop
  decision at +15B must use the paired n=10 + MMLU + LAMBADA-EN, not n=1 (the 36k→37k ±4pt
  swing shows n=1 is noise). Budget two ~2.5h full-block eval windows (+15B, +30B).

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

1. **Tier A go/no-go** (+30B, ~1.7 d train + two 2.5 h n10 eval windows, zero generation,
   exclusive 8-card block, resume r3 step38070, new mix from §3.1, half-point gate). The tokens
   are already packed; the cost is card time and it settles whether broad scale moves the
   coverage count.
2. **Mandatory pre-launch gate (CPU, no GPU):** the 427/HE-164 whitespace-13-gram union rescan
   over starcoder/rp1t/zh and the selected noexec slice (§3.1 decon steps 2-3), and regenerate
   the reused-domain mix rows with `--resume-cursor` at ckpt r3. The build-time gate does not
   cover MBPP-427; CLEAN n is re-derived from this rescan, not assumed to stay 156/338.
3. Pre-authorise the Tier B pool-build (CPU) so it is ready if Tier A plateaus.
4. Rulings needed before launch: LR scale/warmdown (recommend 0.2 / no warmup restart, §3.3),
   card grant in `runs/card_assignment.json`, explicit user go. Nothing here launches anything.

## Appendix — evidence paths (all pod `/work/aupai`, read 2026-09-15)

- r3 ckpt cursor/mix: `ckpt_v41_r3_0914.pt` keys `step` 38070, `total_steps` 38070,
  `cfg.mix = data/mix_v41_r3.json`, `row_cursor`, `row_cursor_srcfp` (dump script used:
  torch.load weights_only=False; values in §2 table).
- r3 mixes: `data/mix_v41_r3.json` (6 domains, weights/epochs in §2),
  `data/mix_v41_r3_anneal.json` (3B: textbooks .5 / stub .3 / keep_p1 .2).
- pools: `/data00/tokens_<domain>.pt` byte sizes (int32 → /4) and `.vocab` (all
  f1f860970d15d623) / `.srcfp` / `.seed`(42) sidecars; values in §2.
- L3 lineage: `data/corpus/code_ultra_l3_noexec_dc/build_corpus_stats.json` (kept 15,308,313,
  26.70B) and `data/corpus/code_ultra_l3_stub_dc/build_corpus_stats.json`
  (`survivor_source_dir …/code_ultra_l3_noexec_dc`, kept 13,682,629, 4.99B).
- decon gate benchmark set: `filters/decontam_ngram.py:45` (MBPP =
  data/eval/mbpp_holdouts.jsonl); holdouts are 974 `mbpp-train-N`, zero task_id intersection
  with `data/eval/sanitized-mbpp.json` 427 (measured), 190/427 prompt-prefix overlaps.
- 427 protection as actually applied: `runs/contam_r3_mbpp427_r3.json` (per-domain union over
  the six r3 domains) → `runs/contam_r3_mbpp_union.json` (89, CLEAN 338, key
  prompt/code+test_list, normaliser fp 0aefe6a2); HE `runs/contam_r3_he_union.json` (8, 156).
- resume semantics: `train.py:2597-2669` (cursor/srcfp/seed refusal; superset accept),
  `train.py:2713`, `train.py:2777` (new domain starts row 0), `train.py:3821` (SCHEDULE MOVED).
- n10 eval cost: `runs/e0_n10.sh` (8 shards, HE+MBPP serial/card); logs `runs/e0_shard0.log`
  from 03:29, merged `data/eval/e0_he_merged…` 04:23 / `e0_mbpp_merged…` 06:00 ≈ 2.5 h.
- gap data: `data/eval/{e0,psft}_{he,mbpp}_merged.n10temp0.2.jsonl`; clustering script
  `scripts/cluster_gap.py` (genB, pod) independently re-derived by 3b.

