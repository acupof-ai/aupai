# Corpus rebuild plan — 2026-08-29 (for aupai-fb approval, then execute)

User decision: rebuild, not archive. Drop the bad data; keep only clean. And the
10.6GB of provenance-unknown sources are to be replaced by re-derivable public
sources, not shelved.

## Principle

Every domain ends up sourced from a reproducible public source or a known
in-repo fetcher/generator. After this, MANIFEST.tsv's frozen tier collapses to
~0: everything becomes `fetched` or `derived`. The two eval holdouts stay
`eval`/archived-compatible.

## What is being dropped (measured bad)

- `belle`: 38.7% defective (5-round manual sample, p=2.87e-05), all four
  failure classes pass eqcheck, so no string check can rescue it. Gone.
- `school_math_r1_zh.jsonl`: another copy of belle. Gone.
- `data/corpus/math/*.jsonl` shards that originate from belle / school_math_r1_zh:
  rebuilt. (mxode/gsm8k_zh shards are clean — 0 defects over 17/18 manual rows —
  and stay.)

## Per-domain target

| domain | today | source (plan) | reproducible? | notes |
|---|---|---|---|---|
| web | 32G | **fineweb2** `cmn_Hani` (preferred) — via `build_corpus.py --source fineweb2`, 3e9 token target | yes (HF) | replaces prov-unknown `pretrain_full.jsonl` + local FW2 parquets; pick over skypile so `metrics against that source` stay comparable to the last clean build. |
| math | 534M | drop belle/school_math; keep `en_math_text.jsonl` (clean, prov-unknown — see decision 2) + `mxode`/`gsm8k_zh` (fetched) + `math_short_v8` synthetic | partially | the pure-math bulk shrinks; volume gap is acceptable at this model size (math <5% of mix), and synthetic math_short_v8 covers the L3/4 band. |
| code | 222M | **`skypile`**-filtered-for-code OR keep `code_filtered.jsonl` (see decision 2) | partial | code_filtered.jsonl has no producer recorded; if we must be fully reproducible it has to go, and the cleanest replacement is skypile narrowed to code-like docs — higher effort. |
| en | 775M | **cosmopedia** (ModelScope OpenCSG/Chinese-Cosmopedia, re-fetch via the ModelScope path `fetch_data.py` already uses) + keep `en_textbook.jsonl` per decision 2 | partial | cosmopedia_extra re-fetchable; en_textbook prov-unknown. |
| chat | 156M | unchanged — `coig` + `alpaca_gpt4_zh` are fetched/known | yes | no bad data found. |

## Decisions I need from you before downloading

1. **web source**: fineweb2-cmn (dfault, comparable) vs skypile (pure-zh, larger
   but different distribution). I recommend **fineweb2** for continuity.
2. **the three prov-unknown but NOT-measured-bad files** — `code_filtered.jsonl`,
   `en_textbook.jsonl`, `en_math_text.jsonl`:
   - (a) keep as frozen (they weren't flagged by the manual audit), OR
   - (b) rebuild them too (fully reproducible, but code/en add effort and
     volume risk).
   User's instruction was "drop the bad data, keep clean" — these three weren't
   bad, so I lean **(a) keep**, accepting they remain frozen-tier. You/owner call.
3. **math volume**: after dropping belle, math domain shrinks (~1/3). Accept the
   shrink (recommended — math is <5% of mix, synthetic covers the hard band) or
   add a replacement math source (e.g. OpenWebMath/MathInstruct). I recommend
   accept the shrink for this round.

## Executable steps (after approval)

1. `build_domains.sh` edits for the chosen sources (frozen/belle paths removed).
2. Run small domains (code/en/chat/math) + web with the new sources; exclude
   the old `data/corpus/{math}/*.jsonl` so stale belle-derived shards aren't kept.
3. Mandatory post-build gate (my own near-dup harness, now a hard requirement):
   - holdout filter + cross-domain exact dedup (build_corpus already does),
   - **≥0.8 near-dup scan of the rebuilt corpus against BOTH eval sets**
     (math_hard_eval_1k + math_test_500), same-answer-aggregated, report pairs.
4. Re-tokenize (`build_tokenizer.py --force`; new vocab → all old packs void via
   the new `vocab_id` fingerprint) + `check_mix.py`.
5. **Anchor baseline**: immediately re-run the SAME SFT mix through `sft_math.c.`
   to reproduce the current math-hard 3.6% against the new corpus, so every
   historical EXPERIMENTS.md number regains a comparable reference.
6. Update `data/PROVENANCE.md` + `data/MANIFEST.tsv` (frozen tier collapses).

## Download / time guesstimate

Fineweb2-cmn at 3e9 tokens ≈ a few GB of parquet pulled through the mirror; the
`iter_source` path downloads one shard at a time and deletes it. Estimate 40-70
min for web at ~3.4K docs/s. Small domains are fast (<10 min each). Total wall
time ~1-1.5h before tokenize.

Approve / adjust, then I start.