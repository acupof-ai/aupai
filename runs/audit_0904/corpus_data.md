---
area: corpus and data
owner: 3b
pair: b0
date: 2026-09-04
status: partial (first report)
---

# Audit: corpus and data (3b, 2026-09-04)

## 1. Scope

Every domain named by `data/mix_200m_*.json`, `data/mix_30b*.json`, and (the guard's default)
`data/mix_500m.json`. Union across the five files: 11 distinct corpus domains.

| domain | named by | pod dir |
|---|---|---|
| code_rp1t | mix_30b_stage2 | code_rp1t (235 sh, 23G) |
| code_py_starcoder | mix_200m/500m/8b | code_py_starcoder (283/284 sh, 28G) |
| code_py_rp1t | mix_200m/500m/8b | code_py_rp1t (15/16 sh, 1.5G) |
| math_owm_stage2 | all five | math_owm_stage2 (206 sh, 21G) |
| en_c4_stage2 | all five | en_c4_stage2 (83 sh, 8.1G) |
| cot | all five | cot (13 sh, 1.3G) |
| zh_web | mix_200m/500m/8b | zh_web (909 sh) |
| textbook_30b | mix_200m/500m/8b, 30b_stage2 | textbook_30b (79 sh, 7.6G) |
| wiki_chat | mix_30b_stage2 | wiki_chat (12 sh) |
| chatml | mix_200m/500m/8b | chatml (2/3 sh, 164M) |
| chat_qa | mix_200m/500m/8b | chat_qa (2/3 sh, 156M) |

`mix_30b.json` itself names no stamped domain (its `domains` is empty; `_blocked` holds
code_rp1t at 10e9 tokens as a target spec). `data/raw/*` inventoried (see 5).

Deliberately excluded: `web_cci3_p*` (24 dirs, see finding CD-6), loose `batch_*.jsonl`
under `data/corpus/`, `data/corpus/sample/`.

## 2. Method

- Corpus stamps read from pod `data/corpus/<dom>/build_corpus_stats.json`.
- Field meanings read from `datagen/build_corpus.py` locals (CANONICAL_STATS_KEYS :593,
  `_write_stats` :621, `kept_tokens=int(kept_chars/CHARS_PER_TOKEN)` :647, tokens measured
  :664-686) and `datagen/corpus_fingerprint.py` (`fp_filters` :49, `fp_dir` :68).
- Supply cross-check: stamped `tokens` vs `facts/corpus_supply.json#cs.*_landed`.
- Request cross-check: per-domain mix `weight` against `total_rows` and the landed `kept`
  rows. Basis: `build_mix` computes `want = int(total_rows * weight)`; the weight is on rows.
- Guard population read from `scripts/harness.py check_corpus_filters_fp` (:4082), which reads
  `cfg_default("mix")` = `data/mix_500m.json`.
- Instrument self-test on real dirs: `fp_dir` re-derived for `zh_web` and `code_rp1t`
  matches the stamped fingerprint exactly (MATCH both) — the fingerprint instrument
  works on the shards it covers.

## 3. Population counts

- 11 audit domains; all 11 present as pod corpus dirs with a `build_corpus_stats.json`.
- 8 carry the canonical 15-key schema; 3 (zh_web, textbook_30b, wiki_chat) carry a
  7-key minimal schema.
- `facts/corpus_supply.json` `*_landed` present for all 11 (via cs.*; math via cs.math_owm_landed).
- 9 domains' stamps read in full; supply cross-checked for 9 (zh_web/textbook/wiki_chat
  minimal stamps cross-checked by token only).

## 4. Findings

| id | sev | claim | evidence | contradicts |
|---|---|---|---|---|
| CD-1 | S2 | zh_web (909 shards, the single largest domain), textbook_30b, wiki_chat carry a minimal 7-key stamp with **no `tokens_config`**, so their stamped `tokens` supply numbers are irreproducible from the stamp. zh_web's 21.29e9 "measured" tokens has no sample size / no convention recorded. | zh_web stamp keys = {domain, filters, filters_fp, fingerprint, tokens, tokens_status, n_shards}; textbook_30b, wiki_chat identical. `datagen/build_corpus.py:664` requires `tokens_config` for a canonical stamp. | the 8 canonical domains all carry full `tokens_config` (e.g. cot 14 keys) |
| CD-2 | S2 | No corpus stamp can record which holdout population its domain was built against (the 4-file EVAL_FILES vs the 13-entry REGISTRY): the canonical schema `CANONICAL_STATS_KEYS` has no holdout field. The registry-incident class (a domain built against a stale holdout set, e.g. `control_sft_text_heldout` missing) is undetectable from any stamp. | `datagen/build_corpus.py:593` CANONICAL_STATS_KEYS = 15 keys, no holdout/hold_out/eval entry. `grep holdout` over the 9 stamps found only en_c4_stage2, and its hold/eval dict is `{}`. `datagen/holdout.py:55` now has a 13-entry REGISTRY, `EVAL_FILES` derived :230 | none found |
| CD-3 | S3 | code_py_starcoder and code_py_rp1t stamp `kept_tokens: 0` / `kept_chars: 0` while carrying real `kept` rows (6.18M / 0.21M) and real `tokens` (8.74e9 / 0.42e9). A silent schema-zero on 2 of 8 canonical domains; `kept_tokens` reads as "tokens kept" = 0 unless the reader knows it is a chars/1.5 estimate. | code_py_starcoder stamp: kept=6180174, kept_chars=0, kept_tokens=0, tokens=8744830156. `build_corpus.py:647` kept_tokens=int(kept_chars/CT). | code_rp1t (kept_tokens=15357296598), math_owm_stage2, cot all have nonzero kept |
| CD-4 | S3 | `filters_fp` is the sha of ALL `filters/*.py`, not of the per-domain `--filters` profile, so it is identical (33462c13868a2194) across domains with different filters (light / starcoder-python-ast / rp1t-python-ast / chatml-render / chat-original). A reader expecting `filters_fp` to identify "which filter ran" would be misled; the profile is recoverable only from the separate `filters:` field. | all 11 stamps `filters_fp=33462c13868a2194`; `filters:` differs by domain. `corpus_fingerprint.py:49` `fp_filters` hashes every `.py` in `filters/`, no profile arg. | the `filters:` field does record the profile name |
| CD-5 | S2 | `check_corpus_filters_fp` verifies only the `cfg_default("mix")` = `data/mix_500m.json` domains (9). code_rp1t and wiki_chat, named only by `mix_30b_stage2`, are outside the guard's population: their `filters_fp`/stamp are never mismatch-checked. A filters/ edit that drifted them would go uncaught. | `scripts/harness.py:4100` `read_mix(cfg_default("mix"))`; mix_500m domains exclude code_rp1t, wiki_chat. | none found (their current filters_fp matches live, so nothing is wrong now; the gap is coverage) |
| CD-6 | S3 | 24 `web_cci3_p*` corpus dirs (21 stamped, some unp0/p22 unstamped) under `data/corpus/` are named by NO mix; 115 loose `batch_*.jsonl` sit directly under `data/corpus/`. Dead corpus footprint (~80G+) not in any training mix; the old-layout batch files are unowned. | `find data/corpus -name web_cci3_p*` = 24 dirs, `grep web_cci3 data/mix_*.json` = 0; 115 loose batch files present. | none of the five mixes mention web_cci3 |

## 5. data/raw inventory

`data/raw` = 263G total. Present (provenance retained): rp1t_github (30G, -> code_rp1t),
ms_starcoder_py (22G, -> code_py_starcoder), cot_open_thoughts (1.1G), skywork_or1 (785M),
hf_numma (+_jsonl 2.5G), hf_om2 (empty), fineweb2_cmn, fable5_2m/sft/traces, gsm8k_chinese,
hf_finemath_4plus, ms_finemath_4plus, mathinstruct_zh, kyara-chinese-math-sft, hf_numma,
rp1t_arxiv, rp1t_c4, plus manifest-only entries (rp1t_github_manifest.txt 8K, rp1t_c4 32K,
rp1t_arxiv 8K, ms_starcoder 4K). The web source raw for zh_web is under `data/raw` as
`web_cci3_p*`? — NOT enumerated against zh_web's 909 shards this pass (the zh web raw
relation to the stamped `zh_web` dir is a blind spot, see 6).

## 6. Blind spots

- zh_web's raw source: the 909-shard stamped `zh_web` dir's upstream raw lineage was not
  traced this pass (whether it is web_cci3_p* or another source).
- `fingerprint` matches shards verified on 2 of 11 domains (zh_web, code_rp1t); the other 9
  not re-derived this pass.
- Rows requested vs supplied (CD under-supply) checked by stamp `kept` only; zh_web /
  textbook_30b / wiki_chat have no `kept` to compare (tied to CD-1).
- No tokenizer re-run of the 3 minimal domains' `tokens` (their sample not recoverable
  from the stamp, which is CD-1).

## 7. Open questions for the controller

1. CD-1 confirms 3 minimal stamps (incl. largest domain). Are these domains on the
   rebuild list to canonical-schema re-stamp, or is zh_web's 21.29B accepted un-re-factored?
2. CD-2: should `CANONICAL_STATS_KEYS` gain a `holdout`/`held_out_registry_fp` field so a
   rebuilt domain records which REGISTRY it was built against?
3. CD-5: is code_rp1t/wiki_chat's filters_fp coverage gap acceptable until mix_30b_stage2
   becomes a default, or should the guard union all mix-named domains?
4. CD-6: are the 24 web_cci3_p* dirs + 115 loose batch files deletion candidates (they are
   in no mix), or are they archived intentionally?