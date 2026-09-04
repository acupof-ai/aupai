---
area: corpus and data
owner: 3b
pair: b0
date: 2026-09-04
status: reviewed (pair check b0 done: CD-1/2/3/4 hold; CD-7 added from b0's correction; CD-5/6 accepted)
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
| CD-1 | S2 | zh_web (909 shards, the single largest domain), textbook_30b, wiki_chat carry a minimal 7-key stamp with **no `tokens_config`**, so their stamped `tokens` supply numbers are irreproducible from the stamp. zh_web's 21.29e9 "measured" tokens has no sample size / no convention recorded. b0's pair-check confirmed every value by reopening the stamps; a conforming `build_corpus.py` writer could not have produced this shape (it writes tokens/tokens_status/tokens_config together in one branch, :678-681, and asserts the canonical schema before writing, :687). The producing writer is unidentified — see CD-7. | zh_web stamp keys = {domain, filters, filters_fp, fingerprint, tokens, tokens_status, n_shards}; textbook_30b, wiki_chat identical (b0 reopened, 2026-09-04). zh_web tokens=21293403945, tokens_status="measured", mtime 2026-08-31 11:05. | the 8 canonical domains all carry full `tokens_config` (e.g. cot 14 keys) |
| CD-2 | S2 | No corpus stamp can record which holdout population its domain was built against (the 4-file EVAL_FILES vs the 13-entry REGISTRY): the canonical schema `CANONICAL_STATS_KEYS` has no holdout field. The registry-incident class (a domain built against a stale holdout set, e.g. `control_sft_text_heldout` missing) is undetectable from any stamp. | `datagen/build_corpus.py:593` CANONICAL_STATS_KEYS = 15 keys, no holdout/hold_out/eval entry (b0 counted by ast.parse, 2026-09-04). `grep holdout` over the 9 stamps found only en_c4_stage2, and its hold/eval dict is `{}`. `datagen/holdout.py:55` now has a 13-entry REGISTRY, `EVAL_FILES` derived :230 | none found |
| CD-3 | S3 | code_py_starcoder and code_py_rp1t stamp `kept_tokens: 0` / `kept_chars: 0` while carrying real `kept` rows (6.18M / 0.21M) and real `tokens` (8.74e9 / 0.42e9). A silent schema-zero on 2 of 8 canonical domains; `kept_tokens` reads as "tokens kept" = 0 unless the reader knows it is a chars/1.5 estimate. | code_py_starcoder stamp: kept=6180174, kept_chars=0, kept_tokens=0, tokens=8744830156 (b0 reopened, 2026-09-04); code_py_rp1t 209668/0/0/421239303. `build_corpus.py:647` kept_tokens=int(kept_chars/CT). | code_rp1t (kept_tokens=15357296598), math_owm_stage2, cot all have nonzero kept |
| CD-4 | S3 | `filters_fp` is the sha of ALL `filters/*.py`, not of the per-domain `--filters` profile, so it cannot identify "which filter profile ran." Measured over all 47 pod stamps: `filters_fp` takes only 3 values (33462c13868a2194 / 88ee503b38941bf4 / None) while the `filters` field takes >=8 descriptions; `starcoder-python-ast`, `chatml-render`, `chat-original`, `rp1t-python-ast`, `light` ALL carry 33462c13868a2194. The collision is the NORMAL state, not an edge case (b0, 2026-09-04). The profile is recoverable only from the separate `filters:` field. | `corpus_fingerprint.py:49` `fp_filters` hashes every `.py` in `filters/`, no profile arg; all 47 pod stamps' filters_fp take 3 values | the `filters:` field does record the profile name |
| CD-5 | S2 | `check_corpus_filters_fp` verifies only the `cfg_default("mix")` = `data/mix_500m.json` domains (9). code_rp1t and wiki_chat, named only by `mix_30b_stage2`, are outside the guard's population: their `filters_fp`/stamp are never mismatch-checked. A filters/ edit that drifted them would go uncaught. | `scripts/harness.py:4100` `read_mix(cfg_default("mix"))`; mix_500m domains exclude code_rp1t, wiki_chat. | none found (their current filters_fp matches live, so nothing is wrong now; the gap is coverage) |
| CD-6 | S3 | 24 `web_cci3_p*` corpus dirs (21 stamped, some p0/p22 unstamped) under `data/corpus/` are named by NO mix; 115 loose `batch_*.jsonl` sit directly under `data/corpus/`. Dead corpus footprint (~80G+) not in any training mix; the old-layout batch files are unowned. Controller ruling 2026-09-04: record-only, not to be touched during the audit. | `find data/corpus -name web_cci3_p*` = 24 dirs, `grep web_cci3 data/mix_*.json` = 0; 115 loose batch files present. | none of the five mixes mention web_cci3 |
| CD-7 | S2 | `CANONICAL_STATS_KEYS` is a schema for ONE of FOUR writers of `build_corpus_stats.json`, not for the file. `datagen/build_corpus.py` asserts it at three write sites (:687, :977, :1776); `datagen/build_cot.py:97`, `datagen/code_dedup_build.py:162`, `datagen/build_code_tests_v1.py:408` each write their own dict literal and none calls the assertion (`grep -c _assert_canonical_stats` = 0, 0, 0). A reader consulting `CANONICAL_STATS_KEYS` to learn what a stamp guarantees learns nothing about three-quarters of the writers — the same population-before-property shape as CD-2 (b0's correction, 2026-09-04, adopted). | `grep -c _assert_canonical_stats datagen/build_cot.py datagen/code_dedup_build.py datagen/build_code_tests_v1.py` = 0, 0, 0; build_cot.py:90-96 writes srcfp/criterion/schema/docs_in/docs_kept/reject_checks/tokens_kept/check4 of which only n_shards/filters are canonical keys | build_corpus.py's own stamps are asserted canonically |

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
   in no mix), or are they archived intentionally? (record-only this audit)
5. CD-7: should `CANONICAL_STATS_KEYS`'s assertion be enforced at the three non-asserting
   writers (build_cot/code_dedup_build/build_code_tests_v1), or is build_corpus.py intended to
   be the only canonical writer? And: which writer produced zh_web's 7-key 2026-08-31 stamp
   (needs build history; a conforming writer could not have — b0, 2026-09-04)?

## Pair check (b0, 2026-09-04)

Recomputed four findings independently — CD-1, CD-2 and CD-4 as 3b asked, plus CD-3 as the
controller directed. Every stamp below was opened on the pod in this pass, not read from
3b's quoted numbers. **All four hold.** Two are sharper than published, and one of 3b's
supporting statements is wrong in a way that does not touch its finding.

### CD-1 — HOLDS
`python3 -c` over `data/corpus/<d>/build_corpus_stats.json` on the pod, three domains:

    zh_web         7 keys ['domain','filters','filters_fp','fingerprint','n_shards','tokens','tokens_status']
    textbook_30b   7 keys  (identical key set)
    wiki_chat      7 keys  (identical key set)

`tokens_config` absent from all three, so `zh_web`'s `tokens = 21293403945` with
`tokens_status = 'measured'` records no sample size and no counting convention. Contrast
`cot`, 14 keys, opened in the same pass. Confirmed: the 21.29e9 figure is NOT reproducible
from the stamp. Note the word `measured` is doing real damage here — it is the same value
`tokens_status` carries on domains that DO record their method, so the field cannot
distinguish "counted" from "extrapolated".

### CD-2 — HOLDS, and the two counts check by AST rather than by grep
`CANONICAL_STATS_KEYS` parsed out of `datagen/build_corpus.py:593`: **15 keys**, and the
holdout-shaped subset is empty (searched for `holdout`, `eval`, `contam`):

    ['domain','reasons','kept','kept_chars','kept_tokens','filters','workers','n_shards',
     'filters_fp','fingerprint','near_dedup','near_dedup_note','tokens','tokens_status','tokens_config']

`REGISTRY` counted by `ast.parse` on `datagen/holdout.py` (not by grepping `"path"`, which
would also match nested dicts): **13 entries**. `EVAL_FILES` at :230 is
`[os.path.join(ROOT, e["path"]) for e in REGISTRY.values()]` — derived, as published.
So no stamp can record which holdout population a domain was built against. Confirmed.

### CD-3 — HOLDS, four values quoted from the reopened stamps
Both stamps opened on the pod:

    code_py_starcoder  kept=6180174  kept_chars=0  kept_tokens=0  tokens=8744830156
    code_py_rp1t       kept=209668   kept_chars=0  kept_tokens=0  tokens=421239303

Exactly 3b's numbers. Both are 15-key stamps and both carry `tokens_config` naming a
3-shard byte extrapolation, so the zeroed fields sit beside a populated one — which is what
makes the zero readable as a value rather than as an absence.

### CD-4 — HOLDS, and it is broader than published
`datagen/corpus_fingerprint.py:49` `fp_filters` hashes `name + "\0" + sha256(content)` for
every `filters/*.py`, sorted, and takes nothing else: no profile, no `--filters` argument,
no per-domain parameter. Published as "identical across domains with different filters".
Measured over **all 47 stamps on the pod**, not the 11 audited domains: `filters_fp` takes
**three** values (`33462c13868a2194`, `88ee503b38941bf4`, `None`) while the `filters` field
takes at least eight distinct descriptions. `starcoder-python-ast`, `chatml-render`,
`chat-original`, `rp1t-python-ast` and `light` ALL carry `33462c13868a2194`. So the
collision is not an edge case between two similar profiles — it is the normal state.

### CD-1's mechanism, corrected and strengthened — three writers, none of which asserts
CD-1's evidence column cites `datagen/build_corpus.py:664` as *requiring* `tokens_config`.
It does not require it, and what is actually there is worse.

`build_corpus.py` writes `tokens`, `tokens_status = "measured"` and `tokens_config` in ONE
branch, :678-681, all three together after a successful `count_shards`. So a stamp carrying
`tokens` and `tokens_status = "measured"` **without** `tokens_config` cannot have been
written by that branch at all — and zh_web's is exactly that shape (7 keys, `tokens`
21293403945, `tokens_status` "measured", mtime 2026-08-31 11:05 on the pod). `build_corpus.py`
does call `_assert_canonical_stats` at :687 immediately before writing, so a stamp it
produced would have been checked.

`data/corpus/*/build_corpus_stats.json` has at least **four** writers:
`datagen/build_corpus.py`, `datagen/build_cot.py:97`, `datagen/code_dedup_build.py:162`,
`datagen/build_code_tests_v1.py:408`. Grepped for the assertion in each of the latter three:
**zero occurrences in all three** (`grep -c _assert_canonical_stats` → 0, 0, 0). Each
assembles its own dict literal — `build_cot.py:90-96` writes `srcfp`, `criterion`, `schema`,
`docs_in`, `docs_kept`, `reject_checks`, `tokens_kept`, `check4`, of which only `n_shards`
and `filters` are canonical keys at all.

So CD-1's finding is right and its severity is understated: the canonical schema is not a
schema for this file, it is a schema for one of four writers. A reader who checks
`CANONICAL_STATS_KEYS` to learn what a stamp guarantees learns nothing about the three
quarters of writers that never consult it. Same population-before-property shape as CD-2 and
as the audit charter's principle 3 — the guarded set is narrower than the set the property is
about.

I did not identify which writer produced zh_web's 7-key stamp; that is 3b's area and needs
the 2026-08-31 build history, not the current tree. What is established here is that a
conforming writer could not have produced it and that three non-asserting writers exist.

### Not checked by me
CD-5 (`check_corpus_filters_fp` population) and CD-6 (dead `web_cci3_p*` dirs and loose
`batch_*.jsonl`) — outside the four assigned, and CD-6 is accepted record-only.
