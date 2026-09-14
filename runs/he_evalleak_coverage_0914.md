# r3 final pre-anneal eval-leak coverage (2026-09-14)

Whitespace-13 word-gram containment (normaliser fp 0aefe6a2aa5e130f) of every registered generative/MC holdout vs all r3 domains. 31,111 problems, 12 corpus domains, shard-level fork-pool scan (660 s). Machine artifacts: `runs/contam_evalscreen_r3_final.json` (per-set raw-hit ids, excluded ids, scoring-clean ids, per-domain row counts) and `runs/evalscreen_examples.json` (example shared grams); per-problem attribution pods `runs/evalscreen_ids_<domain>.json`.

Every one of the 134 raw hit problems was hand-adjudicated by an example shared 13-gram. **Zero shared grams contain CJK** — no Chinese question or instruction was verbatim copied into any domain.

## Coverage matrix

| eval set (n) | screened vs all r3 _dc? | raw ws13 hits | adjudication | scoring CLEAN | fact before |
|---|---|---:|---|---:|---|
| HumanEval 164 | yes (#325) | 8 problems | excluded | **156** | cont.decon_markdown_comment_strip_0913 |
| MBPP-427 | yes (#344) | 89 (71 solution / 24 prompt) | excluded; FP 0/5000 | **338** | cont.r3_mbpp427_union_0914 |
| code_holdout_500 | yes, new | 65 | **FP**: 3 idioms only — Euclid gcd (37 problems), `sqrt`/`**0.5` primality loop (28); synthetic generator reused canonical templates; no Chinese instruction copied | **500** | bigram scans vs crawl/SFT only |
| code_holdout_v2_500 | yes, new | 43 | **FP**: canonical pair-loop idiom (34) + coincidental sorted-number test vectors inside unrelated sort docs; 0 Chinese instructions | **500** | ws13 vs math_owm/en_c4 =0; t51 absent-by-construction |
| math_test_500 | yes, new | 2 | **FP**: shared LaTeX solution steps (`2x = 26 ...`, order-of-ops line), different problems | **500** | ws13 vs math_owm/en_c4 =0 |
| math_hard_eval_1k 1032 | yes | 0 | retired ruler (cont.math_hard_v1_void) | n/a retired | cont.math_hard_v1_void |
| math_hard_eval_v2 1080 | yes, new | 0 | verbatim clean | **1080** | generator-disjoint + ws13 math_owm/en_c4 |
| gsm8k_zh 7473 | yes, new | 0 | verbatim clean (incl. zh_c4/zh_wiki) | **7473** | bigram vs web_hq, 1 FP |
| lambada_en 5153 | yes, new | 2 | resolved: lamben:3235 generic-sentence FP (label "Hamnet" absent); lamben:360 a starcoder ML-example doc contains the verbatim mask prefix but **truncates exactly before the target "driving"** — prompt-only, answer absent | **5153** | ws13 math_owm/en_c4 ≈0 |
| ceval 1050 | yes, new | 0 | verbatim clean | **1050** | none |
| cmmlu 11582 | yes, new | 2 | **FP**: generic LaTeX matrix rows (`1 & 0 \\ 1 & 1...`), no question text | **11582** | none |
| arc-easy 2241 | yes, new | **20** | **REAL question-prose overlap**: 20 distinct science questions verbatim (19 math_owm_stage2 / 26 rows, 1 en_c4); no answer key | **2221** | none |
| MMLU English (14,042) | yes, new (fetched) | **478** | **auxiliary metric, full-excluded** (conservative, arc-standard): 425 distinct shared grams; math/physics/stats/CS textbook questions (statistics 108/216=50%, physics 62/151=41%, hs math 90/270=33%) + humanities source passages; **52 carry the ≥13-word correct answer in the same doc**, 60 repeat in ≥10 corpus docs (textbook), 276 single-row (famous passages + isolated questions, not sub-divided); MC choices <13 words make answer-grams mechanically absent, stated | **13564** | none (fetched 2026-09-14) |
| api_cloze / novel_ops probes | n/a | — | their construct is pool seen/unseen; frozen-set facts stand | — | cont.novel_ops_* | — |

Per-domain raw rows (code500/codev2 idiom hits dominant in UltraData): l3_noexec 2548 rows {code500 65p, codev2 40p}; l3_stub 531 {19p,35p}; l2 877 {65p,3p}; keep_p1 28 {28p,2p}; starcoder 40 {28p,38p, lamben 1}; rp1t 1 {19p}; math_owm 70 {arc 19p, code500 19p, codev2 6p, cmmlu 2p}; en_c4 2 {arc 1, lamben 1}; cot 3 {math500 2p}; textbook_v41 0; zh_c4 0; zh_wiki 0.

## Decisions (fb, 2026-09-14)

1. **arc-easy**: exclude 20 ids (`excluded_ids`) → denominator **2221**. Auxiliary metric, not in the 30% primary criterion (primary HE rstrip CLEAN 156; secondary MBPP 338). Verbatim question prose gives familiarity even without the answer key; 0.9% changes no result but prevents inflation.
2. **MMLU English**: fetched to pod `data/eval/mmlu_test.jsonl` (14,042 rows, sha1 d9c4079e; pod outbound to HF dead, laptop direct to huggingface.co + tn push). Full-exclude 478 → denominator **13564**, same conservative standard, auxiliary-only. Not added to the holdout registry (would push holdout_hashes.txt past the 5 MiB tracked-blob cap); `eval/mmlu.py` asserts the screened file path+sha1 directly (option B). SFT pool decontaminates independently; registry split deferred until MMLU enters an SFT exclusion set.
3. **lambada_en**: resolved — 3235 FP, 360 prompt-only (answer word absent); no exclusion, denominator 5153.

Code/math holdouts need no exclusion: all raw hits are canonical-template collisions the synthetic generators share with open code corpora, confirmed by the 3-idiom/0-CJK collapse.
